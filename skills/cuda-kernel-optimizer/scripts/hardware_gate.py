#!/usr/bin/env python3
"""Strict CUDA hardware/tool discovery gate with bounded retries."""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


EXIT_HARDWARE_UNAVAILABLE = 3


def _run(cmd: list[str], timeout: int = 15) -> tuple[int, str, str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                encoding="utf-8", errors="ignore")
        return result.returncode, result.stdout or "", result.stderr or ""
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, "", str(exc)


def _first_executable(candidates: list[str]) -> str | None:
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate) or candidate
        resolved = os.path.abspath(resolved)
        if resolved in seen:
            continue
        seen.add(resolved)
        if os.path.isfile(resolved) and os.access(resolved, os.X_OK):
            return resolved
    return None


def discover_tools() -> dict[str, str | None]:
    cuda_bins: list[str] = []
    for name in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(name, "").strip()
        if root:
            cuda_bins.append(os.path.join(root, "bin"))
    cuda_bins.extend(["/usr/local/cuda/bin"])
    cuda_bins.extend(sorted(glob.glob("/usr/local/cuda-*/bin"), reverse=True))
    cuda_bins.extend(sorted(glob.glob("/opt/cuda*/bin"), reverse=True))

    ncu_candidates = ["ncu"] + [os.path.join(p, "ncu") for p in cuda_bins]
    ncu_candidates.extend(sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"), reverse=True))
    ncu_candidates.extend(sorted(glob.glob("/opt/nvidia/nsight-compute/*/target/linux-desktop-*/ncu"), reverse=True))
    return {
        "nvcc": _first_executable(["nvcc"] + [os.path.join(p, "nvcc") for p in cuda_bins]),
        "ncu": _first_executable(ncu_candidates),
        "cuobjdump": _first_executable(["cuobjdump"] + [os.path.join(p, "cuobjdump") for p in cuda_bins]),
        "nvidia_smi": _first_executable(["nvidia-smi", "/usr/bin/nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi"]),
    }


def _find_cutlass_include() -> str | None:
    candidates = []
    for name in ("CUTLASS_INCLUDE_DIR", "CUTLASS_PATH"):
        value = os.environ.get(name, "").strip()
        if value:
            candidates.extend([value, os.path.join(value, "include")])
    candidates.extend(sorted(glob.glob("/usr/local/cutlass*/include"), reverse=True))
    candidates.extend(["/usr/local/cutlass/include", "/opt/cutlass/include"])
    for path in candidates:
        if os.path.isdir(os.path.join(path, "cutlass")) and os.path.isdir(os.path.join(path, "cute")):
            return os.path.abspath(path)
    return None


def _python_dependencies() -> dict:
    result = {}
    for name in ("torch", "triton"):
        try:
            module = importlib.import_module(name)
            entry = {"available": True, "version": getattr(module, "__version__", "unknown")}
            if name == "torch":
                entry["cuda_available"] = bool(module.cuda.is_available())
            result[name] = entry
        except Exception as exc:
            result[name] = {"available": False, "version": None, "error": str(exc)}
    return result


def _probe_gpu(tools: dict[str, str | None]) -> dict:
    smi = tools.get("nvidia_smi")
    if smi:
        fields = "index,name,compute_cap,memory.total,driver_version"
        rc, out, err = _run([smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
        if rc == 0 and out.strip():
            gpus = []
            for line in out.splitlines():
                parts = [part.strip() for part in line.split(",")]
                if len(parts) >= 5:
                    major_minor = parts[2].replace(".", "")
                    gpus.append({"index": int(parts[0]), "name": parts[1],
                                 "compute_capability": parts[2], "sm_arch": f"sm_{major_minor}",
                                 "total_memory_mb": int(float(parts[3])), "driver_version": parts[4]})
            if gpus:
                # nvidia-smi visibility is necessary but not sufficient; force CUDA context creation below.
                runtime = _probe_cuda_runtime(tools.get("nvcc"))
                if runtime.get("available"):
                    return {"available": True, "gpus": runtime.get("gpus") or gpus,
                            "source": "cuda_runtime", "nvidia_smi_gpus": gpus}
                torch_runtime = _probe_torch_runtime()
                if torch_runtime.get("available"):
                    return torch_runtime
                return {"available": False, "gpus": gpus, "source": "nvidia-smi-only",
                        "error": runtime.get("error", "CUDA runtime context unavailable")}
        smi_error = (err or out).strip()[:500]
    else:
        smi_error = "nvidia-smi not found"
    runtime = _probe_cuda_runtime(tools.get("nvcc"))
    if runtime.get("available"):
        return runtime
    torch_runtime = _probe_torch_runtime()
    if torch_runtime.get("available"):
        return torch_runtime
    return {"available": False, "gpus": [],
            "error": runtime.get("error") or torch_runtime.get("error") or smi_error}


def _probe_torch_runtime() -> dict:
    try:
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            return {"available": False, "error": "torch CUDA runtime unavailable"}
        torch.cuda.init()
        gpus = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            major, minor = torch.cuda.get_device_capability(index)
            gpus.append({"index": index, "name": props.name,
                         "compute_capability": f"{major}.{minor}", "sm_arch": f"sm_{major}{minor}",
                         "total_memory_mb": props.total_memory // (1024 * 1024)})
        return {"available": True, "gpus": gpus, "source": "torch_cuda_runtime"}
    except Exception as exc:
        return {"available": False, "error": f"torch CUDA context probe failed: {exc}"}


def _probe_cuda_runtime(nvcc: str | None) -> dict:
    if not nvcc:
        return {"available": False, "error": "nvcc not found; CUDA runtime probe cannot be built"}
    source = r'''
#include <cstdio>
#include <cuda_runtime.h>
int main() {
  int count = 0; cudaError_t e = cudaGetDeviceCount(&count);
  if (e != cudaSuccess || count < 1) { std::fprintf(stderr, "%s", cudaGetErrorString(e)); return 2; }
  for (int i=0; i<count; ++i) { cudaDeviceProp p{}; e=cudaGetDeviceProperties(&p,i);
    if(e!=cudaSuccess) return 3; std::printf("%d\t%s\t%d.%d\t%zu\n",i,p.name,p.major,p.minor,p.totalGlobalMem); }
  e=cudaFree(nullptr); return e == cudaSuccess ? 0 : 4;
}'''
    try:
        with tempfile.TemporaryDirectory(prefix="cko-hw-") as td:
            src, exe = Path(td) / "probe.cu", Path(td) / "probe"
            src.write_text(source, encoding="utf-8")
            rc, _, err = _run([nvcc, "-O0", "-o", str(exe), str(src)], timeout=30)
            if rc != 0:
                return {"available": False, "error": f"runtime probe compile failed: {err[-500:]}"}
            rc, out, err = _run([str(exe)], timeout=15)
            if rc != 0:
                return {"available": False, "error": f"CUDA context probe failed: {(err or out)[-500:]}"}
    except OSError as exc:
        return {"available": False, "error": str(exc)}
    gpus = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) == 4:
            cc = parts[2]
            gpus.append({"index": int(parts[0]), "name": parts[1],
                         "compute_capability": cc, "sm_arch": f"sm_{cc.replace('.', '')}",
                         "total_memory_mb": int(parts[3]) // (1024 * 1024)})
    return {"available": bool(gpus), "gpus": gpus, "source": "cuda_runtime"}


def probe_once(*, backend: str, require_torch: bool) -> dict:
    tools = discover_tools()
    errors: list[str] = []
    versions: dict[str, str | None] = {}
    required_tools = ["ncu"] if backend == "triton" else ["nvcc", "ncu", "cuobjdump"]
    for name in required_tools:
        path = tools.get(name)
        if not path:
            errors.append(f"{name} not found")
            versions[name] = None
            continue
        rc, out, err = _run([path, "--version"], timeout=15)
        versions[name] = (out or err).strip().splitlines()[0] if (out or err).strip() else None
        if rc != 0:
            errors.append(f"{name} probe failed rc={rc}: {(err or out)[-300:]}")
        if name == "ncu" and rc == 0:
            query_rc, query_out, query_err = _run([path, "--query-metrics"], timeout=15)
            if query_rc != 0 or not query_out.strip():
                errors.append(f"ncu query failed rc={query_rc}: {(query_err or query_out)[-300:]}")
    for name in set(("nvcc", "ncu", "cuobjdump")) - set(required_tools):
        versions[name] = None

    gpu = _probe_gpu(tools)
    if not gpu.get("available"):
        errors.append(str(gpu.get("error", "CUDA GPU runtime unavailable")))

    python_libs = _python_dependencies()
    if require_torch and not python_libs["torch"].get("available"):
        errors.append("Python dependency torch is not importable")
    elif require_torch and not python_libs["torch"].get("cuda_available"):
        errors.append("Python dependency torch has no usable CUDA backend")
    if backend == "triton" and not python_libs["triton"].get("available"):
        errors.append("Python dependency triton is not importable")
    cutlass_include = _find_cutlass_include()
    if backend == "cutlass" and not cutlass_include:
        errors.append("CUTLASS include tree not found")

    return {"resolved_tools": tools, "tool_versions": versions, "gpu_probe": gpu,
            "python_dependencies": python_libs, "cutlass_include_dir": cutlass_include,
            "errors": errors, "passed": not errors}


def run_gate(*, backend: str = "cuda", require_torch: bool = True,
             attempts: int = 3, delays: tuple[float, ...] = (0.0, 1.0, 2.0)) -> dict:
    records = []
    for index in range(1, max(1, attempts) + 1):
        if index > 1:
            time.sleep(delays[min(index - 1, len(delays) - 1)] if delays else 0)
        record = probe_once(backend=backend, require_torch=require_torch)
        record["attempt"] = index
        records.append(record)
        if record["passed"]:
            break
    last = records[-1]
    gpus = last["gpu_probe"].get("gpus", [])
    smi_gpus = last["gpu_probe"].get("nvidia_smi_gpus", [])
    driver_version = next((gpu.get("driver_version") for gpu in smi_gpus
                           if gpu.get("driver_version")), None)
    return {
        "schema_version": 1, "platform": platform.system().lower(),
        "python": sys.version.split()[0], "attempts": records,
        "required_checks": {"gpu_runtime": True, "nvcc": backend != "triton",
                            "ncu": True, "cuobjdump": backend != "triton",
                            "python_dependencies": require_torch or backend == "triton"},
        "generation_allowed": bool(last["passed"]),
        "stop_reason": None if last["passed"] else "environment_unavailable",
        "hardware_attempts": len(records), "gpus": gpus,
        "primary_sm_arch": gpus[0].get("sm_arch") if gpus else None,
        "nvcc": {"available": bool(last["resolved_tools"].get("nvcc")),
                 "path": last["resolved_tools"].get("nvcc"), "version": last["tool_versions"].get("nvcc")},
        "ncu": {"available": bool(last["resolved_tools"].get("ncu")),
                "path": last["resolved_tools"].get("ncu"), "version": last["tool_versions"].get("ncu"),
                "can_read_counters": bool(last["resolved_tools"].get("ncu")) and not any(e.startswith("ncu ") for e in last["errors"])},
        "cuobjdump": {"available": bool(last["resolved_tools"].get("cuobjdump")),
                      "path": last["resolved_tools"].get("cuobjdump")},
        "driver": {"available": bool(driver_version), "version": driver_version},
        "cutlass": {"available": bool(last.get("cutlass_include_dir")),
                    "include_dir": last.get("cutlass_include_dir")},
        "libs": last["python_dependencies"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./env.json")
    parser.add_argument("--backend", choices=["cuda", "cutlass", "triton"], default="cuda")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--require-torch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostic", action="store_true",
                        help="Report failures but return success; never authorizes generation")
    args = parser.parse_args()
    result = run_gate(backend=args.backend, require_torch=args.require_torch, attempts=args.attempts)
    if args.diagnostic:
        result["probe_passed"] = result["generation_allowed"]
        result["generation_allowed"] = False
        result["diagnostic_only"] = True
        result["stop_reason"] = "diagnostic_only"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["generation_allowed"] and not args.diagnostic:
        raise SystemExit(EXIT_HARDWARE_UNAVAILABLE)


if __name__ == "__main__":
    main()
