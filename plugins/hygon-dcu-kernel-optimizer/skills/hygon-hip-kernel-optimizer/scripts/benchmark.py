#!/usr/bin/env python3
"""Generic HIP operator benchmark with optional correctness validation.

Supported backends:
  - hip: raw HIP/C++ files exposing `extern "C" void solve(...)`
  - ck_tile: CK Tile based HIP/C++ files exposing the same solve entry
  - python: Python modules exposing `setup(...)` and `run_kernel(...)`

The contract intentionally mirrors the CUDA skill's benchmark driver so the
rest of the optimization loop can stay deterministic.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import glob
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch


SUPPORTED_TYPES = {
    "float*": ctypes.c_void_p,
    "double*": ctypes.c_void_p,
    "unsigned char*": ctypes.c_void_p,
    "unsigned short*": ctypes.c_void_p,
    "unsigned int*": ctypes.c_void_p,
    "char*": ctypes.c_void_p,
    "short*": ctypes.c_void_p,
    "long*": ctypes.c_void_p,
    "int*": ctypes.c_void_p,
    "int": ctypes.c_int,
    "long": ctypes.c_long,
    "size_t": ctypes.c_size_t,
    "unsigned int": ctypes.c_uint,
    "unsigned short": ctypes.c_ushort,
    "unsigned char": ctypes.c_ubyte,
    "char": ctypes.c_char,
    "short": ctypes.c_short,
}

DTYPE_MAP = {
    "float*": torch.float32,
    "double*": torch.float64,
    "int*": torch.int32,
    "long*": torch.int64,
    "short*": torch.int16,
    "char*": torch.int8,
    "unsigned char*": torch.uint8,
    "unsigned short*": getattr(torch, "uint16", torch.int16),
    "unsigned int*": getattr(torch, "uint32", torch.int32),
}

INT_TYPES = {"int", "long", "size_t", "unsigned int", "unsigned short", "unsigned char", "char", "short"}


def _write_json(path: str, payload: dict) -> None:
    if not path:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_solve_signature(source_file: str) -> list[tuple[str, str, bool]]:
    src = Path(source_file).read_text(encoding="utf-8", errors="ignore")
    match = re.search(r'extern\s+"C"\s+void\s+solve\s*\(([\s\S]*?)\)\s*\{', src)
    if not match:
        raise ValueError(f'Cannot find `extern "C" void solve(...)` in {source_file}')

    raw = re.sub(r"/\*.*?\*/", "", match.group(1), flags=re.S)
    raw = re.sub(r"//[^\n]*", "", raw)
    raw = " ".join(raw.split())

    params: list[tuple[str, str, bool]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        is_const = "const" in token
        clean = re.sub(r"\s+", " ", token.replace("const", "").strip())
        for key in sorted(SUPPORTED_TYPES, key=len, reverse=True):
            base = key.replace("*", r"\s*\*")
            m = re.match(rf"({base})\s+(\w+)", clean)
            if m:
                params.append((key, m.group(2), is_const))
                break
        else:
            raise ValueError(f"Cannot parse parameter: `{token}`")
    return params


def detect_arch(device_index: int | None = None) -> str:
    if torch.cuda.is_available():
        if device_index is None:
            device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        gcn = getattr(props, "gcnArchName", None)
        if gcn:
            return str(gcn).split(":", 1)[0]
    try:
        r = subprocess.run(["rocminfo"], capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=10)
        m = re.search(r"\b(gfx[0-9a-fA-F]+)\b", r.stdout or "")
        if m:
            return m.group(1)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "gfx938"


def find_ck_tile_include_dir() -> str:
    candidates: list[str] = []
    for var in ("CK_TILE_PATH", "CK_TILE_INCLUDE_DIR", "CK_PATH", "COMPOSABLE_KERNEL_PATH"):
        v = os.environ.get(var, "").strip()
        if v:
            candidates.extend([v, os.path.join(v, "include")])
    candidates.extend(sorted(glob.glob("/opt/dtk*/**/include", recursive=True)))
    candidates.extend(sorted(glob.glob("/opt/rocm*/**/include", recursive=True)))
    candidates.extend(["/opt/dtk/include", "/opt/rocm/include", "/usr/local/include"])
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        resolved = os.path.abspath(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        if os.path.isdir(os.path.join(resolved, "ck_tile")):
            return resolved
    return ""


def infer_backend(solution_file: str, backend: str) -> str:
    if backend != "auto":
        return backend
    suffix = os.path.splitext(solution_file)[1].lower()
    if suffix == ".py":
        return "python"
    if suffix in {".hip", ".cu", ".cpp", ".cc", ".cxx"}:
        text = Path(solution_file).read_text(encoding="utf-8", errors="ignore")
        if "ck_tile/" in text or "ck_tile::" in text:
            return "ck_tile"
        return "hip"
    return "hip"


def compile_hip(source_file: str, output_so: str, arch: str, hipcc_bin: str, backend: str) -> None:
    cmd = [hipcc_bin, "-fPIC", "-shared", "-std=c++17", "-O3", f"--offload-arch={arch}"]
    if os.path.splitext(source_file)[1].lower() in {".cpp", ".cc", ".cxx"}:
        cmd.extend(["-x", "hip"])
    if backend == "ck_tile":
        include_dir = find_ck_tile_include_dir()
        if not include_dir:
            raise RuntimeError(
                "Cannot locate CK Tile headers. Set CK_TILE_PATH, CK_TILE_INCLUDE_DIR, "
                "CK_PATH, or COMPOSABLE_KERNEL_PATH."
            )
        cmd.extend(["-I", include_dir])
    cmd.extend(["-o", output_so, source_file])
    print(f"[compile] {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        raise RuntimeError("Compilation failed:\n" + (r.stderr or r.stdout or "unknown hipcc error"))


def load_module(module_file: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module from {module_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clone_value(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    return copy.deepcopy(value)


def _determine_ptr_elems(int_values: list[int], ptr_size_override: int) -> int:
    if ptr_size_override > 0:
        ptr_elems = ptr_size_override
    elif not int_values:
        ptr_elems = 1024 * 1024
    elif len(int_values) == 1:
        ptr_elems = int_values[0]
    else:
        sv = sorted(int_values, reverse=True)
        ptr_elems = sv[0] * sv[1]
    return min(ptr_elems, 256 * 1024 * 1024)


def _setup_hip(solution_file: str, dims: dict, ptr_size: int, arch: str, hipcc_bin: str, seed: int | None, backend: str) -> dict:
    params = parse_solve_signature(solution_file)
    sig_str = ", ".join(f"{'const ' if c else ''}{t} {n}" for t, n, c in params)
    print(f"[signature] solve({sig_str})\n")

    so_file = os.path.splitext(solution_file)[0] + ".so"
    compile_hip(solution_file, so_file, arch, hipcc_bin, backend)
    lib = ctypes.CDLL(os.path.abspath(so_file))

    for ptype, pname, _ in params:
        if ptype in INT_TYPES and pname not in dims:
            raise ValueError(f"Missing dimension: --{pname}=<value>")

    int_vals = [int(dims[pname]) for ptype, pname, _ in params if ptype in INT_TYPES]
    ptr_elems = _determine_ptr_elems(int_vals, ptr_size)
    if seed is not None:
        torch.manual_seed(seed)

    tensor_inputs: dict[str, torch.Tensor] = {}
    reference_inputs: dict = {}
    pristine_tensors: dict[str, torch.Tensor] = {}
    output_specs: list[tuple[str, str]] = []
    kernel_args = []
    argtypes = []

    for ptype, pname, is_const in params:
        if ptype in DTYPE_MAP:
            dtype = DTYPE_MAP[ptype]
            if is_const:
                tensor = torch.randn(ptr_elems, device="cuda", dtype=dtype) if dtype.is_floating_point else torch.randint(0, 17, (ptr_elems,), device="cuda", dtype=dtype)
            else:
                tensor = torch.zeros(ptr_elems, device="cuda", dtype=dtype)
                output_specs.append((pname, ptype))
            tensor_inputs[pname] = tensor
            pristine_tensors[pname] = tensor.clone()
            reference_inputs[pname] = tensor.clone()
            kernel_args.append(ctypes.c_void_p(tensor.data_ptr()))
            argtypes.append(ctypes.c_void_p)
        else:
            ctype = SUPPORTED_TYPES[ptype]
            val = int(dims[pname])
            reference_inputs[pname] = val
            kernel_args.append(ctype(val))
            argtypes.append(ctype)

    lib.solve.argtypes = argtypes
    lib.solve.restype = None
    total_ptr_bytes = sum(t.nelement() * t.element_size() for t in tensor_inputs.values())
    return {
        "backend": backend,
        "signature": [{"type": t, "name": n, "is_const": c} for t, n, c in params],
        "callable": lambda: lib.solve(*kernel_args),
        "tensor_inputs": tensor_inputs,
        "reference_inputs": reference_inputs,
        "pristine_tensors": pristine_tensors,
        "output_specs": output_specs,
        "ptr_elems": ptr_elems,
        "total_ptr_bytes": total_ptr_bytes,
    }


def _setup_python(solution_file: str, dims: dict, seed: int | None) -> dict:
    module = load_module(solution_file, "_hip_python_kernel")
    if not hasattr(module, "setup") or not hasattr(module, "run_kernel"):
        raise AttributeError(f"{solution_file} must define setup(**kwargs) and run_kernel(**kwargs)")
    prepared = module.setup(**dict(dims, seed=seed))
    if not isinstance(prepared, dict) or not isinstance(prepared.get("inputs"), dict):
        raise TypeError("setup() must return {'inputs': dict, 'outputs': list}")
    inputs = prepared["inputs"]
    outputs = set(prepared.get("outputs", []))
    for name in outputs:
        inputs[name].zero_()
    tensor_inputs = {k: v for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    reference_inputs = {k: clone_value(v) for k, v in inputs.items()}
    pristine_tensors = {k: v.clone() for k, v in tensor_inputs.items()}
    output_specs = [(name, str(inputs[name].dtype).replace("torch.", "")) for name in outputs]
    total_ptr_bytes = sum(t.nelement() * t.element_size() for t in tensor_inputs.values())
    signature = [
        {"type": f"tensor[{str(v.dtype).replace('torch.', '')}]" if isinstance(v, torch.Tensor) else type(v).__name__,
         "name": k,
         "is_const": k not in outputs}
        for k, v in inputs.items()
    ]
    return {
        "backend": "python",
        "signature": signature,
        "callable": lambda: module.run_kernel(**inputs),
        "tensor_inputs": tensor_inputs,
        "reference_inputs": reference_inputs,
        "pristine_tensors": pristine_tensors,
        "output_specs": output_specs,
        "ptr_elems": sum(t.numel() for t in tensor_inputs.values()),
        "total_ptr_bytes": total_ptr_bytes,
    }


def _reset_tensor_inputs(state: dict) -> None:
    for name, tensor in state["tensor_inputs"].items():
        snap = state["pristine_tensors"].get(name)
        if snap is not None:
            tensor.copy_(snap)


def _time_iterations(fn, warmup: int, repeat: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    avg_ms = start.elapsed_time(end) / repeat
    return [avg_ms] * repeat


def _stats(times: list[float]) -> dict:
    ordered = sorted(times)
    avg = sum(times) / len(times)
    return {"average_ms": avg, "median_ms": ordered[len(ordered) // 2], "min_ms": min(times), "max_ms": max(times)}


def _validate_outputs(kernel_tensors: dict, ref_tensors: dict, output_specs: list[tuple[str, str]], atol: float, rtol: float) -> bool:
    all_pass = True
    for name, ptype in output_specs:
        kt = kernel_tensors[name].float()
        rt = ref_tensors[name].float()
        ok = torch.allclose(kt, rt, atol=atol, rtol=rtol)
        all_pass = all_pass and ok
        max_diff = (kt - rt).abs().max().item()
        mean_diff = (kt - rt).abs().mean().item()
        print(f"[validate] {name} ({ptype}) {'PASS' if ok else 'FAIL'} max={max_diff:.6e} mean={mean_diff:.6e}")
    return all_pass


def run(
    solution_file: str,
    ref_file: str,
    dims: dict,
    warmup: int,
    repeat: int,
    ptr_size: int,
    arch: str,
    atol: float,
    rtol: float,
    seed: int,
    json_out: str,
    hipcc_bin: str,
    backend: str,
) -> None:
    resolved_backend = infer_backend(solution_file, backend)
    has_ref = bool(ref_file)
    gpu_index = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_index)
    result = {
        "solution_file": os.path.abspath(solution_file),
        "backend": resolved_backend,
        "ref_file": os.path.abspath(ref_file) if has_ref else "",
        "has_reference": has_ref,
        "dims": dims,
        "warmup": warmup,
        "repeat": repeat,
        "ptr_size_override": ptr_size,
        "gpu_index": gpu_index,
        "gpu_name": gpu_name,
        "arch": arch,
        "seed": seed,
        "correctness": {"checked": has_ref, "passed": None, "atol": atol if has_ref else None, "rtol": rtol if has_ref else None},
        "kernel": None,
        "reference": None,
        "speedup_vs_reference": None,
        "error": None,
    }

    try:
        state = _setup_python(solution_file, dims, seed) if resolved_backend == "python" else _setup_hip(solution_file, dims, ptr_size, arch, hipcc_bin, seed if has_ref else None, resolved_backend)
    except Exception as exc:
        result["error"] = {"code": "setup_failed", "stage": "setup", "message": str(exc)}
        if has_ref:
            result["correctness"]["passed"] = False
        _write_json(json_out, result)
        raise

    result["signature"] = state["signature"]
    result["ptr_elems"] = state["ptr_elems"]
    result["total_ptr_bytes"] = state["total_ptr_bytes"]

    ref_fn = None
    if has_ref:
        ref_mod = load_module(ref_file, "_ref_module")
        if not hasattr(ref_mod, "reference"):
            raise AttributeError(f"{ref_file} must define reference(**kwargs)")
        ref_fn = ref_mod.reference
        atol = float(getattr(ref_mod, "atol", atol))
        rtol = float(getattr(ref_mod, "rtol", rtol))
        ref_inputs = {name: clone_value(value) for name, value in state["reference_inputs"].items()}
        _reset_tensor_inputs(state)
        state["callable"]()
        torch.cuda.synchronize()
        ref_fn(**ref_inputs)
        torch.cuda.synchronize()
        kernel_outputs = {name: state["tensor_inputs"][name] for name, _ in state["output_specs"]}
        ref_outputs = {name: ref_inputs[name] for name, _ in state["output_specs"]}
        passed = _validate_outputs(kernel_outputs, ref_outputs, state["output_specs"], atol, rtol)
        result["correctness"].update({"passed": passed, "atol": atol, "rtol": rtol, "output_tensor_count": len(state["output_specs"])})
        if not passed:
            _write_json(json_out, result)
            sys.exit(1)

    times_ref = None
    if has_ref and ref_fn is not None:
        ref_bench_inputs = {name: clone_value(value) for name, value in state["reference_inputs"].items()}
        times_ref = _time_iterations(lambda: ref_fn(**ref_bench_inputs), warmup, repeat)

    _reset_tensor_inputs(state)
    times_kernel = _time_iterations(state["callable"], warmup, repeat)
    result["kernel"] = _stats(times_kernel)
    if result["kernel"]["average_ms"] > 0:
        result["kernel"]["bandwidth_gbps_rough"] = state["total_ptr_bytes"] / (result["kernel"]["average_ms"] / 1000) / 1e9

    if times_ref is not None:
        result["reference"] = _stats(times_ref)
        if result["reference"]["average_ms"] > 0:
            result["reference"]["bandwidth_gbps_rough"] = state["total_ptr_bytes"] / (result["reference"]["average_ms"] / 1000) / 1e9
        result["speedup_vs_reference"] = result["reference"]["average_ms"] / result["kernel"]["average_ms"]

    print(json.dumps({
        "backend": resolved_backend,
        "gpu": gpu_name,
        "arch": arch,
        "correct": result["correctness"]["passed"],
        "kernel_ms": result["kernel"]["average_ms"],
        "ref_ms": (result.get("reference") or {}).get("average_ms"),
        "json_out": json_out,
    }, indent=2))
    _write_json(json_out, result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic HIP/CK Tile/Python operator benchmark")
    parser.add_argument("solution_file")
    parser.add_argument("--backend", default="auto", choices=["auto", "hip", "ck_tile", "python"])
    parser.add_argument("--ref", default="")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--ptr-size", type=int, default=0)
    parser.add_argument("--arch", default="")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--hipcc-bin", default=os.environ.get("HIPCC", "hipcc"))
    args, unknown = parser.parse_known_args()

    dims = {}
    for item in unknown:
        if item.startswith("--") and "=" in item:
            key, val = item[2:].split("=", 1)
            dims[key] = int(val)
        else:
            print(f"Warning: ignoring unknown arg `{item}`", file=sys.stderr)

    torch.cuda.set_device(args.gpu)
    arch = args.arch or detect_arch(args.gpu)
    run(args.solution_file, args.ref, dims, args.warmup, args.repeat, args.ptr_size, arch, args.atol, args.rtol, args.seed, args.json_out, args.hipcc_bin, args.backend)


if __name__ == "__main__":
    main()
