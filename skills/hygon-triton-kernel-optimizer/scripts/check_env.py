#!/usr/bin/env python3
"""Collect Hygon DCU Triton optimization environment facts."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "hygon-hip-kernel-optimizer/scripts"))
from hcu_targets import HCU_NAMES, describe, normalize_gfx


def _run(cmd: list[str], timeout: int = 15) -> tuple[int, str, str]:
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="ignore",
        )
        return r.returncode, r.stdout or "", r.stderr or ""
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return -1, "", str(exc)


def _tool(name: str, args: list[str] | None = None) -> dict:
    path = shutil.which(name) or shutil.which(os.path.join("/opt/dtk/bin", name))
    if not path:
        return {"available": False, "path": None, "version": None}
    rc, out, err = _run([path] + (args or ["--version"]))
    text = (out or err).strip()
    return {
        "available": True,
        "path": path,
        "returncode": rc,
        "version": text.splitlines()[0] if text else None,
    }


def _torch_info() -> dict:
    info = {"available": False}
    try:
        import torch

        info.update({
            "available": True,
            "version": getattr(torch, "__version__", None),
            "hip": getattr(torch.version, "hip", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "devices": [],
        })
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gcn = getattr(props, "gcnArchName", None)
                arch = normalize_gfx(gcn)
                info["devices"].append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "gcn_arch": arch,
                    "total_memory_mb": props.total_memory // (1024 * 1024),
                    "compute_units": getattr(props, "multi_processor_count", None),
                })
    except Exception as exc:
        info["error"] = str(exc)
    return info


def _triton_info() -> dict:
    try:
        import triton

        return {"available": True, "version": getattr(triton, "__version__", None)}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _rocminfo_arches() -> list[str]:
    rocminfo = shutil.which("rocminfo")
    if not rocminfo:
        return []
    rc, out, _ = _run([rocminfo], timeout=30)
    if rc != 0:
        return []
    return list(dict.fromkeys(normalize_gfx(x) for x in re.findall(r"\b(gfx[0-9a-f]+)\b", out, re.I)))


def collect() -> dict:
    torch_info = _torch_info()
    primary_arch = None
    for dev in torch_info.get("devices", []):
        if dev.get("gcn_arch"):
            primary_arch = dev["gcn_arch"]
            break
    rocminfo_arches = _rocminfo_arches()
    if not primary_arch and rocminfo_arches:
        primary_arch = rocminfo_arches[0]

    env = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "cwd": os.getcwd(),
        "torch": torch_info,
        "triton": _triton_info(),
        "primary_arch": primary_arch,
        "target_identity": describe(primary_arch),
        "rocminfo_arches": rocminfo_arches,
        "tools": {
            "xprof": _tool("xprof", ["--help"]),
            "xcompute": _tool("xcompute", ["--help"]),
            "aicc": _tool("aicc", ["--version"]),
            "dcc": _tool("dcc", ["--version"]),
            "hipcc": _tool("hipcc", ["--version"]),
            "hipprof": _tool("hipprof", ["-h"]),
            "dccobjdump": _tool("dccobjdump", ["--version"]),
            "rocminfo": _tool("rocminfo", []),
            "rocm-smi": _tool("rocm-smi", []),
        },
        "env": {
            "AMDGCN_USE_BUFFER_OPS": os.environ.get("AMDGCN_USE_BUFFER_OPS"),
            "TORCH_LOGS": os.environ.get("TORCH_LOGS"),
            "TORCHINDUCTOR_TRACE": os.environ.get("TORCHINDUCTOR_TRACE"),
            "TRITON_CAPTURE_DIR": os.environ.get("TRITON_CAPTURE_DIR"),
            "TORCHINDUCTOR_CACHE_DIR": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "TRITON_HIP_CLANG_PATH": os.environ.get("TRITON_HIP_CLANG_PATH"),
            "ROCM_PATH": os.environ.get("ROCM_PATH"),
        },
        "warnings": [],
    }

    if env["env"]["AMDGCN_USE_BUFFER_OPS"] == "0":
        env["warnings"].append("AMDGCN_USE_BUFFER_OPS is set to 0; buffer ops may be disabled.")
    if not torch_info.get("cuda_available"):
        env["warnings"].append("torch.cuda is not available; run capture/benchmark on the DCU host.")
    if not primary_arch:
        env["warnings"].append("No gfx arch detected; run on the Hygon DCU host for capture and benchmark.")
    if primary_arch and primary_arch not in HCU_NAMES:
        env["warnings"].append(f"Target is outside the currently named HCU set: {primary_arch}; confirm vendor and toolchain support")
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Hygon DCU Triton optimization environment")
    parser.add_argument("--out", default="./triton_env.json")
    args = parser.parse_args()
    env = collect()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(json.dumps({
        "arch": env.get("primary_arch"),
        "torch": env.get("torch", {}).get("version"),
        "triton": env.get("triton", {}).get("version"),
        "hipcc": env.get("tools", {}).get("hipcc", {}).get("available"),
        "hipprof": env.get("tools", {}).get("hipprof", {}).get("available"),
        "dccobjdump": env.get("tools", {}).get("dccobjdump", {}).get("available"),
        "warnings": env.get("warnings", []),
        "out": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
