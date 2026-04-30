#!/usr/bin/env python3
"""Detect local Hygon DCU / DTK / HIP / CK Tile environment."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], timeout: int = 10) -> tuple[int, str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="ignore")
        return r.returncode, r.stdout or "", r.stderr or ""
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return -1, "", str(exc)


def _detect_gpus() -> list[dict]:
    gpus: list[dict] = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gcn = getattr(props, "gcnArchName", None)
                arch = str(gcn).split(":", 1)[0] if gcn else None
                gpus.append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "gcn_arch": arch,
                    "gfx_arch": arch,
                    "total_memory_mb": props.total_memory // (1024 * 1024),
                    "compute_units": getattr(props, "multi_processor_count", None),
                })
    except Exception as exc:
        gpus.append({"error": f"torch probe failed: {exc}"})

    if any(g.get("gfx_arch") for g in gpus):
        return gpus

    rocminfo = shutil.which("rocminfo")
    if rocminfo:
        rc, out, _ = _run([rocminfo], timeout=20)
        if rc == 0:
            matches = re.findall(r"\b(gfx[0-9a-fA-F]+)\b", out)
            for i, arch in enumerate(dict.fromkeys(matches)):
                gpus.append({"index": i, "name": arch, "gcn_arch": arch, "gfx_arch": arch})
    return gpus


def _detect_tool(name: str, args: list[str] | None = None) -> dict:
    path = shutil.which(name) or shutil.which(os.path.join("/opt/dtk/bin", name))
    if not path:
        return {"available": False, "path": None, "version": None}
    args = args or ["--version"]
    rc, out, err = _run([path] + args)
    text = (out or err).strip()
    return {"available": True, "path": path, "version": text.splitlines()[0] if text else None}


def _detect_hipprof() -> dict:
    info = _detect_tool("hipprof", ["-h"])
    if not info["available"]:
        return info | {"pmc_available": False}
    path = info["path"]
    rc, out, err = _run([path, "--list-basic"], timeout=20)
    info["pmc_available"] = rc == 0
    info["pmc_note"] = None if rc == 0 else (err or out).strip()[:400]
    return info


def _detect_rocm_smi() -> dict:
    info = _detect_tool("rocm-smi", [])
    if not info["available"]:
        return info
    rc, out, err = _run([info["path"]], timeout=20)
    info["raw"] = (out or err).strip()[:2000]
    return info


def _detect_ck_tile() -> dict:
    candidates: list[str] = []
    for var in ("CK_TILE_PATH", "CK_TILE_INCLUDE_DIR", "CK_PATH", "COMPOSABLE_KERNEL_PATH"):
        v = os.environ.get(var, "").strip()
        if v:
            candidates.extend([v, os.path.join(v, "include")])
    candidates.extend(sorted(glob.glob("/opt/dtk*/**/include", recursive=True)))
    candidates.extend(sorted(glob.glob("/opt/rocm*/**/include", recursive=True)))
    candidates.extend(["/opt/dtk/include", "/opt/rocm/include", "/usr/local/include"])
    seen: set[str] = set()
    for c in candidates:
        r = os.path.abspath(c)
        if r in seen:
            continue
        seen.add(r)
        if os.path.isdir(os.path.join(r, "ck_tile")):
            return {"available": True, "include_dir": r}
    return {"available": False, "include_dir": None}


def _detect_python_libs() -> dict:
    libs = {}
    for name in ("torch", "triton"):
        try:
            mod = __import__(name)
            libs[name] = {"available": True, "version": getattr(mod, "__version__", "unknown")}
        except Exception:
            libs[name] = {"available": False, "version": None}
    return libs


def collect_env() -> dict:
    gpus = _detect_gpus()
    primary = next((g.get("gfx_arch") for g in gpus if g.get("gfx_arch")), None)
    return {
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "gpus": gpus,
        "primary_gfx_arch": primary,
        "hipcc": _detect_tool("hipcc", ["--version"]),
        "hipprof": _detect_hipprof(),
        "dccobjdump": _detect_tool("dccobjdump", ["--version"]),
        "rocminfo": _detect_tool("rocminfo", []),
        "rocm_smi": _detect_rocm_smi(),
        "ck_tile": _detect_ck_tile(),
        "libs": _detect_python_libs(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="./env.json")
    args = p.parse_args()
    env = collect_env()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, ensure_ascii=False)
    print(json.dumps({
        "gpu": (env["gpus"][0] or {}).get("name") if env["gpus"] else None,
        "gfx_arch": env["primary_gfx_arch"],
        "hipcc": env["hipcc"].get("path"),
        "hipprof": env["hipprof"].get("available"),
        "hipprof_pmc": env["hipprof"].get("pmc_available"),
        "dccobjdump": env["dccobjdump"].get("available"),
        "ck_tile": env["ck_tile"].get("available"),
        "torch": env["libs"].get("torch", {}).get("version"),
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
