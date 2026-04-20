#!/usr/bin/env python3
"""Preflight: validate that baseline + ref files satisfy benchmark.py's contract
BEFORE we invest in setup / profiling / iterations.

Checks:
  Baseline
    .cu  →  contains `extern "C" void solve(...)` with parseable params
    .py  →  importable, exposes `setup(**kwargs)` AND `run_kernel(**kwargs)`
  Reference
    .py  →  importable, exposes `reference(**kwargs)` (+ optional atol/rtol)
  Dims
    Every int/long/size_t parameter in the CUDA signature has a matching
    --<name>=<value> in the supplied dims dict. (Triton gets a looser check
    since setup() is free-form.)

Exit code: 0 iff everything checks out.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CUDA .cu inspection (mirrors benchmark.py's parse_solve_signature)
# ---------------------------------------------------------------------------

_SIG = re.compile(r'extern\s+"C"\s+void\s+solve\s*\(([\s\S]*?)\)\s*\{')

INT_TYPES = {"int", "long", "size_t", "unsigned int", "unsigned short", "unsigned char", "char", "short"}
PTR_TYPE_PREFIXES = ("float*", "double*", "int*", "long*", "short*", "char*",
                     "unsigned char*", "unsigned short*", "unsigned int*")


def _parse_solve(cu_path: str) -> list[tuple[str, str, bool]]:
    src = Path(cu_path).read_text(encoding="utf-8", errors="ignore")
    m = _SIG.search(src)
    if not m:
        raise ValueError(
            f'{cu_path}: cannot find `extern "C" void solve(...)` — '
            f"benchmark.py will not be able to parse this file."
        )
    raw = re.sub(r"/\*.*?\*/", "", m.group(1))
    raw = re.sub(r"//[^\n]*", "", raw)
    raw = " ".join(raw.split())

    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        is_const = "const" in tok
        clean = re.sub(r"\s+", " ", tok.replace("const", "").strip())
        # Greedily match longest known type prefix
        matched = False
        for prefix in sorted(
            list(PTR_TYPE_PREFIXES) + list(INT_TYPES),
            key=len, reverse=True,
        ):
            base = prefix.replace("*", r"\s*\*")
            mm = re.match(rf"({base})\s+(\w+)", clean)
            if mm:
                out.append((prefix, mm.group(2), is_const))
                matched = True
                break
        if not matched:
            raise ValueError(f"{cu_path}: cannot parse parameter '{tok}' in solve signature")
    return out


# ---------------------------------------------------------------------------
# Python module inspection (ref.py, triton .py)
# ---------------------------------------------------------------------------

def _import_without_executing_cuda(path: str, name: str):
    """Import a .py file. If the import itself has top-level CUDA / torch
    calls that fail on a host without a GPU, surface a helpful error.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot form spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        # Re-raise with richer context; this is often where `import torch`
        # or a Triton dependency fails on a headless CI-style host.
        raise ImportError(f"failed to import {path}: {e.__class__.__name__}: {e}") from e
    return mod


def _check_ref(ref_path: str) -> dict:
    mod = _import_without_executing_cuda(ref_path, "_preflight_ref")
    if not hasattr(mod, "reference"):
        raise AttributeError(f"{ref_path}: must define `reference(**kwargs)`")
    fn = getattr(mod, "reference")
    if not callable(fn):
        raise TypeError(f"{ref_path}: `reference` must be callable")
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
    except (TypeError, ValueError):
        params = []
    return {
        "path": ref_path,
        "params": [p.name for p in params],
        "atol": getattr(mod, "atol", None),
        "rtol": getattr(mod, "rtol", None),
    }


def _check_triton(py_path: str) -> dict:
    mod = _import_without_executing_cuda(py_path, "_preflight_triton")
    missing = []
    for name in ("setup", "run_kernel"):
        if not hasattr(mod, name) or not callable(getattr(mod, name)):
            missing.append(name)
    if missing:
        raise AttributeError(
            f"{py_path}: Triton module missing required callables: {', '.join(missing)}"
        )
    try:
        setup_sig = inspect.signature(mod.setup)
        setup_params = list(setup_sig.parameters.values())
    except (TypeError, ValueError):
        setup_params = []
    return {
        "path": py_path,
        "backend": "triton",
        "setup_params": [p.name for p in setup_params],
    }


# ---------------------------------------------------------------------------
# Dim coverage
# ---------------------------------------------------------------------------

def _check_dims_cuda(sig: list[tuple[str, str, bool]], dims: dict) -> list[str]:
    missing = []
    for ptype, pname, _ in sig:
        if ptype in INT_TYPES and pname not in dims:
            missing.append(pname)
    return missing


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(baseline: str, ref: str, dims: dict, strict_ref_params: bool = False) -> dict:
    report = {"ok": True, "baseline": {}, "ref": {}, "warnings": [], "errors": []}

    # ref first: it's cheaper and always .py
    try:
        report["ref"] = _check_ref(ref)
    except Exception as e:
        report["ok"] = False
        report["errors"].append(f"ref: {e}")
        return report

    ext = os.path.splitext(baseline)[1].lower()
    try:
        if ext == ".cu":
            sig = _parse_solve(baseline)
            report["baseline"] = {
                "path": baseline,
                "backend": "cuda_or_cutlass",
                "signature": [
                    {"type": t, "name": n, "is_const": c} for t, n, c in sig
                ],
            }
            missing = _check_dims_cuda(sig, dims)
            if missing:
                report["ok"] = False
                report["errors"].append(
                    f"baseline: missing dim values for int/long params: "
                    f"{', '.join(missing)} — pass them via --dims"
                )
        elif ext == ".py":
            report["baseline"] = _check_triton(baseline)
            # Triton: we can't statically know which dims setup() needs, but
            # we can warn when the dict is empty.
            if not dims:
                report["warnings"].append(
                    "baseline: Triton module — no dims supplied; setup() will "
                    "be called with seed only. If setup() needs shape args, "
                    "pass them via --dims."
                )
        else:
            report["ok"] = False
            report["errors"].append(
                f"baseline: unsupported extension '{ext}' (expected .cu or .py)"
            )
            return report
    except Exception as e:
        report["ok"] = False
        report["errors"].append(f"baseline: {e}")
        return report

    # Strict mode: warn when dim names don't match ref's signature params
    if strict_ref_params and ext == ".cu":
        ref_params = set(report["ref"].get("params", []))
        sig_names = {n for _, n, _ in sig}
        overlap = sig_names & ref_params
        if not overlap:
            report["warnings"].append(
                "ref.reference() shares no parameter names with solve() — "
                "this is usually wrong (benchmark.py passes dim ints by name to both)."
            )

    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--dims", type=str, default="{}", help="JSON dict of name→int")
    p.add_argument("--strict", action="store_true",
                   help="Also warn when ref.reference() parameters don't overlap solve()'s")
    p.add_argument("--out", type=str, default="",
                   help="Optional path to write the report as JSON")
    args = p.parse_args()

    try:
        dims = json.loads(args.dims)
    except json.JSONDecodeError as e:
        sys.exit(f"--dims must be valid JSON: {e}")

    rep = run(os.path.abspath(args.baseline), os.path.abspath(args.ref), dims, args.strict)
    payload = json.dumps(rep, indent=2, ensure_ascii=False)
    print(payload)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload)

    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
