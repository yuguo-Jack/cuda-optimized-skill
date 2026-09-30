#!/usr/bin/env python3
"""Bounded contract verification for CUDA kernel candidates.

The checker deliberately separates evidence gates.  Static CUDA inspection
covers the ``extern \"C\" void solve(...)`` ABI, scalar dimension coverage,
and common indexing/layout hazards.  Runtime correctness and timing are read
from an existing benchmark JSON.  An optional compute-sanitizer racecheck can
be run against a caller-supplied command.

Each gate is one of ``pass``, ``fail`` or ``inconclusive`` and is exposed both
under ``states`` and as a top-level key for consumers that prefer flat JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


INT_TYPES = {
    "int", "long", "long long", "size_t", "ptrdiff_t",
    "unsigned int", "unsigned long", "unsigned long long",
    "unsigned short", "unsigned char", "char", "short",
}
POINTER_TYPES = {
    "float", "double", "half", "int", "long", "long long", "short",
    "char", "unsigned char", "unsigned short", "unsigned int",
}
_SOLVE_RE = re.compile(
    r'extern\s+"C"\s+void\s+solve\s*\(([^)]*)\)\s*\{', re.S
)
_TYPE_RE = re.compile(
    r"^(?P<const>const\s+)?(?P<type>[A-Za-z_][\w\s]*?)\s*"
    r"(?P<ptr>\*\s*)?(?P<name>[A-Za-z_]\w*)$"
)


def generate_contract_cases(dims: dict[str, Any] | None = None) -> list[dict[str, int]]:
    """Generate cheap shape variants that expose common flattened-index bugs."""
    base = {str(k): int(v) for k, v in (dims or {}).items()
            if isinstance(v, (int, float)) and int(v) > 0}
    cases = [dict(base)] if base else [{}]
    if all(k in base for k in ("B", "H", "S")):
        odd = dict(base)
        odd["B"] = max(2, int(base["B"]))
        odd["H"] = max(3, int(base["H"]))
        odd["S"] = max(3, int(base["S"]) | 1)
        cases.append(odd)
        tail = dict(odd)
        tail["S"] += 1
        cases.append(tail)
    elif "S" in base:
        odd = dict(base)
        odd["S"] = max(3, int(base["S"]) | 1)
        cases.append(odd)
    unique: list[dict[str, int]] = []
    seen: set[tuple[tuple[str, int], ...]] = set()
    for case in cases:
        key = tuple(sorted(case.items()))
        if key not in seen:
            unique.append(case)
            seen.add(key)
    return unique
_PARAM_QUALIFIERS_RE = re.compile(r"\b(?:__restrict__|__restrict|restrict|volatile)\b")


def _state(value: str) -> str:
    if value not in {"pass", "fail", "inconclusive"}:
        raise ValueError(value)
    return value


def parse_solve_signature(source: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Parse the benchmark ABI, accepting common CUDA scalar/pointer types."""
    text = Path(source).read_text(encoding="utf-8", errors="ignore")
    match = _SOLVE_RE.search(text)
    if not match:
        raise ValueError('cannot find `extern "C" void solve(...)`')
    raw = re.sub(r"/\*.*?\*/", "", match.group(1), flags=re.S)
    raw = re.sub(r"//[^\n]*", "", raw)
    params: list[dict[str, Any]] = []
    for token in (part.strip() for part in raw.split(",")):
        if not token:
            continue
        token = " ".join(token.split())
        token = re.sub(r"\b(?:__restrict__|__restrict)\b", "", token)
        token = " ".join(token.split())
        token = _PARAM_QUALIFIERS_RE.sub("", token)
        token = " ".join(token.split())
        parsed = _TYPE_RE.match(token)
        if not parsed:
            raise ValueError(f"cannot parse parameter `{token}`")
        base = parsed.group("type").strip()
        is_ptr = bool(parsed.group("ptr"))
        if base not in INT_TYPES and not (is_ptr and base in POINTER_TYPES):
            raise ValueError(f"unsupported parameter type `{base}{'*' if is_ptr else ''}`")
        params.append({
            "type": f"{base}{'*' if is_ptr else ''}",
            "name": parsed.group("name"),
            "is_const": bool(parsed.group("const")),
            "is_pointer": is_ptr,
        })
    return params


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def inspect_index_layout(source: str | os.PathLike[str], signature: list[dict[str, Any]]) -> dict[str, Any]:
    """Collect conservative indexing/layout signals without pretending to prove safety."""
    text = _strip_comments(Path(source).read_text(encoding="utf-8", errors="ignore"))
    pointer_names = [p["name"] for p in signature if p["is_pointer"]]
    index_tokens = re.findall(r"\b(?:threadIdx|blockIdx|blockDim|gridDim)\s*\.\s*[xyz]\b", text)
    has_kernel = bool(re.search(r"__global__\s+(?:void|[\w:<>]+)\s+\w+\s*\(", text))
    has_guard = bool(re.search(
        r"if\s*\([^\n;]*(?:>=|>|<=|<)\s*[^\n;]+\)\s*(?:\{\s*)?return\s*;",
        text,
    ))
    has_sync = bool(re.search(r"__syncthreads\s*\(|__syncwarp\s*\(", text))
    has_shared = "__shared__" in text
    has_64bit = bool(re.search(r"\b(?:long long|size_t|ptrdiff_t)\b|\(long long\)", text))
    layout_signals = sorted(set(re.findall(
        r"(?:stride\w*|contiguous|is_contiguous|ld[a-zA-Z_]*|pitch|layout)", text,
        re.I,
    )))
    indexed_pointers = [name for name in pointer_names if re.search(rf"\b{re.escape(name)}\s*\[", text)]
    warnings: list[str] = []
    if has_kernel and index_tokens and not has_guard:
        warnings.append("kernel uses thread/block indices without an obvious bounds guard")
    if indexed_pointers and not has_64bit:
        warnings.append("pointer indexing has no obvious 64-bit index arithmetic")
    if has_shared and not has_sync:
        warnings.append("shared memory is used without an obvious synchronization primitive")
    return {
        "kernel_detected": has_kernel,
        "index_tokens": sorted(set(index_tokens)),
        "indexed_pointers": indexed_pointers,
        "bounds_guard": has_guard if has_kernel and index_tokens else None,
        "uses_64bit_index": has_64bit,
        "shared_memory": has_shared,
        "synchronization": has_sync,
        "layout_signals": layout_signals,
        "warnings": warnings,
    }


def _compile_cuda(source: str, nvcc: str, timeout: int) -> dict[str, Any]:
    if not shutil.which(nvcc) and not os.path.isfile(nvcc):
        return {"state": "inconclusive", "note": f"nvcc not found: {nvcc}"}
    with tempfile.TemporaryDirectory(prefix="contract-check-") as td:
        output = os.path.join(td, "kernel.so")
        cmd = [nvcc, "-shared", "-Xcompiler", "-fPIC", "-std=c++17", "-O3", "-o", output, source]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"state": "fail", "command": cmd, "note": str(exc)}
        return {
            "state": "pass" if result.returncode == 0 else "fail",
            "command": cmd,
            "returncode": result.returncode,
            "stdout": (result.stdout or "")[-2000:],
            "stderr": (result.stderr or "")[-4000:],
        }


def _run_racecheck(command: str, timeout: int, tool: str = "racecheck") -> dict[str, Any]:
    sanitizer = shutil.which("compute-sanitizer")
    if not sanitizer:
        return {"state": "inconclusive", "note": "compute-sanitizer not found"}
    cmd = [sanitizer, "--tool", tool, "--error-exitcode", "1"] + shlex.split(command)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"state": "fail", "command": cmd, "note": str(exc)}
    output = ((result.stdout or "") + "\n" + (result.stderr or ""))[-6000:]
    return {"state": "pass" if result.returncode == 0 else "fail", "command": cmd,
            "returncode": result.returncode, "output": output}


def _bench_states(bench: dict[str, Any] | None) -> tuple[str, str]:
    if not bench:
        return "inconclusive", "inconclusive"
    correctness = bench.get("correctness") or {}
    cp = correctness.get("passed")
    correctness_state = "pass" if cp is True else "fail" if cp is False else "inconclusive"
    kernel = bench.get("kernel") or {}
    avg = kernel.get("average_ms")
    timing_state = "pass" if isinstance(avg, (int, float)) and math.isfinite(float(avg)) and float(avg) > 0 else "fail" if avg is not None else "inconclusive"
    if bench.get("error"):
        timing_state = "fail"
    return correctness_state, timing_state


def check_contract(kernel: str, dims: dict[str, Any] | None = None, bench: dict[str, Any] | None = None,
                   compile: bool = False, nvcc: str = "nvcc", sanitizer_command: str = "",
                   timeout: int = 60, sanitizer_tools: str = "racecheck") -> dict[str, Any]:
    """Return a JSON-serializable contract report for one CUDA candidate."""
    dims = dims or {}
    report: dict[str, Any] = {"kernel": os.path.abspath(kernel), "backend": "cuda", "errors": [], "warnings": []}
    report["contract_cases"] = generate_contract_cases(dims)
    signature: list[dict[str, Any]] = []
    try:
        signature = parse_solve_signature(kernel)
        report["signature"] = signature
    except (OSError, ValueError) as exc:
        report["errors"].append(str(exc))
        report["signature"] = []
    if signature:
        missing_dims = [p["name"] for p in signature if not p["is_pointer"] and p["type"] in INT_TYPES and p["name"] not in dims]
        report["dims"] = {"provided": dims, "missing": missing_dims}
        index_layout = inspect_index_layout(kernel, signature)
        report["index_layout"] = index_layout
        report["warnings"].extend(index_layout["warnings"])
        contract_state = "fail" if missing_dims else "pass"
    else:
        report["dims"] = {"provided": dims, "missing": []}
        report["index_layout"] = {}
        contract_state = "fail"
    compile_report = _compile_cuda(kernel, nvcc, timeout) if compile else {"state": "inconclusive", "note": "compile not requested"}
    correctness_state, timing_state = _bench_states(bench)
    tools = [x.strip() for x in str(sanitizer_tools).split(",") if x.strip()] or ["racecheck"]
    sanitizer_reports = {
        tool: (_run_racecheck(sanitizer_command, timeout, tool=tool)
               if sanitizer_command else {"state": "inconclusive", "note": f"{tool} not requested"})
        for tool in tools
    }
    report_states = [item.get("state") for item in sanitizer_reports.values()]
    race_state = "fail" if "fail" in report_states else "pass" if report_states and all(x == "pass" for x in report_states) else "inconclusive"
    race_report = {"state": race_state, "tools": sanitizer_reports}
    states = {
        "compile_pass": _state(compile_report["state"]),
        "contract_pass": _state(contract_state),
        "correctness_pass": _state(correctness_state),
        "race_safe": _state(race_report["state"]),
        "timing_valid": _state(timing_state),
    }
    report.update(states)
    report["states"] = states
    report["compile"] = compile_report
    report["racecheck"] = race_report
    report["sanitizers"] = sanitizer_reports
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True, help="CUDA .cu candidate")
    parser.add_argument("--dims", default="{}", help="JSON scalar dimension values")
    parser.add_argument("--bench", default="", help="Existing benchmark JSON")
    parser.add_argument("--compile", action="store_true", help="Compile with nvcc")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--sanitizer-command", default="", help="Program command to execute under racecheck")
    parser.add_argument("--sanitizer-tools", default="racecheck",
                        help="Comma-separated compute-sanitizer tools (memcheck,racecheck,synccheck,initcheck)")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    try:
        dims = json.loads(args.dims)
        if not isinstance(dims, dict):
            raise ValueError("--dims must decode to an object")
        bench = json.loads(Path(args.bench).read_text()) if args.bench else None
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    result = check_contract(args.kernel, dims, bench, args.compile, args.nvcc, args.sanitizer_command, args.timeout, args.sanitizer_tools)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    # Static contract failures are actionable; inconclusive runtime gates do
    # not make the command fail when no runtime evidence was requested.
    raise SystemExit(1 if result["contract_pass"] == "fail" else 0)


if __name__ == "__main__":
    main()
