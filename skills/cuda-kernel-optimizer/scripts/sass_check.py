#!/usr/bin/env python3
"""Verify that claimed optimization methods are actually present in compiled SASS.

Uses cuobjdump --dump-sass on the compiled .so to grep for expected instruction
patterns defined in references/sass_signatures.json.

Writes iterv{i}/sass_check.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


_DEFAULT_SIGNATURES = Path(__file__).resolve().parent.parent / "references" / "sass_signatures.json"


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_so_file(kernel_path: str) -> str | None:
    """Find the compiled .so corresponding to a .cu kernel."""
    base = os.path.splitext(kernel_path)[0]
    for ext in (".so", ".dll"):
        candidate = base + ext
        if os.path.isfile(candidate):
            return candidate
    return None


def _dump_sass(so_path: str, cuobjdump: str = "cuobjdump") -> str:
    """Run cuobjdump --dump-sass and return output."""
    try:
        r = subprocess.run(
            [cuobjdump, "--dump-sass", so_path],
            capture_output=True, text=True,
            encoding="utf-8", errors="ignore",
            timeout=30,
        )
        return r.stdout or ""
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        return f"ERROR: {e}"


def check_method_sass(
    method_id: str,
    sass_text: str,
    signatures: dict,
    target_symbol: str = "",
) -> dict:
    """Check if a method's expected SASS patterns appear in the disassembly."""
    result = {
        "method_id": method_id,
        "status": "inconclusive",
        "verified": False,
        "patterns_checked": [],
        "patterns_found": [],
        "patterns_missing": [],
    }

    method_sigs = signatures.get("methods", {}).get(method_id, {})
    if target_symbol:
        # cuobjdump emits `Function : <name>` headers. Restrict evidence to
        # the requested kernel when the artifact exposes that symbol.
        blocks = re.split(r"(?=Function\s*:\s*)", sass_text, flags=re.IGNORECASE)
        scoped = [b for b in blocks if re.search(rf"Function\s*:\s*{re.escape(target_symbol)}\b", b, re.IGNORECASE)]
        if not scoped:
            return {"method_id": method_id, "status": "inconclusive", "verified": False,
                    "patterns_checked": method_sigs.get("sass_patterns", []),
                    "patterns_found": [], "patterns_missing": method_sigs.get("sass_patterns", []),
                    "note": f"target_symbol_not_found:{target_symbol}"}
        sass_text = "\n".join(scoped)
    patterns = method_sigs.get("sass_patterns", [])
    require_any = method_sigs.get("require_any", True)  # True = at least one pattern found

    if not patterns:
        result["status"] = "not_applicable"
        result["note"] = "no_sass_verifier_declared"
        return result

    result["patterns_checked"] = patterns

    for pattern in patterns:
        if re.search(pattern, sass_text, re.IGNORECASE):
            result["patterns_found"].append(pattern)
        else:
            result["patterns_missing"].append(pattern)

    if require_any:
        result["verified"] = len(result["patterns_found"]) > 0
    else:
        # require_all
        result["verified"] = len(result["patterns_missing"]) == 0
    result["status"] = "pass" if result["verified"] else "fail"

    return result


def run(state_path: str, iteration: int, signatures_path: str = None) -> dict:
    state = _load_json(state_path)
    run_dir = state["run_dir"]
    iter_dir = os.path.join(run_dir, f"iterv{iteration}")

    # Load method choices
    methods_path = os.path.join(iter_dir, "methods.json")
    if not os.path.isfile(methods_path):
        sys.exit(f"methods.json not found at {methods_path}")
    methods_data = _load_json(methods_path)
    methods_list = methods_data.get("methods", [])

    # Load SASS signatures
    sig_path = signatures_path or str(_DEFAULT_SIGNATURES)
    if os.path.isfile(sig_path):
        signatures = _load_json(sig_path)
    else:
        signatures = {"methods": {}}

    # Find compiled kernel
    kernel_path = None
    for ext in (".cu", ".py"):
        candidate = os.path.join(iter_dir, f"kernel{ext}")
        if os.path.isfile(candidate):
            kernel_path = candidate
            break

    if not kernel_path:
        return {"error": "no_kernel_found", "checks": []}

    # For Triton kernels, SASS check is not directly applicable
    if kernel_path.endswith(".py"):
        checks = []
        for m in methods_list:
            checks.append({
                "method_id": m.get("id", "unknown"),
                "status": "not_applicable",
                "verified": False,
                "note": "triton_kernel_sass_not_applicable",
            })
        result = {"kernel": kernel_path, "backend": "triton", "checks": checks}
        _write_result(iter_dir, result)
        return result

    # CUDA/CUTLASS: find .so and dump SASS
    manifest_path = os.path.join(iter_dir, "kernel.build.json")
    so_path = None
    if os.path.isfile(manifest_path):
        try:
            so_path = _load_json(manifest_path).get("artifact")
        except (OSError, json.JSONDecodeError):
            so_path = None
    so_path = so_path if so_path and os.path.isfile(so_path) else _find_so_file(kernel_path)
    if not so_path:
        checks = [{"method_id": m.get("id", "unknown"), "status": "inconclusive",
                   "verified": False, "note": "artifact_not_found"} for m in methods_list]
        result = {"error": "so_not_found", "kernel": kernel_path, "checks": checks}
        _write_result(iter_dir, result)
        return result

    env_tool = state.get("env", {}).get("cuobjdump", {})
    cuobjdump = env_tool.get("path") if isinstance(env_tool, dict) else None
    sass_text = _dump_sass(so_path, cuobjdump or "cuobjdump")

    if sass_text.startswith("ERROR:"):
        checks = []
        for m in methods_list:
            checks.append({
                "method_id": m.get("id", "unknown"),
                "status": "tool_error",
                "verified": False,
                "note": f"cuobjdump_unavailable: {sass_text}",
            })
        result = {"kernel": kernel_path, "backend": "cuda", "sass_error": sass_text, "checks": checks}
        _write_result(iter_dir, result)
        return result

    # Check each method
    checks = []
    target_symbol = str(state.get("kernel_symbol", ""))
    for m in methods_list:
        mid = m.get("id", "unknown")
        check = check_method_sass(mid, sass_text, signatures, target_symbol)
        checks.append(check)

    result = {
        "kernel": kernel_path,
        "so": so_path,
        "backend": "cuda",
        "target_symbol": target_symbol,
        "sass_lines": len(sass_text.splitlines()),
        "checks": checks,
    }

    _write_result(iter_dir, result)
    print(json.dumps(result, indent=2))
    return result


def _write_result(iter_dir: str, result: dict):
    out_path = os.path.join(iter_dir, "sass_check.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--signatures", default=None)
    args = p.parse_args()
    run(args.state, args.iter, args.signatures)


if __name__ == "__main__":
    main()
