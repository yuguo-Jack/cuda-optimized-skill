#!/usr/bin/env python3
"""Branch-and-Select: compile and benchmark K candidate kernels in parallel.

All K branches share the same method combination (from methods.json) but
differ in hyperparameters (tile size, num_stages, num_warps, etc.).

Claude generates K kernels under iterv{i}/branches/b{1..K}/kernel.<ext>.

This script:
  1. Compiles all K kernels (can be parallelized)
  2. Benchmarks each kernel with validation
  3. Selects champion = fastest valid kernel
  4. Copies champion to iterv{i}/kernel.<ext>
  5. Returns non-champions as frontier candidates in state

Writes iterv{i}/branch_results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


_BUNDLED_BENCHMARK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.py")


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _dims_argv(dims: dict) -> list[str]:
    return [f"--{k}={v}" for k, v in dims.items()]


def _ptr_size_argv(ptr_size: int) -> list[str]:
    return ["--ptr-size", str(ptr_size)] if ptr_size and ptr_size > 0 else []


def _bench_kernel(
    benchmark_py: str,
    kernel_path: str,
    ref_path: str,
    dims: dict,
    ptr_size: int,
    json_out: str,
    warmup: int = 10,
    repeat: int = 20,
) -> dict:
    """Run benchmark.py on a kernel. Returns parsed result or error dict."""
    cmd = [
        sys.executable, benchmark_py, kernel_path,
        "--ref", ref_path,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--json-out", json_out,
    ] + _ptr_size_argv(ptr_size) + _dims_argv(dims)

    Path(json_out).parent.mkdir(parents=True, exist_ok=True)
    stderr_out = json_out.replace(".json", ".stderr.txt")

    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="ignore",
        )
    except OSError as e:
        return {"error": str(e), "passed": False}

    # Save stderr for debugging
    with open(stderr_out, "w", encoding="utf-8") as f:
        f.write("---STDOUT---\n")
        f.write(r.stdout or "")
        f.write("\n---STDERR---\n")
        f.write(r.stderr or "")

    if os.path.isfile(json_out):
        return _load_json(json_out)

    return {
        "error": "no_json_output",
        "stderr": (r.stderr or "")[-2000:],
        "passed": False,
    }


def run(state_path: str, iteration: int, benchmark_py: str = None,
        warmup: int = 10, repeat: int = 20) -> dict:
    state = _load_json(state_path)
    run_dir = state["run_dir"]
    iter_dir = os.path.join(run_dir, f"iterv{iteration}")
    bench_py = benchmark_py or _BUNDLED_BENCHMARK
    branches_dir = os.path.join(iter_dir, "branches")
    ref_file = state["ref_file"]
    dims = state.get("dims", {})
    ptr_size = state.get("ptr_size", 0)
    num_branches = state.get("branches", 4)

    # Discover branches
    branch_dirs = []
    for i in range(1, num_branches + 1):
        bd = os.path.join(branches_dir, f"b{i}")
        if os.path.isdir(bd):
            # Check if there's a kernel file
            kernel = None
            for ext in (".cu", ".py"):
                candidate = os.path.join(bd, f"kernel{ext}")
                if os.path.isfile(candidate):
                    kernel = candidate
                    break
            if kernel:
                branch_dirs.append({"index": i, "dir": bd, "kernel": kernel})

    if not branch_dirs:
        # Fallback: check if there's a single kernel directly in iter_dir
        for ext in (".cu", ".py"):
            candidate = os.path.join(iter_dir, f"kernel{ext}")
            if os.path.isfile(candidate):
                branch_dirs.append({
                    "index": 0, "dir": iter_dir, "kernel": candidate,
                })
                break

    if not branch_dirs:
        sys.exit(f"No branch kernels found under {branches_dir}")

    print(f"[branch_explore] Found {len(branch_dirs)} branches", file=sys.stderr)

    # Benchmark all branches
    results = []
    for branch in branch_dirs:
        idx = branch["index"]
        kernel = branch["kernel"]
        json_out = os.path.join(branch["dir"], "bench.json")

        print(f"[branch {idx}] Benchmarking {os.path.basename(kernel)}...",
              file=sys.stderr)

        bench_result = _bench_kernel(
            bench_py, kernel, ref_file, dims, ptr_size, json_out, warmup, repeat,
        )

        passed = bool(bench_result.get("correctness", {}).get("passed", False))
        ms = None
        if bench_result.get("kernel"):
            ms = bench_result["kernel"].get("average_ms")

        results.append({
            "branch_index": idx,
            "kernel": kernel,
            "passed": passed,
            "ms": ms,
            "error": bench_result.get("error"),
        })

        status = "PASS" if passed else "FAIL"
        ms_str = f"{ms:.4f} ms" if ms else "N/A"
        print(f"[branch {idx}] {status}  {ms_str}", file=sys.stderr)

    # Select champion: fastest valid branch
    valid_results = [r for r in results if r["passed"] and r["ms"] is not None]

    if not valid_results:
        output = {
            "iter": iteration,
            "status": "all_branches_failed",
            "branches": results,
            "champion": None,
        }
        _write_json(os.path.join(iter_dir, "branch_results.json"), output)
        print(json.dumps(output, indent=2))
        sys.exit(2)

    champion = min(valid_results, key=lambda r: r["ms"])

    # Copy champion kernel to iterv{i}/kernel.<ext>
    champ_kernel = champion["kernel"]
    ext = os.path.splitext(champ_kernel)[1]
    dest = os.path.join(iter_dir, f"kernel{ext}")
    if os.path.abspath(champ_kernel) != os.path.abspath(dest):
        shutil.copy2(champ_kernel, dest)

    # Also copy champion bench.json to iter_dir
    champ_bench = os.path.join(os.path.dirname(champ_kernel), "bench.json")
    dest_bench = os.path.join(iter_dir, "bench.json")
    if os.path.isfile(champ_bench) and os.path.abspath(champ_bench) != os.path.abspath(dest_bench):
        shutil.copy2(champ_bench, dest_bench)

    # Build frontier from non-champion valid results
    frontier_entries = []
    for r in valid_results:
        if r["branch_index"] != champion["branch_index"]:
            frontier_entries.append({
                "iter": iteration,
                "branch_index": r["branch_index"],
                "kernel": r["kernel"],
                "ms": r["ms"],
                "delta_from_champion": round(r["ms"] - champion["ms"], 4),
            })

    output = {
        "iter": iteration,
        "status": "champion_selected",
        "champion": {
            "branch_index": champion["branch_index"],
            "kernel": dest,
            "ms": champion["ms"],
        },
        "branches": results,
        "frontier": frontier_entries,
        "total_branches": len(branch_dirs),
        "valid_branches": len(valid_results),
    }

    _write_json(os.path.join(iter_dir, "branch_results.json"), output)
    print(json.dumps(output, indent=2))
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--benchmark", default=None)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=20)
    args = p.parse_args()
    run(args.state, args.iter, args.benchmark, args.warmup, args.repeat)


if __name__ == "__main__":
    main()
