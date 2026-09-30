#!/usr/bin/env python3
"""Run benchmark.py for a given iteration (or for the baseline seed step).

Subcommands:
  seed-baseline   Run benchmark.py on the baseline to capture initial timing.
  benchmark       Run benchmark.py on iterv{i}/kernel.<ext>.

Both write JSON under the appropriate directory. `benchmark` additionally
captures stderr so the agent can inspect it on validation failure.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from experiment import benchmark_gate, run_json, resolve_benchmark, require_open_iteration, iteration_kernel


_BUNDLED_BENCHMARK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.py")
KERNEL_EXTS = (".hip", ".cu", ".cpp", ".cc", ".cxx", ".py")


def _read(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dims_argv(dims: dict) -> list[str]:
    return [f"--{k}={v}" for k, v in dims.items()]


def _ptr_size_argv(ptr_size: int) -> list[str]:
    return ["--ptr-size", str(ptr_size)] if ptr_size and ptr_size > 0 else []


def _run_bench(
    *,
    benchmark_py: str,
    solution: str,
    ref: str,
    dims: dict,
    ptr_size: int,
    json_out: str,
    stderr_out: str,
    warmup: int,
    repeat: int,
) -> int:
    cmd = [
        sys.executable, benchmark_py, solution,
        "--ref", ref,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--json-out", json_out,
    ] + _ptr_size_argv(ptr_size) + _dims_argv(dims)
    print(f"[bench] {' '.join(cmd)}", file=sys.stderr)

    result = run_json(cmd, json_out, stderr_out)
    return 0 if benchmark_gate(result, solution, ref)[0] else 1


def cmd_seed_baseline(args: argparse.Namespace) -> None:
    state = _read(args.state)
    if state.get("best_metric_ms") is not None or state.get("history"):
        sys.exit("Baseline already seeded; start a new run to remeasure it")
    args.benchmark = resolve_benchmark(state, args.benchmark)
    run_dir = state["run_dir"]
    out_dir = os.path.join(run_dir, "baseline")
    os.makedirs(out_dir, exist_ok=True)
    json_out = os.path.join(out_dir, "bench.json")
    stderr_out = os.path.join(out_dir, "bench.stderr.txt")

    _run_bench(
        benchmark_py=os.path.abspath(args.benchmark),
        solution=state["baseline_file"],
        ref=state["ref_file"],
        dims=state.get("dims", {}),
        ptr_size=state.get("ptr_size", 0),
        json_out=json_out,
        stderr_out=stderr_out,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    # Push baseline ms into state via state.py CLI (keep one place that writes state)
    # Runs have a single writer; state.py validates the frozen evidence.
    sibling = os.path.join(os.path.dirname(__file__), "state.py")
    rc = subprocess.call([
        sys.executable, sibling, "set-baseline-metric",
        "--state", args.state, "--bench", json_out,
    ])
    sys.exit(rc)


def cmd_benchmark(args: argparse.Namespace) -> None:
    state = _read(args.state)
    require_open_iteration(state, args.iter)
    args.benchmark = resolve_benchmark(state, args.benchmark)
    run_dir = state["run_dir"]
    iter_dir = os.path.join(run_dir, f"iterv{args.iter}")
    kernel = iteration_kernel(iter_dir)

    json_out = os.path.join(iter_dir, "bench.json")
    stderr_out = os.path.join(iter_dir, "bench.stderr.txt")
    _run_bench(
        benchmark_py=os.path.abspath(args.benchmark),
        solution=kernel,
        ref=state["ref_file"],
        dims=state.get("dims", {}),
        ptr_size=state.get("ptr_size", 0),
        json_out=json_out,
        stderr_out=stderr_out,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    # Return the result summary on stdout for the orchestrator to consume
    res = _read(json_out)
    summary = {
        "iter": args.iter,
        "kernel": kernel,
        "passed": benchmark_gate(res, kernel, state["ref_file"])[0],
        "ms": (res.get("kernel") or {}).get("average_ms"),
        "ref_ms": (res.get("reference") or {}).get("average_ms"),
        "speedup_vs_ref": res.get("speedup_vs_reference"),
        "error": res.get("error"),
        "bench_json": json_out,
        "stderr_log": stderr_out,
    }
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("seed-baseline")
    ps.add_argument("--state", required=True)
    ps.add_argument("--benchmark", default=None,
                    help="Benchmark frozen in state, or bundled for legacy runs")
    ps.add_argument("--warmup", type=int, default=10)
    ps.add_argument("--repeat", type=int, default=20)
    ps.set_defaults(func=cmd_seed_baseline)

    pb = sub.add_parser("benchmark")
    pb.add_argument("--state", required=True)
    pb.add_argument("--iter", type=int, required=True)
    pb.add_argument("--benchmark", default=None,
                    help="Benchmark frozen in state, or bundled for legacy runs")
    pb.add_argument("--warmup", type=int, default=10)
    pb.add_argument("--repeat", type=int, default=20)
    pb.set_defaults(func=cmd_benchmark)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
