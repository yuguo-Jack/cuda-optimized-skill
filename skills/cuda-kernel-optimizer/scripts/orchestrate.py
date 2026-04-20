#!/usr/bin/env python3
"""End-to-end orchestrator. Intended to be driven BY Claude (which pauses at
reasoning steps) but can also be invoked standalone if the user has scripted
the code-generation step elsewhere.

Typical invocation (Claude drives):
  # step 0-2: setup
  python orchestrate.py setup \
    --baseline ./gemm.cu --ref ./ref.py --benchmark ./benchmark.py \
    --iterations 3 --ncu-num 5 --dims '{"M":4096,"N":4096,"K":4096}'

  # (Claude reads profile, writes iterv1/kernel.{cu|py} and methods.json)

  # step 3d-3f: validate + record
  python orchestrate.py close-iter --run-dir <run_dir> --iter 1

For fully automated runs (rare — code generation requires Claude), supply
a `--code-generator` shell command that accepts state.json + iter and
produces iterv{i}/kernel.* + methods.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"[run] {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, text=True, **kw)


def _read(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# setup  —  steps 0, 1, 2, and 3a (profile best_input) for iter 1
# ---------------------------------------------------------------------------

def cmd_setup(args):
    env_json = os.path.abspath(args.env_out or "./env.json")

    # 0. env
    rc = _run([sys.executable, str(SCRIPT_DIR / "check_env.py"), "--out", env_json]).returncode
    if rc != 0:
        sys.exit(f"check_env failed rc={rc}")

    # 0b. preflight — validate baseline + ref contract before we invest anything
    rc = _run([
        sys.executable, str(SCRIPT_DIR / "preflight.py"),
        "--baseline", os.path.abspath(args.baseline),
        "--ref", os.path.abspath(args.ref),
        "--dims", args.dims,
    ]).returncode
    if rc != 0:
        sys.exit("preflight failed — fix baseline/ref/dims above, then retry")

    # 1. init run dir + state
    init = _run([
        sys.executable, str(SCRIPT_DIR / "state.py"), "init",
        "--baseline", os.path.abspath(args.baseline),
        "--ref", os.path.abspath(args.ref),
        "--iterations", str(args.iterations),
        "--ncu-num", str(args.ncu_num),
        "--dims", args.dims,
        "--env", env_json,
        "--noise-threshold-pct", str(args.noise_threshold_pct),
    ], capture_output=True)
    if init.returncode != 0:
        sys.stderr.write(init.stderr or "")
        sys.exit("state init failed")
    sys.stderr.write(init.stderr or "")
    # state.py init prints one pretty-printed JSON object. Try that first;
    # fall back to last-line-is-json in case a future version emits a log
    # preamble.
    init_info = {}
    try:
        init_info = json.loads(init.stdout or "{}")
    except json.JSONDecodeError:
        for line in reversed((init.stdout or "").splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    init_info = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
    run_dir = init_info.get("run_dir")
    state_path = init_info.get("state")
    if not run_dir or not state_path:
        sys.exit(f"could not parse state init output:\n{init.stdout}")

    # 2. seed baseline (timing-only; correctness must pass)
    rc = _run([
        sys.executable, str(SCRIPT_DIR / "run_iteration.py"), "seed-baseline",
        "--state", state_path,
        "--benchmark", os.path.abspath(args.benchmark),
        "--warmup", str(args.warmup),
        "--repeat", str(args.repeat),
    ]).returncode
    if rc != 0:
        sys.exit("baseline seed failed (likely a correctness failure on the baseline itself)")

    # 3a (for iter 1): profile the best_input
    rc = _run([
        sys.executable, str(SCRIPT_DIR / "profile_ncu.py"),
        "--state", state_path,
        "--iter", "1",
        "--which", "best_input",
        "--benchmark", os.path.abspath(args.benchmark),
    ]).returncode
    if rc != 0:
        print("[warn] ncu profiling failed or degraded; see iterv1/*.ncu.log", file=sys.stderr)

    print(json.dumps({
        "run_dir": run_dir,
        "state": state_path,
        "env": env_json,
        "next_step": (
            "Claude should now read iterv1/ncu_top.json and state.json, then write "
            "iterv1/kernel.<ext>, iterv1/methods.json, and iterv1/analysis.md. "
            "After that, run: orchestrate.py close-iter --run-dir <run_dir> --iter 1"
        ),
    }, indent=2))


# ---------------------------------------------------------------------------
# close-iter  —  steps 3d (benchmark) + 3f (state update) + 3a-for-next-iter
# ---------------------------------------------------------------------------

def cmd_close_iter(args):
    state_path = os.path.join(args.run_dir, "state.json")
    if not os.path.isfile(state_path):
        sys.exit(f"state.json missing: {state_path}")
    state = _read(state_path)
    iter_dir = os.path.join(args.run_dir, f"iterv{args.iter}")
    kernel = next((p for p in [os.path.join(iter_dir, "kernel.cu"),
                               os.path.join(iter_dir, "kernel.py")]
                   if os.path.isfile(p)), None)
    methods_json = os.path.join(iter_dir, "methods.json")
    if not kernel:
        sys.exit(f"no kernel found under {iter_dir}")
    if not os.path.isfile(methods_json):
        sys.exit(f"methods.json missing at {methods_json}")

    # d. benchmark
    bench = _run([
        sys.executable, str(SCRIPT_DIR / "run_iteration.py"), "benchmark",
        "--state", state_path,
        "--iter", str(args.iter),
        "--benchmark", os.path.abspath(args.benchmark),
        "--warmup", str(args.warmup),
        "--repeat", str(args.repeat),
    ], capture_output=True)
    sys.stderr.write(bench.stderr or "")
    if bench.returncode not in (0, 1):
        sys.exit(f"bench subprocess crashed rc={bench.returncode}")
    try:
        summary = json.loads(bench.stdout)
    except Exception:
        summary = {"passed": False, "error": "could not parse bench output"}

    passed = bool(summary.get("passed"))
    bench_json = os.path.join(iter_dir, "bench.json")

    if not passed:
        print(json.dumps({
            "iter": args.iter,
            "status": "validation_failed",
            "bench_json": bench_json,
            "stderr_log": os.path.join(iter_dir, "bench.stderr.txt"),
            "guidance": (
                "Claude should read bench.json['correctness'] and bench.stderr.txt, "
                "fix the kernel, and run close-iter again. After --max-retries failed "
                "attempts, pass --abandon to skip this iteration."
            ),
        }, indent=2))
        sys.exit(2)

    # f. update state
    rc = _run([
        sys.executable, str(SCRIPT_DIR / "state.py"), "update",
        "--state", state_path,
        "--iter", str(args.iter),
        "--kernel", kernel,
        "--bench", bench_json,
        "--methods-json", methods_json,
        "--retries", str(args.retries),
    ]).returncode
    if rc != 0:
        sys.exit("state update failed")

    # Profile the new kernel (so next iter's 3a has fresh data if this became best)
    rc = _run([
        sys.executable, str(SCRIPT_DIR / "profile_ncu.py"),
        "--state", state_path,
        "--iter", str(args.iter),
        "--which", "kernel",
        "--benchmark", os.path.abspath(args.benchmark),
        "--promote-if-best",
    ]).returncode

    # 3a for the *next* iteration
    state = _read(state_path)
    next_iter = args.iter + 1
    if next_iter <= state["iterations_total"]:
        _run([
            sys.executable, str(SCRIPT_DIR / "profile_ncu.py"),
            "--state", state_path,
            "--iter", str(next_iter),
            "--which", "best_input",
            "--benchmark", os.path.abspath(args.benchmark),
        ])

    print(json.dumps({
        "iter": args.iter,
        "status": "closed",
        "best_ms": state.get("best_metric_ms"),
        "next_iter": next_iter if next_iter <= state["iterations_total"] else None,
        "state": state_path,
    }, indent=2))


# ---------------------------------------------------------------------------
# finalize  —  step 4
# ---------------------------------------------------------------------------

def cmd_finalize(args):
    state_path = os.path.join(args.run_dir, "state.json")
    summary_path = os.path.join(args.run_dir, "summary.md")
    rc = _run([
        sys.executable, str(SCRIPT_DIR / "summarize.py"),
        "--state", state_path,
        "--out", summary_path,
    ]).returncode
    if rc != 0:
        sys.exit("summarize failed")
    print(json.dumps({"summary": summary_path}, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    _default_bench = str(SCRIPT_DIR / "benchmark.py")

    ps = sub.add_parser("setup")
    ps.add_argument("--baseline", required=True)
    ps.add_argument("--ref", required=True)
    ps.add_argument("--benchmark", default=_default_bench,
                    help="Path to benchmark.py (default: bundled scripts/benchmark.py)")
    ps.add_argument("--iterations", type=int, default=3)
    ps.add_argument("--ncu-num", type=int, default=5)
    ps.add_argument("--dims", required=True, help="JSON dict of name->int")
    ps.add_argument("--noise-threshold-pct", type=float, default=2.0)
    ps.add_argument("--env-out", type=str, default="")
    ps.add_argument("--warmup", type=int, default=10)
    ps.add_argument("--repeat", type=int, default=20)
    ps.set_defaults(func=cmd_setup)

    pc = sub.add_parser("close-iter")
    pc.add_argument("--run-dir", required=True)
    pc.add_argument("--iter", type=int, required=True)
    pc.add_argument("--benchmark", default=_default_bench,
                    help="Path to benchmark.py (default: bundled scripts/benchmark.py)")
    pc.add_argument("--retries", type=int, default=0,
                    help="How many correctness retries were already done "
                         "before this call (recorded in history, doesn't gate).")
    pc.add_argument("--warmup", type=int, default=10)
    pc.add_argument("--repeat", type=int, default=20)
    pc.set_defaults(func=cmd_close_iter)

    pf = sub.add_parser("finalize")
    pf.add_argument("--run-dir", required=True)
    pf.set_defaults(func=cmd_finalize)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
