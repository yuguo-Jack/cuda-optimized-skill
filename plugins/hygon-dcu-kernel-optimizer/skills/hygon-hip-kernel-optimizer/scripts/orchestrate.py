#!/usr/bin/env python3
"""End-to-end orchestrator (v2 — roofline-driven, branch-and-select).

Subcommands:
  setup       Steps 0-2: env check, preflight, init, seed baseline, profile+roofline for iter 1
  open-iter   Prepare an iteration: profile best → dcu_top → roofline → axis budgets
              (Codex then writes K branch kernels + methods.json + analysis.md)
  close-iter  Steps 3e-3j: branch explore → champion → hipprof champion → ablate → ISA check → update
  finalize    Step 4: emit summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
KERNEL_EXTS = (".hip", ".cu", ".cpp", ".cc", ".cxx", ".py")


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"[run] {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, text=True, **kw)


def _read(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# setup  —  steps 0, 1, 2, and open-iter(1)
# ---------------------------------------------------------------------------

def cmd_setup(args):
    env_json = os.path.abspath(args.env_out or "./env.json")

    # 0. env
    rc = _run([sys.executable, str(SCRIPT_DIR / "check_env.py"),
               "--out", env_json]).returncode
    if rc != 0:
        sys.exit(f"check_env failed rc={rc}")

    # 0b. preflight
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
        "--branches", str(args.branches),
        "--dims", args.dims,
        "--env", env_json,
        "--noise-threshold-pct", str(args.noise_threshold_pct),
        "--ptr-size", str(args.ptr_size),
    ], capture_output=True)
    if init.returncode != 0:
        sys.stderr.write(init.stderr or "")
        sys.exit("state init failed")
    sys.stderr.write(init.stderr or "")

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

    # 2. seed baseline
    rc = _run([
        sys.executable, str(SCRIPT_DIR / "run_iteration.py"), "seed-baseline",
        "--state", state_path,
        "--benchmark", os.path.abspath(args.benchmark),
        "--warmup", str(args.warmup),
        "--repeat", str(args.repeat),
    ]).returncode
    if rc != 0:
        sys.exit("baseline seed failed")

    # 3a for iter 1: profile best_input
    rc = _run([
        sys.executable, str(SCRIPT_DIR / "profile_hipprof.py"),
        "--state", state_path,
        "--iter", "1",
        "--which", "best_input",
        "--benchmark", os.path.abspath(args.benchmark),
    ]).returncode
    if rc != 0:
        print("[warn] hipprof profiling failed or degraded", file=sys.stderr)

    # 3b for iter 1: roofline
    rc = _run([
        sys.executable, str(SCRIPT_DIR / "roofline.py"),
        "--state", state_path,
        "--iter", "1",
    ]).returncode

    # Check for early stop
    iter_dir = os.path.join(run_dir, "iterv1")
    roofline_path = os.path.join(iter_dir, "roofline.json")
    early_stop = False
    if os.path.isfile(roofline_path):
        roofline = _read(roofline_path)
        early_stop = roofline.get("near_peak", False)

    print(json.dumps({
        "run_dir": run_dir,
        "state": state_path,
        "env": env_json,
        "early_stop": early_stop,
        "next_step": (
            "Codex should now read iterv1/roofline.json (for axis budgets), "
            "iterv1/dcu_top.json, and state.json, then write "
            f"iterv1/branches/b{{1..K}}/kernel.<ext>, iterv1/methods.json, "
            "and iterv1/analysis.md. "
            "After that, run: orchestrate.py close-iter --run-dir <run_dir> --iter 1"
        ) if not early_stop else "Near roofline — consider stopping.",
    }, indent=2))


# ---------------------------------------------------------------------------
# open-iter  —  profile + roofline for iteration N (if not done by setup)
# ---------------------------------------------------------------------------

def cmd_open_iter(args):
    state_path = os.path.join(args.run_dir, "state.json")
    if not os.path.isfile(state_path):
        sys.exit(f"state.json missing: {state_path}")

    state = _read(state_path)

    # Profile best_input for this iter
    rc = _run([
        sys.executable, str(SCRIPT_DIR / "profile_hipprof.py"),
        "--state", state_path,
        "--iter", str(args.iter),
        "--which", "best_input",
        "--benchmark", os.path.abspath(args.benchmark),
    ]).returncode
    if rc != 0:
        print("[warn] hipprof profiling failed or degraded", file=sys.stderr)

    # Roofline
    rc = _run([
        sys.executable, str(SCRIPT_DIR / "roofline.py"),
        "--state", state_path,
        "--iter", str(args.iter),
    ]).returncode

    # Check early stop
    iter_dir = os.path.join(args.run_dir, f"iterv{args.iter}")
    roofline_path = os.path.join(iter_dir, "roofline.json")
    early_stop = False
    if os.path.isfile(roofline_path):
        roofline = _read(roofline_path)
        early_stop = roofline.get("near_peak", False)

    # Create branch dirs
    num_branches = state.get("branches", 4)
    branches_dir = os.path.join(iter_dir, "branches")
    for b in range(1, num_branches + 1):
        os.makedirs(os.path.join(branches_dir, f"b{b}"), exist_ok=True)

    print(json.dumps({
        "iter": args.iter,
        "early_stop": early_stop,
        "branches_dir": branches_dir,
        "num_branches": num_branches,
        "next_step": (
            f"Codex should read iterv{args.iter}/roofline.json and dcu_top.json, "
            f"write {num_branches} branch kernels under iterv{args.iter}/branches/b{{1..{num_branches}}}/kernel.<ext>, "
            f"plus iterv{args.iter}/methods.json and iterv{args.iter}/analysis.md. "
            f"Then run: orchestrate.py close-iter --run-dir {args.run_dir} --iter {args.iter}"
        ) if not early_stop else "Near roofline — consider stopping.",
    }, indent=2))


# ---------------------------------------------------------------------------
# close-iter  —  branch explore → hipprof champion → ablate → ISA check → update
# ---------------------------------------------------------------------------

def cmd_close_iter(args):
    state_path = os.path.join(args.run_dir, "state.json")
    if not os.path.isfile(state_path):
        sys.exit(f"state.json missing: {state_path}")

    state = _read(state_path)
    iter_dir = os.path.join(args.run_dir, f"iterv{args.iter}")
    methods_json = os.path.join(iter_dir, "methods.json")
    if not os.path.isfile(methods_json):
        sys.exit(f"methods.json missing at {methods_json}")

    # Step 3e: Branch explore — compile + benchmark all branches
    branch_result = _run([
        sys.executable, str(SCRIPT_DIR / "branch_explore.py"),
        "--state", state_path,
        "--iter", str(args.iter),
        "--benchmark", os.path.abspath(args.benchmark),
        "--warmup", str(args.warmup),
        "--repeat", str(args.repeat),
    ], capture_output=True)
    sys.stderr.write(branch_result.stderr or "")

    if branch_result.returncode == 2:
        # All branches failed
        print(json.dumps({
            "iter": args.iter,
            "status": "all_branches_failed",
            "guidance": "Codex should fix the kernels and retry close-iter.",
        }, indent=2))
        sys.exit(2)
    if branch_result.returncode != 0:
        sys.exit(f"branch_explore failed rc={branch_result.returncode}")

    # Find champion kernel
    kernel = None
    for ext in KERNEL_EXTS:
        candidate = os.path.join(iter_dir, f"kernel{ext}")
        if os.path.isfile(candidate):
            kernel = candidate
            break
    if not kernel:
        sys.exit(f"No champion kernel found after branch_explore")

    bench_json = os.path.join(iter_dir, "bench.json")
    if not os.path.isfile(bench_json):
        sys.exit(f"bench.json missing for champion")

    bench = _read(bench_json)
    passed = bool(bench.get("correctness", {}).get("passed", False))

    if not passed:
        print(json.dumps({
            "iter": args.iter,
            "status": "validation_failed",
            "bench_json": bench_json,
            "guidance": "Codex should fix the kernel and re-run close-iter.",
        }, indent=2))
        sys.exit(2)

    # Step 3g: Profile champion with hipprof
    rc = _run([
        sys.executable, str(SCRIPT_DIR / "profile_hipprof.py"),
        "--state", state_path,
        "--iter", str(args.iter),
        "--which", "kernel",
        "--benchmark", os.path.abspath(args.benchmark),
        "--promote-if-best",
    ]).returncode
    if rc != 0:
        print("[warn] hipprof profiling of champion failed", file=sys.stderr)

    # Step 3h: Ablation attribution (optional — runs if ablation kernels exist)
    attribution_path = os.path.join(iter_dir, "attribution.json")
    ablation_dir = os.path.join(iter_dir, "ablations")
    if os.path.isdir(ablation_dir):
        _run([
            sys.executable, str(SCRIPT_DIR / "ablate.py"),
            "--state", state_path,
            "--iter", str(args.iter),
            "--benchmark", os.path.abspath(args.benchmark),
        ])

    # Step 3i: DCU ISA verification
    sass_check_path = os.path.join(iter_dir, "isa_check.json")
    _run([
        sys.executable, str(SCRIPT_DIR / "sass_check.py"),
        "--state", state_path,
        "--iter", str(args.iter),
    ])

    # Step 3j: Update state
    update_cmd = [
        sys.executable, str(SCRIPT_DIR / "state.py"), "update",
        "--state", state_path,
        "--iter", str(args.iter),
        "--kernel", kernel,
        "--bench", bench_json,
        "--methods-json", methods_json,
        "--retries", str(args.retries),
    ]
    if os.path.isfile(attribution_path):
        update_cmd.extend(["--attribution", attribution_path])
    if os.path.isfile(sass_check_path):
        update_cmd.extend(["--sass-check", sass_check_path])

    rc = _run(update_cmd).returncode
    if rc != 0:
        sys.exit("state update failed")

    state = _read(state_path)
    hipprof_output = os.path.join(iter_dir, "kernel.hipprof")
    if os.path.abspath(state.get("best_file", "")) == os.path.abspath(kernel):
        _run([
            sys.executable, str(SCRIPT_DIR / "state.py"), "set-best-hipprof-output",
            "--state", state_path,
            "--hipprof-output", hipprof_output,
        ])

    # Open next iteration if needed
    state = _read(state_path)
    next_iter = args.iter + 1
    if next_iter <= state["iterations_total"]:
        # Profile best_input for next iter + roofline
        _run([
            sys.executable, str(SCRIPT_DIR / "profile_hipprof.py"),
            "--state", state_path,
            "--iter", str(next_iter),
            "--which", "best_input",
            "--benchmark", os.path.abspath(args.benchmark),
        ])
        _run([
            sys.executable, str(SCRIPT_DIR / "roofline.py"),
            "--state", state_path,
            "--iter", str(next_iter),
        ])

        # Check early stop
        roofline_path = os.path.join(state["run_dir"], f"iterv{next_iter}", "roofline.json")
        early_stop = False
        if os.path.isfile(roofline_path):
            roofline = _read(roofline_path)
            early_stop = roofline.get("near_peak", False)
    else:
        early_stop = False

    print(json.dumps({
        "iter": args.iter,
        "status": "closed",
        "best_ms": state.get("best_metric_ms"),
        "next_iter": next_iter if next_iter <= state["iterations_total"] else None,
        "early_stop": early_stop,
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
    ps.add_argument("--benchmark", default=_default_bench)
    ps.add_argument("--iterations", type=int, default=3)
    ps.add_argument("--ncu-num", type=int, default=5)
    ps.add_argument("--branches", type=int, default=4)
    ps.add_argument("--dims", required=True, help="JSON dict of name->int")
    ps.add_argument("--noise-threshold-pct", type=float, default=2.0)
    ps.add_argument("--ptr-size", type=int, default=0)
    ps.add_argument("--env-out", type=str, default="")
    ps.add_argument("--warmup", type=int, default=10)
    ps.add_argument("--repeat", type=int, default=20)
    ps.set_defaults(func=cmd_setup)

    po = sub.add_parser("open-iter")
    po.add_argument("--run-dir", required=True)
    po.add_argument("--iter", type=int, required=True)
    po.add_argument("--benchmark", default=_default_bench)
    po.set_defaults(func=cmd_open_iter)

    pc = sub.add_parser("close-iter")
    pc.add_argument("--run-dir", required=True)
    pc.add_argument("--iter", type=int, required=True)
    pc.add_argument("--benchmark", default=_default_bench)
    pc.add_argument("--warmup", type=int, default=10)
    pc.add_argument("--repeat", type=int, default=20)
    pc.add_argument("--retries", type=int, default=0)
    pc.set_defaults(func=cmd_close_iter)

    pf = sub.add_parser("finalize")
    pf.add_argument("--run-dir", required=True)
    pf.set_defaults(func=cmd_finalize)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
