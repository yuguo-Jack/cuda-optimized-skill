#!/usr/bin/env python3
"""End-to-end strict orchestrator (v3 — hardware-gated branch-and-select).

Subcommands:
  setup       Steps 0-2: env check, preflight, init, seed baseline, profile+roofline for iter 1
  open-iter   Prepare an iteration: profile best → ncu_top → roofline → axis budgets
              (Claude then writes K branch kernels + methods.json + analysis.md)
  close-iter  Steps 3e-3j: branch explore → champion → ncu champion → ablate → sass → update
  finalize    Step 4: emit summary.md
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from hardware_gate import run_gate
from strict_validation import benchmark_gate, ncu_gate, write_stop

SCRIPT_DIR = Path(__file__).resolve().parent


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"[run] {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, text=True, **kw)


def _read(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write(path: str, payload: dict) -> None:
    target = Path(path)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, target)


def _backend(path: str) -> str:
    if path.endswith(".py"):
        return "triton"
    source = Path(path).read_text(encoding="utf-8", errors="ignore")
    return "cutlass" if "cutlass/" in source or "cute/" in source else "cuda"


def _is_bundled_benchmark(path: str) -> bool:
    return os.path.realpath(path) == os.path.realpath(SCRIPT_DIR / "benchmark.py")


def _emit_summary(state_path: str) -> None:
    state = _read(state_path)
    _run([sys.executable, str(SCRIPT_DIR / "summarize.py"), "--state", state_path,
          "--out", os.path.join(state["run_dir"], "summary.md")])


def _stop(state_path: str, *, reason: str, stage: str, iteration: int | None,
          error: str, attempts: int = 0, exit_code: int = 2) -> None:
    payload = write_stop(state_path, reason=reason, stage=stage, iteration=iteration,
                         last_error=error, attempts=attempts, exit_code=exit_code)
    _emit_summary(state_path)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    raise SystemExit(exit_code)


def _refresh_hardware(state_path: str, *, backend: str, require_torch: bool,
                      out_path: str) -> dict:
    result = run_gate(backend=backend, require_torch=require_torch, attempts=3)
    _write(out_path, result)
    state = _read(state_path)
    state["env"] = result
    state["hardware_attempts"] = int(result.get("hardware_attempts", 0))
    _write(state_path, state)
    return result


def _open_iteration(state_path: str, iteration: int, benchmark: str) -> dict:
    state = _read(state_path)
    if int(state.get("schema_version", 0)) < 5:
        raise SystemExit("legacy run is read-only; create a new strict run")
    if state.get("run_status") == "stopped":
        raise SystemExit(f"run is stopped: {state.get('stop_reason')}")
    if iteration < 1 or iteration > int(state.get("iterations_total", 0)):
        raise SystemExit(f"iteration {iteration} is outside configured range")
    expected_previous = iteration - 1
    if not state.get("baseline_verified") or int(state.get("verified_iterations", 0)) != expected_previous:
        _stop(state_path, reason="iteration_timing_invalid", stage="open_iter_precondition",
              iteration=iteration, error="previous best lacks complete strict validation evidence")

    backend = _backend(state["best_file"])
    gate_path = os.path.join(state["run_dir"], f"hardware_iterv{iteration}.json")
    gate = _refresh_hardware(state_path, backend=backend,
                             require_torch=_is_bundled_benchmark(benchmark), out_path=gate_path)
    if not gate.get("generation_allowed"):
        _stop(state_path, reason="iteration_hardware_unavailable", stage="open_iter_hardware",
              iteration=iteration, error="; ".join(gate["attempts"][-1].get("errors", [])),
              attempts=int(gate.get("hardware_attempts", 0)), exit_code=3)

    stage_dir = tempfile.mkdtemp(prefix=f".iterv{iteration}-profile-", dir=state["run_dir"])
    try:
        result = _run([
            sys.executable, str(SCRIPT_DIR / "profile_ncu.py"), "--state", state_path,
            "--iter", str(iteration), "--which", "best_input", "--output-dir", stage_dir,
            "--benchmark", os.path.abspath(benchmark), "--strict",
        ])
        top_path = os.path.join(stage_dir, "ncu_top.json")
        top = _read(top_path) if os.path.isfile(top_path) else {}
        valid, error = ncu_gate(top, os.path.join(stage_dir, "best_input.ncu-rep"))
        if result.returncode != 0 or not valid:
            _stop(state_path, reason="iteration_ncu_failed", stage="open_iter_ncu",
                  iteration=iteration, error=error or f"profile_ncu rc={result.returncode}", exit_code=3)

        iter_dir = os.path.join(state["run_dir"], f"iterv{iteration}")
        os.makedirs(iter_dir, exist_ok=False)
        for item in Path(stage_dir).iterdir():
            shutil.move(str(item), os.path.join(iter_dir, item.name))
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

    roofline_result = _run([sys.executable, str(SCRIPT_DIR / "roofline.py"),
                            "--state", state_path, "--iter", str(iteration)])
    if roofline_result.returncode != 0:
        _stop(state_path, reason="iteration_ncu_failed", stage="roofline",
              iteration=iteration, error=f"roofline rc={roofline_result.returncode}")
    roofline = _read(os.path.join(state["run_dir"], f"iterv{iteration}", "roofline.json"))
    early_stop = bool(roofline.get("near_peak", False))
    if not early_stop:
        for index in range(1, int(state.get("branches", 4)) + 1):
            os.makedirs(os.path.join(state["run_dir"], f"iterv{iteration}", "branches", f"b{index}"))
    state = _read(state_path)
    state["run_status"] = "completed" if early_stop else "running"
    state["generation_allowed"] = not early_stop
    _write(state_path, state)
    return {"iter": iteration, "early_stop": early_stop,
            "branches_dir": os.path.join(state["run_dir"], f"iterv{iteration}", "branches"),
            "num_branches": int(state.get("branches", 4))}


def _init_run(args, env_json: str) -> tuple[str, str]:
    init = _run([
        sys.executable, str(SCRIPT_DIR / "state.py"), "init",
        "--baseline", os.path.abspath(args.baseline), "--ref", os.path.abspath(args.ref),
        "--iterations", str(args.iterations), "--ncu-num", str(args.ncu_num),
        "--branches", str(args.branches), "--dims", args.dims, "--env", env_json,
        "--noise-threshold-pct", str(args.noise_threshold_pct), "--ptr-size", str(args.ptr_size),
        "--compile-jobs", str(args.compile_jobs), "--numerics-mode", args.numerics_mode,
        "--archetype", args.archetype, "--validation-seeds", args.validation_seeds,
        "--workload-matrix", args.workload_matrix, "--max-working-set-mb", str(args.max_working_set_mb),
        "--hard-working-set-mb", str(args.hard_working_set_mb),
        "--adaptive-downscale" if args.adaptive_downscale else "--no-adaptive-downscale",
        "--oom-retries", str(args.oom_retries), "--timing-batches", str(args.timing_batches),
        "--timing-repeats", str(args.timing_repeats),
    ], capture_output=True)
    if init.returncode != 0:
        sys.stderr.write(init.stderr or "")
        raise SystemExit("state init failed")
    try:
        info = json.loads(init.stdout or "{}")
    except json.JSONDecodeError:
        info = {}
    if not info.get("run_dir") or not info.get("state"):
        raise SystemExit(f"could not parse state init output:\n{init.stdout}")
    return info["run_dir"], info["state"]


# ---------------------------------------------------------------------------
# setup  —  steps 0, 1, 2, and open-iter(1)
# ---------------------------------------------------------------------------

def cmd_setup(args):
    env_json = os.path.abspath(args.env_out or "./env.json")
    baseline = os.path.abspath(args.baseline)
    benchmark = os.path.abspath(args.benchmark)
    backend = _backend(baseline)

    # Strict environment gate runs before any run or iteration directory exists.
    env = run_gate(backend=backend, require_torch=_is_bundled_benchmark(benchmark), attempts=3)
    if args.diagnostic:
        env["probe_passed"] = env["generation_allowed"]
        env["generation_allowed"] = False
        env["diagnostic_only"] = True
        env["stop_reason"] = "diagnostic_only"
        _write(env_json, env)
        print(json.dumps(env, indent=2, ensure_ascii=False))
        return
    _write(env_json, env)
    if not env.get("generation_allowed"):
        _, state_path = _init_run(args, env_json)
        _stop(state_path, reason="environment_unavailable", stage="setup_hardware",
              iteration=None, error="; ".join(env["attempts"][-1].get("errors", [])),
              attempts=int(env.get("hardware_attempts", 0)), exit_code=3)

    # 0b. preflight
    preflight = _run([
        sys.executable, str(SCRIPT_DIR / "preflight.py"),
        "--baseline", baseline,
        "--ref", os.path.abspath(args.ref),
        "--dims", args.dims,
    ])
    if preflight.returncode != 0:
        run_dir, state_path = _init_run(args, env_json)
        _stop(state_path, reason="baseline_preflight_failed", stage="baseline_preflight",
              iteration=None, error="baseline/reference contract preflight failed", exit_code=4)

    # 1. init run dir + state
    run_dir, state_path = _init_run(args, env_json)

    # 2. seed baseline
    rc = _run([
        sys.executable, str(SCRIPT_DIR / "run_iteration.py"), "seed-baseline",
        "--state", state_path,
        "--benchmark", benchmark,
        "--warmup", str(args.warmup),
        "--repeat", str(args.repeat),
    ]).returncode
    if rc != 0:
        bench_path = os.path.join(run_dir, "baseline", "bench.json")
        bench = _read(bench_path) if os.path.isfile(bench_path) else {}
        states = bench.get("states", {}) if isinstance(bench.get("states"), dict) else {}
        if states.get("compile_pass") == "fail":
            reason = "baseline_compile_failed"
        elif states.get("correctness_pass") == "fail" or bench.get("correctness", {}).get("passed") is False:
            reason = "baseline_correctness_failed"
        else:
            reason = "baseline_timing_invalid"
        _stop(state_path, reason=reason, stage="baseline_benchmark", iteration=None,
              error=str(bench.get("error") or "strict baseline benchmark failed"))

    bench = _read(os.path.join(run_dir, "baseline", "bench.json"))
    valid, error = benchmark_gate(
        bench, max_cv=float((_read(state_path).get("stability_policy") or {}).get("max_cv", 0.10)),
        require_reference_timing=True)
    if not valid:
        _stop(state_path, reason="baseline_timing_invalid", stage="baseline_benchmark",
              iteration=None, error=error or "invalid benchmark evidence")

    # Baseline NCU evidence is stored under baseline/, never in a future iteration.
    rc = _run([
        sys.executable, str(SCRIPT_DIR / "profile_ncu.py"),
        "--state", state_path,
        "--iter", "0", "--which", "baseline",
        "--output-dir", os.path.join(run_dir, "baseline"),
        "--benchmark", benchmark, "--strict",
    ]).returncode
    top_path = os.path.join(run_dir, "baseline", "ncu_top.json")
    top = _read(top_path) if os.path.isfile(top_path) else {}
    ncu_valid, ncu_error = ncu_gate(top, os.path.join(run_dir, "baseline", "baseline.ncu-rep"))
    if rc != 0 or not ncu_valid:
        _stop(state_path, reason="baseline_ncu_failed", stage="baseline_ncu", iteration=None,
              error=ncu_error or f"profile_ncu rc={rc}", exit_code=3)

    state = _read(state_path)
    state["baseline_ncu_rep"] = os.path.join(run_dir, "baseline", "baseline.ncu-rep")
    state["baseline_ncu_top"] = top_path
    state["run_status"] = "ready"
    state["generation_allowed"] = True
    _write(state_path, state)

    opened = _open_iteration(state_path, 1, benchmark)
    early_stop = opened["early_stop"]

    print(json.dumps({
        "run_dir": run_dir,
        "state": state_path,
        "env": env_json,
        "early_stop": early_stop,
        "next_step": (
            "Claude should now read iterv1/roofline.json (for axis budgets), "
            "iterv1/ncu_top.json, and state.json, then write "
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

    opened = _open_iteration(state_path, args.iter, os.path.abspath(args.benchmark))
    early_stop = opened["early_stop"]
    num_branches = opened["num_branches"]
    branches_dir = opened["branches_dir"]

    print(json.dumps({
        "iter": args.iter,
        "early_stop": early_stop,
        "branches_dir": branches_dir,
        "num_branches": num_branches,
        "next_step": (
            f"Claude should read iterv{args.iter}/roofline.json and ncu_top.json, "
            f"write {num_branches} branch kernels under iterv{args.iter}/branches/b{{1..{num_branches}}}/kernel.<ext>, "
            f"plus iterv{args.iter}/methods.json and iterv{args.iter}/analysis.md. "
            f"Then run: orchestrate.py close-iter --run-dir {args.run_dir} --iter {args.iter}"
        ) if not early_stop else "Near roofline — consider stopping.",
    }, indent=2))


# ---------------------------------------------------------------------------
# close-iter  —  branch explore → ncu champion → ablate → sass → update
# ---------------------------------------------------------------------------

def cmd_close_iter(args):
    state_path = os.path.join(args.run_dir, "state.json")
    if not os.path.isfile(state_path):
        sys.exit(f"state.json missing: {state_path}")

    state = _read(state_path)
    if int(state.get("schema_version", 0)) < 5:
        raise SystemExit("legacy run is read-only; create a new strict run")
    iter_dir = os.path.join(args.run_dir, f"iterv{args.iter}")
    methods_json = os.path.join(iter_dir, "methods.json")
    if not os.path.isfile(methods_json):
        _stop(state_path, reason="iteration_no_valid_branch", stage="iteration_input",
              iteration=args.iter, error=f"methods.json missing at {methods_json}", exit_code=4)

    # Step 3e: Branch explore — compile + benchmark all branches
    branch_result = _run([
        sys.executable, str(SCRIPT_DIR / "branch_explore.py"),
        "--state", state_path,
        "--iter", str(args.iter),
        "--benchmark", os.path.abspath(args.benchmark),
        "--warmup", str(args.warmup),
        "--repeat", str(args.repeat),
        "--compile-jobs", str(args.compile_jobs),
    ], capture_output=True)
    sys.stderr.write(branch_result.stderr or "")

    if branch_result.returncode == 2:
        _stop(state_path, reason="iteration_no_valid_branch", stage="branch_explore",
              iteration=args.iter, error=(branch_result.stderr or "all branches failed")[-2000:])
    if branch_result.returncode != 0:
        _stop(state_path, reason="iteration_no_valid_branch", stage="branch_explore",
              iteration=args.iter, error=f"branch_explore failed rc={branch_result.returncode}")

    # Find champion kernel
    kernel = None
    for ext in (".cu", ".py"):
        candidate = os.path.join(iter_dir, f"kernel{ext}")
        if os.path.isfile(candidate):
            kernel = candidate
            break
    if not kernel:
        _stop(state_path, reason="iteration_no_valid_branch", stage="branch_explore",
              iteration=args.iter, error="No champion kernel found after branch_explore")

    bench_json = os.path.join(iter_dir, "bench.json")
    if not os.path.isfile(bench_json):
        _stop(state_path, reason="iteration_timing_invalid", stage="champion_benchmark",
              iteration=args.iter, error="bench.json missing for champion")

    bench = _read(bench_json)
    valid, error = benchmark_gate(
        bench, max_cv=float((state.get("stability_policy") or {}).get("max_cv", 0.10)),
        require_reference_timing=True)
    if not valid:
        reason = ("iteration_correctness_failed" if "correctness" in (error or "")
                  else "iteration_timing_invalid")
        _stop(state_path, reason=reason, stage="champion_benchmark", iteration=args.iter,
              error=error or "champion failed strict benchmark gate")

    # Step 3g: Profile champion with ncu (MANDATORY full report)
    rc = _run([
        sys.executable, str(SCRIPT_DIR / "profile_ncu.py"),
        "--state", state_path,
        "--iter", str(args.iter),
        "--which", "kernel",
        "--benchmark", os.path.abspath(args.benchmark),
        "--promote-if-best", "--strict",
    ]).returncode
    kernel_ncu_rep = os.path.join(iter_dir, "kernel.ncu-rep")
    ncu_top_path = os.path.join(iter_dir, "ncu_top.json")
    ncu_top = _read(ncu_top_path) if os.path.isfile(ncu_top_path) else {}
    ncu_valid, ncu_error = ncu_gate(ncu_top, kernel_ncu_rep)
    if rc != 0 or not ncu_valid:
        _stop(state_path, reason="iteration_ncu_failed", stage="champion_ncu",
              iteration=args.iter, error=ncu_error or f"profile_ncu rc={rc}", exit_code=3)

    # Step 3h: Ablation attribution (optional — runs if ablation kernels exist)
    attribution_path = os.path.join(iter_dir, "attribution.json")
    ablation_dir = os.path.join(iter_dir, "ablations")
    if os.path.isdir(ablation_dir):
        _run([
            sys.executable, str(SCRIPT_DIR / "ablate.py"),
            "--state", state_path,
            "--iter", str(args.iter),
            "--benchmark", os.path.abspath(args.benchmark),
            "--compile-jobs", str(state.get("compile_jobs", "auto")),
        ])

    # Step 3i: SASS verification
    sass_check_path = os.path.join(iter_dir, "sass_check.json")
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
        "--kernel-ncu-rep", kernel_ncu_rep,
    ]
    if os.path.isfile(attribution_path):
        update_cmd.extend(["--attribution", attribution_path])
    if os.path.isfile(sass_check_path):
        update_cmd.extend(["--sass-check", sass_check_path])

    rc = _run(update_cmd).returncode
    if rc != 0:
        _stop(state_path, reason="iteration_timing_invalid", stage="state_update",
              iteration=args.iter, error=f"state update failed rc={rc}")

    # Open the next iteration only after state.update records this one as verified.
    state = _read(state_path)
    next_iter = args.iter + 1
    if next_iter <= state["iterations_total"]:
        opened = _open_iteration(state_path, next_iter, os.path.abspath(args.benchmark))
        early_stop = opened["early_stop"]
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
    ps.add_argument("--compile-jobs", default="auto")
    ps.add_argument("--numerics-mode", choices=["reference", "strict", "approximate"], default="reference")
    ps.add_argument("--archetype", default="generic",
                    help="Workload archetype, e.g. attention, gemm, reduction")
    ps.add_argument("--validation-seeds", default="7,19,41,73,101",
                    help="Comma-separated validation seeds")
    ps.add_argument("--workload-matrix", default="")
    ps.add_argument("--max-working-set-mb", type=int, default=384)
    ps.add_argument("--hard-working-set-mb", type=int, default=512)
    ps.add_argument("--adaptive-downscale", action=argparse.BooleanOptionalAction, default=True)
    ps.add_argument("--oom-retries", type=int, default=3)
    ps.add_argument("--timing-batches", type=int, default=5)
    ps.add_argument("--timing-repeats", type=int, default=30)
    ps.add_argument("--env-out", type=str, default="")
    ps.add_argument("--warmup", type=int, default=10)
    ps.add_argument("--repeat", type=int, default=20)
    ps.add_argument("--diagnostic", action="store_true",
                    help="Only run hardware discovery; never create a run or iteration code.")
    ps.set_defaults(func=cmd_setup)

    po = sub.add_parser("open-iter")
    po.add_argument("--run-dir", required=True)
    po.add_argument("--iter", type=int, required=True)
    po.add_argument("--benchmark", default=_default_bench)
    po.add_argument("--compile-jobs", default="")
    po.set_defaults(func=cmd_open_iter)

    pc = sub.add_parser("close-iter")
    pc.add_argument("--run-dir", required=True)
    pc.add_argument("--iter", type=int, required=True)
    pc.add_argument("--benchmark", default=_default_bench)
    pc.add_argument("--warmup", type=int, default=10)
    pc.add_argument("--repeat", type=int, default=20)
    pc.add_argument("--retries", type=int, default=0)
    pc.add_argument("--compile-jobs", default="")
    pc.set_defaults(func=cmd_close_iter)

    pf = sub.add_parser("finalize")
    pf.add_argument("--run-dir", required=True)
    pf.set_defaults(func=cmd_finalize)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
