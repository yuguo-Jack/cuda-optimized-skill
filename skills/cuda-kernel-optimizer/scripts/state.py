#!/usr/bin/env python3
"""Global state manager for the optimization loop (v4 — gated multi-scale).

Subcommands:
  init               create run_YYYYMMDD_HHMMSS/, seed state.json
  update             after a successful iteration, merge methods into
                     selected / effective / ineffective / implementation_failed
                     lists using attribution + SASS verification data
  set-baseline-metric  called by run_iteration.py seed-baseline
  set-best-ncu-rep   helper called by profile_ncu after promoting best
  show               pretty-print current state (debug)

state.json schema (all paths stored absolute; older runs are read compatibly):
{
  "run_dir": str,
  "baseline_file": str,
  "ref_file": str,
  "best_file": str,
  "best_metric_ms": float | null,
  "best_ncu_rep": str | null,
  "env": {...},
  "iterations_total": int,
  "ncu_num": int,
  "branches": int,
  "noise_threshold_pct": float,
  "ptr_size": int,
  "dims": dict,
  "selected_methods":   [ {id, name, axis, iter} ],
  "effective_methods":  [ {id, name, axis, iter, attribution_ms} ],
  "ineffective_methods":[ {id, name, axis, iter} ],
  "implementation_failed_methods": [ {id, name, axis, iter, note} ],
  "history": [ per-iteration records ],
  "roofline_history": [ {iter, delta_c, delta_m, delta_l, bound, budget} ],
  "frontier": [ {iter, branch, kernel, ms, methods} ]
}
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def _read(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> None:
    baseline = os.path.abspath(args.baseline)
    ref = os.path.abspath(args.ref)
    if not os.path.isfile(baseline):
        sys.exit(f"baseline not found: {baseline}")
    if not os.path.isfile(ref):
        sys.exit(f"ref not found: {ref}")

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.path.dirname(baseline), f"run_{ts}")
    os.makedirs(run_dir, exist_ok=False)

    baseline_copy_dir = os.path.join(run_dir, "baseline")
    os.makedirs(baseline_copy_dir, exist_ok=True)
    baseline_copy = os.path.join(baseline_copy_dir, os.path.basename(baseline))
    shutil.copy2(baseline, baseline_copy)

    env = {}
    if args.env and os.path.isfile(args.env):
        env = _read(args.env)

    try:
        dims = json.loads(args.dims) if args.dims else {}
    except json.JSONDecodeError as e:
        sys.exit(f"--dims must be valid JSON: {e}")

    state = {
        "run_dir": run_dir,
        "baseline_file": baseline_copy,
        "baseline_file_original": baseline,
        "ref_file": ref,
        "best_file": baseline_copy,
        "best_metric_ms": None,
        "best_ncu_rep": None,
        "env": env,
        "iterations_total": int(args.iterations),
        "ncu_num": int(args.ncu_num),
        "branches": int(args.branches),
        "noise_threshold_pct": float(args.noise_threshold_pct),
        "ptr_size": int(args.ptr_size),
        "schema_version": 4,
        "gpu": int(args.gpu),
        "gpu_uuid": ((env.get("gpus") or [{}])[int(args.gpu)].get("uuid") or (env.get("gpus") or [{}])[int(args.gpu)].get("gpu_uuid", "")
                     if isinstance(env.get("gpus"), list) and int(args.gpu) < len(env.get("gpus")) else ""),
        "compile_jobs": str(args.compile_jobs),
        "numerics_mode": args.numerics_mode,
        "archetype": args.archetype,
        "validation_seeds": args.validation_seeds,
        "workload_matrix": args.workload_matrix or "",
        "max_working_set_mb": int(args.max_working_set_mb),
        "hard_working_set_mb": int(args.hard_working_set_mb),
        "adaptive_downscale": bool(args.adaptive_downscale),
        "oom_retries": int(args.oom_retries),
        "timing_batches": int(args.timing_batches),
        "timing_repeats": int(args.timing_repeats),
        "scale_weights": {"small": 0.20, "medium": 0.30, "large": 0.50},
        "stability_policy": {"stable_cv": 0.05, "max_cv": 0.10, "max_large_regression_pct": 5.0},
        "large_regression_gate": {"max_pct": 5.0, "required": True},
        "timing_policy": {
            "batches": int(args.timing_batches), "repeats": int(args.timing_repeats),
            "stable_cv": 0.05, "max_cv": 0.10, "robust_cv_primary": True,
            "calibration_target_ms": 40.0,
        },
        "profiling_policy": {
            "full_max_duration_ms": 10.0, "light_max_duration_ms": 50.0,
            "timeout_sec": 120.0, "bundle": None,
        },
        "global_champion": None,
        "contract_policy": {
            "required_states": ["compile_pass", "contract_pass", "correctness_pass", "race_safe", "timing_valid"],
            "block_timing_on_contract_fail": True,
        },
        "build_cache_dir": os.path.join(run_dir, ".build-cache"),
        "dims": dims,
        "selected_methods": [],
        "effective_methods": [],
        "ineffective_methods": [],
        "implementation_failed_methods": [],
        "unverified_methods": [],
        "history": [],
        "roofline_history": [],
        "frontier": [],
        "created_at": ts,
    }
    state_path = os.path.join(run_dir, "state.json")
    _write(state_path, state)

    for i in range(1, state["iterations_total"] + 1):
        os.makedirs(os.path.join(run_dir, f"iterv{i}"), exist_ok=True)

    print(json.dumps({"run_dir": run_dir, "state": state_path}, indent=2))


# ---------------------------------------------------------------------------
# update  (v2: uses attribution + sass_check)
# ---------------------------------------------------------------------------

def _method_key(m: dict) -> str:
    if "id" in m and m["id"]:
        return str(m["id"]).strip().lower()
    return f"{str(m.get('name','')).strip().lower()}::{str(m.get('axis','')).strip().lower()}"


def _merge_unique(bag: list[dict], new_items: list[dict]) -> list[dict]:
    seen = {_method_key(m) for m in bag}
    for m in new_items:
        k = _method_key(m)
        if k not in seen:
            bag.append(m)
            seen.add(k)
    return bag


def cmd_update(args: argparse.Namespace) -> None:
    state = _read(args.state)
    # Older runs remain readable; missing v4 collections are initialized and
    # never retroactively migrated or reinterpreted.
    for key in ("selected_methods", "effective_methods", "ineffective_methods",
                "implementation_failed_methods", "unverified_methods",
                "history", "roofline_history", "frontier"):
        state.setdefault(key, [])
    bench = _read(args.bench)
    methods = _read(args.methods_json)

    if not isinstance(methods, dict) or "methods" not in methods:
        sys.exit("methods-json must contain a top-level 'methods' list")
    methods_list = methods["methods"]

    # --- Priority-compliance validation ---
    if not args.skip_validation:
        validator = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "validate_methods.py")
        cmd = [
            sys.executable, validator,
            "--methods", args.methods_json,
            "--state", args.state,
        ]
        if args.allow_ineffective:
            cmd.append("--allow-ineffective")
        rv = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="ignore")
        if rv.returncode != 0:
            sys.stderr.write(
                "\n[state update] methods.json failed validation:\n"
            )
            sys.stderr.write(rv.stdout or "")
            sys.stderr.write(rv.stderr or "")
            sys.exit(1)

    validation_passed = bool(bench.get("correctness", {}).get("passed", True))
    bench_states = bench.get("states", {}) if isinstance(bench.get("states", {}), dict) else {}
    if any(bench_states.get(name) == "fail" for name in ("compile_pass", "contract_pass", "correctness_pass", "timing_valid")):
        validation_passed = False
    new_ms = None
    ref_ms = None
    if bench.get("kernel"):
        new_ms = bench["kernel"].get("average_ms")
    # Launch/fusion/overlap strategies are selected on end-to-end latency when
    # an adapter supplied that measurement; ordinary kernels remain
    # kernel-only for backward compatibility.
    method_ids = {str(m.get("id", "")) for m in methods_list}
    e2e_methods = {"latency.cuda_graphs", "latency.static_launch_grid_graph",
                   "latency.independent_kernel_overlap", "memory.kernel_fusion",
                   "latency.grouped_gemm_scheduler"}
    if method_ids & e2e_methods and bench.get("end_to_end_ms") is not None:
        new_ms = bench.get("end_to_end_ms")
    if bench.get("reference"):
        ref_ms = bench["reference"].get("average_ms")

    # Load attribution and sass_check if provided
    attribution_data = {}
    if args.attribution and os.path.isfile(args.attribution):
        attr = _read(args.attribution)
        for a in attr.get("attributions", []):
            attribution_data[a["method_id"]] = a

    sass_data = {}
    if args.sass_check and os.path.isfile(args.sass_check):
        sass = _read(args.sass_check)
        for c in sass.get("checks", []):
            sass_data[c["method_id"]] = c

    # Decide improvement
    best_before = state.get("best_metric_ms")
    threshold = 1.0 - (state.get("noise_threshold_pct", 2.0) / 100.0)
    improved = False
    speedup_vs_best_before = None
    if validation_passed and new_ms and new_ms > 0:
        if best_before is None:
            improved = True
        else:
            speedup_vs_best_before = best_before / new_ms
            improved = new_ms < best_before * threshold

    # Annotate each method
    for m in methods_list:
        m.setdefault("id", _method_key(m))
        m["iter"] = int(args.iter)

    # Always: add to selected
    _merge_unique(state["selected_methods"], methods_list)

    # Classify each method based on attribution + SASS
    for m in methods_list:
        mid = m["id"]
        attr_info = attribution_data.get(mid, {})
        sass_info = sass_data.get(mid, {})

        sass_status = sass_info.get("status")
        sass_verified = sass_info.get("verified", False) if sass_status is None else sass_status == "pass"
        sass_inconclusive = sass_status in {None, "inconclusive", "not_applicable", "tool_error"}
        contributed = attr_info.get("contributed", None)
        attr_ms = attr_info.get("attribution_ms", None)

        m_entry = dict(m)

        if sass_status == "fail":
            # SASS signature missing — implementation failed
            m_entry["note"] = f"SASS patterns not found: {sass_info.get('patterns_missing', [])}"
            state["implementation_failed_methods"].append(m_entry)
        elif sass_inconclusive:
            m_entry["note"] = f"verification_status={sass_status or 'missing'}"
            state["unverified_methods"].append(m_entry)
        elif contributed is True or contributed is None:
            # Contributed (or no ablation data — assume effective if overall improved)
            if validation_passed and improved:
                if attr_ms is not None:
                    m_entry["attribution_ms"] = attr_ms
                if speedup_vs_best_before is not None:
                    m_entry["speedup_vs_best_before"] = speedup_vs_best_before
                state["effective_methods"].append(m_entry)
            elif validation_passed:
                state["ineffective_methods"].append(m_entry)
        elif contributed is False:
            # Attribution says it didn't help
            m_entry["note"] = f"attribution_ms={attr_ms}"
            state["ineffective_methods"].append(m_entry)

    # Update best
    if validation_passed and improved:
        state["best_file"] = os.path.abspath(args.kernel)
        state["best_metric_ms"] = new_ms
        if bench.get("artifact_manifest"):
            state["best_build_manifest"] = bench["artifact_manifest"]
        if args.kernel_ncu_rep and os.path.isfile(args.kernel_ncu_rep):
            state["best_ncu_rep"] = os.path.abspath(args.kernel_ncu_rep)
        state["global_champion"] = {
            "iter": int(args.iter),
            "kernel": state["best_file"],
            "average_ms": new_ms,
            "artifact_manifest": state.get("best_build_manifest"),
        }

    # Load roofline data if available
    iter_dir = os.path.join(state["run_dir"], f"iterv{args.iter}")
    roofline_path = os.path.join(iter_dir, "roofline.json")
    if os.path.isfile(roofline_path):
        roofline = _read(roofline_path)
        state["roofline_history"].append({
            "iter": int(args.iter),
            "delta_compute": roofline.get("delta_compute"),
            "delta_memory": roofline.get("delta_memory"),
            "delta_latency": roofline.get("delta_latency"),
            "bound": roofline.get("bound"),
            "axis_budget": roofline.get("axis_budget"),
        })

    # Load frontier from branch_results if available
    branch_results_path = os.path.join(iter_dir, "branch_results.json")
    if os.path.isfile(branch_results_path):
        br = _read(branch_results_path)
        for fe in br.get("frontier", []):
            fe["methods"] = [m["id"] for m in methods_list]
            state["frontier"].append(fe)

    status = (
        "improved" if (validation_passed and improved)
        else "regressed" if validation_passed
        else "failed_validation"
    )
    state["history"].append({
        "iter": int(args.iter),
        "kernel_file": os.path.abspath(args.kernel),
        "methods": [m["id"] for m in methods_list],
        "method_names": [m.get("name") for m in methods_list],
        "ms": new_ms,
        "ref_ms": ref_ms,
        "speedup_vs_ref": (ref_ms / new_ms) if (ref_ms and new_ms and new_ms > 0) else None,
        "speedup_vs_best_before": speedup_vs_best_before,
        "validation_passed": validation_passed,
        "status": status,
        "retries": int(args.retries),
        "states": bench_states,
        "kernel_only_ms": bench.get("kernel_only_ms", new_ms),
        "end_to_end_ms": bench.get("end_to_end_ms", new_ms),
    })

    _write(args.state, state)
    print(json.dumps({
        "iter": args.iter,
        "status": status,
        "new_ms": new_ms,
        "best_ms": state["best_metric_ms"],
        "improved": improved,
        "speedup_vs_best_before": speedup_vs_best_before,
    }, indent=2))


# ---------------------------------------------------------------------------
# set-best-ncu-rep
# ---------------------------------------------------------------------------

def cmd_set_best_ncu(args: argparse.Namespace) -> None:
    state = _read(args.state)
    state["best_ncu_rep"] = os.path.abspath(args.ncu_rep)
    _write(args.state, state)
    print(json.dumps({"best_ncu_rep": state["best_ncu_rep"]}, indent=2))


# ---------------------------------------------------------------------------
# seed baseline metric
# ---------------------------------------------------------------------------

def cmd_set_baseline_metric(args: argparse.Namespace) -> None:
    state = _read(args.state)
    bench = _read(args.bench)
    if not bench.get("correctness", {}).get("passed", True):
        sys.exit("Baseline failed correctness validation — cannot proceed.")
    states = bench.get("states", {}) if isinstance(bench.get("states", {}), dict) else {}
    if any(states.get(k) == "fail" for k in ("compile_pass", "contract_pass", "correctness_pass", "timing_valid")):
        sys.exit("Baseline failed a required contract/timing gate — cannot proceed.")
    ms = bench.get("kernel", {}).get("average_ms")
    if ms is None:
        sys.exit("Baseline bench has no kernel timing.")
    state["best_metric_ms"] = ms
    if bench.get("artifact_manifest"):
        state["best_build_manifest"] = bench["artifact_manifest"]
    _write(args.state, state)
    print(json.dumps({"baseline_ms": ms}, indent=2))


def cmd_show(args: argparse.Namespace) -> None:
    state = _read(args.state)
    print(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init")
    pi.add_argument("--baseline", required=True)
    pi.add_argument("--ref", required=True)
    pi.add_argument("--iterations", type=int, default=3)
    pi.add_argument("--ncu-num", type=int, default=5)
    pi.add_argument("--branches", type=int, default=4)
    pi.add_argument("--dims", type=str, default="{}", help="JSON dict of dim name -> int")
    pi.add_argument("--env", type=str, default="")
    pi.add_argument("--noise-threshold-pct", type=float, default=2.0)
    pi.add_argument("--ptr-size", type=int, default=0)
    pi.add_argument("--gpu", type=int, default=0)
    pi.add_argument("--compile-jobs", default="auto")
    pi.add_argument("--numerics-mode", choices=["reference", "strict", "approximate"], default="reference")
    pi.add_argument("--archetype", default="generic")
    pi.add_argument("--validation-seeds", default="7,19,41,73,101",
                    help="Comma-separated seeds passed to every candidate validation")
    pi.add_argument("--workload-matrix", default="")
    pi.add_argument("--max-working-set-mb", type=int, default=384)
    pi.add_argument("--hard-working-set-mb", type=int, default=512)
    pi.add_argument("--adaptive-downscale", action=argparse.BooleanOptionalAction, default=True)
    pi.add_argument("--oom-retries", type=int, default=3)
    pi.add_argument("--timing-batches", type=int, default=5)
    pi.add_argument("--timing-repeats", type=int, default=30)
    pi.set_defaults(func=cmd_init)

    pu = sub.add_parser("update")
    pu.add_argument("--state", required=True)
    pu.add_argument("--iter", required=True, type=int)
    pu.add_argument("--kernel", required=True)
    pu.add_argument("--bench", required=True)
    pu.add_argument("--methods-json", required=True)
    pu.add_argument("--attribution", type=str, default=None,
                    help="Path to attribution.json from ablation step")
    pu.add_argument("--sass-check", type=str, default=None,
                    help="Path to sass_check.json from SASS verification step")
    pu.add_argument("--retries", type=int, default=0)
    pu.add_argument("--skip-validation", action="store_true")
    pu.add_argument("--allow-ineffective", action="store_true")
    pu.add_argument("--kernel-ncu-rep", type=str, default="")
    pu.set_defaults(func=cmd_update)

    pb = sub.add_parser("set-baseline-metric")
    pb.add_argument("--state", required=True)
    pb.add_argument("--bench", required=True)
    pb.set_defaults(func=cmd_set_baseline_metric)

    pbn = sub.add_parser("set-best-ncu-rep")
    pbn.add_argument("--state", required=True)
    pbn.add_argument("--ncu-rep", required=True)
    pbn.set_defaults(func=cmd_set_best_ncu)

    ps = sub.add_parser("show")
    ps.add_argument("--state", required=True)
    ps.set_defaults(func=cmd_show)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
