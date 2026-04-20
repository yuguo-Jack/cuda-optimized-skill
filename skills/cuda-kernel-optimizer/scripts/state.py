#!/usr/bin/env python3
"""Global state manager for the optimization loop.

Subcommands:
  init           create run_YYYYMMDD_HHMMSS/ next to the baseline file,
                 seed state.json with empty method lists.
  update         after a successful iteration, merge the new methods into
                 selected / effective / ineffective lists, update best_file
                 and best_metric_ms when the new kernel is strictly faster.
  show           pretty-print current state (debug).

state.json schema (all paths stored absolute):
{
  "run_dir": str,
  "baseline_file": str,
  "ref_file": str,
  "best_file": str,
  "best_metric_ms": float | null,
  "best_ncu_rep": str | null,
  "env": {...},                      # inlined snapshot for reproducibility
  "iterations_total": int,
  "ncu_num": int,
  "noise_threshold_pct": float,      # speedup must exceed this to count as improvement
  "dims": dict,
  "selected_methods":   [ {id, name, axis, iter} ],
  "effective_methods":  [ {id, name, axis, iter, speedup} ],
  "ineffective_methods":[ {id, name, axis, iter} ],
  "history": [
     {iter, kernel_file, status, methods:[ids],
      ms, ref_ms, speedup_vs_ref, speedup_vs_best_before,
      validation_passed, retries}
  ]
}
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
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
        "noise_threshold_pct": float(args.noise_threshold_pct),
        "dims": dims,
        "selected_methods": [],
        "effective_methods": [],
        "ineffective_methods": [],
        "history": [],
        "created_at": ts,
    }
    state_path = os.path.join(run_dir, "state.json")
    _write(state_path, state)

    for i in range(1, state["iterations_total"] + 1):
        os.makedirs(os.path.join(run_dir, f"iterv{i}"), exist_ok=True)

    print(json.dumps({"run_dir": run_dir, "state": state_path}, indent=2))


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

def _method_key(m: dict) -> str:
    # Prefer explicit id; fall back to lowercased name+axis
    if "id" in m and m["id"]:
        return str(m["id"]).strip().lower()
    return f"{str(m.get('name','')).strip().lower()}::{str(m.get('axis','')).strip().lower()}"


def _merge_unique(bag: list[dict], new_items: list[dict]) -> list[dict]:
    seen = { _method_key(m) for m in bag }
    for m in new_items:
        k = _method_key(m)
        if k not in seen:
            bag.append(m)
            seen.add(k)
    return bag


def cmd_update(args: argparse.Namespace) -> None:
    state = _read(args.state)
    bench = _read(args.bench)
    methods = _read(args.methods_json)  # expected: {"methods": [ {id,name,axis,priority,...}, ... ]}

    if not isinstance(methods, dict) or "methods" not in methods:
        sys.exit("methods-json must contain a top-level 'methods' list")
    methods_list = methods["methods"]
    if len(methods_list) != 3:
        print(f"[warn] expected 3 methods, got {len(methods_list)}", file=sys.stderr)

    # --- NEW: priority-compliance validation ---
    # Run validate_methods.py against the registry + state. Skip only if the
    # caller explicitly passes --skip-validation (emergency escape hatch,
    # should never be used in normal runs).
    if not args.skip_validation:
        import subprocess as _sp
        validator = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "validate_methods.py")
        cmd = [
            sys.executable, validator,
            "--methods", args.methods_json,
            "--state", args.state,
        ]
        if args.allow_ineffective:
            cmd.append("--allow-ineffective")
        rv = _sp.run(cmd, capture_output=True, text=True,
                     encoding="utf-8", errors="ignore")
        if rv.returncode != 0:
            sys.stderr.write(
                "\n[state update] methods.json failed priority-compliance validation:\n"
            )
            sys.stderr.write(rv.stdout or "")
            sys.stderr.write(rv.stderr or "")
            sys.stderr.write(
                "\nFix methods.json (re-scan from P1, document all skipped "
                "higher-priority methods in skipped_higher) and re-run.\n"
            )
            sys.exit(1)

    validation_passed = bool(bench.get("correctness", {}).get("passed", True))
    new_ms = None
    ref_ms = None
    if bench.get("kernel"):
        new_ms = bench["kernel"].get("average_ms")
    if bench.get("reference"):
        ref_ms = bench["reference"].get("average_ms")

    # Decide improvement
    best_before = state.get("best_metric_ms")
    threshold = 1.0 - (state.get("noise_threshold_pct", 2.0) / 100.0)  # e.g. 0.98
    improved = False
    speedup_vs_best_before = None
    if validation_passed and new_ms and new_ms > 0:
        if best_before is None:
            improved = True  # first timing
            speedup_vs_best_before = None
        else:
            speedup_vs_best_before = best_before / new_ms
            improved = new_ms < best_before * threshold

    # Annotate each method
    for m in methods_list:
        m.setdefault("id", _method_key(m))
        m["iter"] = int(args.iter)

    # Always: selected
    _merge_unique(state["selected_methods"], methods_list)

    if validation_passed and improved:
        for m in methods_list:
            m_e = dict(m)
            if speedup_vs_best_before is not None:
                m_e["speedup_vs_best_before"] = speedup_vs_best_before
            state["effective_methods"].append(m_e)
        state["best_file"] = os.path.abspath(args.kernel)
        state["best_metric_ms"] = new_ms
        # best_ncu_rep may be set separately by profile_ncu.py when profiling the new kernel
    elif validation_passed:
        state["ineffective_methods"].extend(methods_list)

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
# set-best-ncu-rep  (helper called by profile_ncu after we promote best)
# ---------------------------------------------------------------------------

def cmd_set_best_ncu(args: argparse.Namespace) -> None:
    state = _read(args.state)
    state["best_ncu_rep"] = os.path.abspath(args.ncu_rep)
    _write(args.state, state)
    print(json.dumps({"best_ncu_rep": state["best_ncu_rep"]}, indent=2))


# ---------------------------------------------------------------------------
# seed baseline metric (called by run_iteration.py seed-baseline)
# ---------------------------------------------------------------------------

def cmd_set_baseline_metric(args: argparse.Namespace) -> None:
    state = _read(args.state)
    bench = _read(args.bench)
    if not bench.get("correctness", {}).get("passed", True):
        sys.exit("Baseline failed correctness validation — cannot proceed.")
    ms = bench.get("kernel", {}).get("average_ms")
    if ms is None:
        sys.exit("Baseline bench has no kernel timing.")
    state["best_metric_ms"] = ms
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
    pi.add_argument("--dims", type=str, default="{}", help="JSON dict of dim name -> int")
    pi.add_argument("--env", type=str, default="")
    pi.add_argument("--noise-threshold-pct", type=float, default=2.0)
    pi.set_defaults(func=cmd_init)

    pu = sub.add_parser("update")
    pu.add_argument("--state", required=True)
    pu.add_argument("--iter", required=True, type=int)
    pu.add_argument("--kernel", required=True)
    pu.add_argument("--bench", required=True)
    pu.add_argument("--methods-json", required=True)
    pu.add_argument("--retries", type=int, default=0)
    pu.add_argument("--skip-validation", action="store_true",
                    help="DO NOT USE in normal runs. Skips the priority-compliance "
                         "check on methods.json. Emergency escape hatch only.")
    pu.add_argument("--allow-ineffective", action="store_true",
                    help="Allow re-selecting a method that is in "
                         "state.ineffective_methods. analysis.md must document "
                         "why the bottleneck profile has changed.")
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
