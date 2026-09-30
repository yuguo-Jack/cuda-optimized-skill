#!/usr/bin/env python3
"""Compute DCU roofline-like gaps and allocate per-axis method budgets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


TOTAL_BUDGET = 3
MAX_PER_AXIS = 2
TIE_BREAK_ORDER = ["memory", "latency", "compute"]


def compute_deltas(dcu_top: dict, env: dict) -> dict:
    """Only consume explicitly defined normalized metrics; raw counts stay raw.

    normalized.<axis> must include value, unit (percent/ratio), definition,
    source and scope. Compute/memory are achieved/appropriate attainable peak;
    latency is a measured stall fraction, never an instruction occurrence count.
    These values guide hypotheses, not a mathematical proof of the limiting roof.
    """
    import math
    values, evidence = {}, {}
    for axis in ("compute", "memory", "latency"):
        item = (dcu_top.get("normalized") or {}).get(axis, {})
        value = item.get("value")
        if (not dcu_top.get("degraded") and isinstance(value, (int, float))
                and not isinstance(value, bool) and math.isfinite(value)
                and all(item.get(k) for k in ("definition", "source", "scope"))
                and item.get("unit") in ("percent", "ratio")):
            value = value / 100 if item["unit"] == "percent" else value
            if 0 <= value <= 1:
                values[axis] = value
                evidence[axis] = item
    gaps = {a: (v if a == "latency" else 1 - v) for a, v in values.items()}
    return {**{f"delta_{a}": gaps.get(a) for a in ("compute", "memory", "latency")},
            "compute_util_pct": values.get("compute", 0) * 100 if "compute" in values else None,
            "memory_util_pct": values.get("memory", 0) * 100 if "memory" in values else None,
            "max_stall_pct": values.get("latency", 0) * 100 if "latency" in values else None,
            "degraded": len(values) != 3, "metric_evidence": evidence,
            "note": "Unknown values are not zero. Raw PMC counts, ISA counts and allocation bytes are not utilization."}


def allocate_budget(delta_c: float, delta_m: float, delta_l: float) -> dict:
    deltas = {"compute": delta_c, "memory": delta_m, "latency": delta_l}
    for axis in deltas:
        if deltas[axis] < 0.10:
            deltas[axis] = 0.0
    total = sum(deltas.values())
    if total < 0.01:
        return {"compute": 1, "memory": 1, "latency": 1}
    raw = {axis: TOTAL_BUDGET * deltas[axis] / total for axis in deltas}
    budgets = {axis: int(round(raw[axis])) for axis in deltas}
    overflow = 0
    for axis in budgets:
        if budgets[axis] > MAX_PER_AXIS:
            overflow += budgets[axis] - MAX_PER_AXIS
            budgets[axis] = MAX_PER_AXIS
    for axis in sorted(deltas, key=lambda a: -deltas[a]):
        if overflow <= 0:
            break
        if budgets[axis] < MAX_PER_AXIS:
            take = min(overflow, MAX_PER_AXIS - budgets[axis])
            budgets[axis] += take
            overflow -= take
    while sum(budgets.values()) != TOTAL_BUDGET:
        if sum(budgets.values()) < TOTAL_BUDGET:
            best = max((a for a in TIE_BREAK_ORDER if budgets[a] < MAX_PER_AXIS), key=lambda a: raw[a] - budgets[a], default=None)
            if best is None:
                break
            budgets[best] += 1
        else:
            worst = min((a for a in reversed(TIE_BREAK_ORDER) if budgets[a] > 0), key=lambda a: raw[a] - budgets[a], default=None)
            if worst is None:
                break
            budgets[worst] -= 1
    return budgets


def run(state_path: str, iteration: int) -> dict:
    with open(state_path, "r", encoding="utf-8-sig") as f:
        state = json.load(f)
    iter_dir = os.path.join(state["run_dir"], f"iterv{iteration}")
    top_path = os.path.join(iter_dir, "dcu_top.json")
    if not os.path.isfile(top_path):
        sys.exit(f"dcu_top.json not found at {top_path}")
    with open(top_path, "r", encoding="utf-8-sig") as f:
        dcu_top = json.load(f)

    deltas = compute_deltas(dcu_top, state.get("env", {}))
    measured = {a: deltas[f"delta_{a}"] for a in ("compute", "memory", "latency")}
    # A balanced budget is a search starting point when any denominator is unknown.
    budget = allocate_budget(*measured.values()) if not deltas["degraded"] else dict.fromkeys(measured, 1)
    result = {**deltas, "bound": "unresolved", "near_peak": False,
              "axis_budget": budget, "budget_kind": "advisory_maximum",
              "early_stop_note": "Stop using measured task targets/plateau and regression gates, not counter magnitude."}
    out_path = os.path.join(iter_dir, "roofline.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--iter", type=int, required=True)
    args = p.parse_args()
    run(args.state, args.iter)


if __name__ == "__main__":
    main()
