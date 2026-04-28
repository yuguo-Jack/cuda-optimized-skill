#!/usr/bin/env python3
"""Compute DCU roofline-like gaps and allocate per-axis method budgets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


_GPU_SPECS = {
    "gfx936": {"peak_flops_tflops": 180, "peak_bw_gbs": 1600},
    "gfx938": {"peak_flops_tflops": 240, "peak_bw_gbs": 2000},
}
_DEFAULT_SPEC = {"peak_flops_tflops": 200, "peak_bw_gbs": 1800}

TOTAL_BUDGET = 3
MAX_PER_AXIS = 2
NEAR_PEAK_THRESHOLD = 0.15
TIE_BREAK_ORDER = ["memory", "latency", "compute"]


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_gpu_spec(env: dict) -> dict:
    gpu = (env.get("gpus") or [{}])[0]
    arch = gpu.get("gfx_arch") or gpu.get("gcn_arch") or env.get("primary_gfx_arch") or "gfx938"
    spec = _GPU_SPECS.get(arch, _DEFAULT_SPEC).copy()
    for key in ("peak_flops_tflops", "peak_bw_gbs"):
        if key in gpu:
            spec[key] = gpu[key]
    return spec


def _find_metric(dcu_top: dict, patterns: list[str], axis: str | None = None) -> float:
    axes = [axis] if axis else ["compute", "memory", "latency"]
    for ax in axes:
        for item in dcu_top.get(ax, []):
            name = item.get("name") or item.get("metric") or ""
            lname = name.lower()
            if any(p.lower() in lname for p in patterns):
                return _safe_float(item.get("value"))
    return 0.0


def compute_deltas(dcu_top: dict, env: dict) -> dict:
    if dcu_top.get("degraded", False):
        return {"delta_compute": 0.50, "delta_memory": 0.50, "delta_latency": 0.50, "degraded": True}

    sq_busy = _find_metric(dcu_top, ["SQ_BUSY_CYCLES", "GRBM_GUI_ACTIVE"], "compute")
    sq_cycles = _find_metric(dcu_top, ["SQ_CYCLES", "GRBM_COUNT"], "compute")
    mmop = _find_metric(dcu_top, ["SQ_INSTS_MMOP", "MMOP", "MMAC"], "compute")
    if sq_busy > 0 and sq_cycles > 0:
        compute_util = min(1.0, sq_busy / sq_cycles)
    elif sq_busy > 1:
        compute_util = min(1.0, sq_busy / 100.0)
    else:
        compute_util = 0.30 if mmop > 0 else 0.15
    delta_c = max(0.0, 1.0 - compute_util)

    tcc_busy = _find_metric(dcu_top, ["TCC_BUSY", "TCP_TOTAL_CACHE_ACCESSES", "TCP_TCC_READ_REQ"], "memory")
    memory_util = min(1.0, (tcc_busy / 100.0) if tcc_busy > 1 else tcc_busy)
    if memory_util <= 0:
        memory_util = 0.50
    delta_m = max(0.0, 1.0 - memory_util)

    stalls = []
    for item in dcu_top.get("latency", []):
        name = (item.get("name") or "").lower()
        if any(token in name for token in ("stall", "latency", "wait", "wave_cycles", "barrier")):
            value = _safe_float(item.get("value"))
            if value > 1:
                value = min(value, 100.0) / 100.0
            stalls.append(max(0.0, min(1.0, value)))
    delta_l = max(stalls) if stalls else 0.50

    sqtt = dcu_top.get("sqtt_analysis") or {}
    if isinstance(sqtt, dict) and not sqtt.get("error"):
        waitcnt_count = _safe_float(sqtt.get("waitcnt_count"))
        branch_count = _safe_float(sqtt.get("branch_count"))
        stall_hits = sqtt.get("stall_like_hits") or []
        if waitcnt_count > 0 or stall_hits:
            delta_l = max(delta_l, 0.65)
        if branch_count > 0:
            delta_l = max(delta_l, 0.55)

    codeobj = dcu_top.get("codeobj_analyze") or {}
    if isinstance(codeobj, dict):
        flags = set(codeobj.get("pressure_flags") or [])
        if "high_vgpr" in flags or "high_sgpr" in flags:
            delta_c = max(delta_c, 0.55)
            delta_l = max(delta_l, 0.55)

    spec = _get_gpu_spec(env)
    ai_ridge = (spec["peak_flops_tflops"] * 1e12) / (spec["peak_bw_gbs"] * 1e9)
    return {
        "delta_compute": round(delta_c, 4),
        "delta_memory": round(delta_m, 4),
        "delta_latency": round(delta_l, 4),
        "compute_util_pct": round(compute_util * 100, 2),
        "memory_util_pct": round(memory_util * 100, 2),
        "max_stall_pct": round(delta_l * 100, 2),
        "ai_ridge": round(ai_ridge, 2),
        "sqtt_informed": bool(sqtt and not sqtt.get("error")),
        "codeobj_informed": bool(codeobj and codeobj.get("available")),
        "degraded": False,
    }


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
    dc, dm, dl = deltas["delta_compute"], deltas["delta_memory"], deltas["delta_latency"]
    near_peak = dc < NEAR_PEAK_THRESHOLD and dm < NEAR_PEAK_THRESHOLD and dl < NEAR_PEAK_THRESHOLD
    max_delta = max(dc, dm, dl)
    bound = "near_peak" if near_peak else "compute" if max_delta == dc else "bandwidth" if max_delta == dm else "latency"
    result = {**deltas, "bound": bound, "near_peak": near_peak, "axis_budget": allocate_budget(dc, dm, dl)}
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
