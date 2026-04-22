#!/usr/bin/env python3
"""Compute roofline gaps and allocate axis budgets for the current iteration.

Reads ncu_top.json and env.json, computes Δ_c / Δ_m / Δ_l (pure
observable ratios with NO tunable parameters), allocates per-axis method
budgets proportionally with a per-axis cap of 2 and total B=3.

Writes iterv{i}/roofline.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# GPU peak specs (fallback table when env.json lacks bandwidth/flops info).
# Values in TFLOPS (FP16 tensor) and GB/s.
# ---------------------------------------------------------------------------

_GPU_SPECS: dict[str, dict] = {
    # sm_arch -> {peak_flops_tflops_fp16, peak_bw_gbs}
    "sm_70": {"peak_flops_tflops": 125,  "peak_bw_gbs": 900},   # V100
    "sm_75": {"peak_flops_tflops": 65,   "peak_bw_gbs": 672},   # T4
    "sm_80": {"peak_flops_tflops": 312,  "peak_bw_gbs": 2039},  # A100 SXM
    "sm_86": {"peak_flops_tflops": 150,  "peak_bw_gbs": 936},   # A6000
    "sm_89": {"peak_flops_tflops": 330,  "peak_bw_gbs": 1008},  # L40S / 4090
    "sm_90": {"peak_flops_tflops": 990,  "peak_bw_gbs": 3350},  # H100 SXM
}

# Default fallback
_DEFAULT_SPEC = {"peak_flops_tflops": 200, "peak_bw_gbs": 1500}


def _get_gpu_spec(env: dict) -> dict:
    """Extract or look up peak FLOPS and bandwidth."""
    gpus = env.get("gpus") or [{}]
    gpu = gpus[0]
    sm = gpu.get("sm_arch", "sm_80")

    spec = _GPU_SPECS.get(sm, _DEFAULT_SPEC).copy()

    # Allow env.json to override if check_env.py populated these
    if "peak_flops_tflops" in gpu:
        spec["peak_flops_tflops"] = gpu["peak_flops_tflops"]
    if "peak_bw_gbs" in gpu:
        spec["peak_bw_gbs"] = gpu["peak_bw_gbs"]

    return spec


# ---------------------------------------------------------------------------
# Δ computation from ncu metrics
# ---------------------------------------------------------------------------

def _safe_float(v, default=0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


def _find_metric(ncu_top: dict, patterns: list[str], axis: str = None) -> float:
    """Find the first matching metric value from ncu_top.json axes."""
    axes_to_search = [axis] if axis else ["compute", "memory", "latency"]
    for ax in axes_to_search:
        metrics_list = ncu_top.get(ax, [])
        for m in metrics_list:
            name = m.get("metric", m.get("name", ""))
            for pat in patterns:
                if pat in name:
                    return _safe_float(m.get("value", 0.0))
    return 0.0


def compute_deltas(ncu_top: dict, env: dict) -> dict:
    """Compute Δ_c, Δ_m, Δ_l from ncu_top.json metrics.

    All values are pure ratios of observables — no tunable parameters.
    """
    degraded = ncu_top.get("degraded", False)

    if degraded:
        # No real ncu data — use uniform gaps
        return {
            "delta_compute": 0.50,
            "delta_memory": 0.50,
            "delta_latency": 0.50,
            "degraded": True,
        }

    # --- Compute gap ---
    # Primary: tensor core utilization for GEMM-like; FP32 pipe for others
    tensor_pct = _find_metric(ncu_top, [
        "pipe_tensor_op_hmma_cycles_active",
        "pipe_tensor_op_imma_cycles_active",
    ], "compute")
    fp32_pct = _find_metric(ncu_top, [
        "pipe_fp32_cycles_active",
    ], "compute")
    sm_throughput = _find_metric(ncu_top, [
        "sm__throughput",
    ], "compute")

    # Use the best available compute utilization metric
    compute_util = max(tensor_pct, fp32_pct, sm_throughput) / 100.0
    delta_c = max(0.0, 1.0 - compute_util)

    # --- Memory gap ---
    dram_throughput_pct = _find_metric(ncu_top, [
        "dram__throughput",
        "gpu__compute_memory_throughput",
    ], "memory") / 100.0
    delta_m = max(0.0, 1.0 - dram_throughput_pct)

    # --- Latency gap ---
    # Take the maximum stall percentage across all stall types
    stall_metrics = []
    for m in ncu_top.get("latency", []):
        name = m.get("metric", m.get("name", ""))
        name_l = name.lower()
        # Ignore pcsamp counters (counts), use pct-based stall metrics only.
        if "pcsamp" in name_l:
            continue
        if "pct" not in name_l:
            continue
        if "stalled" in name_l or "warp_latency" in name_l:
            v = _safe_float(m.get("value", 0.0))
            stall_metrics.append(min(max(v, 0.0), 100.0))

    if stall_metrics:
        max_stall_pct = max(stall_metrics) / 100.0
    else:
        max_stall_pct = 0.5  # Default if no stall data

    delta_l = min(1.0, max(0.0, max_stall_pct))

    # --- Determine bound ---
    spec = _get_gpu_spec(env)
    ai_ridge = (spec["peak_flops_tflops"] * 1e12) / (spec["peak_bw_gbs"] * 1e9)

    return {
        "delta_compute": round(delta_c, 4),
        "delta_memory": round(delta_m, 4),
        "delta_latency": round(delta_l, 4),
        "compute_util_pct": round(compute_util * 100, 2),
        "memory_util_pct": round(dram_throughput_pct * 100, 2),
        "max_stall_pct": round(max_stall_pct * 100, 2),
        "ai_ridge": round(ai_ridge, 2),
        "degraded": False,
    }


# ---------------------------------------------------------------------------
# Axis budget allocation
# ---------------------------------------------------------------------------

TOTAL_BUDGET = 3
MAX_PER_AXIS = 2
NEAR_PEAK_THRESHOLD = 0.15
# Tie-break order: memory > latency > compute (memory changes shift roofline
# position most; this is a structural choice, not a tunable param)
TIE_BREAK_ORDER = ["memory", "latency", "compute"]


def allocate_budget(delta_c: float, delta_m: float, delta_l: float) -> dict:
    """Allocate method budget per axis.

    Rules:
      - Proportional to Δ, rounded
      - Per-axis cap = 2
      - Total = 3
      - At least 2 axes covered (consequence of cap=2 with total=3)
      - Axes with Δ < 0.10 get 0 (negligible gap)
    """
    deltas = {"compute": delta_c, "memory": delta_m, "latency": delta_l}

    # Zero out negligible axes
    for axis in deltas:
        if deltas[axis] < 0.10:
            deltas[axis] = 0.0

    total_delta = sum(deltas.values())

    # Edge case: all deltas are negligible
    if total_delta < 0.01:
        return {"compute": 1, "memory": 1, "latency": 1}

    # Proportional allocation (float)
    raw = {axis: TOTAL_BUDGET * deltas[axis] / total_delta for axis in deltas}

    # Round to integers
    budgets = {axis: int(round(raw[axis])) for axis in deltas}

    # Step 1: Cap at MAX_PER_AXIS
    overflow = 0
    for axis in budgets:
        if budgets[axis] > MAX_PER_AXIS:
            overflow += budgets[axis] - MAX_PER_AXIS
            budgets[axis] = MAX_PER_AXIS

    # Step 2: Redistribute overflow to non-saturated axes, ordered by Δ
    for axis in sorted(deltas, key=lambda a: -deltas[a]):
        if overflow <= 0:
            break
        if budgets[axis] < MAX_PER_AXIS:
            room = MAX_PER_AXIS - budgets[axis]
            take = min(overflow, room)
            budgets[axis] += take
            overflow -= take

    # Step 3: Fix total to exactly TOTAL_BUDGET
    current = sum(budgets.values())
    while current != TOTAL_BUDGET:
        if current < TOTAL_BUDGET:
            # Add 1 to axis with highest fractional remainder that isn't capped
            best_axis = None
            best_remainder = -1.0
            for axis in TIE_BREAK_ORDER:
                if budgets[axis] < MAX_PER_AXIS:
                    remainder = raw[axis] - budgets[axis]
                    if remainder > best_remainder:
                        best_remainder = remainder
                        best_axis = axis
            if best_axis is None:
                # All axes capped — shouldn't happen with cap=2, B=3
                break
            budgets[best_axis] += 1
        else:
            # Remove 1 from axis with lowest fractional remainder
            worst_axis = None
            worst_remainder = 999.0
            for axis in reversed(TIE_BREAK_ORDER):
                if budgets[axis] > 0:
                    remainder = raw[axis] - budgets[axis]
                    if remainder < worst_remainder:
                        worst_remainder = remainder
                        worst_axis = axis
            if worst_axis is None:
                break
            budgets[worst_axis] -= 1
        current = sum(budgets.values())

    return budgets


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(state_path: str, iteration: int) -> dict:
    with open(state_path, "r") as f:
        state = json.load(f)

    run_dir = state["run_dir"]
    iter_dir = os.path.join(run_dir, f"iterv{iteration}")
    ncu_top_path = os.path.join(iter_dir, "ncu_top.json")

    if not os.path.isfile(ncu_top_path):
        sys.exit(f"ncu_top.json not found at {ncu_top_path}")

    with open(ncu_top_path, "r") as f:
        ncu_top = json.load(f)

    env = state.get("env", {})

    # Compute deltas
    deltas = compute_deltas(ncu_top, env)

    dc = deltas["delta_compute"]
    dm = deltas["delta_memory"]
    dl = deltas["delta_latency"]

    # Check near-peak
    near_peak = (dc < NEAR_PEAK_THRESHOLD and
                 dm < NEAR_PEAK_THRESHOLD and
                 dl < NEAR_PEAK_THRESHOLD)

    # Determine primary bound
    max_delta = max(dc, dm, dl)
    if near_peak:
        bound = "near_peak"
    elif max_delta == dc:
        bound = "compute"
    elif max_delta == dm:
        bound = "bandwidth"
    else:
        bound = "latency"

    # Allocate budgets
    axis_budget = allocate_budget(dc, dm, dl)

    result = {
        **deltas,
        "bound": bound,
        "near_peak": near_peak,
        "axis_budget": axis_budget,
    }

    # Write roofline.json
    out_path = os.path.join(iter_dir, "roofline.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--iter", type=int, required=True)
    args = p.parse_args()
    run(args.state, args.iter)


if __name__ == "__main__":
    main()
