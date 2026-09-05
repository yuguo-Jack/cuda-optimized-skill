"""Validation predicates and durable stop-state helpers for strict runs."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path


def benchmark_gate(bench: dict, *, max_cv: float = 0.10,
                   require_reference_timing: bool = True) -> tuple[bool, str | None]:
    states = bench.get("states", {}) if isinstance(bench.get("states"), dict) else {}
    for name in ("compile_pass", "contract_pass", "correctness_pass", "timing_valid"):
        if states.get(name) != "pass":
            return False, f"{name}={states.get(name, 'missing')}"
    if bench.get("error"):
        return False, f"benchmark_error={bench['error']}"
    if bench.get("correctness", {}).get("passed") is not True:
        return False, "correctness.passed is not true"
    kernel = bench.get("kernel") or {}
    try:
        kernel_ms = float(kernel.get("average_ms"))
    except (TypeError, ValueError):
        return False, "kernel.average_ms is missing or invalid"
    if not math.isfinite(kernel_ms) or kernel_ms <= 0:
        return False, "kernel.average_ms must be finite and positive"
    if kernel.get("stability") == "unstable":
        return False, "kernel timing is unstable"
    cv = kernel.get("robust_cv")
    if cv is None:
        return False, "kernel robust_cv is missing"
    try:
        if not math.isfinite(float(cv)) or float(cv) > max_cv:
            return False, f"kernel robust_cv exceeds {max_cv}"
    except (TypeError, ValueError):
        return False, "kernel robust_cv is invalid"
    if require_reference_timing:
        reference = bench.get("reference") or {}
        try:
            ref_ms = float(reference.get("average_ms"))
        except (TypeError, ValueError):
            return False, "reference.average_ms is missing or invalid"
        if not math.isfinite(ref_ms) or ref_ms <= 0:
            return False, "reference.average_ms must be finite and positive"
    return True, None


def ncu_gate(top: dict, rep_path: str) -> tuple[bool, str | None]:
    if not os.path.isfile(rep_path) or os.path.getsize(rep_path) <= 0:
        return False, "NCU report is missing or empty"
    if top.get("degraded") is not False:
        return False, f"NCU result degraded: {top.get('reason', 'unknown reason')}"
    attempts = top.get("profile_attempts") or []
    if not attempts or attempts[-1].get("rc") != 0:
        return False, "NCU collection did not finish with rc=0"
    try:
        metric_count = int(top.get("metric_count_collected", 0))
    except (TypeError, ValueError):
        metric_count = 0
    if metric_count <= 0:
        return False, "NCU import produced no metrics"
    return True, None


def write_stop(state_path: str, *, reason: str, stage: str, iteration: int | None,
               last_error: str = "", attempts: int = 0, exit_code: int = 1) -> dict:
    state_file = Path(state_path)
    state = json.loads(state_file.read_text(encoding="utf-8"))
    validation_reasons = {"baseline_compile_failed", "baseline_correctness_failed",
                          "baseline_timing_invalid", "iteration_no_valid_branch",
                          "iteration_correctness_failed", "iteration_timing_invalid"}
    iteration_status = "failed_validation" if reason in validation_reasons else "stopped"
    state.update({"run_status": "stopped", "generation_allowed": False,
                  "stop_reason": reason, "stop_stage": stage,
                  "stop_iteration": iteration,
                  "terminal_iteration_status": iteration_status})
    tmp = state_file.with_suffix(state_file.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, state_file)
    payload = {
        "status": "stopped", "iteration": iteration, "stage": stage,
        "reason": reason, "iteration_status": iteration_status,
        "attempts": attempts, "last_error": last_error,
        "generated_iterations": int(state.get("verified_iterations", 0)),
        "next_iteration": None, "exit_code": exit_code,
    }
    stop_path = Path(state["run_dir"]) / "stop.json"
    stop_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload
