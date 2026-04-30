#!/usr/bin/env python3
"""Validate iterv{i}/methods.json against registry, state, and roofline budgets.

v2 changes from v1:
- Axis distribution must match roofline.json axis_budget (not fixed 1:1:1)
- Per-axis cap of 2 is enforced
- Total method count must equal sum of axis budgets (typically 3)
- Priority scan compliance still enforced within each axis

Exit 0 when valid; exit 1 and print violations otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


_DEFAULT_REGISTRY = Path(__file__).resolve().parent.parent / "references" / "method_registry.json"


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_sm_arch(arch: str | None) -> int:
    if not arch:
        return 0
    m = re.match(r"(?:sm_|gfx)(\d+)", arch)
    return int(m.group(1)) if m else 0


def _higher_priority_ids(registry: dict, axis: str, priority: int) -> list[tuple[str, int]]:
    out = []
    for mid, meta in registry["methods"].items():
        if meta["axis"] == axis and meta["priority"] < priority:
            out.append((mid, meta["priority"]))
    return sorted(out, key=lambda x: x[1])


def validate(
    methods_path: str,
    state_path: str,
    registry_path: str = None,
    allow_ineffective: bool = False,
) -> tuple[bool, list[str]]:
    registry = _load_json(registry_path or str(_DEFAULT_REGISTRY))
    state = _load_json(state_path)
    methods_data = _load_json(methods_path)

    errors: list[str] = []

    if "methods" not in methods_data:
        return False, ["Top-level 'methods' key missing"]

    methods_list = methods_data["methods"]

    # Detect gfx arch
    gpus = state.get("env", {}).get("gpus", [{}])
    detected_sm = _parse_sm_arch((gpus[0].get("gfx_arch") or gpus[0].get("gcn_arch")) if gpus else None)

    # Load roofline budget if available
    iter_num = methods_data.get("iter", 1)
    run_dir = state.get("run_dir", "")
    roofline_path = os.path.join(run_dir, f"iterv{iter_num}", "roofline.json")
    axis_budget = {"compute": 1, "memory": 1, "latency": 1}  # default
    if os.path.isfile(roofline_path):
        roofline = _load_json(roofline_path)
        axis_budget = roofline.get("axis_budget", axis_budget)

    # Validate total count matches budget
    expected_total = sum(axis_budget.values())
    if len(methods_list) != expected_total:
        errors.append(
            f"Expected {expected_total} methods (budget: {axis_budget}), "
            f"got {len(methods_list)}"
        )

    # Validate axis distribution
    axis_counts = {"compute": 0, "memory": 0, "latency": 0}
    for m in methods_list:
        ax = m.get("axis", "unknown")
        if ax in axis_counts:
            axis_counts[ax] += 1
        else:
            errors.append(f"Unknown axis '{ax}' for method {m.get('id')}")

    for axis in ["compute", "memory", "latency"]:
        if axis_counts[axis] != axis_budget.get(axis, 0):
            errors.append(
                f"Axis '{axis}': expected {axis_budget.get(axis, 0)} methods "
                f"(from roofline budget), got {axis_counts[axis]}"
            )
        if axis_counts[axis] > 2:
            errors.append(
                f"Axis '{axis}': {axis_counts[axis]} methods exceeds per-axis cap of 2"
            )

    # Validate each method
    all_submitted_ids = {m.get("id", "") for m in methods_list}
    coupled_pairs = registry.get("coupled_methods", [])

    for idx, m in enumerate(methods_list):
        prefix = f"methods[{idx}]"

        for field in ("id", "axis", "priority"):
            if field not in m:
                errors.append(f"{prefix}: missing required field '{field}'")
        if any(f not in m for f in ("id", "axis", "priority")):
            continue

        mid = m["id"]
        axis = m["axis"]
        priority = m["priority"]

        # id must exist in registry
        if mid not in registry["methods"]:
            errors.append(
                f"{prefix}: id '{mid}' not in registry. Known ids on '{axis}': "
                f"{sorted(k for k,v in registry['methods'].items() if v['axis']==axis)}"
            )
            continue

        reg = registry["methods"][mid]

        # axis & priority must match
        if reg["axis"] != axis:
            errors.append(f"{prefix}: axis '{axis}' != registry '{reg['axis']}'")
        if reg["priority"] != priority:
            errors.append(f"{prefix}: P{priority} != registry P{reg['priority']} for '{mid}'")

        # arch compatibility
        if reg["min_sm"] > detected_sm > 0:
            errors.append(f"{prefix}: '{mid}' requires gfx{reg['min_sm']}+ but have gfx{detected_sm}")

        # already selected?
        selected_ids = {item.get("id") for item in state.get("selected_methods", [])}
        if mid in selected_ids:
            errors.append(f"{prefix}: '{mid}' already in selected_methods")

        # ineffective check
        if not allow_ineffective:
            ineff_ids = {item.get("id") for item in state.get("ineffective_methods", [])}
            if mid in ineff_ids:
                errors.append(
                    f"{prefix}: '{mid}' in ineffective_methods (use --allow-ineffective "
                    "if bottleneck profile fundamentally changed)"
                )

        # implementation_failed check
        impl_failed_ids = {item.get("id") for item in state.get("implementation_failed_methods", [])}
        if mid in impl_failed_ids:
            errors.append(
                f"{prefix}: '{mid}' previously failed DCU ISA verification. "
                "Ensure implementation is corrected before re-selecting."
            )

        # skipped_higher — must account for all higher-priority on this axis
        higher = _higher_priority_ids(registry, axis, priority)
        skipped = m.get("skipped_higher", [])
        skipped_ids_set = {s.get("id") for s in skipped}
        valid_reasons = {"already_selected", "arch_incompatible", "feature_unavailable", "skip_condition", "no_trigger"}

        for hid, hpri in higher:
            if hid in all_submitted_ids:
                continue  # selected, not skipped
            if hid not in skipped_ids_set:
                errors.append(
                    f"{prefix}: higher-priority '{hid}' (P{hpri}) on axis '{axis}' "
                    "not in skipped_higher and not selected"
                )

        for s in skipped:
            reason = s.get("reason", "")
            if reason not in valid_reasons:
                errors.append(
                    f"{prefix}: skipped_higher '{s.get('id')}' has invalid reason '{reason}'. "
                    f"Valid: {valid_reasons}"
                )

    # Coupled pairs check
    for pair in coupled_pairs:
        pair_ids = set(pair.get("ids", []))
        if pair_ids.issubset(all_submitted_ids):
            errors.append(
                f"Coupled pair both selected: {pair_ids}. "
                f"Note: {pair.get('note', '')}"
            )

    return (len(errors) == 0, errors)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", required=True)
    p.add_argument("--state", required=True)
    p.add_argument("--registry", default=str(_DEFAULT_REGISTRY))
    p.add_argument("--allow-ineffective", action="store_true")
    args = p.parse_args()

    ok, errors = validate(
        methods_path=args.methods,
        state_path=args.state,
        registry_path=args.registry,
        allow_ineffective=args.allow_ineffective,
    )

    if ok:
        print(json.dumps({"valid": True}, indent=2))
        sys.exit(0)
    else:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
