#!/usr/bin/env python3
"""Validate iterv{i}/methods.json against registry, state, and roofline budgets.

v2 changes from v1:
- Roofline axis budgets are advisory, not proof of a bottleneck
- Per-axis cap of 2 is enforced
- Choose 1..3 methods without filling unused budget
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


def _parse_gfx_arch(arch: str | None) -> str:
    if not arch:
        return ""
    value = arch.split(":", 1)[0].lower()
    return value if re.fullmatch(r"gfx[0-9a-f]+", value) else ""


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
    if not isinstance(methods_list, list) or not all(isinstance(m, dict) for m in methods_list):
        return False, ["methods must be a list of objects"]
    if len({m.get("id") for m in methods_list}) != len(methods_list):
        errors.append("method IDs must be unique")

    # Detect gfx arch
    gpus = state.get("env", {}).get("gpus", [{}])
    detected_arch = _parse_gfx_arch((gpus[0].get("gfx_arch") or gpus[0].get("gcn_arch")) if gpus else None)

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
    if not 1 <= len(methods_list) <= expected_total:
        errors.append(
            f"Expected 1..{expected_total} methods (advisory budget: {axis_budget}), "
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

        # gfx numbers are identifiers, not a monotonic feature level.
        evidence = m.get("target_evidence", "")
        if reg.get("requires_target_probe") and not (isinstance(evidence, str) and evidence.strip()):
            errors.append(f"{prefix}: exact-target header/source + compile/ISA probe evidence is required")
        if reg.get("allowed_arches") and detected_arch not in reg["allowed_arches"]:
            errors.append(f"{prefix}: method is scoped to {reg['allowed_arches']}; observed {detected_arch or 'unknown target'}")
        prior_ids = {item.get("id") for item in state.get("selected_methods", [])}
        if mid in prior_ids and not m.get("retry_reason"):
            errors.append(f"{prefix}: repeat requires retry_reason explaining changed implementation, shape or bottleneck")

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
            reviews = methods_data.get("coupling_reviews", [])
            if isinstance(reviews, list) and any(isinstance(r, dict) and set(r.get("ids", [])) == pair_ids
                    and all(isinstance(r.get(k), str) and r[k].strip() for k in ("separate_deltas", "validation_plan"))
                    for r in reviews):
                continue
            errors.append(
                f"Coupled pair both selected: {pair_ids}. "
                f"Provide coupling_reviews with separate_deltas and validation_plan, or count one change once. Note: {pair.get('note', '')}"
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
