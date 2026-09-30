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
    m = re.match(r"sm_(\d+)", arch)
    return int(m.group(1)) if m else 0


def _higher_priority_ids(registry: dict, axis: str, priority: int) -> list[tuple[str, int]]:
    out = []
    for mid, meta in registry["methods"].items():
        if meta["axis"] == axis and meta["priority"] < priority:
            out.append((mid, meta["priority"]))
    return sorted(out, key=lambda x: x[1])


def _semantic_effect(reg: dict, method_id: str) -> str:
    if reg.get("semantic_effect"):
        return reg["semantic_effect"]
    if method_id in {
        "compute.mixed_precision", "compute.two_level_accumulation_promotion",
        "compute.fp8_fast_accumulation_mode", "compute.block_scaled_precision",
        "compute.tf32_emulation_3xtf32_bf16x6", "compute.mufu_ex2_softmax_replacement",
        "compute.fma_and_fast_math", "memory.split_k_parallel_reduce",
    }:
        return "tolerance_changing"
    if method_id in {"memory.kernel_fusion", "latency.online_recomputation"}:
        return "preconditioned"
    return "bitwise_preserving"


def _effective_meta(registry: dict, meta: dict) -> dict:
    defaults = registry.get("$metadata_defaults", {})
    merged = dict(defaults)
    merged.update(meta)
    return merged


def _version_tuple(value: str) -> tuple[int, ...]:
    nums = re.findall(r"\d+", str(value or ""))
    return tuple(int(x) for x in nums[:3]) or (0,)


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

    # Detect sm_arch
    gpus = state.get("env", {}).get("gpus", [{}])
    detected_sm = _parse_sm_arch(gpus[0].get("sm_arch") if gpus else None)
    backend = str(state.get("backend") or state.get("archetype_backend") or "cuda").lower()
    toolchain = state.get("toolchain", {}) if isinstance(state.get("toolchain", {}), dict) else {}
    archetype = str(methods_data.get("archetype") or state.get("archetype") or "generic")

    # Load roofline budget if available
    iter_num = methods_data.get("iter", 1)
    run_dir = state.get("run_dir", "")
    roofline_path = os.path.join(run_dir, f"iterv{iter_num}", "roofline.json")
    axis_budget = {"compute": 1, "memory": 1, "latency": 1}  # default
    if os.path.isfile(roofline_path):
        roofline = _load_json(roofline_path)
        axis_budget = roofline.get("axis_budget", axis_budget)

    # A method list may intentionally use fewer slots when no eligible method
    # has sufficient evidence; never force an unsafe filler method.
    expected_total = sum(axis_budget.values())
    if len(methods_list) > expected_total:
        errors.append(
            f"Expected {expected_total} methods (budget: {axis_budget}), "
            f"got {len(methods_list)}"
        )

    # Validate axis distribution
    axis_counts = {"compute": 0, "memory": 0, "latency": 0}
    selected_ids_for_budget = {m.get("id", "") for m in methods_list}
    grouped_ids: set[str] = set()
    for rel in registry.get("relationships", []):
        if rel.get("kind") == "same_budget_group":
            ids = [x for x in rel.get("ids", []) if x in selected_ids_for_budget]
            if len(ids) > 1:
                grouped_ids.update(ids[1:])
    for m in methods_list:
        ax = m.get("axis", "unknown")
        if ax in axis_counts and m.get("id") not in grouped_ids:
            axis_counts[ax] += 1
        else:
            errors.append(f"Unknown axis '{ax}' for method {m.get('id')}")

    for axis in ["compute", "memory", "latency"]:
        if axis_counts[axis] > axis_budget.get(axis, 0):
            errors.append(
                f"Axis '{axis}': expected {axis_budget.get(axis, 0)} methods "
                f"(from roofline budget), got {axis_counts[axis]}"
            )

    for axis in ["compute", "memory", "latency"]:
        if axis_counts[axis] > 2:
            errors.append(f"Axis '{axis}': {axis_counts[axis]} methods exceeds per-axis cap of 2")

    # Validate each method
    all_submitted_ids = {m.get("id", "") for m in methods_list}
    if len(all_submitted_ids) != len(methods_list):
        errors.append("Duplicate method ids are not allowed")
    coupled_pairs = registry.get("coupled_methods", [])
    relationships = registry.get("relationships", [])
    for rel in relationships:
        ids = rel.get("ids", [])
        if len(ids) == 2 and set(ids).issubset(all_submitted_ids) and rel.get("kind") in {"conflicts", "subsumes"}:
            errors.append(f"Relationship {rel.get('kind')} forbids selecting both: {ids}")

    for idx, m in enumerate(methods_list):
        prefix = f"methods[{idx}]"

        for field in ("id",):
            if field not in m:
                errors.append(f"{prefix}: missing required field '{field}'")
        if "id" not in m:
            continue

        mid = m["id"]

        # id must exist in registry
        if mid not in registry["methods"]:
            errors.append(
                f"{prefix}: id '{mid}' not in registry. Known ids: "
                f"{sorted(registry['methods'])}"
            )
            continue

        reg = _effective_meta(registry, registry["methods"][mid])
        axis = m.get("axis", reg["axis"])
        priority = m.get("priority", reg["priority"])

        # axis & priority must match
        if reg["axis"] != axis:
            errors.append(f"{prefix}: axis '{axis}' != registry '{reg['axis']}'")
        if reg["priority"] != priority:
            errors.append(f"{prefix}: P{priority} != registry P{reg['priority']} for '{mid}'")

        # arch compatibility
        if reg["min_sm"] > detected_sm > 0:
            errors.append(f"{prefix}: '{mid}' requires sm_{reg['min_sm']}+ but have sm_{detected_sm}")
        feature_map = registry.get("arch_feature_map", {})
        arch_key = f"sm_{detected_sm}" if detected_sm else ""
        probed = (state.get("capabilities") or state.get("env", {}).get("capabilities") or {})
        if isinstance(probed, dict) and probed:
            available_features = {k for k, v in probed.items() if v is True or (isinstance(v, dict) and v.get("available") is True)}
        else:
            available_features = set(feature_map.get(arch_key, []))
        missing_features = set(reg.get("required_features", [])) - available_features
        if detected_sm and missing_features:
            errors.append(f"{prefix}: '{mid}' requires unavailable features {sorted(missing_features)} on {arch_key}")

        support = reg.get("backend_support", {}).get(backend, "unsupported") if isinstance(reg.get("backend_support"), dict) else "unsupported"
        if support in {"unsupported", "unavailable", "experimental"} and backend not in {"", "unknown"}:
            errors.append(f"{prefix}: '{mid}' backend '{backend}' is {support}")
        archtypes = set(reg.get("workload_archetypes", ["generic"]))
        if archetype not in archtypes and "generic" not in archtypes:
            errors.append(f"{prefix}: '{mid}' is not enabled for archetype '{archetype}'")
        for key, version_key in (("min_cuda", "cuda"), ("min_cutlass", "cutlass"), ("min_triton", "triton")):
            required = reg.get(key)
            actual = toolchain.get(version_key, "")
            if required and actual and _version_tuple(actual) < _version_tuple(required):
                errors.append(f"{prefix}: '{mid}' requires {version_key}>={required}, have {actual}")

        semantic = _semantic_effect(reg, mid)
        mode = methods_data.get("numerics_mode", state.get("numerics_mode", "reference"))
        if mode == "strict" and semantic not in {"bitwise_preserving", "preconditioned"}:
            errors.append(f"{prefix}: '{mid}' has semantic_effect={semantic}, disallowed in strict mode")

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
                f"{prefix}: '{mid}' previously failed SASS verification. "
                "Ensure implementation is corrected before re-selecting."
            )

        # skipped_higher — must account for all higher-priority on this axis
        higher = _higher_priority_ids(registry, axis, priority)
        skipped = m.get("skipped_higher", [])
        skipped_ids_set = {s.get("id") for s in skipped}
        valid_reasons = set(registry.get("skip_reason_codes", [])) or {
            "already_selected", "arch_incompatible", "feature_unavailable", "skip_condition", "no_trigger"
        }

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

    # Backward-compatible relation check for registries without typed edges.
    # Typed relationships above take precedence when present.
    if not relationships:
        # Coupled pairs check
        for pair in coupled_pairs:
            pair_ids = set(pair.get("ids", []))
            note = str(pair.get("note", "")).lower()
            is_conflict = any(word in note for word in ("mutually exclusive", "mutually", "pick one", "conflict"))
            if is_conflict and pair_ids.issubset(all_submitted_ids):
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
