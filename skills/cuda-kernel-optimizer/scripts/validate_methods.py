#!/usr/bin/env python3
"""Validate iterv{i}/methods.json against the method registry and state.json.

Enforces the priority-scan discipline that SKILL.md step 3b describes:
- Every method id exists in references/method_registry.json
- Submitted axis / priority match the registry exactly
- skipped_higher lists every higher-priority method on the same axis, each
  annotated with a valid skip_reason_code
- Selected method is not already in state.selected_methods
- Selected method's min_sm is ≤ detected sm_arch
- The coupled_methods rule is respected (e.g. memory.multi_stage_pipeline
  and latency.async_pipeline cannot both appear)

Exit 0 when methods.json is valid; exit 1 and print the violations otherwise.
Designed to be called from state.py update before any state mutation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Registry + state loading
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRY = Path(__file__).resolve().parent.parent / "references" / "method_registry.json"


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_sm_arch(arch: str | None) -> int:
    """Parse 'sm_80' / 'sm_90' → 80 / 90. Returns 0 if unknown."""
    if not arch:
        return 0
    m = re.match(r"sm_(\d+)", arch)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------

def _higher_priority_ids(registry: dict, axis: str, priority: int) -> list[tuple[str, int]]:
    """Return [(id, priority)] for all methods with strictly higher priority
    on the same axis (numerically: priority < p means higher rank)."""
    out = []
    for mid, meta in registry["methods"].items():
        if meta["axis"] == axis and meta["priority"] < priority:
            out.append((mid, meta["priority"]))
    return sorted(out, key=lambda x: x[1])


def _check_single_method(
    *,
    idx: int,
    m: dict,
    registry: dict,
    state: dict,
    detected_sm: int,
    all_submitted_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    prefix = f"methods[{idx}]"

    # 1. Required fields
    for field in ("id", "axis", "priority"):
        if field not in m:
            errors.append(f"{prefix}: missing required field '{field}'")
    if errors:
        return errors

    mid = m["id"]
    axis = m["axis"]
    priority = m["priority"]

    # 2. id must exist in registry
    if mid not in registry["methods"]:
        errors.append(
            f"{prefix}: id '{mid}' is not in method_registry.json. "
            f"Known ids on '{axis}' axis: "
            f"{sorted(k for k,v in registry['methods'].items() if v['axis']==axis)}"
        )
        return errors

    reg = registry["methods"][mid]

    # 3. axis & priority must match registry
    if reg["axis"] != axis:
        errors.append(f"{prefix}: submitted axis '{axis}' != registry axis '{reg['axis']}'")
    if reg["priority"] != priority:
        errors.append(
            f"{prefix}: submitted priority P{priority} != registry P{reg['priority']} "
            f"for id '{mid}'"
        )

    # 4. arch compatibility
    if reg["min_sm"] > detected_sm > 0:
        errors.append(
            f"{prefix}: id '{mid}' requires sm_{reg['min_sm']}+ "
            f"but detected arch is sm_{detected_sm}"
        )

    # 5. already selected?
    selected_ids = {item.get("id") for item in state.get("selected_methods", [])}
    if mid in selected_ids:
        errors.append(
            f"{prefix}: id '{mid}' is already in state.selected_methods — "
            f"cannot re-select across iterations"
        )

    # 6. in ineffective list? (hard block unless ncu profile changed —
    #    we don't have the evidence here, so we only WARN via error with
    #    a specific code the caller can suppress if needed)
    ineffective_ids = {item.get("id") for item in state.get("ineffective_methods", [])}
    if mid in ineffective_ids:
        errors.append(
            f"{prefix}: id '{mid}' is in state.ineffective_methods — "
            f"re-selecting requires documented evidence the bottleneck has "
            f"fundamentally changed (override with --allow-ineffective)"
        )

    # 7. skipped_higher must cover all higher-priority methods on this axis
    #    (unless priority == 1, in which case nothing to skip)
    if priority > 1:
        higher = _higher_priority_ids(registry, axis, priority)
        skipped_list = m.get("skipped_higher") or []
        skipped_ids = {s.get("id") for s in skipped_list}

        missing = [(hid, hp) for hid, hp in higher if hid not in skipped_ids]
        if missing:
            errors.append(
                f"{prefix}: priority-scan incomplete. Selected P{priority} but did not "
                f"account for higher-priority methods: "
                + ", ".join(f"{hid}(P{hp})" for hid, hp in missing)
                + ". Every higher-priority method must appear in skipped_higher "
                f"with a valid reason."
            )

        # 8. each skipped entry must have a valid reason code
        valid_codes = set(registry.get("skip_reason_codes", []))
        for sidx, sentry in enumerate(skipped_list):
            if not isinstance(sentry, dict):
                errors.append(f"{prefix}.skipped_higher[{sidx}]: not an object")
                continue
            for field in ("id", "reason"):
                if field not in sentry:
                    errors.append(
                        f"{prefix}.skipped_higher[{sidx}]: missing '{field}'"
                    )
            reason = sentry.get("reason", "")
            if reason and reason not in valid_codes:
                errors.append(
                    f"{prefix}.skipped_higher[{sidx}]: reason '{reason}' is not in "
                    f"{sorted(valid_codes)}"
                )

            # 9. sanity-check that the skipped id is actually higher priority
            sid = sentry.get("id")
            if sid in registry["methods"]:
                sreg = registry["methods"][sid]
                if sreg["axis"] != axis:
                    errors.append(
                        f"{prefix}.skipped_higher[{sidx}]: id '{sid}' is on axis "
                        f"'{sreg['axis']}', not '{axis}'"
                    )
                elif sreg["priority"] >= priority:
                    errors.append(
                        f"{prefix}.skipped_higher[{sidx}]: id '{sid}' is P{sreg['priority']} "
                        f"which is not strictly higher than the selected P{priority}"
                    )

    return errors


def _check_coupled_methods(methods: list[dict], registry: dict) -> list[str]:
    """If any pair in registry.coupled_methods[*].ids is both in `methods`,
    that's a violation — Claude should have picked a different method on the
    second axis."""
    errors: list[str] = []
    ids_submitted = {m.get("id") for m in methods}
    for rule in registry.get("coupled_methods", []):
        group = set(rule["ids"])
        overlap = group & ids_submitted
        if len(overlap) > 1:
            errors.append(
                f"coupled methods conflict: both {sorted(overlap)} selected. "
                f"These count as one optimization — replace one with a different "
                f"method on its axis. ({rule.get('note', '')})"
            )
    return errors


def _check_axis_coverage(methods: list[dict]) -> list[str]:
    errors: list[str] = []
    axes = [m.get("axis") for m in methods]
    expected = {"compute", "memory", "latency"}
    seen = set(axes)

    if len(methods) != 3:
        errors.append(f"expected exactly 3 methods, got {len(methods)}")
    if seen != expected:
        missing = expected - seen
        extra = seen - expected
        if missing:
            errors.append(f"missing axis coverage: {sorted(missing)}")
        if extra:
            errors.append(f"unexpected axis values: {sorted(extra)}")
    # duplicate axes
    from collections import Counter
    c = Counter(axes)
    dups = [a for a, n in c.items() if n > 1]
    if dups:
        errors.append(f"duplicate axis: {dups} — must be one method per axis")
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate(
    *,
    methods_path: str,
    state_path: str,
    registry_path: str,
    allow_ineffective: bool = False,
) -> tuple[bool, list[str]]:
    registry = _load_json(registry_path)
    state = _load_json(state_path)
    methods_doc = _load_json(methods_path)

    errors: list[str] = []
    if "methods" not in methods_doc or not isinstance(methods_doc["methods"], list):
        errors.append("methods.json must contain a top-level 'methods' list")
        return False, errors

    methods = methods_doc["methods"]
    detected_sm = _parse_sm_arch(
        (state.get("env") or {}).get("primary_sm_arch")
    )

    errors.extend(_check_axis_coverage(methods))

    submitted_ids = {m.get("id") for m in methods}
    for i, m in enumerate(methods):
        errors.extend(_check_single_method(
            idx=i, m=m, registry=registry, state=state,
            detected_sm=detected_sm,
            all_submitted_ids=submitted_ids,
        ))

    errors.extend(_check_coupled_methods(methods, registry))

    # Optional override for ineffective re-selection
    if allow_ineffective:
        errors = [e for e in errors if "ineffective_methods" not in e]

    return len(errors) == 0, errors


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", required=True, help="Path to iterv{i}/methods.json")
    p.add_argument("--state", required=True, help="Path to run_*/state.json")
    p.add_argument("--registry", default=str(_DEFAULT_REGISTRY),
                   help="Path to method_registry.json (defaults to bundled)")
    p.add_argument("--allow-ineffective", action="store_true",
                   help="Allow re-selecting methods in state.ineffective_methods "
                        "(use only when analysis.md documents that ncu bottleneck "
                        "has fundamentally changed)")
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
