#!/usr/bin/env python3
"""Check registry invariants used by method selection and documentation."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

KINDS = {"conflicts", "subsumes", "requires", "complements", "same_budget_group"}

def lint(path: str, metric_path: str | None = None, catalog_path: str | None = None) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    methods = data.get("methods", {})
    errors = []
    seen = set()
    for mid, meta in methods.items():
        if mid in seen or not re.match(r"^(compute|memory|latency)\.[a-z0-9_]+$", mid):
            errors.append(f"invalid/duplicate method id: {mid}")
        seen.add(mid)
        if meta.get("axis") not in {"compute", "memory", "latency"}:
            errors.append(f"{mid}: invalid axis")
        if not isinstance(meta.get("priority"), int) or meta["priority"] < 1:
            errors.append(f"{mid}: invalid priority")
    for rel in data.get("relationships", []):
        if rel.get("kind") not in KINDS:
            errors.append(f"invalid relationship kind: {rel.get('kind')}")
        for mid in rel.get("ids", []):
            if mid not in methods:
                errors.append(f"relationship references unknown id: {mid}")
    for arch, features in data.get("arch_feature_map", {}).items():
        if not re.match(r"^sm_[0-9]+[a-z]*$", arch) or not isinstance(features, list):
            errors.append(f"invalid arch feature map entry: {arch}")
    defaults = data.get("$metadata_defaults", {})
    required_meta = ["backend_support", "min_cuda", "min_cutlass", "min_triton", "semantic_effect", "verification", "workload_archetypes", "tested_matrix", "confidence", "experimental", "evidence_requirements"]
    for field in required_meta:
        if field not in defaults:
            errors.append(f"$metadata_defaults missing {field}")
    allowed_effects = {"bitwise_preserving", "reordered_equivalent", "tolerance_changing", "preconditioned"}
    for mid, meta in methods.items():
        effect = meta.get("semantic_effect", defaults.get("semantic_effect"))
        if effect not in allowed_effects:
            errors.append(f"{mid}: invalid semantic_effect {effect}")
        if not isinstance(meta.get("verification", defaults.get("verification")), list):
            errors.append(f"{mid}: verification must be a list")
    if metric_path:
        metrics = json.loads(Path(metric_path).read_text(encoding="utf-8")).get("metrics", {})
        if not metrics:
            errors.append("metric registry is empty")
        for name, spec in metrics.items():
            if not spec.get("aliases") or "missing" not in spec:
                errors.append(f"metric {name}: aliases and missing policy are required")
    if catalog_path:
        catalog = Path(catalog_path).read_text(encoding="utf-8")
        catalog_ids = set(re.findall(r"`((?:compute|memory|latency)\.[A-Za-z0-9_]+)`", catalog))
        missing = sorted(set(methods) - catalog_ids)
        if missing:
            errors.append(f"methods missing from catalog: {missing}")
    return errors

def main() -> int:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(); p.add_argument("--registry", default=str(root / "references/method_registry.json")); p.add_argument("--metrics", default=str(root / "references/metric_registry.json")); p.add_argument("--catalog", default=str(root / "references/optimization_catalog.md")); args = p.parse_args()
    errors = lint(args.registry, args.metrics, args.catalog)
    if errors:
        print("\n".join(errors)); return 1
    print(f"registry ok: {args.registry}"); return 0

if __name__ == "__main__":
    raise SystemExit(main())
