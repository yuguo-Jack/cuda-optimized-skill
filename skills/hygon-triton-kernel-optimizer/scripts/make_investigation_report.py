#!/usr/bin/env python3
"""Create a Triton investigation report from parsed artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: str) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _pick_kernel(summary: dict, kernel: str) -> dict:
    kernels = summary.get("kernels", [])
    if kernel:
        for item in kernels:
            if item.get("kernel") == kernel:
                return item
    return kernels[0] if kernels else {}


def _metric(item: dict, key: str) -> str:
    return str(item.get("metrics", {}).get(key, ""))


def build_report(autotune: dict, meta: dict, isa: dict, kernel: str) -> str:
    item = _pick_kernel(autotune, kernel)
    kernel_name = kernel or item.get("kernel") or ", ".join(meta.get("inductor_kernel_names", [])) or "unknown"
    best = next((c for c in item.get("config_timings", []) if c.get("best")), {})
    ptr_args = meta.get("ptr_hint_status") or meta.get("ptr_args") or []
    isa_cls = isa.get("classification", {})
    totals = isa.get("totals", {})

    input_lines = []
    for inp in item.get("inputs", []):
        input_lines.append(f"`{inp['name']}` {inp['shape']} {inp['dtype']} stride={inp['stride']}")
    inputs_text = "<br>".join(input_lines)

    ptr_text = "<br>".join(
        f"`{p.get('name')}` index={p.get('index')} type={p.get('type')} "
        f"div={p.get('has_divisibility', '')} range={p.get('has_pointer_range', '')}"
        for p in ptr_args
    )

    load_summary = ", ".join(f"{k}={v}" for k, v in totals.items() if v)

    lines = [
        f"# Triton Investigation: {kernel_name}",
        "",
        "## 1. Profile",
        "",
        "| Question | Result |",
        "| --- | --- |",
        f"| Kernel name and profile rank/time share | `{kernel_name}`; rank/time share: TODO |",
        "| Graph IR or log snippet connecting graph to kernel | TODO |",
        f"| Current timing | `{best.get('time_ms', '')}` ms standalone best; profile timing: TODO |",
        f"| Input shapes/dtypes/strides | {inputs_text or 'TODO'} |",
        f"| Autotune configs and best config | configs=`{item.get('configs', '')}`; best=`{best.get('config', '')}` |",
        "| Captured artifact path | TODO |",
        "",
        "## 2. Hints and Instructions",
        "",
        "| Question | Result |",
        "| --- | --- |",
        f"| Pointer arguments | {ptr_text or 'TODO'} |",
        f"| `tt.divisibility` status | indices `{meta.get('divisibility_indices', [])}` |",
        f"| `tt.pointer_range` status | indices `{meta.get('pointer_range_indices', [])}` |",
        f"| Dominant AMDGCN load/store families | {load_summary or 'TODO'} |",
        f"| `buffer_load/store_dwordx4` observed | `{isa_cls.get('wide_buffer_observed', '')}` |",
        f"| `global_*` or `flat_*` still dominant | `{not isa_cls.get('buffer_ops_dominant', False) if isa_cls else ''}` |",
        "",
        "## 3. Kernel Tuning",
        "",
        "| Question | Result |",
        "| --- | --- |",
        f"| `tl.assume` added or already present | count `{meta.get('counts', {}).get('tl_assume', '')}` |",
        f"| `tl.multiple_of` added or already present | count `{meta.get('counts', {}).get('tl_multiple_of', '')}` |",
        "| Config or block-size variants tested | TODO |",
        f"| Timing and bandwidth delta | effective bandwidth `{_metric(item, 'effective_bandwidth')}`; utilization `{_metric(item, 'bandwidth_utilization')}` |",
        "| Instruction delta | TODO |",
        "| Continue Triton tuning? | TODO |",
        "",
        "## 4. Avoid Generated Kernel",
        "",
        "| Question | Result |",
        "| --- | --- |",
        "| Model code that triggers the kernel | TODO |",
        "| Eager fallback or compile boundary | TODO |",
        "| End-to-end performance delta | TODO |",
        "",
        "## 5. Model Rewrite",
        "",
        "| Question | Result |",
        "| --- | --- |",
        "| Rewrite opportunity | TODO |",
        "| Reason it should help | TODO |",
        "| Correctness impact | TODO |",
        "| Final performance | TODO |",
        "",
        "## Notes",
        "",
    ]
    for note in meta.get("notes", []):
        lines.append(f"- {note}")
    if isa_cls.get("atomic_observed"):
        lines.append("- Atomic instructions observed in AMDGCN scan.")
    if not meta.get("notes") and not isa_cls.get("atomic_observed"):
        lines.append("- TODO")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a Triton kernel investigation report")
    parser.add_argument("--autotune-summary", default="")
    parser.add_argument("--meta", default="")
    parser.add_argument("--isa", default="")
    parser.add_argument("--kernel", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = build_report(
        _read_json(args.autotune_summary),
        _read_json(args.meta),
        _read_json(args.isa),
        args.kernel,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(json.dumps({"out": args.out}, indent=2))


if __name__ == "__main__":
    main()
