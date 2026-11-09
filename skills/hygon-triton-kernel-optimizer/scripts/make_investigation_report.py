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
        return {}
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
        "Draft aligned with the installed Skill's references/report_template.md; parsed fields are evidence inputs, not validation conclusions.",
        "",
        "## 1. Problem and Contract",
        "",
        "Target model/operator, numerical/layout/alias/stream contract, independent oracle and performance goal: TODO",
        "",
        "## 2. Environment and Sources",
        "",
        "Exact gfx/Torch/Triton/compiler/repository SHA, source IDs and capture hash: TODO",
        "If HCU cases/PRs or NVIDIA KernelWiki informed a candidate: cite source/commit, original conditions, HCU differences, hypothesis and measured outcome; otherwise omit.",
        "",
        "## 3. Code and Dispatch",
        "",
        "| Question | Result |",
        "| --- | --- |",
        f"| Kernel name | `{kernel_name}` |",
        "| Graph IR or log snippet connecting graph to kernel | TODO |",
        "| Repository paths, caller and actual dispatch/loaded artifact | TODO |",
        "",
        "## 4. Capture and Reproduction",
        "",
        "| Question | Result |",
        "| --- | --- |",
        f"| Input shapes/dtypes/strides | {inputs_text or 'TODO'} |",
        "| Source/capture hash, pre-execution storage/offset/alias and launcher | TODO |",
        "| Capture completeness or standalone handwritten-kernel repro | TODO |",
        "",
        "## 5. Hotspot and Analysis Choice",
        "",
        "Actual profile tool/version, dispatch, rank/time share, artifact, metric definitions and observations: TODO. Tool blocks remain unresolved; static scans are not dynamic profiling.",
        "Follow the shared HCU contract's analysis-depth guidance: select relevant ceiling, fusion/dataflow or ISA-pipeline analysis by complexity, evidence and progress; reuse comparable evidence, not all three each round. Connect findings to the next experiment; note reason/reopen condition for triggered-but-deferred issues only.",
        "",
        "## 6. Candidates and Correctness",
        "",
        "Candidate/source differences, hypothesis and expected observation, oracle results for all configurations, seeds/shapes/tails, race and bounds checks: TODO",
        "Fusion, fallback, compile-boundary and model rewrites are optional implementation choices, not required stages; retain original semantics and numerical policy.",
        "",
        "## 7. Performance and Integration",
        "",
        "| Question | Result |",
        "| --- | --- |",
        f"| Autotune screening time (not independent confirmation) | `{best.get('time_ms', '')}` ms |",
        f"| Autotune configs and selected config | configs=`{item.get('configs', '')}`; best=`{best.get('config', '')}` |",
        f"| Footprint rate estimate | `{_metric(item, 'footprint_rate_estimate')}`; not measured HBM utilization |",
        "| Independent before/after samples, timing scope and noise | TODO |",
        "| Compile/JIT cost, actual model dispatch and end-to-end result | TODO |",
        "",
        "## 8. Mechanism and Feedback",
        "",
        "Compare the expected observation with measured results; retain attribution gaps and revise the hypothesis on counterexamples. ISA analysis may guide source changes without code-object rewriting. Direct rewriting additionally needs object identities, ABI/resource checks, verified loading and differential coverage under the original oracle policy.",
        "",
        "| Question | Result |",
        "| --- | --- |",
        f"| Pointer arguments | {ptr_text or 'TODO'} |",
        f"| `tt.divisibility` status | indices `{meta.get('divisibility_indices', [])}` |",
        f"| `tt.pointer_range` status | indices `{meta.get('pointer_range_indices', [])}` |",
        f"| Static AMDGCN load/store counts | {load_summary or 'TODO'} |",
        f"| `buffer_load/store_dwordx4` observed | `{isa_cls.get('wide_buffer_observed', '')}` |",
        f"| Static buffer-ops heuristic | `{isa_cls.get('buffer_ops_dominant', '')}`; not a dynamic bottleneck classification |",
        f"| `tl.assume` added or already present | count `{meta.get('counts', {}).get('tl_assume', '')}` |",
        f"| `tl.multiple_of` added or already present | count `{meta.get('counts', {}).get('tl_multiple_of', '')}` |",
        "| Before/after instruction/resource/timeline changes and valid ablation | TODO |",
        "",
        "## 9. Conclusion, Rollback and Next Steps",
        "",
        "Adopted implementation and supported scope, failures/untested items, stopping or reopening conditions, rollback and task artifact paths: TODO. Report tested hardware separately from static-only preparation.",
        "",
        "## Notes",
        "",
    ]
    if kernel and autotune.get("kernels") and not item:
        lines.append("- Requested kernel was not found in the autotune summary; no other kernel's timings or inputs were reused.")
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
