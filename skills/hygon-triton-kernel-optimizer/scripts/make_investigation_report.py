#!/usr/bin/env python3
"""Create a Triton investigation report from parsed artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: str) -> dict:
    if not path:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _pick_kernel(summary: dict, kernel: str, record_index: int | None = None) -> dict:
    kernels = summary.get("kernels", [])
    if record_index is not None:
        if not 0 <= record_index < len(kernels):
            raise ValueError("--record-index is outside the autotune summary")
        item = kernels[record_index]
        if kernel and item.get("kernel") != kernel:
            raise ValueError("--record-index does not match --kernel")
        return item
    matches = [item for item in kernels if not kernel or item.get("kernel") == kernel]
    if len(matches) > 1:
        raise ValueError("Ambiguous autotune records; select --kernel and/or --record-index (zero-based summary position)")
    return matches[0] if matches else {}


def _scoped_evidence(meta: dict, isa: dict, kernel: str, notes: list[str]) -> tuple[dict, dict]:
    """Avoid attributing file/directory aggregates to a particular kernel.

    Names only establish a reporting scope, not matching builds/inputs. Actual
    source, capture and loaded-object identity still require the investigation.
    """
    names = set(meta.get("inductor_kernel_names") or meta.get("defined_functions") or [])
    if meta and names != {kernel}:
        notes.append("Metadata kernel identity is missing, ambiguous or differs from the target; metadata fields were omitted. Inspect a single target source.")
        meta = {}
    files = isa.get("files", [])
    if isa:
        for item in isa.get("unresolved", []):
            notes.append(f"Unresolved ISA: `{item.get('file', '')}`: {item.get('reason', 'unknown')}")
        if len(files) != 1:
            notes.append("ISA evidence has no resolved file or multiple assembly artifacts; aggregate counts were omitted. Select the actual target build/configuration.")
            isa = {}
        else:
            item = files[0]
            selected = item.get("selected_symbol")
            matches = ((selected and (selected == kernel or isa.get("kernel_filter") == kernel))
                       or (not selected and set(item.get("symbols", [])) == {kernel}))
            if not matches:
                notes.append("ISA symbol scope is unbound or differs from the target; counts were omitted. Rescan using the actual target symbol and verify its dispatch mapping.")
                isa = {}
    return meta, isa


def _metric(item: dict, key: str) -> str:
    return str(item.get("metrics", {}).get(key, ""))


def build_report(autotune: dict, meta: dict, isa: dict, kernel: str, record_index: int | None = None) -> str:
    item = _pick_kernel(autotune, kernel, record_index)
    names = set(meta.get("inductor_kernel_names") or meta.get("defined_functions") or [])
    kernel_name = kernel or item.get("kernel") or (next(iter(names)) if len(names) == 1 else "unknown")
    notes = []
    if kernel and autotune.get("kernels") and not item:
        notes.append("Requested kernel was not found in the autotune summary; no other kernel's timings or inputs were reused.")
    selected_index = next((i for i, value in enumerate(autotune.get("kernels", [])) if value is item), "")
    source_file, source_hash = meta.get("file", ""), meta.get("source_sha256", "")
    isa_root, isa_status = isa.get("root", ""), isa.get("status", "unknown" if isa else "not supplied")
    meta, isa = _scoped_evidence(meta, isa, kernel_name, notes)
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
    if not load_summary and isa:
        load_summary = "No matching tracked instruction families in the resolved instructions"
    assembly_sources = "; ".join(f"{f.get('file', '')} / {f.get('sha256', '')}" for f in isa.get("files", []))

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
        f"Autotune log: `{autotune.get('log', '')}`; record index: `{selected_index}`; log line: `{item.get('log_line', '')}`.",
        f"Captured source: `{item.get('artifacts', {}).get('kernel_source_saved', '')}`; inputs: `{item.get('artifacts', {}).get('inputs_saved', '')}`.",
        f"Metadata input: `{source_file}`; SHA256: `{source_hash}`. ISA input: `{isa_root}`; scan status: `{isa_status}`.",
        "Matching names alone do not establish matching builds or inputs; verify capture/configuration, source/ISA hashes and the actually loaded artifact before joining evidence.",
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
        f"| Static AMDGCN instruction-family counts | {load_summary or 'TODO'} |",
        f"| Selected assembly file / SHA256 | {assembly_sources or 'TODO'} |",
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
    lines.extend(f"- {note}" for note in notes)
    for note in meta.get("notes", []):
        lines.append(f"- {note}")
    if isa_cls.get("atomic_observed"):
        lines.append("- Atomic instructions observed in AMDGCN scan.")
    if not notes and not meta.get("notes") and not isa_cls.get("atomic_observed"):
        lines.append("- TODO")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Make a Triton kernel investigation report")
    parser.add_argument("--autotune-summary", default="")
    parser.add_argument("--meta", default="")
    parser.add_argument("--isa", default="")
    parser.add_argument("--kernel", default="")
    parser.add_argument("--record-index", type=int, default=None, help="Zero-based record position in the autotune summary")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        report = build_report(
            _read_json(args.autotune_summary),
            _read_json(args.meta),
            _read_json(args.isa),
            args.kernel,
            args.record_index,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(json.dumps({"out": args.out}, indent=2))


if __name__ == "__main__":
    main()
