#!/usr/bin/env python3
"""Render run_dir/summary.md from state.json + per-iteration artifacts (v2).

Includes roofline history, axis budget tracking, attribution results,
and SASS verification status.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _read(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_ms(v) -> str:
    return f"{v:.4f} ms" if v is not None else "—"


def _fmt_speedup(v) -> str:
    return f"{v:.2f}×" if v is not None else "—"


def _method_bullets(methods: list[dict]) -> str:
    if not methods:
        return "_(none)_"
    lines = []
    for m in methods:
        axis = m.get("axis", "?")
        name = m.get("name", m.get("id", "?"))
        extra = ""
        if "iter" in m:
            extra = f" _(iter {m['iter']})_"
        if "attribution_ms" in m and m["attribution_ms"] is not None:
            extra += f" — attribution +{m['attribution_ms']:.3f}ms"
        if "speedup_vs_best_before" in m and m["speedup_vs_best_before"] is not None:
            extra += f" — speedup {m['speedup_vs_best_before']:.2f}×"
        if "note" in m:
            extra += f" [{m['note']}]"
        lines.append(f"- **[{axis}]** {name}{extra}")
    return "\n".join(lines)


def _timeline_table(state: dict) -> str:
    rows = [
        "| Iter | Status | Methods | ms | Speedup vs prev best | Speedup vs ref |",
        "|------|--------|---------|----|----------------------|----------------|",
    ]
    for h in state.get("history", []):
        methods = ", ".join(h.get("method_names") or h.get("methods") or [])
        rows.append(
            f"| {h['iter']} "
            f"| {h['status']} "
            f"| {methods} "
            f"| {_fmt_ms(h.get('ms'))} "
            f"| {_fmt_speedup(h.get('speedup_vs_best_before'))} "
            f"| {_fmt_speedup(h.get('speedup_vs_ref'))} |"
        )
    return "\n".join(rows)


def _roofline_table(state: dict) -> str:
    rows = [
        "| Iter | Bound | Δ_compute | Δ_memory | Δ_latency | Budget (c,m,l) |",
        "|------|-------|-----------|----------|-----------|----------------|",
    ]
    for r in state.get("roofline_history", []):
        budget = r.get("axis_budget", {})
        budget_str = f"({budget.get('compute','-')},{budget.get('memory','-')},{budget.get('latency','-')})"
        rows.append(
            f"| {r['iter']} "
            f"| {r.get('bound', '?')} "
            f"| {r.get('delta_compute', '?')} "
            f"| {r.get('delta_memory', '?')} "
            f"| {r.get('delta_latency', '?')} "
            f"| {budget_str} |"
        )
    return "\n".join(rows)


def render(state_path: str, out_path: str) -> None:
    state = _read(state_path)
    run_dir = state["run_dir"]

    baseline_bench_path = os.path.join(run_dir, "baseline", "bench.json")
    baseline_ms = None
    if os.path.isfile(baseline_bench_path):
        b = _read(baseline_bench_path)
        baseline_ms = (b.get("kernel") or {}).get("average_ms")

    best_ms = state.get("best_metric_ms")
    final_speedup = (baseline_ms / best_ms) if (baseline_ms and best_ms and best_ms > 0) else None

    env = state.get("env", {})
    gpu = (env.get("gpus") or [{}])[0]

    lines = []
    lines.append("# CUDA Kernel Optimization Summary (v2 — Roofline-Driven)")
    lines.append("")
    lines.append(f"- **Run dir**: `{run_dir}`")
    lines.append(f"- **Baseline**: `{state.get('baseline_file_original', state.get('baseline_file'))}`")
    lines.append(f"- **Reference**: `{state.get('ref_file')}`")
    lines.append(f"- **Dims**: `{json.dumps(state.get('dims', {}))}`")
    lines.append(f"- **Iterations**: {len(state.get('history', []))} / {state.get('iterations_total')}")
    lines.append(f"- **Branches per iter**: {state.get('branches', 4)}")
    lines.append("")

    lines.append("## Environment")
    lines.append("")
    lines.append(f"- GPU: **{gpu.get('name', '?')}** ({gpu.get('sm_arch','?')}, cc {gpu.get('compute_capability','?')})")
    lines.append(f"- nvcc: `{env.get('nvcc',{}).get('version','?')}`")
    lines.append(f"- ncu: `{env.get('ncu',{}).get('version','?')}` (can_read_counters={env.get('ncu',{}).get('can_read_counters')})")
    lines.append(f"- cutlass include: `{env.get('cutlass',{}).get('include_dir','—')}`")
    lines.append("")

    lines.append("## Headline")
    lines.append("")
    lines.append(f"- **Baseline time**: {_fmt_ms(baseline_ms)}")
    lines.append(f"- **Best time**: {_fmt_ms(best_ms)}")
    lines.append(f"- **Overall speedup vs baseline**: {_fmt_speedup(final_speedup)}")
    lines.append(f"- **Best kernel**: `{state.get('best_file')}`")
    if state.get("best_ncu_rep"):
        lines.append(f"- **Best kernel ncu-rep**: `{state['best_ncu_rep']}`")
    lines.append("")

    lines.append("## Roofline History")
    lines.append("")
    lines.append(_roofline_table(state))
    lines.append("")

    lines.append("## Iteration Timeline")
    lines.append("")
    lines.append(_timeline_table(state))
    lines.append("")

    lines.append("## Effective Methods (attribution-verified)")
    lines.append("")
    lines.append(_method_bullets(state.get("effective_methods", [])))
    lines.append("")

    lines.append("## Ineffective Methods")
    lines.append("")
    lines.append(_method_bullets(state.get("ineffective_methods", [])))
    lines.append("")

    lines.append("## Implementation-Failed Methods (SASS verification failed)")
    lines.append("")
    lines.append(_method_bullets(state.get("implementation_failed_methods", [])))
    lines.append("")

    lines.append("## All Methods Tried")
    lines.append("")
    lines.append(_method_bullets(state.get("selected_methods", [])))
    lines.append("")

    if state.get("frontier"):
        lines.append("## Frontier (unexplored branch candidates)")
        lines.append("")
        for fe in state["frontier"][:10]:
            lines.append(
                f"- iter {fe.get('iter')} branch {fe.get('branch_index')}: "
                f"{_fmt_ms(fe.get('ms'))} "
                f"(+{fe.get('delta_from_champion', '?')}ms from champion)"
            )
        lines.append("")

    lines.append("## Retrospective")
    lines.append("")
    lines.append("_Claude will fill this section in:_")
    lines.append("")
    lines.append("- Which optimizations moved the needle and why (tie to ncu + attribution evidence).")
    lines.append("- Which ones were no-ops or regressions, and plausible reasons.")
    lines.append("- How roofline gaps shifted across iterations (did the bound type change?).")
    lines.append("- Next steps if the user wants more iterations.")
    lines.append("")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(json.dumps({"summary": out_path}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    render(args.state, args.out)


if __name__ == "__main__":
    main()
