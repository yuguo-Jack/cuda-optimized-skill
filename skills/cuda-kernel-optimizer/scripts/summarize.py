#!/usr/bin/env python3
"""Render run_dir/summary.md from state.json + per-iteration artifacts.

Claude is expected to append a brief retrospective paragraph (what worked,
what didn't, what to try next) by editing the produced file. This script
lays down everything structural.
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
    if v is None:
        return "—"
    return f"{v:.4f} ms"


def _fmt_speedup(v) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}×"


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
        if "speedup_vs_best_before" in m and m["speedup_vs_best_before"] is not None:
            extra += f" — local speedup {m['speedup_vs_best_before']:.2f}×"
        lines.append(f"- **[{axis}]** {name}{extra}")
    return "\n".join(lines)


def _timeline_table(state: dict) -> str:
    rows = ["| Iter | Status | Methods | ms | Speedup vs prev best | Speedup vs ref |",
            "|------|--------|---------|----|----------------------|----------------|"]
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


def render(state_path: str, out_path: str) -> None:
    state = _read(state_path)
    run_dir = state["run_dir"]

    # Final numbers
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
    lines.append(f"# CUDA Kernel Optimization Summary")
    lines.append("")
    lines.append(f"- **Run dir**: `{run_dir}`")
    lines.append(f"- **Baseline**: `{state.get('baseline_file_original', state.get('baseline_file'))}`")
    lines.append(f"- **Reference**: `{state.get('ref_file')}`")
    lines.append(f"- **Dims**: `{json.dumps(state.get('dims', {}))}`")
    lines.append(f"- **Iterations**: {len(state.get('history', []))} / {state.get('iterations_total')}")
    lines.append("")
    lines.append(f"## Environment")
    lines.append("")
    lines.append(f"- GPU: **{gpu.get('name', '?')}** ({gpu.get('sm_arch','?')}, cc {gpu.get('compute_capability','?')})")
    lines.append(f"- nvcc: `{env.get('nvcc',{}).get('version','?')}`")
    lines.append(f"- ncu: `{env.get('ncu',{}).get('version','?')}` (can_read_counters={env.get('ncu',{}).get('can_read_counters')})")
    lines.append(f"- cutlass include: `{env.get('cutlass',{}).get('include_dir','—')}`")
    lines.append(f"- torch: `{env.get('libs',{}).get('torch',{}).get('version','?')}` — triton: `{env.get('libs',{}).get('triton',{}).get('version','?')}`")
    lines.append("")
    lines.append(f"## Headline")
    lines.append("")
    lines.append(f"- **Baseline time**: {_fmt_ms(baseline_ms)}")
    lines.append(f"- **Best time**: {_fmt_ms(best_ms)}")
    lines.append(f"- **Overall speedup vs baseline**: {_fmt_speedup(final_speedup)}")
    lines.append(f"- **Best kernel**: `{state.get('best_file')}`")
    if state.get("best_ncu_rep"):
        lines.append(f"- **Best kernel ncu-rep**: `{state['best_ncu_rep']}`")
    lines.append("")
    lines.append(f"## Iteration timeline")
    lines.append("")
    lines.append(_timeline_table(state))
    lines.append("")
    lines.append(f"## Effective methods")
    lines.append("")
    lines.append(_method_bullets(state.get("effective_methods", [])))
    lines.append("")
    lines.append(f"## Ineffective methods")
    lines.append("")
    lines.append(_method_bullets(state.get("ineffective_methods", [])))
    lines.append("")
    lines.append(f"## All methods tried")
    lines.append("")
    lines.append(_method_bullets(state.get("selected_methods", [])))
    lines.append("")
    lines.append(f"## Retrospective")
    lines.append("")
    lines.append(f"_Claude will fill this section in:_")
    lines.append("")
    lines.append(f"- Which optimizations moved the needle and why (tie to ncu evidence).")
    lines.append(f"- Which ones were no-ops or regressions, and plausible reasons.")
    lines.append(f"- Next steps if the user wants more iterations (methods not yet tried, plausible arch-specific tactics).")
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
