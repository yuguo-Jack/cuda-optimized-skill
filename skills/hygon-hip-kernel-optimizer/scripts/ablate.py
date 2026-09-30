#!/usr/bin/env python3
"""Single-method ablation for attribution.

For each method applied in the champion kernel, this script expects the agent
to have generated an ablated kernel (champion minus that one method) under
  iterv{i}/ablations/{method_id}/kernel.<ext>

This script benchmarks each ablated kernel and computes attribution:
  attribution(m) = ms_ablated(m) - ms_champion

Positive means the method helped (removing it slowed things down).
Near-zero or inconsistent repetitions remain inconclusive; consistent negative
contribution means the method hurt this particular workload.

Writes iterv{i}/attribution.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from experiment import benchmark_gate, resolve_benchmark, require_open_iteration, iteration_kernel
from paired_measurement import compare


KERNEL_EXTS = (".hip", ".cu", ".cpp", ".cc", ".cxx", ".py")


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(state_path: str, iteration: int, benchmark_py: str = None) -> dict:
    state = _load_json(state_path)
    require_open_iteration(state, iteration)
    run_dir = state["run_dir"]
    iter_dir = os.path.join(run_dir, f"iterv{iteration}")
    bench_py = resolve_benchmark(state, benchmark_py)

    # Load champion timing
    champion_bench = os.path.join(iter_dir, "bench.json")
    if not os.path.isfile(champion_bench):
        sys.exit(f"Champion bench.json not found at {champion_bench}")
    champion_data = _load_json(champion_bench)
    champion_ms = (champion_data.get("kernel") or {}).get("average_ms")
    if not benchmark_gate(champion_data, iteration_kernel(iter_dir), state["ref_file"])[0]:
        sys.exit("Champion lacks valid correctness/timing evidence")

    # Load methods
    methods_path = os.path.join(iter_dir, "methods.json")
    if not os.path.isfile(methods_path):
        sys.exit(f"methods.json not found at {methods_path}")
    methods_data = _load_json(methods_path)
    methods_list = methods_data.get("methods", [])

    dims = state.get("dims", {})
    ptr_size = state.get("ptr_size", 0)
    noise_threshold = state.get("noise_threshold_pct", 2.0)

    attributions = []
    ablation_dir = os.path.join(iter_dir, "ablations")

    for m in methods_list:
        mid = m.get("id", "unknown")
        method_dir = os.path.join(ablation_dir, mid.replace(".", "_"))

        # Find ablated kernel
        entries = [str(Path(method_dir) / ("kernel" + ext)) for ext in KERNEL_EXTS
                   if (Path(method_dir) / ("kernel" + ext)).is_file()]
        if len(entries) > 1:
            raise SystemExit(f"Ambiguous ablation entry points: {entries}")
        ablated_kernel = entries[0] if entries else None

        if ablated_kernel is None:
            # No ablated kernel: attribution remains unknown.
            attributions.append({
                "method_id": mid,
                "ablated_kernel": None,
                "ablated_ms": None,
                "champion_ms": champion_ms,
                "attribution_ms": None,
                "attribution_pct": None,
                "contributed": None,
                "note": "no_ablated_kernel_provided",
            })
            continue

        cases = [{"id": "primary", "dims": dims, "ptr_size": ptr_size,
                  "seeds": [champion_data.get("seed", 42)]}]
        comparison = compare(state, iteration_kernel(iter_dir), ablated_kernel, bench_py,
                             Path(method_dir) / "paired", champion_data.get("warmup", 10),
                             champion_data.get("repeat", 20), cases)
        assessment = comparison["assessment"]
        valid = assessment.get("valid") is True
        direction = assessment.get("direction")
        contributed = True if direction == "beneficial" else False if direction == "harmful" else None
        current_ms = assessment.get("candidate_ms")
        delta = assessment.get("difference_ms")
        attributions.append({
            "method_id": mid, "ablated_kernel": ablated_kernel,
            "ablated_ms": assessment.get("control_ms"), "champion_ms": current_ms,
            "attribution_ms": delta, "attribution_pct": delta / current_ms * 100 if valid else None,
            "contributed": contributed, "validation_passed": valid,
            "comparison": comparison,
            "note": "primary workload only; inconclusive repetitions do not prove ineffectiveness" if valid
                    else "invalid_ablation_not_performance_evidence",
        })

    output = {
        "iter": iteration,
        "champion_source_sha256": champion_data.get("source_sha256"),
        "champion_ms": round(champion_ms, 4),
        "noise_threshold_pct": noise_threshold,
        "attributions": attributions,
    }

    out_path = os.path.join(iter_dir, "attribution.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(json.dumps(output, indent=2))
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--benchmark", default=None)
    args = p.parse_args()
    run(args.state, args.iter, args.benchmark)


if __name__ == "__main__":
    main()
