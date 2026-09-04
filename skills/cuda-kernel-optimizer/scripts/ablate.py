#!/usr/bin/env python3
"""Single-method ablation for attribution.

For each method applied in the champion kernel, this script expects Claude
to have generated an ablated kernel (champion minus that one method) under
  iterv{i}/ablations/{method_id}/kernel.<ext>

This script benchmarks each ablated kernel and computes attribution:
  attribution(m) = ms_ablated(m) - ms_champion

Positive → the method helped (removing it slowed things down).
Zero/Negative → the method did not help or actually hurt.

Writes iterv{i}/attribution.json.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

from build import BuildSpec, build


_BUNDLED_BENCHMARK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.py")


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dims_argv(dims: dict) -> list[str]:
    return [f"--{k}={v}" for k, v in dims.items()]


def _ptr_size_argv(ptr_size: int) -> list[str]:
    return ["--ptr-size", str(ptr_size)] if ptr_size and ptr_size > 0 else []


def _bench_kernel(
    benchmark_py: str,
    kernel_path: str,
    ref_path: str,
    dims: dict,
    ptr_size: int,
    json_out: str,
    warmup: int = 5,
    repeat: int = 15,
    artifact_manifest: str = "",
    arch: str = "",
    gpu: int = 0,
    numerics_mode: str = "reference",
    backend: str = "auto",
) -> dict | None:
    """Run benchmark.py on a single kernel and return the parsed JSON result."""
    cmd = [
        sys.executable, benchmark_py, kernel_path,
        "--ref", ref_path,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--json-out", json_out,
        "--gpu", str(gpu),
        "--numerics-mode", numerics_mode,
        "--backend", backend,
    ] + _ptr_size_argv(ptr_size) + _dims_argv(dims)
    if arch:
        cmd += ["--arch", arch]
    if artifact_manifest:
        cmd += ["--artifact-manifest", artifact_manifest]

    Path(json_out).parent.mkdir(parents=True, exist_ok=True)

    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="ignore",
        )
    except OSError as e:
        print(f"[ablate] benchmark failed: {e}", file=sys.stderr)
        return None

    if os.path.isfile(json_out):
        return _load_json(json_out)
    return None


def run(state_path: str, iteration: int, benchmark_py: str = None, compile_jobs: str = "") -> dict:
    state = _load_json(state_path)
    run_dir = state["run_dir"]
    iter_dir = os.path.join(run_dir, f"iterv{iteration}")
    bench_py = benchmark_py or _BUNDLED_BENCHMARK

    # Load champion timing
    champion_bench = os.path.join(iter_dir, "bench.json")
    if not os.path.isfile(champion_bench):
        sys.exit(f"Champion bench.json not found at {champion_bench}")
    champion_data = _load_json(champion_bench)
    champion_ms = (champion_data.get("kernel") or {}).get("average_ms")
    if champion_ms is None:
        sys.exit("Champion has no timing data")

    # Load methods
    methods_path = os.path.join(iter_dir, "methods.json")
    if not os.path.isfile(methods_path):
        sys.exit(f"methods.json not found at {methods_path}")
    methods_data = _load_json(methods_path)
    methods_list = methods_data.get("methods", [])

    ref_file = state["ref_file"]
    dims = state.get("dims", {})
    ptr_size = state.get("ptr_size", 0)
    noise_threshold = state.get("noise_threshold_pct", 2.0)
    gpu = int(state.get("gpu", 0))
    arch = (state.get("env", {}).get("primary_sm_arch")
            or ((state.get("env", {}).get("gpus") or [{}])[0].get("sm_arch"))
            or "sm_80")
    nvcc = (state.get("env", {}).get("nvcc") or {}).get("path") or "nvcc"
    numerics_mode = state.get("numerics_mode", "reference")
    cache_dir = os.path.join(run_dir, ".build-cache")

    attributions = []
    ablation_dir = os.path.join(iter_dir, "ablations")

    candidates = []
    for m in methods_list:
        mid = m.get("id", "unknown")
        method_dir = os.path.join(ablation_dir, mid.replace(".", "_"))

        # Find ablated kernel
        ablated_kernel = None
        for ext in (".cu", ".py"):
            candidate = os.path.join(method_dir, f"kernel{ext}")
            if os.path.isfile(candidate):
                ablated_kernel = candidate
                break

        if ablated_kernel is None:
            # Missing ablation is not evidence that the method was neutral.
            attributions.append({
                "method_id": mid,
                "ablated_kernel": None,
                "ablated_ms": None,
                "champion_ms": champion_ms,
                "attribution_ms": None,
                "attribution_pct": None,
                "contributed": None,
                "status": "inconclusive",
                "note": "no_ablated_kernel_provided",
            })
            continue

        candidates.append((m, ablated_kernel, method_dir))

    bundled = os.path.realpath(bench_py) == os.path.realpath(_BUNDLED_BENCHMARK)
    kind = "cuda"
    if bundled and candidates and any("cutlass/" in Path(k).read_text(encoding="utf-8", errors="ignore") or "cute/" in Path(k).read_text(encoding="utf-8", errors="ignore") for _, k, _ in candidates):
        kind = "cutlass"
    build_results = {}
    if bundled and candidates and all(k.endswith(".cu") for _, k, _ in candidates):
        kind = "cutlass" if any("cutlass/" in Path(k).read_text(encoding="utf-8", errors="ignore")
                                 or "cute/" in Path(k).read_text(encoding="utf-8", errors="ignore")
                                 for _, k, _ in candidates) else "cuda"
        from branch_explore import _compile_jobs
        jobs = _compile_jobs(compile_jobs or state.get("compile_jobs", "auto"), len(candidates), kind)
        def do_build(item):
            method, kernel, _ = item
            return method.get("id", "unknown"), build(
                BuildSpec(os.path.abspath(kernel), kind, arch, nvcc), cache_dir)
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            for future in [pool.submit(do_build, item) for item in candidates]:
                mid, result = future.result()
                build_results[mid] = result

    # Benchmark ablations in methods.json order on the single ranking GPU.
    for m, ablated_kernel, method_dir in candidates:
        mid = m.get("id", "unknown")

        # Benchmark ablated kernel
        ablated_json_out = os.path.join(method_dir, "bench.json")
        Path(ablated_json_out).unlink(missing_ok=True)
        build_result = build_results.get(mid, {})
        if bundled and ablated_kernel.endswith(".cu") and not build_result.get("ok"):
            result = None
        else:
            result = _bench_kernel(
                bench_py, ablated_kernel, ref_file, dims, ptr_size, ablated_json_out,
                artifact_manifest=build_result.get("manifest", ""), arch=arch, gpu=gpu,
                numerics_mode=numerics_mode, backend=kind,
            )

        if result is None or not result.get("correctness", {}).get("passed", False):
            # A failed ablation cannot establish causality.
            attributions.append({
                "method_id": mid,
                "ablated_kernel": ablated_kernel,
                "ablated_ms": None,
                "champion_ms": champion_ms,
                "attribution_ms": None,
                "attribution_pct": None,
                "contributed": None,
                "status": "inconclusive",
                "note": "ablated_kernel_failed_validation",
            })
            continue

        ablated_ms = (result.get("kernel") or {}).get("average_ms")
        if ablated_ms is None:
            attributions.append({
                "method_id": mid,
                "ablated_kernel": ablated_kernel,
                "ablated_ms": None,
                "champion_ms": champion_ms,
                "attribution_ms": 0.0,
                "attribution_pct": 0.0,
                "contributed": None,
                "status": "inconclusive",
                "note": "no_timing_in_ablated_bench",
            })
            continue

        # Compute attribution
        attr_ms = ablated_ms - champion_ms
        attr_pct = (attr_ms / champion_ms * 100) if champion_ms > 0 else 0.0
        contributed = attr_pct > noise_threshold

        attributions.append({
            "method_id": mid,
            "ablated_kernel": ablated_kernel,
            "ablated_ms": round(ablated_ms, 4),
            "champion_ms": round(champion_ms, 4),
            "attribution_ms": round(attr_ms, 4),
            "attribution_pct": round(attr_pct, 2),
            "contributed": contributed,
            "build": build_result,
        })

    output = {
        "iter": iteration,
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
    p.add_argument("--compile-jobs", default="")
    args = p.parse_args()
    run(args.state, args.iter, args.benchmark, args.compile_jobs)


if __name__ == "__main__":
    main()
