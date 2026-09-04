#!/usr/bin/env python3
"""Branch-and-Select: compile and benchmark K candidate kernels in parallel.

All K branches share the same method combination (from methods.json) but
differ in hyperparameters (tile size, num_stages, num_warps, etc.).

Claude generates K kernels under iterv{i}/branches/b{1..K}/kernel.<ext>.

This script:
  1. Compiles all K kernels (can be parallelized)
  2. Benchmarks each kernel with validation
  3. Selects champion = highest weighted multi-scale speedup (or fastest custom result)
  4. Copies champion to iterv{i}/kernel.<ext>
  5. Returns non-champions as frontier candidates in state

Writes iterv{i}/branch_results.json.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from build import BuildSpec, build
    from workload_matrix import generate_workload_matrix, shrink_workload
    from contract_check import check_contract
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build import BuildSpec, build
    from workload_matrix import generate_workload_matrix, shrink_workload
    from contract_check import check_contract


_BUNDLED_BENCHMARK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.py")


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


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
    warmup: int = 10,
    repeat: int = 20,
    artifact_manifest: str = "",
    arch: str = "",
    gpu: int = 0,
    seed: int = 42,
    numerics_mode: str = "reference",
    backend: str = "auto",
    timing_batches: int = 1,
    validation_seeds: str = "",
) -> dict:
    """Run benchmark.py on a kernel. Returns parsed result or error dict."""
    attempt_json = f"{json_out}.attempt-{os.getpid()}"
    Path(attempt_json).unlink(missing_ok=True)
    cmd = [
        sys.executable, benchmark_py, kernel_path,
        "--ref", ref_path,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--json-out", attempt_json,
        "--gpu", str(gpu),
        "--seed", str(seed),
        "--numerics-mode", numerics_mode,
        "--backend", backend,
        "--timing-batches", str(timing_batches),
    ] + _ptr_size_argv(ptr_size) + _dims_argv(dims)
    if arch:
        cmd += ["--arch", arch]
    if artifact_manifest:
        cmd += ["--artifact-manifest", artifact_manifest]
    if validation_seeds:
        cmd += ["--validation-seeds", validation_seeds]

    Path(json_out).parent.mkdir(parents=True, exist_ok=True)
    stderr_out = json_out.replace(".json", ".stderr.txt")

    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="ignore",
        )
    except OSError as e:
        return {"error": str(e), "passed": False}

    # Save stderr for debugging
    with open(stderr_out, "w", encoding="utf-8") as f:
        f.write("---STDOUT---\n")
        f.write(r.stdout or "")
        f.write("\n---STDERR---\n")
        f.write(r.stderr or "")

    if os.path.isfile(attempt_json):
        os.replace(attempt_json, json_out)
        return _load_json(json_out)

    return {
        "error": "no_json_output",
        "stderr": (r.stderr or "")[-2000:],
        "passed": False,
    }


def run(state_path: str, iteration: int, benchmark_py: str = None,
        warmup: int = 10, repeat: int = 20, compile_jobs: str = "") -> dict:
    state = _load_json(state_path)
    run_dir = state["run_dir"]
    iter_dir = os.path.join(run_dir, f"iterv{iteration}")
    bench_py = benchmark_py or _BUNDLED_BENCHMARK
    branches_dir = os.path.join(iter_dir, "branches")
    ref_file = state["ref_file"]
    dims = state.get("dims", {})
    ptr_size = state.get("ptr_size", 0)
    num_branches = state.get("branches", 4)
    gpu = int(state.get("gpu", 0))
    arch = (state.get("env", {}).get("primary_sm_arch")
            or ((state.get("env", {}).get("gpus") or [{}])[0].get("sm_arch"))
            or "sm_80")
    nvcc = (state.get("env", {}).get("nvcc") or {}).get("path") or "nvcc"
    numerics_mode = state.get("numerics_mode", "reference")
    timing_batches = int(state.get("timing_batches", 1))
    timing_repeats = int(state.get("timing_repeats", repeat))
    validation_seeds = state.get("validation_seeds", "")
    methods_path = os.path.join(iter_dir, "methods.json")
    try:
        methods_payload = _load_json(methods_path) if os.path.isfile(methods_path) else {}
    except (OSError, json.JSONDecodeError):
        methods_payload = {}
    method_ids = {str(m.get("id", "")) for m in methods_payload.get("methods", [])}
    use_e2e = bool(method_ids & {"latency.cuda_graphs", "latency.static_launch_grid_graph",
                                 "latency.independent_kernel_overlap", "memory.kernel_fusion",
                                 "latency.grouped_gemm_scheduler"})
    stability_policy = state.get("stability_policy", {}) if isinstance(state.get("stability_policy", {}), dict) else {}
    max_large_regression_pct = float((state.get("large_regression_gate", {}) or {}).get("max_pct", stability_policy.get("max_large_regression_pct", 5.0)))
    max_cv = float(stability_policy.get("max_cv", 0.10))
    cache_dir = os.path.join(run_dir, ".build-cache")

    # Discover branches
    branch_dirs = []
    for i in range(1, num_branches + 1):
        bd = os.path.join(branches_dir, f"b{i}")
        if os.path.isdir(bd):
            # Check if there's a kernel file
            kernel = None
            for ext in (".cu", ".py"):
                candidate = os.path.join(bd, f"kernel{ext}")
                if os.path.isfile(candidate):
                    kernel = candidate
                    break
            if kernel:
                branch_dirs.append({"index": i, "dir": bd, "kernel": kernel})

    if not branch_dirs:
        # Fallback: check if there's a single kernel directly in iter_dir
        for ext in (".cu", ".py"):
            candidate = os.path.join(iter_dir, f"kernel{ext}")
            if os.path.isfile(candidate):
                branch_dirs.append({
                    "index": 0, "dir": iter_dir, "kernel": candidate,
                })
                break

    if not branch_dirs:
        sys.exit(f"No branch kernels found under {branches_dir}")

    print(f"[branch_explore] Found {len(branch_dirs)} branches", file=sys.stderr)

    # External benchmark scripts cannot consume artifacts, so preserve the
    # original serialized compile+execute behavior for compatibility.
    bundled = os.path.realpath(bench_py) == os.path.realpath(_BUNDLED_BENCHMARK)
    kind = "cuda"
    build_results = {}
    if bundled and all(b["kernel"].endswith(".cu") for b in branch_dirs):
        kind = "cutlass" if any("cutlass/" in Path(b["kernel"]).read_text(encoding="utf-8", errors="ignore")
                                 or "cute/" in Path(b["kernel"]).read_text(encoding="utf-8", errors="ignore")
                                 for b in branch_dirs) else "cuda"
        jobs = _compile_jobs(compile_jobs or state.get("compile_jobs", "auto"), len(branch_dirs), kind)
        print(f"[branch_explore] compiling {len(branch_dirs)} branches with {jobs} workers", file=sys.stderr)
        def do_build(branch):
            spec = BuildSpec(os.path.abspath(branch["kernel"]), kind, arch, nvcc)
            return branch["index"], build(spec, cache_dir)
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(do_build, b) for b in branch_dirs]
            for future in futures:
                idx, result = future.result()
                build_results[idx] = result
        # A resource-killed compiler can succeed when retried alone after the
        # rest of the pool has drained. Ordinary compiler errors are retained.
        for branch in branch_dirs:
            idx = branch["index"]
            result = build_results.get(idx, {})
            err = str(result.get("error", "")).lower()
            oom = result.get("returncode") in (137, -9) or "out of memory" in err or "cannot allocate" in err
            if oom:
                spec = BuildSpec(os.path.abspath(branch["kernel"]), kind, arch, nvcc)
                build_results[idx] = build(spec, cache_dir, force=True)

    # Static contract gate runs after compilation and before any GPU work.
    # It catches ABI/dimension/index-layout mistakes that otherwise look like
    # successful builds and would contaminate timing comparisons.
    contract_results = {}
    for branch in branch_dirs:
        idx = branch["index"]
        if bundled and branch["kernel"].endswith(".cu"):
            report = check_contract(branch["kernel"], dims=dims, compile=False)
        else:
            report = {
                "kernel": os.path.abspath(branch["kernel"]),
                "contract_pass": "inconclusive",
                "compile_pass": "inconclusive",
                "correctness_pass": "inconclusive",
                "race_safe": "inconclusive",
                "timing_valid": "inconclusive",
                "warnings": ["custom/Triton benchmark contract requires runtime handshake"],
            }
        contract_results[idx] = report

    # GPU benchmark remains strictly ordered and serialized. Bundled CUDA
    # runs use the same three scale cases for every branch; custom benchmark
    # protocols retain the historical single-case fallback.
    matrix = generate_workload_matrix(
        dims, ptr_size, archetype=state.get("archetype", "generic"),
        profile=state.get("workload_matrix") or None,
        target_mb=int(state.get("max_working_set_mb", 384)),
        hard_mb=int(state.get("hard_working_set_mb", 512)),
        adaptive_downscale=bool(state.get("adaptive_downscale", True)),
    ) if bundled else [{"scale": "default", "realized_dims": dims,
                        "realized_ptr_size": ptr_size, "weight": 1.0,
                        "downscaled": False}]
    # Measure baseline once using the exact realized matrix so the global
    # score can be expressed as speedup and large-scale regressions are visible.
    baseline_scales = {}
    if bundled and os.path.isfile(state.get("baseline_file", "")):
        baseline_kernel = state["baseline_file"]
        baseline_build = build_results.get("__baseline__", {})
        if baseline_kernel.endswith(".cu") and not baseline_build:
            baseline_build = build(BuildSpec(os.path.abspath(baseline_kernel), kind, arch, nvcc), cache_dir)
        for case in matrix:
            scale = case["scale"]
            baseline_json = os.path.join(iter_dir, f"baseline_{scale}.json")
            Path(baseline_json).unlink(missing_ok=True)
            if baseline_kernel.endswith(".cu") and not baseline_build.get("ok"):
                continue
            baseline_result = _bench_kernel(
                bench_py, baseline_kernel, ref_file, case.get("realized_dims", dims),
                case.get("realized_ptr_size", ptr_size), baseline_json, warmup,
                timing_repeats, artifact_manifest=baseline_build.get("manifest", ""),
                arch=arch, gpu=gpu, numerics_mode=numerics_mode, backend=kind,
                timing_batches=timing_batches, validation_seeds=validation_seeds)
            bms = (baseline_result.get("kernel") or {}).get("average_ms")
            if bms is not None and baseline_result.get("correctness", {}).get("passed", False):
                baseline_scales[scale] = bms
        build_results["__baseline__"] = baseline_build
    results = []
    for branch in sorted(branch_dirs, key=lambda b: b["index"]):
        idx = branch["index"]
        kernel = branch["kernel"]
        json_out = os.path.join(branch["dir"], "bench.json")
        # Never let a failed/current attempt consume a prior PASS.
        Path(json_out).unlink(missing_ok=True)

        print(f"[branch {idx}] Benchmarking {os.path.basename(kernel)}...",
              file=sys.stderr)

        build_result = build_results.get(idx, {})
        per_scale = []
        for case in matrix:
            scale = case["scale"]
            scale_json = os.path.join(branch["dir"], f"bench_{scale}.json")
            Path(scale_json).unlink(missing_ok=True)
            attempts = []
            active_case = case
            for attempt in range(4):
                if active_case.get("resource_unavailable"):
                    bench_result = {"correctness": {"passed": False},
                                    "error": {"code": "resource_unavailable", "stage": "workload_fit",
                                               "message": "working set remains above hard limit"}}
                elif bundled and branch["kernel"].endswith(".cu") and not build_result.get("ok"):
                    bench_result = {"correctness": {"passed": False}, "error": build_result.get("error", "compile_failed")}
                elif contract_results[idx].get("contract_pass") == "fail":
                    bench_result = {"correctness": {"passed": False},
                                    "error": {"code": "contract_failed", "stage": "contract_check",
                                               "message": "; ".join(contract_results[idx].get("errors", []))}}
                else:
                    bench_result = _bench_kernel(
                        bench_py, kernel, ref_file, active_case.get("realized_dims", dims),
                        active_case.get("realized_ptr_size", ptr_size), scale_json, warmup,
                        timing_repeats if bundled else repeat,
                        artifact_manifest=build_result.get("manifest", ""), arch=arch, gpu=gpu,
                        numerics_mode=numerics_mode, backend=kind,
                        timing_batches=timing_batches if bundled else 1,
                        validation_seeds=validation_seeds if bundled else "",
                    )
                # Attach the exact requested/realized case to the attempt
                # result even when benchmark.py was invoked without
                # --workload-scale (the branch runner owns the matrix).
                if isinstance(bench_result, dict):
                    bench_result["workload"] = dict(active_case)
                    if scale_json:
                        _write_json(scale_json, bench_result)
                attempts.append({"attempt": attempt, "workload": dict(active_case),
                                 "error": bench_result.get("error"),
                                 "correctness": bench_result.get("correctness"),
                                 "timing": bench_result.get("kernel")})
                error_text = str(bench_result.get("error", "")).lower()
                oom = any(x in error_text for x in ("out of memory", "cuda error: out of memory", "resource exhausted"))
                if not oom or attempt >= 3:
                    break
                active_case = shrink_workload(active_case, 0.8)
            passed_scale = bool(bench_result.get("correctness", {}).get("passed", False))
            scale_ms = (bench_result.get("end_to_end_ms") if use_e2e else
                        (bench_result.get("kernel") or {}).get("average_ms"))
            per_scale.append({"scale": scale, "workload": case,
                              "passed": passed_scale, "ms": scale_ms,
                              "stability": (bench_result.get("kernel") or {}).get("stability"),
                              "robust_cv": (bench_result.get("kernel") or {}).get("robust_cv"),
                              "kernel_only_ms": (bench_result.get("kernel") or {}).get("average_ms"),
                              "end_to_end_ms": bench_result.get("end_to_end_ms"),
                              "result": bench_result, "attempts": attempts,
                              "executed_workload": active_case})
        passed = all(x["passed"] and x["ms"] is not None for x in per_scale)
        weighted = {x["scale"]: x["ms"] for x in per_scale if x["ms"] is not None}
        weights = state.get("scale_weights", {"small": .2, "medium": .3, "large": .5})
        baseline_ms = None
        score = None
        weighted_latency = None
        speedups = {}
        large_regression_pct = None
        if passed:
            for s, value in weighted.items():
                if s in baseline_scales and value > 0:
                    speedups[s] = baseline_scales[s] / value
            if len(speedups) == len(weighted):
                # Cross-scale objective is weighted geometric mean *speedup*.
                # Keep the legacy field name for readers, but expose an
                # unambiguous score as well.
                score = math.prod(max(speedups[s], 1e-12) ** float(weights.get(s, 0.0)) for s in speedups)
                weighted_latency = math.prod(max(weighted[s], 1e-12) ** float(weights.get(s, 0.0)) for s in weighted)
                if "large" in speedups:
                    large_regression_pct = (1.0 / speedups["large"] - 1.0) * 100.0
        ms = weighted.get("large") or weighted.get("medium") or weighted.get("small")

        results.append({
            "branch_index": idx,
            "kernel": kernel,
            "passed": passed,
            "ms": ms,
            "error": next((s["result"].get("error") for s in per_scale if s["result"].get("error")), None),
            "build": build_result,
            "contract": contract_results[idx],
            "states": {
                "compile_pass": ("pass" if build_result.get("ok") else "fail") if bundled else "inconclusive",
                "contract_pass": contract_results[idx].get("contract_pass", "inconclusive"),
                "correctness_pass": "pass" if passed else "fail",
                "race_safe": "inconclusive",
                "timing_valid": "pass" if passed and all(s.get("stability") != "unstable" for s in per_scale) else "fail",
            },
            "scales": per_scale,
            "requested_dims": {x["scale"]: x["workload"].get("requested_dims", {}) for x in per_scale},
            "realized_dims": {x["scale"]: x["executed_workload"].get("realized_dims", {}) for x in per_scale},
            "working_set_bytes": {x["scale"]: x["executed_workload"].get("realized_working_set_bytes", 0) for x in per_scale},
            "oom_attempts": {x["scale"]: x.get("attempts", []) for x in per_scale},
            "weighted_geomean_ms": score,
            "weighted_geomean_speedup": score,
            "weighted_latency_geomean_ms": weighted_latency,
            "speedups": speedups,
            "large_regression_pct": large_regression_pct,
        })

        status = "PASS" if passed else "FAIL"
        ms_str = f"{ms:.4f} ms" if ms else "N/A"
        print(f"[branch {idx}] {status}  {ms_str}", file=sys.stderr)

    # Select champion: fastest valid branch
    valid_results = [r for r in results if r["passed"] and r["ms"] is not None and
                     r.get("states", {}).get("compile_pass") != "fail" and
                     r.get("states", {}).get("contract_pass") != "fail" and
                     (r.get("large_regression_pct") is None or r.get("large_regression_pct") <= max_large_regression_pct) and
                     all((s.get("stability") not in {"unstable"}) and
                         (s.get("robust_cv") is None or s.get("robust_cv") <= max_cv)
                         for s in r.get("scales", []))]

    if not valid_results:
        output = {
            "iter": iteration,
            "status": "all_branches_failed",
            "branches": results,
            "champion": None,
        }
        _write_json(os.path.join(iter_dir, "branch_results.json"), output)
        print(json.dumps(output, indent=2))
        sys.exit(2)

    # Higher weighted speedup wins.  For a custom benchmark without a
    # baseline, retain the historical latency minimization and branch-index
    # tie-break.
    if any(r.get("weighted_geomean_speedup") is not None for r in valid_results):
        champion = max(valid_results, key=lambda r: (r.get("weighted_geomean_speedup") or 0.0, -r["branch_index"]))
    else:
        champion = min(valid_results, key=lambda r: (r.get("ms") or float("inf"), r["branch_index"]))

    # Copy champion kernel to iterv{i}/kernel.<ext>
    champ_kernel = champion["kernel"]
    ext = os.path.splitext(champ_kernel)[1]
    dest = os.path.join(iter_dir, f"kernel{ext}")
    if os.path.abspath(champ_kernel) != os.path.abspath(dest):
        shutil.copy2(champ_kernel, dest)
    if champion.get("build", {}).get("manifest"):
        shutil.copy2(champion["build"]["manifest"], os.path.join(iter_dir, "kernel.build.json"))

    # Also copy champion bench.json to iter_dir
    champ_bench = os.path.join(os.path.dirname(champ_kernel), "bench.json")
    dest_bench = os.path.join(iter_dir, "bench.json")
    if not os.path.isfile(champ_bench):
        # Keep the legacy consumer contract: bench.json represents the large
        # case (or the only custom case), while per-scale files retain all data.
        large_case = os.path.join(os.path.dirname(champ_kernel), "bench_large.json")
        if os.path.isfile(large_case):
            champ_bench = large_case
    if os.path.isfile(champ_bench) and os.path.abspath(champ_bench) != os.path.abspath(dest_bench):
        shutil.copy2(champ_bench, dest_bench)

    # Build frontier from non-champion valid results
    frontier_entries = []
    for r in valid_results:
        if r["branch_index"] != champion["branch_index"]:
            frontier_entries.append({
                "iter": iteration,
                "branch_index": r["branch_index"],
                "kernel": r["kernel"],
                "ms": r["ms"],
                "delta_from_champion": round(r["ms"] - champion["ms"], 4),
            })

    output = {
        "iter": iteration,
        "status": "champion_selected",
        "champion": {
            "branch_index": champion["branch_index"],
            "kernel": dest,
            "ms": champion["ms"],
            "weighted_geomean_ms": champion.get("weighted_geomean_ms"),
            "weighted_geomean_speedup": champion.get("weighted_geomean_speedup"),
            "weighted_latency_geomean_ms": champion.get("weighted_latency_geomean_ms"),
        },
        "branches": results,
        "frontier": frontier_entries,
        "total_branches": len(branch_dirs),
        "valid_branches": len(valid_results),
        "execution_order": [b["branch_index"] for b in sorted(branch_dirs, key=lambda b: b["index"])],
        "compile_jobs": _compile_jobs(compile_jobs or state.get("compile_jobs", "auto"), len(branch_dirs), kind if bundled else "cuda"),
        "backend": kind if bundled else "custom",
        "baseline_scales": baseline_scales,
        "contract_cases": next(iter(contract_results.values()), {}).get("contract_cases", []),
        "scale_weights": state.get("scale_weights", {"small": .2, "medium": .3, "large": .5}),
    }

    _write_json(os.path.join(iter_dir, "branch_results.json"), output)
    print(json.dumps(output, indent=2))
    return output


def _compile_jobs(value: str | int, count: int, kind: str) -> int:
    if isinstance(value, int) or str(value).isdigit():
        return max(1, min(count, int(value)))
    try:
        cpu = max(1, (os.cpu_count() or 1) // 2)
        available = int(Path("/proc/meminfo").read_text().split("MemAvailable:", 1)[1].split()[0]) // 1024
        per_job = 4096 if kind == "cutlass" else 2048
        mem = max(1, (available - 2048) // per_job)
        return max(1, min(count, cpu, mem, 4))
    except (OSError, ValueError, IndexError):
        return max(1, min(count, 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--benchmark", default=None)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=20)
    p.add_argument("--compile-jobs", default="")
    args = p.parse_args()
    run(args.state, args.iter, args.benchmark, args.warmup, args.repeat, args.compile_jobs)


if __name__ == "__main__":
    main()
