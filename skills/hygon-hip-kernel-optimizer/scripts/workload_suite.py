"""Explicit workload regression gates. Shapes are never silently resized."""
from __future__ import annotations
import json
import hashlib
import math
from pathlib import Path
import re
import sys
from experiment import benchmark_gate, positive, run_json, file_sha256, resolve_benchmark


def load_cases(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    if not cases or not isinstance(cases, list):
        raise ValueError("workloads must contain a nonempty cases list")
    seen = set()
    for c in cases:
        name = c.get("id", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in seen:
            raise ValueError("case IDs must be unique safe path components")
        seen.add(name)
        if not isinstance(c.get("dims"), dict) or any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in c["dims"].values()):
            raise ValueError("case dims must be explicit nonnegative integers")
        if any(not re.fullmatch(r"[A-Za-z_]\w*", k) for k in c["dims"]):
            raise ValueError("invalid dimension name")
        limit = c.get("max_regression_pct", 5)
        if not positive(c.get("weight", 1)) or isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit < 0:
            raise ValueError("weights must be positive; regression limits must be finite and nonnegative")
        seeds = c.get("seeds", [42, 123, 2026])
        if not isinstance(seeds, list) or not seeds or not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds):
            raise ValueError("seeds must be a nonempty integer list")
        size = c.get("ptr_size", 0)
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("ptr_size must be a nonnegative integer")
    return cases


def suite_identity(state, source, benchmark):
    """Bind a summary to the measured code and frozen workload definition."""
    config = {"cases": state.get("workloads", []), "ptr_size": state.get("ptr_size", 0)}
    return {"source_sha256": file_sha256(source),
            "baseline_sha256": file_sha256(state["baseline_file"]),
            "reference_sha256": file_sha256(state["ref_file"]),
            "benchmark_sha256": file_sha256(benchmark),
            "workloads_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()}


def evaluate(state, source, benchmark, out_dir, warmup, repeat):
    """Remeasure baseline and candidate serially for comparable case/seed pairs."""
    cases = state.get("workloads", [])
    benchmark = resolve_benchmark(state, benchmark)
    identity = suite_identity(state, source, benchmark)
    rows = []
    folder = Path(out_dir)
    for case in cases:
        for seed in case.get("seeds", [42, 123, 2026]):
            results = {}
            for role, path in (("baseline", state["baseline_file"]), ("candidate", source)):
                target = folder / case["id"] / str(seed) / (role + ".json")
                cmd = [sys.executable, benchmark, path, "--ref", state["ref_file"],
                       "--seed", str(seed), "--warmup", str(warmup), "--repeat", str(repeat), "--json-out", str(target),
                       "--ptr-size", str(case.get("ptr_size", state.get("ptr_size", 0)))]
                cmd += [f"--{k}={v}" for k, v in case["dims"].items()]
                bench = run_json(cmd, target, target.with_suffix(".log"))
                valid, reason = benchmark_gate(bench, path, state["ref_file"])
                results[role] = {"passed": valid, "reason": reason,
                                 "ms": (bench.get("kernel") or {}).get("average_ms"), "artifact": str(target)}
            passed = all(r["passed"] for r in results.values())
            ratio = results["baseline"]["ms"] / results["candidate"]["ms"] if passed else None
            passed = passed and ratio >= 1 / (1 + case.get("max_regression_pct", 5) / 100)
            rows.append({"case": case["id"], "seed": seed, "dims": case["dims"], "passed": passed,
                         "speedup": ratio, "weight": case.get("weight", 1) / len(case.get("seeds", [42, 123, 2026])),
                         "measurements": results})
    passed = bool(rows) and all(r["passed"] for r in rows)
    score = math.exp(sum(r["weight"] * math.log(r["speedup"]) for r in rows) / sum(r["weight"] for r in rows)) if passed else None
    result = {"passed": passed, "weighted_speedup": score, "cases": rows,
              "identity": identity,
              "scope": "explicit workloads only; failure/OOM does not downsize the requested shape"}
    if identity != suite_identity(state, source, benchmark):
        result.update(passed=False, weighted_speedup=None, error="inputs changed during workload measurement")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "suite.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
