"""Shared, CPU-testable evidence gates for HCU experiments."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_benchmark(state, requested=None):
    path = Path(requested or state.get("benchmark_file") or Path(__file__).with_name("benchmark.py")).resolve()
    expected = state.get("benchmark_sha256")
    if expected and expected != file_sha256(path):
        raise SystemExit("Benchmark changed since setup; start a new run and remeasure baseline")
    return str(path)


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def timing_stats(times):
    if len(times) < 1 or not all(positive(t) for t in times):
        raise ValueError("timing samples must be finite and positive")
    ordered = sorted(times)
    median = statistics.median(times)
    cv = statistics.pstdev(times) / statistics.mean(times)
    mad_cv = 1.4826 * statistics.median(abs(t - median) for t in times) / median
    return {"average_ms": statistics.mean(times), "median_ms": median,
            "min_ms": min(times), "max_ms": max(times), "p90_ms": ordered[math.ceil(.9 * len(times)) - 1],
            "samples_ms": times, "sample_count": len(times), "cv": cv, "robust_cv": mad_cv,
            "stability": "stable" if len(times) >= 5 and cv <= .10 else "unstable",
            "timing_scope": "device_events_one_invocation; reset outside interval; warm cache"}


def benchmark_gate(bench, source=None):
    if bench.get("error") or bench.get("process_returncode", 0) != 0:
        return False, "benchmark execution failed"
    correctness = bench.get("correctness") or {}
    if correctness.get("checked") is not True or correctness.get("passed") is not True:
        return False, "correctness not checked and passed"
    kernel = bench.get("kernel") or {}
    if not positive(kernel.get("average_ms")):
        return False, "invalid kernel timing"
    samples = kernel.get("samples_ms")
    if not isinstance(samples, list) or len(samples) < 5 or not all(positive(t) for t in samples):
        return False, "at least five real timing samples required"
    if timing_stats(samples)["stability"] != "stable":
        return False, "unstable timing; rerun or revise measurement protocol"
    if not math.isclose(statistics.mean(samples), kernel["average_ms"], rel_tol=1e-6, abs_tol=1e-9):
        return False, "average_ms disagrees with raw samples"
    if source and bench.get("source_sha256") != file_sha256(source):
        return False, "benchmark source hash differs or is missing"
    return True, None


def run_json(cmd, output, log=None, timeout=600):
    """Never reuse an older success after a crash/nonzero subprocess exit."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        rc, stdout, stderr = proc.returncode, proc.stdout or "", proc.stderr or ""
    except (OSError, subprocess.TimeoutExpired) as exc:
        rc, stdout, stderr = -1, "", str(exc)
    if log:
        Path(log).write_text(stdout + "\n---STDERR---\n" + stderr, encoding="utf-8")
    try:
        result = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("expected result object")
    except (OSError, ValueError):
        result = {"error": "missing_or_invalid_benchmark_json"}
    result["process_returncode"] = rc
    if rc:
        result["error"] = result.get("error") or f"benchmark exited {rc}"
        result["correctness"] = {"checked": False, "passed": False}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
