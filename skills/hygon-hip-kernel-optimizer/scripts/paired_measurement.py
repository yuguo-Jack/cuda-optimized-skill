"""Fresh, counterbalanced comparisons; conservative repeatability, not a CI.

Branch exploration is a selection experiment. Promotion is a separate four-round
AB/BA comparison against the current best, using the frozen workload contract.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from experiment import benchmark_gate, file_sha256, resolve_benchmark, run_json


def cases_for(state):
    if state.get("workloads"):
        return [{**case, "max_regression_pct": case.get("max_regression_pct", 5)} for case in state["workloads"]]
    return [{"id": "primary", "dims": state.get("dims", {}),
             "seeds": [42], "ptr_size": state.get("ptr_size", 0)}]


def identity(state, candidate, control, benchmark, protocol):
    return {"candidate_sha256": file_sha256(candidate), "control_sha256": file_sha256(control),
            "reference_sha256": file_sha256(state["ref_file"]), "benchmark_sha256": file_sha256(benchmark),
            "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()}


def _context(bench):
    # Custom harnesses must report the same fields to enable input/device checks.
    context = {key: bench.get(key) for key in ("dims", "seed", "warmup", "repeat", "ptr_size_override",
            "gpu_index", "gpu_name", "arch", "inputs_sha256", "signature", "numerical_policy")}
    context["tolerances"] = {key: (bench.get("correctness") or {}).get(key) for key in ("atol", "rtol", "equal_nan")}
    context["timing_scope"] = (bench.get("kernel") or {}).get("timing_scope")
    return context


def assess(report, state, candidate, control, benchmark, expected_cases=None):
    """Re-open hashed raw measurements. Never trust an edited summary verdict."""
    try:
        protocol = report["protocol"]
        rounds = protocol["rounds"]
        if rounds != 4 or protocol["cases"] != (expected_cases if expected_cases is not None else cases_for(state)):
            raise ValueError("comparison protocol differs from the frozen cases or four-round policy")
        if protocol["warmup"] < 0 or protocol["repeat"] < 5:
            raise ValueError("invalid warmup/repeat")
        if report["identity"] != identity(state, candidate, control, benchmark, protocol):
            raise ValueError("comparison source/reference/benchmark/protocol changed")
        expected = [(r, c, s) for r in range(rounds) for c in protocol["cases"] for s in c.get("seeds", [42, 123, 2026])]
        if len(report["pairs"]) != len(expected):
            raise ValueError("incomplete comparison")
        by_round = [[] for _ in range(rounds)]
        guards = True
        differences, control_times, candidate_times = [], [], []
        coverage = True
        for row, (r, case, seed) in zip(report["pairs"], expected):
            order = ["control", "candidate"] if r % 2 == 0 else ["candidate", "control"]
            if (row["round"], row["case"], row["seed"], row["order"]) != (r, case["id"], seed, order):
                raise ValueError("case/seed/order mismatch")
            values = {}
            for role, source in (("control", control), ("candidate", candidate)):
                item = row[role]
                if file_sha256(item["artifact"]) != item["sha256"]:
                    raise ValueError("raw measurement changed")
                bench = json.loads(Path(item["artifact"]).read_text(encoding="utf-8"))
                valid, reason = benchmark_gate(bench, source, state["ref_file"])
                if not valid:
                    raise ValueError(f"{role}: {reason}")
                values[role] = bench
            if _context(values["control"]) != _context(values["candidate"]):
                raise ValueError("input/device/measurement context differs")
            coverage = coverage and all(v.get("inputs_sha256") and v.get("arch") and v.get("gpu_name")
                                       and v.get("gpu_index") is not None and v.get("signature")
                                       and all(k in v for k in ("dims", "seed", "warmup", "repeat", "ptr_size_override"))
                                       for v in values.values())
            for v in values.values():
                for k, wanted in (("dims", case["dims"]), ("seed", seed),
                                  ("warmup", protocol["warmup"]), ("repeat", protocol["repeat"]),
                                  ("ptr_size_override", case.get("ptr_size", state.get("ptr_size", 0)))):
                    if k in v and v[k] != wanted:
                        raise ValueError(f"benchmark ignored requested {k}")
            a, b = [values[k]["kernel"]["average_ms"] for k in ("control", "candidate")]
            ratio = a / b
            limit = case.get("max_regression_pct")
            guards = guards and (limit is None or ratio >= 1 / (1 + limit / 100))
            weight = case.get("weight", 1) / len(case.get("seeds", [42, 123, 2026]))
            by_round[r].append((ratio, weight))
            differences.append(a - b)
            control_times.append(a)
            candidate_times.append(b)
        scores = [math.exp(sum(w * math.log(v) for v, w in rows) / sum(w for _, w in rows)) for rows in by_round]
        if not coverage:
            raise ValueError("comparison requires input fingerprint, device, signature and measurement protocol metadata")
        noise = state.get("noise_threshold_pct", 2.0) / 100
        if not 0 <= noise < 1:
            raise ValueError("noise threshold must be in [0, 100)")
        beneficial = guards and all(score > 1 / (1 - noise) for score in scores)
        harmful = all(score < 1 - noise for score in scores)
        return {"valid": True, "improved": beneficial, "direction": "beneficial" if beneficial else "harmful" if harmful else "inconclusive",
                "round_speedups": scores, "case_regression_guards_passed": guards,
                "control_ms": statistics.median(control_times), "candidate_ms": statistics.median(candidate_times),
                "difference_ms": statistics.median(differences), "input_device_metadata_complete": bool(coverage),
                "scope": "four alternating paired rounds; repeatability gate, not statistical confidence or cross-device proof"}
    except (KeyError, TypeError, ValueError, OSError, ZeroDivisionError) as exc:
        return {"valid": False, "improved": False, "direction": "unverified", "reason": str(exc)}


def compare(state, candidate, control, benchmark, folder, warmup=10, repeat=20, cases=None):
    if warmup < 0 or repeat < 5:
        raise ValueError("paired comparison requires nonnegative warmup and at least five repeats")
    benchmark = resolve_benchmark(state, benchmark)
    protocol = {"rounds": 4, "warmup": warmup, "repeat": repeat,
                "cases": cases if cases is not None else cases_for(state)}
    report = {"protocol": protocol, "identity": identity(state, candidate, control, benchmark, protocol), "pairs": []}
    folder = Path(folder).resolve()
    for r in range(4):
        for case in protocol["cases"]:
            for seed in case.get("seeds", [42, 123, 2026]):
                order = ["control", "candidate"] if r % 2 == 0 else ["candidate", "control"]
                row = {"round": r, "case": case["id"], "seed": seed, "order": order}
                for role in order:
                    source = control if role == "control" else candidate
                    target = folder / str(r) / case["id"] / str(seed) / f"{role}.json"
                    cmd = [sys.executable, benchmark, str(source), "--ref", state["ref_file"], "--seed", str(seed),
                           "--warmup", str(warmup), "--repeat", str(repeat), "--json-out", str(target),
                           "--ptr-size", str(case.get("ptr_size", state.get("ptr_size", 0)))]
                    cmd += [f"--{k}={v}" for k, v in case["dims"].items()]
                    run_json(cmd, target, target.with_suffix(".log"))
                    row[role] = {"artifact": str(target), "sha256": file_sha256(target)}
                report["pairs"].append(row)
    report["assessment"] = assess(report, state, candidate, control, benchmark, protocol["cases"])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
