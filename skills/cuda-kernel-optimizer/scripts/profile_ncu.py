#!/usr/bin/env python3
"""Profile a kernel with Nsight Compute and extract top metrics per axis.

Usage:
  profile_ncu.py --state state.json --iter 1 --which best_input \
                 --benchmark ./benchmark.py

Writes (under {run_dir}/iterv{i}/):
  {which}.ncu-rep   — full ncu report (binary)
  ncu_top.json      — top-ncu_num metrics per axis
                      (compute / memory / latency) with rank + raw value

`--which` values:
  best_input   profile state.best_file (snapshot going INTO the iter)
  kernel       profile iterv{i}/kernel.<ext> (the iteration's product)
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Axis rubric — which metric ID patterns belong to which axis, and whether
# higher values are "worse" (i.e. indicate a bottleneck on that axis).
# Order inside each list matters: earlier patterns win when classifying.
# ---------------------------------------------------------------------------

# (regex, axis, higher_is_worse)
# "higher_is_worse" is used for ranking: we sort by "severity" on each axis.
METRIC_RUBRIC: list[tuple[str, str, bool]] = [
    # ------------- COMPUTE / utilization --------------
    (r"sm__pipe_tensor.*\.pct_of_peak", "compute", False),     # tensor core busy — low is a miss
    (r"sm__pipe_fp32_cycles_active.*\.pct_of_peak", "compute", False),
    (r"sm__pipe_fp64_cycles_active.*\.pct_of_peak", "compute", False),
    (r"sm__inst_executed\..*per_cycle_active", "compute", False),  # IPC
    (r"smsp__sass_thread_inst_executed_op_(f|i|h)(add|mul|fma).*sum", "compute", False),
    (r"sm__warps_active.*pct_of_peak_sustained_active", "compute", False),  # occupancy
    (r"sm__cycles_active.*pct_of_peak", "compute", False),
    (r"launch__occupancy_limit", "compute", False),
    (r"sm__throughput.*pct_of_peak", "compute", False),

    # ------------- MEMORY (dram / l2 / l1 / shared) --------------
    (r"dram__throughput.*pct_of_peak", "memory", True),
    (r"dram__bytes.*sum", "memory", True),
    (r"lts__t_sector_hit_rate\.pct", "memory", False),   # low hit rate = bad
    (r"l1tex__t_sector_hit_rate\.pct", "memory", False),
    (r"l1tex__data_bank_conflicts.*sum", "memory", True),
    (r"smsp__inst_executed_op_shared_(ld|st).*sum", "memory", True),
    (r"gpu__compute_memory_throughput.*pct_of_peak", "memory", True),
    (r"l1tex__throughput.*pct_of_peak", "memory", True),
    (r"lts__throughput.*pct_of_peak", "memory", True),

    # ------------- LATENCY / stalls --------------
    (r"smsp__average_warp_latency.*pct", "latency", True),
    (r"smsp__warp_issue_stalled.*per_inst_issued", "latency", True),
    (r"smsp__pcsamp_warps_issue_stalled_.*", "latency", True),
    (r"smsp__warp_issue_stalled_.*long_scoreboard.*", "latency", True),
    (r"smsp__warp_issue_stalled_.*short_scoreboard.*", "latency", True),
    (r"smsp__warp_issue_stalled_.*barrier.*", "latency", True),
    (r"smsp__warp_issue_stalled_.*mio_throttle.*", "latency", True),
    (r"smsp__warp_issue_stalled_.*lg_throttle.*", "latency", True),
    (r"smsp__warp_issue_stalled_.*wait.*", "latency", True),
    (r"gpc__cycles_elapsed\.max", "latency", True),
    (r"sm__cycles_elapsed\..*(avg|max)", "latency", True),
]


# Metrics we explicitly ask ncu to capture. Extending --set full already
# gives us most of these, but listing them explicitly keeps the import
# step fast even when ncu collected a huge superset.
EXPLICIT_METRICS = [
    # compute
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_fp32_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active",
    "sm__inst_executed.avg.per_cycle_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    # memory
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes.sum",
    "lts__t_sector_hit_rate.pct",
    "l1tex__t_sector_hit_rate.pct",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "smsp__inst_executed_op_shared_ld.sum",
    "smsp__inst_executed_op_shared_st.sum",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    # latency / stalls
    "smsp__average_warp_latency_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__average_warp_latency_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__average_warp_latency_issue_stalled_barrier_per_warp_active.pct",
    "smsp__average_warp_latency_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__average_warp_latency_issue_stalled_lg_throttle_per_warp_active.pct",
    "smsp__average_warp_latency_issue_stalled_wait_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "gpc__cycles_elapsed.max",
    "sm__cycles_elapsed.avg",
]


_BUNDLED_BENCHMARK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.py")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_state(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _detect_backend(file: str) -> str:
    ext = os.path.splitext(file)[1].lower()
    if ext == ".py":
        return "triton"
    return "cuda"  # cuda / cutlass indistinguishable here — both use -k solve


def _dims_argv(dims: dict) -> list[str]:
    return [f"--{k}={v}" for k, v in dims.items()]


def _ptr_size_argv(ptr_size: int) -> list[str]:
    return ["--ptr-size", str(ptr_size)] if ptr_size and ptr_size > 0 else []


def _classify(metric_name: str) -> tuple[str | None, bool]:
    for pat, axis, higher_is_worse in METRIC_RUBRIC:
        if re.search(pat, metric_name):
            return axis, higher_is_worse
    return None, False


def _kernel_duration(row: dict) -> float:
    for key in ("gpu__time_duration.sum", "gpu__time_duration.avg"):
        value = _to_float(row.get(key, ""))
        if value is not None:
            return value
    return float("-inf")


def _select_target_kernel_rows(rows: list[dict], kernel_name_hints: list[str] | None = None) -> list[dict]:
    if not rows or "Kernel Name" not in rows[0]:
        return rows

    by_kernel: dict[str, list[dict]] = {}
    for row in rows:
        kernel_name = row.get("Kernel Name", "").strip()
        if not kernel_name:
            continue
        by_kernel.setdefault(kernel_name, []).append(row)

    if not by_kernel:
        return rows

    # Prefer kernels originating from the target solution file when possible.
    # This avoids picking unrelated PyTorch init kernels (e.g. RNG) that may run
    # before the operator under test.
    if kernel_name_hints:
        hinted = []
        for kname, krows in by_kernel.items():
            if any(hint in kname for hint in kernel_name_hints):
                hinted.append((kname, krows))
        if hinted:
            target_kernel = max(
                hinted,
                key=lambda item: max(_kernel_duration(row) for row in item[1]),
            )[0]
            return by_kernel[target_kernel]

    target_kernel = max(
        by_kernel.items(),
        key=lambda item: max(_kernel_duration(row) for row in item[1]),
    )[0]
    return by_kernel[target_kernel]


def _parse_ncu_csv(csv_text: str, kernel_name_hints: list[str] | None = None) -> list[dict]:
    """Parse both old long-form and newer wide-form ncu CSV exports.

    Returns a normalized long-form list with Metric Name / Value / Unit / Kernel Name.
    """
    rows: list[dict] = []
    reader = csv.reader(io.StringIO(csv_text))
    header = None
    buffered_rows: list[dict] = []

    for row in reader:
        if not row:
            continue
        cols = [c.strip() for c in row]
        if header is None:
            if "Metric Name" in cols and "Metric Value" in cols:
                header = cols
                continue
            if "Kernel Name" in cols:
                header = cols
                continue
            continue

        if len(cols) != len(header):
            continue
        entry = dict(zip(header, cols))

        if "Metric Name" in entry and "Metric Value" in entry:
            rows.append(entry)
            continue

        buffered_rows.append(entry)

    if rows:
        return rows

    if not buffered_rows or header is None:
        return []

    # When using the bundled benchmark, all kernels spawned by the solution
    # share a common substring from the .cu file name (e.g. "batchnorm"), while
    # unrelated framework kernels typically do not.
    hints = []
    if kernel_name_hints:
        hints = list(kernel_name_hints)
    target_rows = _select_target_kernel_rows(buffered_rows, kernel_name_hints=hints)
    normalized: list[dict] = []
    for entry in target_rows:
        kernel_name = entry.get("Kernel Name", "")
        for metric_name, metric_value in entry.items():
            axis, _ = _classify(metric_name)
            if axis is None:
                continue
            normalized.append(
                {
                    "Kernel Name": kernel_name,
                    "Metric Name": metric_name,
                    "Metric Value": metric_value,
                    "Metric Unit": "",
                }
            )
    return normalized


def _to_float(s: str) -> float | None:
    if s is None:
        return None
    s = s.strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _aggregate_across_kernels(rows: list[dict]) -> dict[str, dict]:
    """Collapse multiple kernel launches into one average per metric."""
    agg: dict[str, dict] = {}
    for r in rows:
        name = r.get("Metric Name")
        if not name:
            continue
        v = _to_float(r.get("Metric Value", ""))
        unit = r.get("Metric Unit", "")
        if v is None:
            continue
        a = agg.setdefault(name, {"sum": 0.0, "n": 0, "unit": unit, "kernels": set()})
        a["sum"] += v
        a["n"] += 1
        if r.get("Kernel Name"):
            a["kernels"].add(r["Kernel Name"])
    out = {}
    for name, a in agg.items():
        out[name] = {
            "value": a["sum"] / a["n"] if a["n"] else None,
            "unit": a["unit"],
            "samples": a["n"],
            "kernels": sorted(a["kernels"]),
        }
    return out


def _run_ncu_profile(
    *,
    ncu_bin: str,
    rep_path: str,
    benchmark_py: str,
    solution: str,
    ref_file: str,
    dims: dict,
    backend: str,
    ptr_size: int,
    warmup: int,
    repeat: int,
    launch_count: int,
    no_kernel_filter: bool,
) -> tuple[int, str]:
    cmd = [
        ncu_bin,
        "--set", "full",
        "-o", rep_path,
        "-f",
        "--target-processes", "all",
        "--launch-count", str(launch_count),
    ]
    # Do not filter by a fixed kernel name here. CUDA wrappers often launch a
    # device kernel whose symbol is unrelated to the exported host `solve`
    # entrypoint, and library-backed implementations (for example cuBLAS) use
    # internal kernel names entirely. We instead profile the launches and let
    # the CSV import path select the dominant kernel by duration.

    cmd += [
        sys.executable, benchmark_py, solution,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
    ] + _ptr_size_argv(ptr_size) + _dims_argv(dims)

    print(f"[ncu profile] {' '.join(cmd)}", file=sys.stderr)
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore",
        )
    except OSError as e:
        return -1, f"failed to invoke ncu: {e}"
    log = (r.stdout or "") + "\n---STDERR---\n" + (r.stderr or "")
    return r.returncode, log


def _import_metrics_csv(ncu_bin: str, rep_path: str) -> tuple[int, str, str]:
    """Export the ncu-rep to CSV for our top-K analysis.

    We ask for both explicit metrics AND let ncu include everything it
    collected (we'll just filter on our side). Using the `raw` page makes
    the CSV tractable.
    """
    cmd = [
        ncu_bin, "--import", rep_path,
        "--csv", "--page", "raw",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    except OSError as e:
        return -1, "", f"failed to invoke ncu --import: {e}"
    return r.returncode, r.stdout or "", r.stderr or ""


def _rank_by_axis(agg: dict[str, dict], ncu_num: int) -> dict[str, list]:
    """Pick up to ncu_num metrics per axis.

    We prefer a small, stable set of high-signal metrics (utilization, IPC,
    occupancy, stall %), then fill remaining slots by severity.
    """

    def _severity(metric_name: str, value: float) -> float:
        axis, higher_is_worse = _classify(metric_name)
        if axis is None:
            return float("-inf")
        return value if higher_is_worse else (100.0 - value)

    def _add_worst_matching(out: list, seen: set[str], patterns: list[str], *, exclude_fp64: bool = False):
        for pat in patterns:
            matches = []
            for name, info in agg.items():
                if name in seen:
                    continue
                if exclude_fp64 and "fp64" in name:
                    continue
                if re.search(pat, name):
                    v = info.get("value")
                    if v is None:
                        continue
                    matches.append((name, float(v)))
            if not matches:
                continue
            # choose the worst instance among matches
            name, v = max(matches, key=lambda nv: _severity(nv[0], nv[1]))
            axis, higher_is_worse = _classify(name)
            out.append({
                "name": name,
                "value": v,
                "unit": agg[name].get("unit"),
                "higher_is_worse": higher_is_worse,
                "samples": agg[name].get("samples"),
            })
            seen.add(name)
            if len(out) >= ncu_num:
                return

    preferred = {
        "compute": [
            r"sm__pipe_tensor_op_.*pct_of_peak",
            r"sm__pipe_fp32_cycles_active.*pct_of_peak",
            r"sm__inst_executed\..*per_cycle_active",
            r"sm__warps_active.*pct_of_peak",
            r"sm__cycles_active.*pct_of_peak",
        ],
        "memory": [
            r"dram__throughput.*pct_of_peak",
            r"gpu__compute_memory_throughput.*pct_of_peak",
            r"lts__t_sector_hit_rate\.pct",
            r"l1tex__t_sector_hit_rate\.pct",
            r"l1tex__data_bank_conflicts.*sum",
        ],
        "latency": [
            r"smsp__warp_issue_stalled_long_scoreboard.*pct",
            r"smsp__warp_issue_stalled_barrier.*pct",
            r"smsp__warp_issue_stalled_mio_throttle.*pct",
            r"smsp__warp_issue_stalled_wait.*pct",
            r"smsp__average_warp_latency.*pct",
        ],
    }

    by_axis: dict[str, list] = {"compute": [], "memory": [], "latency": []}
    seen: set[str] = set()

    # 1) preferred picks
    _add_worst_matching(by_axis["compute"], seen, preferred["compute"], exclude_fp64=True)
    _add_worst_matching(by_axis["memory"], seen, preferred["memory"])
    _add_worst_matching(by_axis["latency"], seen, preferred["latency"])

    # 2) fill by severity within axis
    for axis in ("compute", "memory", "latency"):
        if len(by_axis[axis]) >= ncu_num:
            continue
        candidates = []
        for name, info in agg.items():
            ax, higher_is_worse = _classify(name)
            if ax != axis:
                continue
            if name in seen:
                continue
            if axis == "compute" and "fp64" in name:
                continue
            v = info.get("value")
            if v is None:
                continue
            v = float(v)
            sev = v if higher_is_worse else (100.0 - v)
            candidates.append((sev, name, v, higher_is_worse, info.get("unit"), info.get("samples")))

        candidates.sort(reverse=True)
        for sev, name, v, higher_is_worse, unit, samples in candidates:
            by_axis[axis].append({
                "name": name,
                "value": v,
                "unit": unit,
                "higher_is_worse": higher_is_worse,
                "samples": samples,
            })
            seen.add(name)
            if len(by_axis[axis]) >= ncu_num:
                break

    return by_axis


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--iter", required=True, type=int)
    p.add_argument("--which", required=True, choices=["best_input", "kernel"])
    p.add_argument("--benchmark", default=_BUNDLED_BENCHMARK,
                   help="Path to benchmark.py (default: bundled)")
    p.add_argument("--launch-count", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--no-kernel-filter", action="store_true")
    p.add_argument("--ncu-bin", type=str, default="")
    p.add_argument("--promote-if-best", action="store_true",
                   help="When --which=kernel, if this kernel is the current best, "
                        "also update state.best_ncu_rep to this report.")
    args = p.parse_args()

    state = _read_state(args.state)
    run_dir = state["run_dir"]
    iter_dir = os.path.join(run_dir, f"iterv{args.iter}")
    os.makedirs(iter_dir, exist_ok=True)

    # Pick the solution file to profile
    if args.which == "best_input":
        solution = state["best_file"]
        rep_name = "best_input.ncu-rep"
    else:
        # iterv{i}/kernel.*  — find whichever extension is present
        candidates = [
            os.path.join(iter_dir, "kernel.cu"),
            os.path.join(iter_dir, "kernel.py"),
        ]
        solution = next((c for c in candidates if os.path.isfile(c)), None)
        if not solution:
            sys.exit(f"No iterv{args.iter}/kernel.(cu|py) found — generate it first.")
        rep_name = "kernel.ncu-rep"

    rep_path = os.path.join(iter_dir, rep_name)

    env_ncu = state.get("env", {}).get("ncu")
    env_ncu_path = env_ncu.get("path", "") if isinstance(env_ncu, dict) else ""
    ncu_bin = args.ncu_bin or env_ncu_path or shutil.which("ncu") or "ncu"

    backend = _detect_backend(solution)

    # If state says ncu is unavailable, emit a degraded top-K straight away.
    ncu_info = state.get("env", {}).get("ncu", {}) or {}
    ncu_available = (
        ncu_info.get("available", True)
        if isinstance(ncu_info, dict)
        else bool(ncu_info) if isinstance(ncu_info, bool)
        else True
    )
    if not ncu_available or shutil.which(ncu_bin) is None:
        top = {
            "degraded": True,
            "reason": "ncu not available on this system",
            "profiled_file": solution,
            "backend": backend,
            "compute": [], "memory": [], "latency": [],
        }
        _write_json(os.path.join(iter_dir, "ncu_top.json"), top)
        print(json.dumps({"degraded": True, "reason": top["reason"]}, indent=2))
        return

    # 1) collect
    rc, log = _run_ncu_profile(
        ncu_bin=ncu_bin,
        rep_path=rep_path,
        benchmark_py=os.path.abspath(args.benchmark),
        solution=solution,
        ref_file=state["ref_file"],
        dims=state.get("dims", {}),
        backend=backend,
        ptr_size=state.get("ptr_size", 0),
        warmup=args.warmup,
        repeat=args.repeat,
        launch_count=args.launch_count,
        no_kernel_filter=args.no_kernel_filter,
    )
    log_path = os.path.join(iter_dir, f"{os.path.splitext(rep_name)[0]}.ncu.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log)
    if rc != 0 or not os.path.isfile(rep_path):
        top = {
            "degraded": True,
            "reason": f"ncu exited {rc}; see {log_path}",
            "profiled_file": solution,
            "backend": backend,
            "compute": [], "memory": [], "latency": [],
        }
        _write_json(os.path.join(iter_dir, "ncu_top.json"), top)
        print(json.dumps({"degraded": True, "rc": rc, "log": log_path}, indent=2))
        return

    # 2) import → csv
    rc2, csv_text, err2 = _import_metrics_csv(ncu_bin, rep_path)
    if rc2 != 0 or not csv_text:
        top = {
            "degraded": True,
            "reason": f"ncu --import failed rc={rc2}: {err2[:400]}",
            "profiled_file": solution,
            "backend": backend,
            "ncu_rep": rep_path,
            "compute": [], "memory": [], "latency": [],
        }
        _write_json(os.path.join(iter_dir, "ncu_top.json"), top)
        print(json.dumps({"degraded": True, "rc": rc2}, indent=2))
        return

    # 3) parse → aggregate → rank
    kernel_hints = []
    if backend in {"cuda", "cutlass"}:
        try:
            text = Path(solution).read_text(encoding="utf-8", errors="ignore")
            kernel_hints = re.findall(r"__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)", text)
        except OSError:
            kernel_hints = []
    rows = _parse_ncu_csv(csv_text, kernel_name_hints=kernel_hints)
    agg = _aggregate_across_kernels(rows)
    by_axis = _rank_by_axis(agg, state.get("ncu_num", 5))

    top = {
        "degraded": False,
        "profiled_file": solution,
        "backend": backend,
        "ncu_rep": rep_path,
        "metric_count_collected": len(agg),
        **by_axis,
    }
    _write_json(os.path.join(iter_dir, "ncu_top.json"), top)

    # Optionally promote this as best_ncu_rep
    if args.which == "kernel" and args.promote_if_best:
        # The caller (run_iteration) is responsible for deciding if this
        # kernel is "best" — but if it already pointed state.best_file to
        # our path, we record the rep here.
        if os.path.abspath(solution) == os.path.abspath(state.get("best_file", "")):
            state["best_ncu_rep"] = rep_path
            with open(args.state, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "ncu_rep": rep_path,
        "ncu_top": os.path.join(iter_dir, "ncu_top.json"),
        "compute_count": len(by_axis["compute"]),
        "memory_count": len(by_axis["memory"]),
        "latency_count": len(by_axis["latency"]),
    }, indent=2))


if __name__ == "__main__":
    main()
