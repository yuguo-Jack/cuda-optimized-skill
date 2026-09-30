#!/usr/bin/env python3
"""Profile a HIP kernel with DTK hipprof and extract top DCU metrics.

Writes under {run_dir}/iterv{i}/:
  {which}.hipprof/   hipprof output directory
  dcu_top.json       top metrics per compute / memory / latency axis
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from analyze_sqtt import analyze as analyze_sqtt_json
from profile_records import save_profile


_BUNDLED_BENCHMARK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.py")
KERNEL_EXTS = (".hip", ".cu", ".cpp", ".cc", ".cxx", ".py")

METRIC_RUBRIC: list[tuple[str, str, bool]] = [
    (r"SQ_INSTS_MMOP|MMAC|MATRIX|VALU_FMA|VALU_ADD|VALU_MUL|SQ_BUSY|GRBM_GUI_ACTIVE", "compute", True),
    (r"TCC|TCP|TA_|TD_|READ_REQ|WRITE_REQ|CACHE|L2|L1|BW|BANDWIDTH", "memory", True),
    (r"STALL|LATENCY|WAIT|OCCUPANCY|WAVE[ _]?CYCLES|ACTIVE[ _]?CYCLES|BARRIER|ATOMIC|SQ_WAVES", "latency", True),
    (r"LDS|BANK_CONFLICT|DS_READ|DS_WRITE", "memory", True),
]


def _read(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _dims_argv(dims: dict) -> list[str]:
    return [f"--{k}={v}" for k, v in dims.items()]


def _ptr_size_argv(ptr_size: int) -> list[str]:
    return ["--ptr-size", str(ptr_size)] if ptr_size and ptr_size > 0 else []


def _detect_backend(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".py":
        return "python"
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        return "ck_tile" if ("ck_tile/" in text or "ck_tile::" in text) else "hip"
    except OSError:
        return "hip"


def _classify(name: str) -> tuple[str | None, bool]:
    for pat, axis, higher_is_worse in METRIC_RUBRIC:
        if re.search(pat, name, re.IGNORECASE):
            return axis, higher_is_worse
    return None, True


def _to_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        value = float(text)
        return value if math.isfinite(value) else None
    except ValueError:
        return None


def _measured_binary(reports: list[str], solution: str) -> str | None:
    """Use the captured benchmark's build receipt, never a neighboring stale .so."""
    from experiment import file_sha256
    for path in reversed(reports):
        try:
            bench = _read(path)
            if not isinstance(bench, dict):
                continue
            build = bench.get("build") or {}
            if not isinstance(build, dict):
                continue
            binary = build.get("binary")
            source_sha = file_sha256(solution)
            if (bench.get("source_sha256") == source_sha == build.get("source_sha256")
                    and binary and Path(binary).is_file()
                    and build.get("binary_sha256") == file_sha256(binary)):
                return binary
        except (OSError, ValueError, TypeError):
            continue
    return None


def _run_hipprof(
    *,
    hipprof_bin: str,
    out_prefix: str,
    benchmark_py: str,
    solution: str,
    dims: dict,
    ptr_size: int,
    warmup: int,
    repeat: int,
    kernel_name: str = "",
    collect_flag: str = "--pmc",
    pmc_type: str = "3",
    pmc_group: str = "",
    benchmark_json: str = "",
    sqtt_type: str = "",
    output_type: str = "",
    data_dir: str = "",
    env: dict[str, str] | None = None,
) -> tuple[int, str]:
    cmd = [
        hipprof_bin,
        "-o", out_prefix,
    ]
    if data_dir:
        if not data_dir.endswith(("/", "\\")):
            data_dir = data_dir + os.sep
        cmd.extend(["-d", data_dir])
    if output_type:
        cmd.extend(["--output-type", output_type])
    cmd.append(collect_flag)
    if pmc_group:
        if collect_flag != "--pmc" or pmc_group not in PMC_GROUPS:
            raise ValueError("A named PMC group requires --pmc and a supported group")
        cmd.append(pmc_group)
    if collect_flag.startswith("--pmc"):
        cmd.extend(["--pmc-type", pmc_type])
    if collect_flag == "--sqtt" and sqtt_type:
        cmd.extend(["--sqtt-type", sqtt_type])
    if kernel_name:
        cmd.extend(["--kernel-name", kernel_name])
    cmd.extend([
        sys.executable, benchmark_py, solution,
        "--warmup", str(warmup),
        "--repeat", str(repeat),
    ])
    cmd.extend(_ptr_size_argv(ptr_size))
    cmd.extend(_dims_argv(dims))
    if benchmark_json:
        cmd.extend(["--json-out", benchmark_json])
    print(f"[hipprof] {' '.join(cmd)}", file=sys.stderr)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", env=env, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, str(exc)
    return r.returncode, (r.stdout or "") + "\n---STDERR---\n" + (r.stderr or "")


def _run_codeobj_analyze(*, hipprof_bin: str, binary: str, out_log: str) -> dict:
    cmd = [hipprof_bin, "--codeobj-analyze", binary]
    print(f"[codeobj] {' '.join(cmd)}", file=sys.stderr)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=120)
        log = (r.stdout or "") + "\n---STDERR---\n" + (r.stderr or "")
        Path(out_log).write_text(log, encoding="utf-8")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "binary": binary, "error": str(exc), "log": out_log}
    return _parse_codeobj_log(log, r.returncode, binary, out_log)


def _sqtt_env_with_llvm_objdump() -> tuple[dict[str, str], str | None]:
    """Return an env where hipprof SQTT can find llvm-objdump if DTK ships it.

    This is only for hipprof's internal SQTT trace export. The optimizer's own
    ISA verification remains dccobjdump-based.
    """
    env = os.environ.copy()
    existing = shutil.which("llvm-objdump", path=env.get("PATH"))
    if existing:
        return env, existing

    candidates: list[Path] = []
    for root in (Path("/opt"), Path("/public/software"), Path("/opt/hpc/software")):
        if root.is_dir():
            candidates.extend(root.glob("**/llvm-objdump"))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            env["PATH"] = str(candidate.parent) + os.pathsep + env.get("PATH", "")
            return env, str(candidate)
    return env, None


def _parse_codeobj_log(log: str, rc: int, binary: str, out_log: str) -> dict:
    def max_for(pattern: str) -> int | None:
        vals = []
        for m in re.finditer(pattern, log, re.IGNORECASE):
            try:
                vals.append(int(m.group(1)))
            except ValueError:
                pass
        return max(vals) if vals else None

    vgpr = max_for(r"\bVGPR\w*[^0-9]{0,20}([0-9]+)")
    sgpr = max_for(r"\bSGPR\w*[^0-9]{0,20}([0-9]+)")
    lds = max_for(r"\b(?:LDS|shared)[^0-9]{0,20}([0-9]+)")
    pressure = []
    if vgpr is not None and vgpr >= 128:
        pressure.append("high_vgpr")
    if sgpr is not None and sgpr >= 96:
        pressure.append("high_sgpr")
    return {
        "available": rc == 0,
        "returncode": rc,
        "binary": binary,
        "log": out_log,
        "max_vgpr": vgpr,
        "max_sgpr": sgpr,
        "max_lds": lds,
        "pressure_flags": pressure,
    }


def _find_csv_files(root: str) -> list[str]:
    if os.path.isfile(root) and root.endswith(".csv"):
        return [root]
    base = os.path.dirname(root) or "."
    prefix = os.path.basename(root)
    files = []
    for path in glob_walk(base):
        name = os.path.basename(path)
        if name.endswith(".csv") and prefix in path:
            files.append(path)
    return sorted(files)


def glob_walk(base: str):
    for dirpath, _, filenames in os.walk(base):
        for filename in filenames:
            yield os.path.join(dirpath, filename)


def _parse_csv_metrics(files: list[str]) -> dict[str, dict]:
    agg: dict[str, dict] = {}
    for file in files:
        try:
            with open(file, "r", encoding="utf-8", errors="ignore", newline="") as f:
                rows = list(csv.DictReader(f))
        except OSError:
            continue
        for row in rows:
            for key, raw in row.items():
                value = _to_float(raw)
                if value is None:
                    continue
                axis, higher_is_worse = _classify(key)
                if axis is None:
                    continue
                item = agg.setdefault(key, {"sum": 0.0, "n": 0, "axis": axis, "higher_is_worse": higher_is_worse, "source_files": set()})
                item["sum"] += value
                item["n"] += 1
                item["source_files"].add(os.path.basename(file))
    out = {}
    for name, item in agg.items():
        out[name] = {
            "value": item["sum"] / item["n"] if item["n"] else None,
            "axis": item["axis"],
            "higher_is_worse": item["higher_is_worse"],
            "samples": item["n"],
            "source_files": sorted(item["source_files"]),
        }
    return out


def _rank_by_axis(agg: dict[str, dict], top_n: int) -> dict[str, list]:
    out = {"compute": [], "memory": [], "latency": []}
    for axis in out:
        candidates = []
        for name, item in agg.items():
            if item["axis"] != axis or item.get("value") is None:
                continue
            value = float(item["value"])
            severity = value if item.get("higher_is_worse", True) else (100.0 - value)
            candidates.append((severity, name, value, item))
        candidates.sort(key=lambda row: row[1])  # discovery only, incompatible units cannot be severity ranked
        for _, name, value, item in candidates[:top_n]:
            out[axis].append({
                "name": name,
                "value": value,
                "unit": "unknown",
                "interpretation": "raw discovery aggregate; scope and denominator unverified",
                "higher_is_worse": item.get("higher_is_worse", True),
                "samples": item.get("samples"),
                "source_files": item.get("source_files", []),
            })
    return out


PMC_GROUPS = ("default", "read", "write", "compute", "wave", "util", "memory")


def _pmc_plan(mode: str, out_prefix: str, group: str = "") -> list[tuple[str, str, str]]:
    if group:
        if mode != "pmc" or group not in PMC_GROUPS:
            raise ValueError("--pmc-group cannot be combined with legacy --pmc-mode read/write/all/none")
        return [("pmc_" + group, "--pmc", out_prefix + ".pmc_" + group)]
    plans = {
        "none": [],
        "pmc": [("pmc", "--pmc", out_prefix)],
        "read": [("pmc_read", "--pmc-read", out_prefix + ".pmc_read")],
        "write": [("pmc_write", "--pmc-write", out_prefix + ".pmc_write")],
        "all": [
            ("pmc", "--pmc", out_prefix),
            ("pmc_read", "--pmc-read", out_prefix + ".pmc_read"),
            ("pmc_write", "--pmc-write", out_prefix + ".pmc_write"),
        ],
    }
    return plans[mode]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--iter", required=True, type=int)
    p.add_argument("--which", required=True, choices=["best_input", "kernel"])
    p.add_argument("--benchmark", default=None)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--hipprof-bin", default="")
    p.add_argument("--kernel-name", default="")
    p.add_argument("--pmc-mode", default="pmc", choices=["none", "pmc", "read", "write", "all"])
    p.add_argument("--pmc-type", default="3")
    p.add_argument("--pmc-group", default="", choices=["", *PMC_GROUPS],
                   help="DTK 26.10 named group; use only when installed hipprof help lists it")
    p.add_argument("--sqtt-type", default="", help="Optional SQTT collection type, e.g. '1', 'stat_stall', 'stat_valu', or 'all'")
    p.add_argument("--sqtt-output-type", default="", choices=["", "0", "1", "2"], help="Optional hipprof --output-type; verify its effect on SQTT in the installed version")
    p.add_argument("--sqtt-data-dir", default="", help="Optional hipprof -d data directory for SQTT trace artifacts")
    p.add_argument("--no-codeobj-analyze", action="store_true")
    p.add_argument("--promote-if-best", action="store_true")
    args = p.parse_args()
    if args.pmc_group and args.pmc_mode != "pmc":
        p.error("--pmc-group requires --pmc-mode pmc (the default)")

    state = _read(args.state)
    from experiment import resolve_benchmark, require_open_iteration, iteration_kernel
    require_open_iteration(state, args.iter)
    args.benchmark = resolve_benchmark(state, args.benchmark)
    run_dir = state["run_dir"]
    iter_dir = os.path.join(run_dir, f"iterv{args.iter}")
    os.makedirs(iter_dir, exist_ok=True)

    if args.which == "best_input":
        solution = state["best_file"]
        rep_name = "best_input.hipprof"
    else:
        selected = os.path.join(iter_dir, "branch_results.json")
        if not os.path.isfile(selected):
            sys.exit("Run branch selection first; a filename alone does not identify the champion")
        solution = iteration_kernel(iter_dir)
        rep_name = "kernel.hipprof"

    hipprof_info = state.get("env", {}).get("hipprof", {}) or {}
    hipprof_bin = args.hipprof_bin or hipprof_info.get("path") or shutil.which("hipprof") or "hipprof"
    capture_dir = os.path.join(iter_dir, rep_name, datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    os.makedirs(capture_dir, exist_ok=True)
    out_prefix = os.path.join(capture_dir, "capture")
    log_path = os.path.join(capture_dir, "collection.log")
    provenance = {"tool": "hipprof", "profiled_file": solution, "raw_directory": capture_dir,
                  "backend": _detect_backend(solution), "pmc_group": args.pmc_group or None}

    if not shutil.which(hipprof_bin) and not os.path.isfile(hipprof_bin):
        top = {
            **provenance,
            "degraded": True,
            "collection_status": "unavailable",
            "reason": "hipprof not available",
            "compute": [], "memory": [], "latency": [],
        }
        save_profile(iter_dir, args.which, top, state, args.benchmark)
        print(json.dumps(top, indent=2))
        return

    try:
        probe = subprocess.run([hipprof_bin, "-h"], capture_output=True, text=True, timeout=30)
        help_text = (probe.stdout or "") + (probe.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        help_text = str(exc)
    Path(capture_dir, "help.txt").write_text(help_text, encoding="utf-8")
    required = [flag for _, flag, _ in _pmc_plan(args.pmc_mode, out_prefix, args.pmc_group)]
    if args.pmc_mode != "none":
        required.append("--pmc-type")
    if args.sqtt_type:
        required.extend(["--sqtt", "--sqtt-type"])
        if args.sqtt_output_type:
            required.append("--output-type")
    if args.kernel_name:
        required.append("--kernel-name")
    if (any(flag not in help_text for flag in required)
            or (args.pmc_group and not re.search(r"\b" + re.escape(args.pmc_group) + r"\b", help_text, re.I))):
        top = {**provenance, "degraded": True, "reason": "Installed hipprof help does not confirm requested flags", "compute": [], "memory": [], "latency": [], "help": str(Path(capture_dir, "help.txt"))}
        top["collection_status"] = "unavailable"
        save_profile(iter_dir, args.which, top, state, args.benchmark)
        print(json.dumps(top)); return
    logs = []
    collection_results = []
    rc_values = []
    profile_outputs = []
    benchmark_reports = []
    for label, flag, prefix in _pmc_plan(args.pmc_mode, out_prefix, args.pmc_group):
        benchmark_json = prefix + ".benchmark.json"
        rc, log = _run_hipprof(
            hipprof_bin=hipprof_bin,
            out_prefix=prefix,
            benchmark_py=os.path.abspath(args.benchmark),
            solution=solution,
            dims=state.get("dims", {}),
            ptr_size=state.get("ptr_size", 0),
            warmup=args.warmup,
            repeat=args.repeat,
            kernel_name=args.kernel_name,
            collect_flag=flag,
            pmc_type=args.pmc_type,
            pmc_group=args.pmc_group,
            benchmark_json=benchmark_json,
        )
        rc_values.append(rc)
        profile_outputs.append(prefix)
        if rc == 0:
            benchmark_reports.append(benchmark_json)
        collection_results.append({"label": label, "flag": flag, "pmc_group": args.pmc_group or None,
                                   "output": prefix, "benchmark_json": benchmark_json, "returncode": rc})
        logs.append(f"===== {label} ({flag}) rc={rc} output={prefix} =====\n{log}")

    sqtt_summary = None
    sqtt_prefix = ""
    if args.sqtt_type:
        sqtt_prefix = out_prefix + ".sqtt"
        if args.sqtt_data_dir:
            os.makedirs(args.sqtt_data_dir, exist_ok=True)
        sqtt_env, llvm_objdump = _sqtt_env_with_llvm_objdump()
        rc, log = _run_hipprof(
            hipprof_bin=hipprof_bin,
            out_prefix=sqtt_prefix,
            benchmark_py=os.path.abspath(args.benchmark),
            solution=solution,
            dims=state.get("dims", {}),
            ptr_size=state.get("ptr_size", 0),
            warmup=max(1, args.warmup),
            repeat=1,
            kernel_name=args.kernel_name,
            collect_flag="--sqtt",
            sqtt_type=args.sqtt_type,
            output_type=args.sqtt_output_type,
            data_dir=args.sqtt_data_dir,
            env=sqtt_env,
            benchmark_json=sqtt_prefix + ".benchmark.json",
        )
        rc_values.append(rc)
        profile_outputs.append(sqtt_prefix)
        if rc == 0:
            benchmark_reports.append(sqtt_prefix + ".benchmark.json")
        collection_results.append({
            "label": "sqtt",
            "flag": "--sqtt",
            "sqtt_type": args.sqtt_type,
            "sqtt_output_type": args.sqtt_output_type,
            "sqtt_data_dir": args.sqtt_data_dir,
            "llvm_objdump_for_hipprof_export": llvm_objdump,
            "output": sqtt_prefix,
            "returncode": rc,
        })
        logs.append(f"===== sqtt (--sqtt-type {args.sqtt_type}) rc={rc} output={sqtt_prefix} =====\n{log}")
        try:
            sqtt_paths = [sqtt_prefix]
            if args.sqtt_data_dir:
                sqtt_paths.append(args.sqtt_data_dir)
            sqtt_summary = analyze_sqtt_json(sqtt_paths)
            _write_json(os.path.join(capture_dir, "sqtt_analysis.json"), sqtt_summary)
        except Exception as exc:  # noqa: BLE001 - profiling should still produce dcu_top
            sqtt_summary = {"error": str(exc), "output": sqtt_prefix}

    Path(log_path).write_text("\n\n".join(logs), encoding="utf-8")

    csv_files = []
    for prefix in profile_outputs or [out_prefix]:
        csv_files.extend(_find_csv_files(prefix))
    csv_files = sorted(set(csv_files))
    agg = _parse_csv_metrics(csv_files)
    by_axis = _rank_by_axis(agg, state.get("ncu_num", state.get("dcu_num", 5)))

    codeobj = None
    if not args.no_codeobj_analyze and not solution.endswith(".py"):
        binary = _measured_binary(benchmark_reports, solution)
        if binary:
            codeobj = _run_codeobj_analyze(
                hipprof_bin=hipprof_bin,
                binary=binary,
                out_log=os.path.join(capture_dir, "codeobj_analyze.log"),
            )
        else:
            codeobj = {"available": False, "reason": "captured_benchmark_build_receipt_missing_or_changed", "kernel": solution}

    sqtt_unavailable = bool(args.sqtt_type) and (not sqtt_summary or bool(sqtt_summary.get("error"))
        or bool(sqtt_summary.get("parse_errors"))
        or not (sqtt_summary.get("file_count") or sqtt_summary.get("csv_files")))
    degraded = (not rc_values or any(rc != 0 for rc in rc_values)
                or (args.pmc_mode != "none" and not agg) or sqtt_unavailable)
    top = {
        **provenance,
        "degraded": degraded,
        "collection_status": "partial_or_failed" if degraded else "collected",
        "reason": f"hipprof rc={rc_values}; csv metrics={len(agg)}; sqtt_unavailable={sqtt_unavailable}; see {log_path}" if degraded else None,
        "hipprof_output": out_prefix,
        "hipprof_log": log_path,
        "collections": collection_results,
        "csv_files": csv_files,
        "metric_count_collected": len(agg),
        "metric_scope": "unverified aggregate; select exact dispatch before interpretation",
        "all_raw_metrics": agg,
        "codeobj_analyze": codeobj,
        "sqtt_analysis": sqtt_summary,
        **by_axis,
    }
    top_name = "dcu_top.json" if args.pmc_mode != "none" else f"{rep_name}.top.json"
    save_profile(iter_dir, args.which, top, state, args.benchmark, update_top=args.pmc_mode != "none")
    _write_json(os.path.join(iter_dir, top_name), top)

    if args.which == "kernel" and args.promote_if_best and os.path.abspath(solution) == os.path.abspath(state.get("best_file", "")):
        state["best_hipprof_output"] = out_prefix
        with open(args.state, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "tool": "hipprof",
        "hipprof_output": out_prefix,
        "dcu_top": os.path.join(iter_dir, top_name),
        "degraded": degraded,
        "metrics": len(agg),
    }, indent=2))


if __name__ == "__main__":
    main()
