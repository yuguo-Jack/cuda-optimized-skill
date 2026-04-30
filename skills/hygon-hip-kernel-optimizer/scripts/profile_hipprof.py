#!/usr/bin/env python3
"""Profile a HIP kernel with DTK hipprof and extract top DCU metrics.

Writes under {run_dir}/iterv{i}/:
  {which}.hipprof/   hipprof output directory
  dcu_top.json       top metrics per compute / memory / latency axis
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from analyze_sqtt import analyze as analyze_sqtt_json


_BUNDLED_BENCHMARK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.py")
KERNEL_EXTS = (".hip", ".cu", ".cpp", ".cc", ".cxx", ".py")

METRIC_RUBRIC: list[tuple[str, str, bool]] = [
    (r"SQ_INSTS_MMOP|MMAC|MATRIX|VALU_FMA|VALU_ADD|VALU_MUL|SQ_BUSY|GRBM_GUI_ACTIVE", "compute", True),
    (r"TCC|TCP|TA_|TD_|READ_REQ|WRITE_REQ|CACHE|L2|L1|BW|BANDWIDTH", "memory", True),
    (r"STALL|LATENCY|WAIT|WAVE_CYCLES|BARRIER|ATOMIC|SQ_WAVES", "latency", True),
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
        return float(text)
    except ValueError:
        return None


def _find_binary(kernel_path: str) -> str | None:
    base = os.path.splitext(kernel_path)[0]
    for ext in (".so", ""):
        candidate = base + ext
        if os.path.isfile(candidate):
            return candidate
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
    print(f"[hipprof] {' '.join(cmd)}", file=sys.stderr)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", env=env)
    except OSError as exc:
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
        if name.endswith(".csv") and (prefix in path or "pmc" in name.lower()):
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
        candidates.sort(reverse=True)
        for _, name, value, item in candidates[:top_n]:
            out[axis].append({
                "name": name,
                "value": value,
                "unit": "",
                "higher_is_worse": item.get("higher_is_worse", True),
                "samples": item.get("samples"),
                "source_files": item.get("source_files", []),
            })
    return out


def _pmc_plan(mode: str, out_prefix: str) -> list[tuple[str, str, str]]:
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
    p.add_argument("--benchmark", default=_BUNDLED_BENCHMARK)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--hipprof-bin", default="")
    p.add_argument("--kernel-name", default="")
    p.add_argument("--pmc-mode", default="all", choices=["none", "pmc", "read", "write", "all"])
    p.add_argument("--pmc-type", default="3")
    p.add_argument("--sqtt-type", default="", help="Optional SQTT collection type, e.g. '1', 'stat_stall', 'stat_valu', or 'all'")
    p.add_argument("--sqtt-output-type", default="", choices=["", "0", "1", "2"], help="Optional hipprof SQTT export type: 0=json, 1=html, 2=perfetto when supported")
    p.add_argument("--sqtt-data-dir", default="", help="Optional hipprof -d data directory for SQTT trace artifacts")
    p.add_argument("--no-codeobj-analyze", action="store_true")
    p.add_argument("--promote-if-best", action="store_true")
    args = p.parse_args()

    state = _read(args.state)
    run_dir = state["run_dir"]
    iter_dir = os.path.join(run_dir, f"iterv{args.iter}")
    os.makedirs(iter_dir, exist_ok=True)

    if args.which == "best_input":
        solution = state["best_file"]
        rep_name = "best_input.hipprof"
    else:
        solution = next((os.path.join(iter_dir, f"kernel{ext}") for ext in KERNEL_EXTS if os.path.isfile(os.path.join(iter_dir, f"kernel{ext}"))), None)
        if not solution:
            sys.exit(f"No iterv{args.iter}/kernel.(hip|cu|cpp|cc|cxx|py) found.")
        rep_name = "kernel.hipprof"

    hipprof_info = state.get("env", {}).get("hipprof", {}) or {}
    hipprof_bin = args.hipprof_bin or hipprof_info.get("path") or shutil.which("hipprof") or "hipprof"
    out_prefix = os.path.join(iter_dir, rep_name)
    log_path = os.path.join(iter_dir, f"{rep_name}.log")

    if not shutil.which(hipprof_bin) and not os.path.isfile(hipprof_bin):
        top = {
            "degraded": True,
            "reason": "hipprof not available",
            "profiled_file": solution,
            "backend": _detect_backend(solution),
            "compute": [], "memory": [], "latency": [],
        }
        _write_json(os.path.join(iter_dir, "dcu_top.json"), top)
        print(json.dumps(top, indent=2))
        return

    logs = []
    collection_results = []
    rc_values = []
    pmc_rc_values = []
    profile_outputs = []
    for label, flag, prefix in _pmc_plan(args.pmc_mode, out_prefix):
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
        )
        rc_values.append(rc)
        pmc_rc_values.append(rc)
        profile_outputs.append(prefix)
        collection_results.append({"label": label, "flag": flag, "output": prefix, "returncode": rc})
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
        )
        rc_values.append(rc)
        profile_outputs.append(sqtt_prefix)
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
            _write_json(os.path.join(iter_dir, f"{rep_name}.sqtt_analysis.json"), sqtt_summary)
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
        binary = _find_binary(solution)
        if binary:
            codeobj = _run_codeobj_analyze(
                hipprof_bin=hipprof_bin,
                binary=binary,
                out_log=os.path.join(iter_dir, f"{rep_name}.codeobj_analyze.log"),
            )
        else:
            codeobj = {"available": False, "reason": "binary_not_found_after_benchmark", "kernel": solution}

    degraded = any(rc != 0 for rc in pmc_rc_values) or (args.pmc_mode != "none" and not agg)
    top = {
        "degraded": degraded,
        "reason": f"hipprof rc={rc_values}; csv metrics={len(agg)}; see {log_path}" if degraded else None,
        "profiled_file": solution,
        "backend": _detect_backend(solution),
        "hipprof_output": out_prefix,
        "hipprof_log": log_path,
        "collections": collection_results,
        "csv_files": csv_files,
        "metric_count_collected": len(agg),
        "codeobj_analyze": codeobj,
        "sqtt_analysis": sqtt_summary,
        **by_axis,
    }
    top_name = "dcu_top.json" if args.pmc_mode != "none" else f"{rep_name}.top.json"
    _write_json(os.path.join(iter_dir, top_name), top)

    if args.which == "kernel" and args.promote_if_best and os.path.abspath(solution) == os.path.abspath(state.get("best_file", "")):
        state["best_hipprof_output"] = out_prefix
        with open(args.state, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    print(json.dumps({
        "hipprof_output": out_prefix,
        "dcu_top": os.path.join(iter_dir, top_name),
        "degraded": degraded,
        "metrics": len(agg),
    }, indent=2))


if __name__ == "__main__":
    main()
