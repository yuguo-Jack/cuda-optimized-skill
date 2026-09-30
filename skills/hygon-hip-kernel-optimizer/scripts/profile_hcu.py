#!/usr/bin/env python3
"""Capability-gated HCU profiling. Preserve raw artifacts; do not invent metrics."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from experiment import resolve_benchmark


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--which", choices=["best_input", "kernel"], required=True)
    p.add_argument("--benchmark", default=None)
    p.add_argument("--promote-if-best", action="store_true", help="Compatibility flag; state promotion belongs to state.py")
    args = p.parse_args()
    state = json.loads(Path(args.state).read_text(encoding="utf-8"))
    args.benchmark = resolve_benchmark(state, args.benchmark)
    folder = Path(state["run_dir"]) / f"iterv{args.iter}"
    folder.mkdir(parents=True, exist_ok=True)
    which = state.get("profiler", "auto")
    xprof = shutil.which("xprof")
    if which == "hipprof" or (which == "auto" and not xprof):
        cmd = [sys.executable, str(Path(__file__).with_name("profile_hipprof.py")),
               "--state", args.state, "--iter", str(args.iter), "--which", args.which,
               "--benchmark", args.benchmark]
        raise SystemExit(subprocess.call(cmd))
    if args.which == "best_input":
        source = state["best_file"]
    else:
        source = json.loads((folder / "branch_results.json").read_text(encoding="utf-8"))["champion"]["kernel"]
    tool = "none" if which == "none" else "xprof"
    out = folder / (args.which + "." + tool) / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True)
    result = {"tool": tool, "requested_tool": which, "profiled_file": source, "raw_directory": str(out),
              "degraded": True, "compute": [], "memory": [], "latency": [],
              "reason": "Raw .perf needs dispatch-specific XCompute review and metric definitions; no automatic utilization conversion"}
    try:
        if which == "none":
            result["reason"] = "Profiling explicitly disabled; hypothesis remains unverified"
        elif not xprof:
            result["reason"] = "Requested xprof is unavailable"
        else:
            help_run = subprocess.run([xprof, "--help"], capture_output=True, text=True, timeout=30)
            help_text = (help_run.stdout or "") + (help_run.stderr or "")
            (out / "help.txt").write_text(help_text, encoding="utf-8")
            if not all(flag in help_text for flag in ("--sections", "--output-dir")):
                result["reason"] = "Installed xprof flags differ; inspect saved help before adapting the command"
            else:
                cmd = [xprof, "--sections", "speed_of_light,instruction_statistics", "--output-dir", str(out),
                       sys.executable, args.benchmark, source, "--warmup", "1", "--repeat", "1"]
                if state.get("ptr_size", 0):
                    cmd += ["--ptr-size", str(state["ptr_size"])]
                cmd += [f"--{k}={v}" for k, v in state.get("dims", {}).items()]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                (out / "collection.log").write_text((proc.stdout or "") + "\n" + (proc.stderr or ""), encoding="utf-8")
                result.update(command=cmd, returncode=proc.returncode,
                              artifacts=[str(f) for f in out.rglob("*") if f.is_file()])
                if proc.returncode:
                    result["reason"] = "xprof collection failed; inspect collection.log"
    except (OSError, subprocess.TimeoutExpired) as exc:
        result["reason"] = str(exc)
    (folder / "dcu_top.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
