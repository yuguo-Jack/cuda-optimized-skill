#!/usr/bin/env python3
"""Backward-compatible informational wrapper around the strict hardware gate."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from hardware_gate import run_gate
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from hardware_gate import run_gate


def collect_env() -> dict:
    result = run_gate(backend="cuda", require_torch=False, attempts=1)
    result.update({"platform": sys.platform, "python": sys.version.split()[0]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./env.json")
    args = parser.parse_args()
    env = collect_env()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    gpu = (env.get("gpus") or [{}])[0]
    print(json.dumps({
        "gpu": gpu.get("name"), "sm_arch": env.get("primary_sm_arch"),
        "nvcc": env.get("nvcc", {}).get("version"),
        "ncu": env.get("ncu", {}).get("available"),
        "ncu_can_read_counters": env.get("ncu", {}).get("can_read_counters"),
        "cutlass": env.get("cutlass", {}).get("available"),
        "torch": env.get("libs", {}).get("torch", {}).get("version"),
        "triton": env.get("libs", {}).get("triton", {}).get("version"),
        "out": args.out,
    }, indent=2))


if __name__ == "__main__":
    main()
