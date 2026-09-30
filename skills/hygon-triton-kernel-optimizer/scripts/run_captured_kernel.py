#!/usr/bin/env python3
"""Run a captured standalone Triton kernel and parse timing output."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


FIELD_RE = re.compile(r"^\s*(?P<key>Total tensor bytes|Best config time|Footprint rate estimate|HW peak bandwidth|Footprint/peak ratio \(not HBM utilization\))\s+:\s+(?P<value>.+?)\s*$", re.M)


def _float_prefix(value: str) -> float | None:
    m = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value)
    return float(m.group(0)) if m else None


def parse_output(text: str) -> dict:
    metrics = {}
    for m in FIELD_RE.finditer(text):
        key = m.group("key").lower().replace(" ", "_")
        value = m.group("value").strip()
        metrics[key] = value
        number = _float_prefix(value)
        if number is not None:
            metrics[key + "_value"] = number
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a captured Triton kernel standalone")
    parser.add_argument("kernel_py")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--env", action="append", default=[], help="Extra KEY=VALUE environment entries")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    env = os.environ.copy()
    for item in args.env:
        if "=" not in item:
            raise SystemExit(f"--env must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        env[key] = value

    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    command = [sys.executable, str(Path(args.kernel_py).resolve())]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=args.timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
        proc = subprocess.CompletedProcess(command, 124, stdout, stderr + "\nCaptured kernel timed out")
    except OSError as exc:
        proc = subprocess.CompletedProcess(command, 1, "", str(exc))
    metrics = parse_output((proc.stdout or "") + "\n" + (proc.stderr or ""))
    result = {
        "kernel_py": str(Path(args.kernel_py).resolve()),
        "correctness": "not_checked",
        "measurement_scope": "autotune diagnostics, not validated optimization evidence",
        "returncode": proc.returncode,
        "metrics": metrics,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "returncode": proc.returncode,
        "best_ms": metrics.get("best_config_time_value"),
        "footprint_rate_gbps": metrics.get("footprint_rate_estimate_value"),
        "json_out": args.json_out or None,
    }, indent=2))
    if proc.returncode != 0:
        sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
