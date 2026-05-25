#!/usr/bin/env python3
"""Parse autotune_capture_patch.py logs into JSON or Markdown."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


START_RE = re.compile(r"\[(?P<kind>AUTOTUNE|SINGLE-CONFIG)\]\s+kernel=(?P<kernel>\S+)\s+configs=(?P<configs>\d+)")
INPUT_RE = re.compile(r"^\s*(?P<name>[A-Za-z_]\w*)\s*:\s+shape=(?P<shape>\[[^\]]*\])\s+dtype=(?P<dtype>\S+)\s+stride=(?P<stride>\[[^\]]*\])")
SCALAR_RE = re.compile(r"^\s*(?P<name>[A-Za-z_]\w*)\s*:\s+(?P<value>[-+A-Za-z0-9_./]+)\s*$")
KV_RE = re.compile(r"^\s*(?P<key>Total tensor bytes|Best config time|Effective bandwidth|HW peak bandwidth|Bandwidth utilization)\s+:\s+(?P<value>.+?)\s*$")
SAVED_RE = re.compile(r"^\s*(?P<key>Kernel source saved|Inputs saved|Run standalone)\s+:\s+(?P<value>.+?)\s*$")
TS_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+")


def _float_prefix(value: str) -> float | None:
    m = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    return float(m.group(0)) if m else None


def parse_log(path: str) -> dict:
    kernels: list[dict] = []
    current: dict | None = None
    in_table = False

    for raw in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        line = TS_PREFIX_RE.sub("", raw.strip("\n"))
        m = START_RE.search(line)
        if m:
            current = {
                "kind": m.group("kind"),
                "kernel": m.group("kernel"),
                "configs": int(m.group("configs")),
                "inputs": [],
                "scalars": {},
                "config_timings": [],
                "metrics": {},
                "artifacts": {},
            }
            kernels.append(current)
            in_table = False
            continue

        if current is None:
            continue

        m = INPUT_RE.match(line)
        if m:
            current["inputs"].append({
                "name": m.group("name"),
                "shape": m.group("shape"),
                "dtype": m.group("dtype"),
                "stride": m.group("stride"),
            })
            continue

        if "Config" in line and "Time(ms)" in line:
            in_table = True
            continue
        if in_table:
            if not line.strip() or "Total tensor bytes" in line:
                in_table = False
            elif set(line.strip()) == {"-"}:
                continue
            else:
                mtime = re.match(r"^\s*(?P<config>.+)\s+(?P<ms>\d+(?:\.\d+)?)\s*(?P<best>.*BEST.*)?$", line)
                if mtime:
                    current["config_timings"].append({
                        "config": mtime.group("config").strip(),
                        "time_ms": float(mtime.group("ms")),
                        "best": bool(mtime.group("best")),
                    })
                    continue

        m = KV_RE.match(line)
        if m:
            key = m.group("key").lower().replace(" ", "_")
            current["metrics"][key] = m.group("value").strip()
            number = _float_prefix(m.group("value"))
            if number is not None:
                current["metrics"][key + "_value"] = number
            continue

        m = SAVED_RE.match(line)
        if m:
            current["artifacts"][m.group("key").lower().replace(" ", "_")] = m.group("value").strip()
            continue

        m = SCALAR_RE.match(line)
        if m and not line.lstrip().startswith(("[", "Total", "Best", "Effective", "HW", "Bandwidth")):
            current["scalars"][m.group("name")] = m.group("value")

    return {"log": str(Path(path).resolve()), "kernels": kernels}


def write_markdown(summary: dict, out: str) -> None:
    lines = ["# Autotune Summary", ""]
    for item in summary["kernels"]:
        best = next((c for c in item["config_timings"] if c.get("best")), None)
        lines.append(f"## {item['kernel']}")
        lines.append("")
        lines.append(f"- kind: `{item['kind']}`")
        lines.append(f"- configs: `{item['configs']}`")
        if best:
            lines.append(f"- best: `{best['time_ms']:.4f} ms` with `{best['config']}`")
        for key in ("effective_bandwidth", "hw_peak_bandwidth", "bandwidth_utilization"):
            if key in item["metrics"]:
                lines.append(f"- {key}: `{item['metrics'][key]}`")
        if item["inputs"]:
            lines.append("")
            lines.append("| Arg | Shape | Dtype | Stride |")
            lines.append("| --- | --- | --- | --- |")
            for inp in item["inputs"]:
                lines.append(f"| `{inp['name']}` | `{inp['shape']}` | `{inp['dtype']}` | `{inp['stride']}` |")
        lines.append("")
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a captured TorchInductor Triton autotune log")
    parser.add_argument("log")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--markdown-out", default="")
    args = parser.parse_args()
    summary = parse_log(args.log)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.markdown_out:
        write_markdown(summary, args.markdown_out)
    print(json.dumps({
        "kernels": len(summary["kernels"]),
        "json_out": args.json_out or None,
        "markdown_out": args.markdown_out or None,
    }, indent=2))


if __name__ == "__main__":
    main()
