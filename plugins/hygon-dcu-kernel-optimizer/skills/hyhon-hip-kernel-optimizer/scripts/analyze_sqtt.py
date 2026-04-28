#!/usr/bin/env python3
"""Summarize DTK SQTT JSON instruction-stream artifacts.

The SQTT JSON schema can vary by DTK release and selected --sqtt-type. This
parser is intentionally permissive: it walks every JSON object, extracts
instruction-like mnemonics, trace-event names, durations, and stall/wait text,
then emits a compact summary that can be merged into dcu_top.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MNEMONIC_RE = re.compile(
    r"\b(?:s|v|ds|buffer|flat|global|matrix|image|exp)_[A-Za-z0-9_]+",
    re.IGNORECASE,
)
STALL_RE = re.compile(r"\b(?:stall|bubble|idle|wait|latency|scoreboard)\b", re.IGNORECASE)
BRANCH_RE = re.compile(r"\b(?:s_cbranch\w*|\w*branch\w*|jump\w*|jmp\w*)\b", re.IGNORECASE)
WAITCNT_RE = re.compile(r"\bs_waitcnt\b", re.IGNORECASE)


def _looks_like_sqtt_json(path: Path) -> bool:
    name = path.name.lower()
    return name.startswith("thread_trace") or "sqtt" in name or "trace" in name


def _iter_sqtt_files(paths: list[str]) -> tuple[list[Path], list[Path]]:
    json_files: list[Path] = []
    csv_files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.suffix.lower() == ".json":
            json_files.append(p)
        elif p.is_file() and p.suffix.lower() == ".csv":
            csv_files.append(p)
        elif p.is_dir():
            preferred = sorted(p.rglob("thread_trace*.json"))
            json_files.extend(preferred)
            if not preferred:
                json_files.extend(x for x in sorted(p.rglob("*.json")) if _looks_like_sqtt_json(x))
            csv_files.extend(x for x in sorted(p.rglob("*.csv")) if "sqtt" in x.name.lower())
        elif not p.exists():
            base = p.parent if str(p.parent) else Path(".")
            stem = p.name
            if base.is_dir():
                json_files.extend(sorted(base.glob(f"{stem}*.json")))
                csv_files.extend(sorted(base.glob(f"{stem}*.csv")))

    def dedupe(items: list[Path]) -> list[Path]:
        seen = set()
        out = []
        for item in items:
            resolved = str(item.resolve())
            if resolved not in seen:
                seen.add(resolved)
                out.append(item)
        return out

    return dedupe(json_files), dedupe(csv_files)


def _summarize_sqtt_csv(files: list[Path]) -> dict:
    kernels: Counter[str] = Counter()
    se_bytes: Counter[str] = Counter()
    numeric_sums: defaultdict[str, float] = defaultdict(float)
    numeric_counts: Counter[str] = Counter()
    rows = 0
    parse_errors = []

    for path in files:
        try:
            with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows += 1
                    kernel = row.get("KernelName") or row.get("kernel") or row.get("Name") or ""
                    if kernel:
                        kernels[kernel] += 1
                    for key, raw in row.items():
                        if raw is None:
                            continue
                        text = str(raw).strip().replace(",", "")
                        try:
                            value = float(text)
                        except ValueError:
                            continue
                        low = key.lower()
                        if low.startswith("sqtt_se") and low.endswith("_size"):
                            se_bytes[key] += value
                        if low in {"grd", "wgr", "lds", "scr", "arch_vgpr", "accum_vgpr", "sgpr", "wave_size", "maxclk"}:
                            numeric_sums[key] += value
                            numeric_counts[key] += 1
        except Exception as exc:  # noqa: BLE001 - report and keep parsing siblings
            parse_errors.append({"file": str(path), "error": str(exc)})

    return {
        "csv_files": [str(p) for p in files],
        "csv_row_count": rows,
        "csv_parse_errors": parse_errors,
        "top_kernels": kernels.most_common(20),
        "sqtt_se_size_total": sum(se_bytes.values()),
        "sqtt_se_size_by_column": dict(se_bytes),
        "kernel_resource_averages": [
            {
                "name": name,
                "avg": numeric_sums[name] / numeric_counts[name] if numeric_counts[name] else None,
                "count": numeric_counts[name],
            }
            for name, _ in numeric_counts.most_common()
        ],
    }


def _json_files_with_trace_data(files: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in files:
        if path.name.lower().startswith("thread_trace"):
            out.append(path)
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig", errors="ignore"))
        except Exception:
            out.append(path)
            continue
        if isinstance(data, dict) and ("traceEvents" in data or "displayTimeUnit" in data):
            out.append(path)
    return out


def _walk(obj: Any, key: str = ""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, str(k))
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, key)
    else:
        yield key, obj


def _category(mnemonic: str) -> str:
    m = mnemonic.lower()
    if m.startswith(("global_", "buffer_", "flat_")):
        return "vmem"
    if m.startswith("ds_"):
        return "lds"
    if "mmac" in m or m.startswith("matrix_"):
        return "matrix"
    if m.startswith("s_"):
        return "salu"
    if m.startswith("v_"):
        return "valu"
    return "other"


def _trace_events(obj: Any) -> tuple[Counter, float, int]:
    events = Counter()
    total_dur = 0.0
    count = 0
    if not isinstance(obj, dict):
        return events, total_dur, count
    trace_events = obj.get("traceEvents")
    if not isinstance(trace_events, list):
        return events, total_dur, count
    for ev in trace_events:
        if not isinstance(ev, dict):
            continue
        name = str(ev.get("name") or ev.get("cat") or "?")
        events[name] += 1
        count += 1
        try:
            total_dur += float(ev.get("dur", 0) or 0)
        except (TypeError, ValueError):
            pass
    return events, total_dur, count


def analyze(paths: list[str]) -> dict:
    files, csv_files = _iter_sqtt_files(paths)
    files = _json_files_with_trace_data(files)
    mnemonics: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    event_names: Counter[str] = Counter()
    stall_hits: Counter[str] = Counter()
    numeric_sums: defaultdict[str, float] = defaultdict(float)
    numeric_counts: Counter[str] = Counter()
    total_trace_duration = 0.0
    total_trace_events = 0
    parse_errors: list[dict] = []

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig", errors="ignore"))
        except Exception as exc:  # noqa: BLE001 - keep parsing the remaining traces
            parse_errors.append({"file": str(path), "error": str(exc)})
            continue

        events, dur, n_events = _trace_events(data)
        event_names.update(events)
        total_trace_duration += dur
        total_trace_events += n_events

        for key, value in _walk(data):
            text = str(value)
            for mnemonic in MNEMONIC_RE.findall(text):
                mn = mnemonic.lower()
                mnemonics[mn] += 1
                categories[_category(mn)] += 1
            if STALL_RE.search(key) or STALL_RE.search(text):
                stall_hits[key or "<value>"] += 1
            if isinstance(value, (int, float)) and any(token in key.lower() for token in ("cycle", "clock", "duration", "dur")):
                numeric_sums[key] += float(value)
                numeric_counts[key] += 1

    waitcnt = sum(count for mn, count in mnemonics.items() if WAITCNT_RE.search(mn))
    branches = sum(count for mn, count in mnemonics.items() if BRANCH_RE.search(mn))
    result = {
        "json_files": [str(p) for p in files],
        "file_count": len(files),
        "parse_errors": parse_errors,
        "trace_event_count": total_trace_events,
        "trace_event_duration_sum": total_trace_duration,
        "top_trace_events": event_names.most_common(20),
        "instruction_count": sum(mnemonics.values()),
        "top_mnemonics": mnemonics.most_common(40),
        "category_counts": dict(categories),
        "waitcnt_count": waitcnt,
        "branch_count": branches,
        "stall_like_hits": stall_hits.most_common(30),
        "numeric_cycle_fields": [
            {
                "name": name,
                "sum": numeric_sums[name],
                "count": numeric_counts[name],
                "avg": numeric_sums[name] / numeric_counts[name] if numeric_counts[name] else None,
            }
            for name, _ in numeric_counts.most_common(30)
        ],
    }
    result.update(_summarize_sqtt_csv(csv_files))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", help="SQTT JSON files or directories")
    p.add_argument("--out", default="")
    args = p.parse_args()
    result = analyze(args.paths)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
