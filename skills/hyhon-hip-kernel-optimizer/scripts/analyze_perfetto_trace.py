#!/usr/bin/env python3
"""Analyze SQTT Chrome-trace JSON with Perfetto Trace Processor.

This is an optional companion to analyze_sqtt.py. DTK SQTT JSON is Chrome
trace JSON, which Perfetto Trace Processor can ingest into SQL tables. Keep the
queries intentionally simple so the script remains useful across DTK releases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _trace_files(paths: list[str], max_files: int) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file() and path.suffix.lower() == ".json":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*thread_trace*.json")))
    out: list[Path] = []
    seen = set()
    for item in files:
        resolved = str(item.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(item)
        if max_files and len(out) >= max_files:
            break
    return out


def _query_rows(tp, sql: str) -> list[dict]:
    rows = []
    for row in tp.query(sql):
        rows.append({name: getattr(row, name) for name in row.__dict__.keys()})
    return rows


def analyze(paths: list[str], max_files: int) -> dict:
    try:
        from perfetto.trace_processor import TraceProcessor
    except Exception as exc:  # noqa: BLE001 - optional dependency
        return {
            "available": False,
            "error": f"perfetto trace_processor import failed: {exc}",
            "files": [],
        }

    files = _trace_files(paths, max_files)
    summaries = []
    for path in files:
        try:
            tp = TraceProcessor(trace=str(path))
            top_slices = _query_rows(
                tp,
                """
                select name, count(*) as count, coalesce(sum(dur), 0) as total_dur
                from slice
                group by name
                order by count desc
                limit 30
                """,
            )
            stall_rows = _query_rows(
                tp,
                """
                select name, count(*) as count, coalesce(sum(dur), 0) as total_dur
                from slice
                where lower(name) like '%stall%' or lower(name) like '%wait%'
                group by name
                order by count desc
                limit 20
                """,
            )
            totals = _query_rows(
                tp,
                "select count(*) as slice_count, coalesce(sum(dur), 0) as total_dur from slice",
            )
            tp.close()
            summaries.append({
                "file": str(path),
                "top_slices": top_slices,
                "stall_or_wait_slices": stall_rows,
                "totals": totals[0] if totals else {},
            })
        except Exception as exc:  # noqa: BLE001 - keep remaining traces usable
            summaries.append({"file": str(path), "error": str(exc)})

    return {
        "available": True,
        "files": [str(p) for p in files],
        "file_count": len(files),
        "summaries": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="SQTT thread_trace JSON files or directories")
    parser.add_argument("--max-files", type=int, default=4, help="Limit files because SQTT can emit one large JSON per SE")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    result = analyze(args.paths, args.max_files)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
