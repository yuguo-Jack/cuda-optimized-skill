#!/usr/bin/env python3
"""Inspect generated Triton source for metadata, hints, and code patterns."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


SIGNATURE_RE = re.compile(r"['\"]signature['\"]\s*:\s*(\{[^{}]*\})", re.S)
KERNEL_NAME_RE = re.compile(r"['\"]kernel_name['\"]\s*:\s*['\"]([^'\"]+)['\"]")
DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", re.M)


def _literal_dict(raw: str) -> dict:
    try:
        value = ast.literal_eval(raw)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _parse_tuple_indices(raw: str) -> list[int]:
    out: list[int] = []
    for token in re.findall(r"\d+", raw or ""):
        out.append(int(token))
    return sorted(set(out))


def _extract_hint_indices(text: str, key: str) -> list[int]:
    found: list[int] = []
    for m in re.finditer(rf"['\"]{re.escape(key)}['\"]\s*:\s*\(([^)]*)\)", text):
        found.extend(_parse_tuple_indices(m.group(1)))
    for m in re.finditer(rf"['\"]{re.escape(key)}['\"]", text):
        prefix = text[max(0, m.start() - 200):m.start()]
        old_style_indices = re.findall(r"\((\d+),?\)\s*:", prefix)
        if old_style_indices:
            found.append(int(old_style_indices[-1]))
    return sorted(set(found))


def inspect_file(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    signature = {}
    m = SIGNATURE_RE.search(text)
    if m:
        signature = _literal_dict(m.group(1))

    arg_order = list(signature.keys())
    ptr_args = [
        {"index": idx, "name": name, "type": str(tp)}
        for idx, (name, tp) in enumerate(signature.items())
        if str(tp).startswith("*")
    ]
    divisibility = _extract_hint_indices(text, "tt.divisibility")
    pointer_range = _extract_hint_indices(text, "tt.pointer_range")

    counts = {
        "tl_load": len(re.findall(r"\btl\.load\s*\(", text)),
        "tl_store": len(re.findall(r"\btl\.store\s*\(", text)),
        "tl_atomic_add": len(re.findall(r"\btl\.atomic_add\s*\(", text)),
        "tl_assume": len(re.findall(r"\btl\.assume\s*\(", text)),
        "tl_multiple_of": len(re.findall(r"\btl\.multiple_of\s*\(", text)),
        "make_block_ptr": len(re.findall(r"\btl\.make_block_ptr\s*\(", text)),
    }

    ptr_hint_status = []
    for item in ptr_args:
        idx = item["index"]
        ptr_hint_status.append({
            **item,
            "has_divisibility": idx in divisibility,
            "has_pointer_range": idx in pointer_range,
        })

    notes = []
    if counts["tl_atomic_add"]:
        notes.append("atomic_add present; inspect contention, traffic and ordering with target profiling before identifying a bottleneck.")
    if ptr_args and not any(p["has_pointer_range"] for p in ptr_hint_status):
        notes.append("No pointer_range hint detected for pointer args.")
    if ptr_args and counts["tl_assume"] == 0:
        notes.append("No tl.assume detected; add assumptions only when proven by the input contract, never infer signed range from pointer syntax.")

    return {
        "file": str(Path(path).resolve()),
        "defined_functions": DEF_RE.findall(text),
        "inductor_kernel_names": KERNEL_NAME_RE.findall(text),
        "signature": signature,
        "arg_order": arg_order,
        "ptr_args": ptr_args,
        "divisibility_indices": divisibility,
        "pointer_range_indices": pointer_range,
        "ptr_hint_status": ptr_hint_status,
        "counts": counts,
        "notes": notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect generated Triton source metadata and hints")
    parser.add_argument("source")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    result = inspect_file(args.source)
    payload = json.dumps(result, indent=2)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
