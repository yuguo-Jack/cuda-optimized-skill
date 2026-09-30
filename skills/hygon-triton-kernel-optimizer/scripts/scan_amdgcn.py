#!/usr/bin/env python3
"""Scan AMDGCN/ISA dumps for memory and compute instruction families."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EXTS = {".amdgcn", ".isa", ".s", ".asm"}
PATTERNS = {
    "buffer_load_dwordx4": r"\bbuffer_load_dwordx4\b",
    "buffer_load_dwordx2": r"\bbuffer_load_dwordx2\b",
    "buffer_load_dword": r"\bbuffer_load_dword\b",
    "buffer_store_dwordx4": r"\bbuffer_store_dwordx4\b",
    "buffer_store_dwordx2": r"\bbuffer_store_dwordx2\b",
    "buffer_store_dword": r"\bbuffer_store_dword\b",
    "global_load": r"\bglobal_load",
    "global_store": r"\bglobal_store",
    "flat_load": r"\bflat_load",
    "flat_store": r"\bflat_store",
    "atomic": r"\b(?:global_|buffer_|flat_)?atomic",
    "ds_read": r"\bds_read",
    "ds_write": r"\bds_write",
    "v_mmac": r"\bv_mmac",
    "s_waitcnt": r"\bs_waitcnt\b",
}


def _iter_files(root: Path) -> list[Path]:
    def eligible(path: Path) -> bool:
        return path.suffix.lower() != ".ll" and (path.suffix.lower() in EXTS or "amdgcn" in path.name.lower())
    if root.is_file():
        return [root] if eligible(root) else []
    return [p for p in root.rglob("*") if p.is_file() and eligible(p)]


def scan(path: str, kernel: str = "") -> dict:
    root = Path(path)
    totals = {name: 0 for name in PATTERNS}
    files = []
    for file in _iter_files(root):
        text = file.read_text(encoding="utf-8", errors="ignore")
        if kernel and kernel not in file.name and kernel not in text:
            continue
        counts = {name: len(re.findall(pattern, text)) for name, pattern in PATTERNS.items()}
        if not any(counts.values()):
            continue
        for name, value in counts.items():
            totals[name] += value
        files.append({"file": str(file), "counts": counts})

    buffer_ops = sum(totals[k] for k in totals if k.startswith("buffer_load") or k.startswith("buffer_store"))
    global_flat_ops = totals["global_load"] + totals["global_store"] + totals["flat_load"] + totals["flat_store"]
    wide_buffer_ops = totals["buffer_load_dwordx4"] + totals["buffer_store_dwordx4"]
    classification = {
        "buffer_ops": buffer_ops,
        "global_flat_ops": global_flat_ops,
        "wide_buffer_ops": wide_buffer_ops,
        "buffer_ops_dominant": buffer_ops > global_flat_ops,
        "wide_buffer_observed": wide_buffer_ops > 0,
        "atomic_observed": totals["atomic"] > 0,
    }
    return {
        "root": str(root.resolve()),
        "kernel_filter": kernel,
        "files_scanned": len(files),
        "totals": totals,
        "classification": classification,
        "scope": "static instruction presence, not dynamic hotness or performance; LLVM IR excluded",
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan AMDGCN/ISA dumps")
    parser.add_argument("path")
    parser.add_argument("--kernel", default="")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()
    result = scan(args.path, args.kernel)
    payload = json.dumps(result, indent=2)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
