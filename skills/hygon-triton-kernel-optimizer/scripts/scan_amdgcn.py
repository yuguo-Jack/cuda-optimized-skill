#!/usr/bin/env python3
"""Scan AMDGCN/ISA dumps for memory and compute instruction families."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "hygon-hip-kernel-optimizer/scripts"))
from isa import parse as parse_isa


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
    "s_waitcnt_vbcnt": r"\bs_waitcnt_vbcnt\b",
    "matrix_load": r"\bmatrix_load(?:_[a-z0-9]+)+\b",
    "matrix_store": r"\bmatrix_store(?:_[a-z0-9]+)+\b",
    "tensor_load": r"\btensor_load\b",
    "v_mmac_scale": r"\bv_mmac_scale_",
    "v_cvt_scale": r"\bv_cvt_scale_",
    "ds_scale_copy": r"\bds_scale_copy_",
    "s_set_vgpr_size": r"\bs_set_vgpr_size(?:_prsv)?\b",
    "s_abarrier": r"\bs_abarrier_",
    "s_ebarrier": r"\bs_ebarrier_",
    "multimem": r"\bmultimem_",
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
    unresolved = []
    for file in _iter_files(root):
        text = file.read_text(encoding="utf-8", errors="ignore")
        parsed = parse_isa(text, kernel)
        if parsed["reason"]:
            unresolved.append({"file": str(file), "reason": parsed["reason"], "symbols": parsed["symbols"]})
            continue
        instruction_text = "\n".join(r["text"] for r in parsed["instructions"])
        counts = {name: len(re.findall(pattern, instruction_text)) for name, pattern in PATTERNS.items()}
        if not any(counts.values()):
            continue
        for name, value in counts.items():
            totals[name] += value
        files.append({"file": str(file), "counts": counts, "scope": parsed["scope"], "symbols": parsed["symbols"],
                      "selected_symbol": parsed["selected_symbol"], "instruction_sites": len(parsed["instructions"])})

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
        "unresolved": unresolved,
        "limitations": "Recognized syntax only; excluded macros are not expanded. Without --kernel all symbols are included; families overlap, so totals cannot be summed as executed instructions.",
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
