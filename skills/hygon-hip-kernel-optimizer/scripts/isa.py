"""Conservative HCU/DCU assembly discovery, never a semantic ISA verifier."""
from __future__ import annotations
import re

INSTRUCTION = re.compile(r"^((?:s|v|ds|buffer|flat|global|matrix|tensor|multimem|image)_[a-z0-9_]+|exp|exp_[a-z0-9_]+)\b(.*)$", re.I)


def parse(text, symbol=""):
    # Keep line positions while excluding block comments, diagnostics and macro bodies.
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m[0].count("\n"), text, flags=re.S)
    text = "\n".join(re.split(r"//|;|#", line, maxsplit=1)[0] for line in text.splitlines())
    declared = set(re.findall(r"\.type\s+([^,\s]+)\s*,\s*[@%]function", text))
    declared.update(re.findall(r"\.amdgpu_hsa_kernel\s+(\S+)", text))
    declared.update(re.findall(r"\.amdhsa_kernel\s+(\S+)", text))
    records, symbols, current, macro_depth = [], set(), None, 0
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = re.split(r"//|;|#", raw, maxsplit=1)[0].strip()
        if line.startswith(".macro"):
            macro_depth += 1
        if line.startswith(".endm"):
            macro_depth = max(0, macro_depth - 1)
            continue
        if macro_depth:
            continue
        obj_label = re.fullmatch(r"[0-9a-fA-F]+\s+<([^>]+)>:", line)
        asm_label = re.fullmatch(r"([\w.$@]+):", line)
        if obj_label:
            name = obj_label[1]
            if "+0x" not in name:
                current = name
                symbols.add(name)
            continue
        if asm_label:
            name = asm_label[1]
            if name in declared or (not declared and not name.startswith((".L", "LBB", "Ltmp"))):
                current = name
                symbols.add(name)
            continue
        if line.startswith(".size"):
            current = None
        # Accept raw ISA or address + encoded words emitted by a disassembler.
        line = re.sub(r"^(?:0x)?[0-9a-fA-F]+:\s*", "", line)
        line = re.sub(r"^(?:(?:[0-9a-fA-F]{2}|[0-9a-fA-F]{8}|[0-9a-fA-F]{16})\s+)+", "", line)
        match = INSTRUCTION.match(line)
        if match:
            records.append({"line": line_no, "symbol": current, "mnemonic": match[1].lower(),
                            "text": match[1].lower() + match[2]})
    selected = None
    if symbol:
        matches = [symbol] if symbol in symbols else sorted(s for s in symbols if symbol in s)
        if len(matches) != 1:
            return {"instructions": [], "symbols": sorted(symbols), "scope": "unresolved_symbol",
                    "reason": "kernel filter must identify one assembly symbol", "matches": matches}
        selected = matches[0]
        records = [r for r in records if r["symbol"] == selected]
    return {"instructions": records, "symbols": sorted(symbols), "selected_symbol": selected,
            "scope": "selected_symbol" if selected else "whole_file", "reason": None,
            "note": "Static instruction sites only; comments/macros excluded, no macro expansion or dynamic execution count"}
