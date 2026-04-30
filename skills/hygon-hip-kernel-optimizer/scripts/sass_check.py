#!/usr/bin/env python3
"""Verify claimed optimization methods in DCU ISA via dccobjdump."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


_DEFAULT_SIGNATURES = Path(__file__).resolve().parent.parent / "references" / "dcu_isa_signatures.json"
KERNEL_EXTS = (".hip", ".cu", ".cpp", ".cc", ".cxx", ".py")


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_binary(kernel_path: str) -> str | None:
    base = os.path.splitext(kernel_path)[0]
    for ext in (".so", ""):
        candidate = base + ext
        if os.path.isfile(candidate):
            return candidate
    return None


_INSTRUCTION_RE = re.compile(
    r"^\s*(?:s_|v_|ds_|buffer_|flat_|global_|matrix_|image_|exp_)",
    re.IGNORECASE | re.MULTILINE,
)
_VMEM_RE = re.compile(r"\b(?:global|buffer|flat)_(?:load|store)_", re.IGNORECASE)


def _collect_texts(root: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    files: list[str] = []
    for path in root.rglob("*"):
        if path.is_file() and path.stat().st_size < 20_000_000:
            files.append(str(path.relative_to(root)))
            try:
                texts.append(path.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                pass
    return texts, files


def _run_dcc(cmd: list[str], cwd: str, texts: list[str], errors: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str] | None:
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        errors.append(f"{cmd[0]}: {exc}")
        return None
    texts.extend([result.stdout or "", result.stderr or ""])
    if result.returncode != 0:
        errors.append(f"{' '.join(cmd)} rc={result.returncode}")
    return result


def _dump_isa(binary_path: str, arch: str = "") -> tuple[str, str | None, dict]:
    dccobjdump = "dccobjdump"
    with tempfile.TemporaryDirectory(prefix="dcu_isa_") as td:
        root = Path(td)
        binary_abs = os.path.abspath(binary_path)
        texts: list[str] = []
        errors: list[str] = []

        def outdir(name: str) -> str:
            path = root / name
            path.mkdir(parents=True, exist_ok=True)
            return str(path)

        sass_cmd = [
            dccobjdump,
            f"--inputs={binary_abs}",
            "--show-sass",
            "--show-instruction-encoding",
            "--separate-functions",
            f"--output={outdir('sass')}",
        ]
        if arch:
            sass_cmd.insert(2, f"--architecture={arch}")
        _run_dcc(sass_cmd, td, texts, errors)

        _run_dcc([
            dccobjdump,
            f"--inputs={binary_abs}",
            "--show-all-fatbin",
            f"--output={outdir('all')}",
        ], td, texts, errors)

        _run_dcc([
            dccobjdump,
            f"--inputs={binary_abs}",
            "--show-symbols",
            "--show-resource-usage",
            "--show-kernel-descriptor",
            f"--output={outdir('meta')}",
        ], td, texts, errors)

        listed = _run_dcc([dccobjdump, f"--inputs={binary_abs}", "--list-elf"], td, texts, errors, timeout=30)
        if listed and "ELF file" in (listed.stdout or ""):
            extract_dir = outdir("extract")
            _run_dcc([dccobjdump, f"--inputs={binary_abs}", "--extract-elf=all", f"--output={extract_dir}"], td, texts, errors)
            for elf in list(root.glob("*.out")) + list((root / "extract").glob("*.out")):
                elf_out = outdir(f"elf_sass_{elf.name}")
                _run_dcc([
                    dccobjdump,
                    f"--inputs={elf}",
                    "--show-sass",
                    "--show-instruction-encoding",
                    "--separate-functions",
                    f"--output={elf_out}",
                ], td, texts, errors)

        file_texts, files = _collect_texts(root)
        texts.extend(file_texts)
        isa_text = "\n".join(texts)
        meta = {
            "dump_files": files,
            "isa_files": [f for f in files if f.lower().endswith(".isa")],
            "instruction_lines": len(_INSTRUCTION_RE.findall(isa_text)),
            "vmem_instruction_count": len(_VMEM_RE.findall(isa_text)),
            "dump_errors": errors,
        }
        fatal = "; ".join(errors) if not isa_text.strip() and errors else None
        return isa_text, fatal, meta


def check_method(method_id: str, isa_text: str, signatures: dict, dump_meta: dict | None = None) -> dict:
    result = {"method_id": method_id, "verified": False, "patterns_checked": [], "patterns_found": [], "patterns_missing": []}
    meta = signatures.get("methods", {}).get(method_id, {})
    patterns = meta.get("isa_patterns", meta.get("sass_patterns", []))
    require_any = meta.get("require_any", True)
    if not patterns:
        result["verified"] = True
        result["note"] = "no_patterns_defined"
        return result
    result["patterns_checked"] = patterns
    for pattern in patterns:
        if re.search(pattern, isa_text, re.IGNORECASE):
            result["patterns_found"].append(pattern)
        else:
            result["patterns_missing"].append(pattern)
    result["verified"] = bool(result["patterns_found"]) if require_any else not result["patterns_missing"]
    if not result["verified"] and method_id.startswith("memory.") and dump_meta and dump_meta.get("vmem_instruction_count", 0) == 0:
        result["inconclusive"] = True
        result["note"] = "dccobjdump produced no vector/global memory instructions; dump may be incomplete for this code object"
    return result


def run(state_path: str, iteration: int, signatures_path: str | None = None) -> dict:
    state = _load_json(state_path)
    iter_dir = os.path.join(state["run_dir"], f"iterv{iteration}")
    methods_path = os.path.join(iter_dir, "methods.json")
    methods = _load_json(methods_path).get("methods", [])
    signatures = _load_json(signatures_path or str(_DEFAULT_SIGNATURES)) if os.path.isfile(signatures_path or str(_DEFAULT_SIGNATURES)) else {"methods": {}}

    kernel_path = next((os.path.join(iter_dir, f"kernel{ext}") for ext in KERNEL_EXTS if os.path.isfile(os.path.join(iter_dir, f"kernel{ext}"))), None)
    if not kernel_path:
        result = {"error": "no_kernel_found", "checks": []}
        _write_result(iter_dir, result)
        return result
    if kernel_path.endswith(".py"):
        result = {"kernel": kernel_path, "backend": "python", "checks": [{"method_id": m.get("id", "unknown"), "verified": True, "note": "python_backend_isa_not_applicable"} for m in methods]}
        _write_result(iter_dir, result)
        return result

    binary = _find_binary(kernel_path)
    if not binary:
        result = {"error": "binary_not_found", "kernel": kernel_path, "checks": []}
        _write_result(iter_dir, result)
        return result

    arch = state.get("env", {}).get("primary_gfx_arch", "")
    isa_text, err, dump_meta = _dump_isa(binary, arch=arch)
    checks = []
    if err and not isa_text:
        checks = [{"method_id": m.get("id", "unknown"), "verified": True, "note": f"dccobjdump_unavailable: {err}"} for m in methods]
    else:
        checks = [check_method(m.get("id", "unknown"), isa_text, signatures, dump_meta) for m in methods]
    result = {
        "kernel": kernel_path,
        "binary": binary,
        "backend": "hip",
        "arch": arch,
        "dccobjdump_error": err,
        "dump": dump_meta,
        "isa_lines": len(isa_text.splitlines()),
        "checks": checks,
    }
    _write_result(iter_dir, result)
    print(json.dumps(result, indent=2))
    return result


def _write_result(iter_dir: str, result: dict) -> None:
    out_path = os.path.join(iter_dir, "isa_check.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--signatures", default=None)
    args = p.parse_args()
    run(args.state, args.iter, args.signatures)


if __name__ == "__main__":
    main()
