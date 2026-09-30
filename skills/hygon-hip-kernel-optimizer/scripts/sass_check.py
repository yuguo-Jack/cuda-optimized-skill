#!/usr/bin/env python3
"""Verify claimed optimization methods in DCU ISA via dccobjdump."""

from __future__ import annotations

import argparse
import hashlib
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from isa import parse as parse_isa


_DEFAULT_SIGNATURES = Path(__file__).resolve().parent.parent / "references" / "dcu_isa_signatures.json"
KERNEL_EXTS = (".hip", ".cu", ".cpp", ".cc", ".cxx", ".py")


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
                content = path.read_text(encoding="utf-8")
                if "\x00" not in content:
                    texts.append(content)
            except (OSError, UnicodeError):
                pass
    return texts, files


def _collect_assembly_texts(root: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    files: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.stat().st_size >= 20_000_000:
            continue
        if path.suffix.lower() not in {".s", ".isa", ".asm"}:
            continue
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
    if result.returncode == 0 and "--show-sass" in cmd:
        texts.append(result.stdout or "")
    if result.returncode != 0:
        errors.append(f"{' '.join(cmd)} rc={result.returncode}")
    return result


def _detect_backend(path: str) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return "hip"
    return "ck_tile" if ("ck_tile/" in text or "ck_tile::" in text) else "hip"


def _find_ck_tile_include_dir() -> str:
    candidates: list[str] = []
    for var in ("CK_TILE_PATH", "CK_TILE_INCLUDE_DIR", "CK_PATH", "COMPOSABLE_KERNEL_PATH"):
        value = os.environ.get(var, "").strip()
        if value:
            candidates.extend([value, os.path.join(value, "include")])
    candidates.extend(sorted(glob.glob("/opt/dtk*/**/include", recursive=True)))
    candidates.extend(sorted(glob.glob("/opt/rocm*/**/include", recursive=True)))
    candidates.extend(["/opt/dtk/include", "/opt/rocm/include", "/usr/local/include"])
    seen: set[str] = set()
    for candidate in candidates:
        resolved = os.path.abspath(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        if os.path.isdir(os.path.join(resolved, "ck_tile")):
            return resolved
    return ""


def _compile_save_temps_isa(kernel_path: str, arch: str = "") -> tuple[str, dict]:
    hipcc = os.environ.get("HIPCC", "hipcc")
    with tempfile.TemporaryDirectory(prefix="dcu_save_temps_") as td:
        root = Path(td)
        source_abs = os.path.abspath(kernel_path)
        output_so = root / "save_temps.so"
        cmd = [hipcc, "-fPIC", "-shared", "-std=c++17", "-O3"]
        if arch:
            cmd.append(f"--offload-arch={arch}")
        if os.path.splitext(kernel_path)[1].lower() in {".cpp", ".cc", ".cxx"}:
            cmd.extend(["-x", "hip"])
        if _detect_backend(kernel_path) == "ck_tile":
            include_dir = _find_ck_tile_include_dir()
            if include_dir:
                cmd.extend(["-I", include_dir])
        cmd.extend(["-save-temps=obj", "-o", str(output_so), source_abs])

        try:
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(source_abs) or ".",
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=120,
            )
            log = (result.stdout or "") + "\n---STDERR---\n" + (result.stderr or "")
            returncode = result.returncode
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
            log = str(exc)
            returncode = -1

        texts, files = _collect_assembly_texts(root)
        asm_text = "\n".join(texts)
        meta = {
            "attempted": True,
            "hipcc": shutil.which(hipcc) or hipcc,
            "command": " ".join(str(part) for part in cmd),
            "returncode": returncode,
            "assembly_files": files,
            "instruction_lines": len(_INSTRUCTION_RE.findall(asm_text)),
            "vmem_instruction_count": len(_VMEM_RE.findall(asm_text)),
            "log_excerpt": log.strip()[:2000],
        }
        return asm_text, meta


def _dump_isa(binary_path: str, arch: str = "", kernel_path: str | None = None) -> tuple[str, str | None, dict]:
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

        _, files = _collect_texts(root)
        # Prefer one dump path. show-all-fatbin repeats the same instructions;
        # binary ELF and diagnostic/resource output must not inflate the scan.
        file_texts, _ = _collect_texts(root / "sass")
        file_texts = [t for t in file_texts if parse_isa(t)["instructions"]]
        if not file_texts:
            for folder in sorted(root.glob("elf_sass_*")):
                found, _ = _collect_texts(folder)
                file_texts.extend(t for t in found if parse_isa(t)["instructions"])
        if not file_texts:
            file_texts = [t for t in texts if parse_isa(t)["instructions"]]
        texts = list(dict.fromkeys(file_texts))
        dcc_text = "\n".join(texts)
        save_temps_meta: dict = {"attempted": False}
        if kernel_path and not parse_isa(dcc_text)["instructions"]:
            save_temps_text, save_temps_meta = _compile_save_temps_isa(kernel_path, arch=arch)
            if save_temps_text:
                texts.append(save_temps_text)

        isa_text = "\n".join(texts)
        meta = {
            "evidence_kind": "executed_binary_dump" if parse_isa(dcc_text)["instructions"] else
                             "auxiliary_recompile" if parse_isa(isa_text)["instructions"] else "unavailable",
            "scope": "module; select the actual kernel symbol before drawing per-kernel conclusions",
            "dump_files": files,
            "isa_files": [f for f in files if f.lower().endswith(".isa")],
            "dccobjdump_instruction_lines": len(parse_isa(dcc_text)["instructions"]),
            "instruction_lines": len(parse_isa(isa_text)["instructions"]),
            "vmem_instruction_count": sum(bool(_VMEM_RE.search(r["mnemonic"])) for r in parse_isa(isa_text)["instructions"]),
            "dump_errors": errors,
            "save_temps": save_temps_meta,
        }
        fatal = "; ".join(errors) if meta["instruction_lines"] == 0 and errors else None
        return isa_text, fatal, meta


def check_method(method_id: str, isa_text: str, signatures: dict, dump_meta: dict | None = None) -> dict:
    result = {"method_id": method_id, "verified": False, "patterns_checked": [], "patterns_found": [], "patterns_missing": []}
    meta = signatures.get("methods", {}).get(method_id, {})
    patterns = meta.get("isa_patterns", meta.get("sass_patterns", []))
    require_any = meta.get("require_any", True)
    if not patterns:
        result["inconclusive"] = True
        result["note"] = "no_patterns_defined; inspect the appropriate source/resource/timeline evidence"
        return result
    result["patterns_checked"] = patterns
    # Builtin declarations, comments, metadata and diagnostic logs are not ISA.
    isa_text = "\n".join(r["text"] for r in parse_isa(isa_text)["instructions"])
    for pattern in patterns:
        if re.search(pattern, isa_text, re.IGNORECASE | re.MULTILINE):
            result["patterns_found"].append(pattern)
        else:
            result["patterns_missing"].append(pattern)
    result["verified"] = bool(result["patterns_found"]) if require_any else not result["patterns_missing"]
    if not result["verified"] and method_id.startswith("memory.") and dump_meta and dump_meta.get("vmem_instruction_count", 0) == 0:
        result["inconclusive"] = True
        result["note"] = "dccobjdump produced no vector/global memory instructions; dump may be incomplete for this code object"
    result["pattern_presence"] = result["verified"]
    result["verified"] = False
    result["inconclusive"] = True
    result["note"] = "Pattern scan is supporting evidence; verify exact kernel, target and mechanism in mechanism-review.json"
    return result


def run(state_path: str, iteration: int, signatures_path: str | None = None) -> dict:
    state = _load_json(state_path)
    from experiment import require_open_iteration, check_frozen_inputs, iteration_kernel
    require_open_iteration(state, iteration)
    check_frozen_inputs(state)
    iter_dir = os.path.join(state["run_dir"], f"iterv{iteration}")
    methods_path = os.path.join(iter_dir, "methods.json")
    methods = _load_json(methods_path).get("methods", [])
    signatures = _load_json(signatures_path or str(_DEFAULT_SIGNATURES)) if os.path.isfile(signatures_path or str(_DEFAULT_SIGNATURES)) else {"methods": {}}

    kernel_path = iteration_kernel(iter_dir)
    if kernel_path.endswith(".py"):
        result = {"kernel": kernel_path, "backend": "python", "checks": [{"method_id": m.get("id", "unknown"), "verified": False, "inconclusive": True, "note": "python backend requires actual generated kernel ISA/resource evidence"} for m in methods]}
        _write_result(iter_dir, result)
        return result

    bench_path = Path(iter_dir) / "bench.json"
    bench = _load_json(str(bench_path)) if bench_path.is_file() else {}
    build = bench.get("build") or {}
    binary = build.get("binary")
    from experiment import file_sha256
    if (bench.get("source_sha256") != file_sha256(kernel_path) or build.get("source_sha256") != file_sha256(kernel_path) or not binary
            or not Path(binary).is_file() or build.get("binary_sha256") != file_sha256(binary)):
        result = {"error": "measured_binary_missing_or_changed", "kernel": kernel_path, "checks": [],
                  "note": "Inspect the benchmark build record; a neighboring .so is not evidence of the executed binary"}
        _write_result(iter_dir, result)
        return result

    arch = build.get("arch") or bench.get("arch") or state.get("env", {}).get("primary_gfx_arch", "")
    isa_text, err, dump_meta = _dump_isa(binary, arch=arch, kernel_path=kernel_path)
    Path(iter_dir, "isa_dump.txt").write_text(isa_text, encoding="utf-8")
    checks = []
    if err and not isa_text:
        checks = [{"method_id": m.get("id", "unknown"), "verified": False, "inconclusive": True, "note": f"dccobjdump_unavailable: {err}"} for m in methods]
    else:
        checks = [check_method(m.get("id", "unknown"), isa_text, signatures, dump_meta) for m in methods]
    result = {
        "kernel": kernel_path,
        "binary": binary,
        "binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
        "isa_artifact": str(Path(iter_dir, "isa_dump.txt")),
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
