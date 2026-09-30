#!/usr/bin/env python3
"""Collect TorchInductor Triton artifacts for a kernel investigation."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import uuid
from pathlib import Path


OUTPUT_CODE_RE = re.compile(r"Output code written to:\s*(?P<path>\S+)")
TRITON_CACHE_RE = re.compile(r"Triton cache dir:\s*(?P<path>\S+)")
BEST_CONFIG_RE = re.compile(r"Save heuristic tuning result to\s+(?P<path>\S+\.best_config)")


def _copy_file(src: Path, dst_dir: Path) -> str | None:
    if not src.is_file():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        stem = dst.stem
        suffix = dst.suffix
        dst = dst_dir / f"{stem}_{uuid.uuid4().hex}{suffix}"
    shutil.copy2(src, dst)
    return str(dst)


def _copy_tree(src: Path, dst_dir: Path) -> str | None:
    if dst_dir.resolve() == src.resolve() or src.resolve() in dst_dir.resolve().parents:
        raise ValueError("Artifact output cannot be inside its source cache/capture directory")
    if not src.is_dir():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        dst = dst_dir / f"{src.name}_{uuid.uuid4().hex}"
    shutil.copytree(src, dst)
    return str(dst)


def _paths_from_log(log_path: Path) -> dict[str, list[str]]:
    found = {"output_code": [], "triton_cache_dirs": [], "best_config": []}
    if not log_path.is_file():
        return found
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    found["output_code"] = list(dict.fromkeys(m.group("path") for m in OUTPUT_CODE_RE.finditer(text)))
    found["triton_cache_dirs"] = list(dict.fromkeys(m.group("path") for m in TRITON_CACHE_RE.finditer(text)))
    found["best_config"] = list(dict.fromkeys(m.group("path") for m in BEST_CONFIG_RE.finditer(text)))
    return found


def _find_kernel_cache_matches(cache_root: Path, kernel: str) -> list[Path]:
    if not kernel or not cache_root.is_dir():
        return []
    matches: list[Path] = []
    for path in cache_root.rglob("*"):
        if not path.is_file():
            continue
        if kernel in path.name:
            matches.append(path)
            continue
        if path.suffix.lower() in {".amdgcn", ".ttir", ".ttgir", ".ll", ".s", ".py"}:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if kernel in text:
                matches.append(path)
    return matches


def collect(args) -> dict:
    out = Path(args.out).resolve()
    log_path = Path(args.log) if args.log else None
    paths = _paths_from_log(log_path) if log_path else {"output_code": [], "triton_cache_dirs": [], "best_config": []}
    for root in [args.capture_dir, args.cache_root, *paths["triton_cache_dirs"]]:
        if root and (out == Path(root).resolve() or Path(root).resolve() in out.parents):
            raise ValueError("Artifact output cannot be inside a source cache/capture directory")
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "out": str(out),
        "kernel": args.kernel,
        "copied": {"logs": [], "capture_dir": None, "output_code": [], "cache_dirs": [], "best_config": [], "kernel_matches": []},
        "missing": [],
    }

    if log_path and log_path.is_file():
        copied = _copy_file(log_path, out / "logs")
        if copied:
            manifest["copied"]["logs"].append(copied)
    elif log_path:
        manifest["missing"].append(str(log_path))

    capture_dir = Path(args.capture_dir) if args.capture_dir else None
    copied_tree = _copy_tree(capture_dir, out) if capture_dir else None
    if copied_tree:
        manifest["copied"]["capture_dir"] = copied_tree
    elif args.capture_dir:
        manifest["missing"].append(str(capture_dir))

    for p in paths["output_code"]:
        copied = _copy_file(Path(p), out / "generated_python")
        if copied:
            manifest["copied"]["output_code"].append(copied)
        else:
            manifest["missing"].append(p)

    for p in paths["best_config"]:
        copied = _copy_file(Path(p), out / "best_config")
        if copied:
            manifest["copied"]["best_config"].append(copied)
        else:
            manifest["missing"].append(p)

    for p in paths["triton_cache_dirs"]:
        copied = _copy_tree(Path(p), out / "triton_cache_dirs")
        if copied:
            manifest["copied"]["cache_dirs"].append(copied)
        else:
            manifest["missing"].append(p)

    for p in _find_kernel_cache_matches(Path(args.cache_root), args.kernel):
        dst_dir = out / "kernel_matches" / p.parent.name
        copied = _copy_file(p, dst_dir)
        if copied:
            manifest["copied"]["kernel_matches"].append(copied)

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect TorchInductor Triton artifacts")
    parser.add_argument("--log", default="./log_profile.txt")
    parser.add_argument("--capture-dir", default="./autotune_kernels")
    parser.add_argument("--cache-root", default=os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_root"))
    parser.add_argument("--kernel", default="")
    parser.add_argument("--out", default="./triton_artifacts")
    args = parser.parse_args()
    manifest = collect(args)
    print(json.dumps({
        "out": manifest["out"],
        "manifest": manifest["manifest"],
        "kernel_matches": len(manifest["copied"]["kernel_matches"]),
        "missing": len(manifest["missing"]),
    }, indent=2))


if __name__ == "__main__":
    main()
