#!/usr/bin/env python3
"""Back up and copy only the three Hygon Skills; verify every source-file hash."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

NAMES = ("hygon-hip-baseline-generator", "hygon-hip-kernel-optimizer", "hygon-triton-kernel-optimizer")
ROOT = Path(__file__).resolve().parents[1]


def manifest(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def install(skills_dir, backup_root=None):
    target = Path(skills_dir).resolve()
    source = (ROOT / "skills").resolve()
    backup_base = Path(backup_root or ROOT / "hygon_tmp/skill-backups").resolve()
    if target == source or source in target.parents or target in source.parents:
        raise ValueError("Installed directory must be separate from repository skills")
    if target == backup_base or target in backup_base.parents or backup_base in target.parents:
        raise ValueError("Backup and install directories must be separate")
    backup = backup_base / datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for name in NAMES:
        src, dst = source / name, target / name
        if not (src / "SKILL.md").is_file():
            raise FileNotFoundError(src / "SKILL.md")
        if dst.resolve().parent != target or any(p.is_symlink() for p in dst.rglob("*")) or dst.is_symlink():
            raise ValueError(f"Refusing linked install path: {dst}")
    backup.mkdir(parents=True)
    report = {"target": str(target), "backup": str(backup), "skills": {}}
    # Back up all old versions before changing any of them.
    for name in NAMES:
        dst = target / name
        if dst.exists():
            shutil.copytree(dst, backup / name)
    for name in NAMES:
        src, dst = source / name, target / name
        before = manifest(dst) if dst.exists() else {}
        expected = manifest(src)
        for rel in expected:
            dest = dst / rel
            if not dest.resolve().is_relative_to(target):
                raise ValueError(f"Destination escapes Skill directory: {dest}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, dest)
        after = manifest(dst)
        if any(after.get(k) != v for k, v in expected.items()):
            raise RuntimeError(f"Installed hash verification failed: {name}; backup at {backup}")
        # Preserve user-local extras instead of deleting files without need.
        report["skills"][name] = {"files_verified": len(expected), "previous": before,
                                 "installed": expected, "preserved_extra_files": sorted(set(after) - set(expected))}
    (backup / "install-manifest.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skills-dir", default=str(Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "skills"))
    parser.add_argument("--backup-root", default="")
    args = parser.parse_args()
    report = install(args.skills_dir, args.backup_root or None)
    print(json.dumps({"target": report["target"], "backup": report["backup"],
                      "skills": {name: {"files_verified": info["files_verified"], "preserved_extra_files": info["preserved_extra_files"]}
                                 for name, info in report["skills"].items()}}, indent=2))


if __name__ == "__main__":
    main()
