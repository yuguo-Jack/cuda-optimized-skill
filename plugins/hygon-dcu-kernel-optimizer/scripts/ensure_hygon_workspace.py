#!/usr/bin/env python3
"""Prepare the target project workspace for Hygon DCU plugin work."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def ensure(root: str, tmp_name: str, update_gitignore: bool) -> dict[str, str | bool]:
    root_path = Path(root).resolve()
    tmp_path = root_path / tmp_name
    tmp_path.mkdir(parents=True, exist_ok=True)

    changed_gitignore = False
    gitignore_path = root_path / ".gitignore"
    entry = f"{tmp_name.rstrip('/')}/"
    if update_gitignore:
        existing = ""
        if gitignore_path.exists():
            existing = gitignore_path.read_text(encoding="utf-8", errors="ignore")
        lines = {line.strip() for line in existing.splitlines()}
        if entry not in lines and tmp_name.rstrip("/") not in lines:
            prefix = "" if not existing or existing.endswith(("\n", "\r")) else "\n"
            with gitignore_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(f"{prefix}{entry}\n")
            changed_gitignore = True

    return {
        "root": str(root_path),
        "tmp_dir": str(tmp_path),
        "gitignore": str(gitignore_path),
        "gitignore_updated": changed_gitignore,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensure target project has a Hygon scratch directory")
    parser.add_argument("--root", default=os.getcwd(), help="Target project root, default current directory")
    parser.add_argument("--tmp-name", default="hygon_tmp", help="Scratch directory name")
    parser.add_argument("--no-gitignore", action="store_true", help="Do not add the scratch dir to .gitignore")
    args = parser.parse_args()

    result = ensure(args.root, args.tmp_name, not args.no_gitignore)
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
