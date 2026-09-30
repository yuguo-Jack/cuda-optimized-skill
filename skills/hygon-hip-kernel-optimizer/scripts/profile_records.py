"""Keep before/after profiling attempts separate from interpretation and timing."""
from __future__ import annotations

import datetime
import json
from pathlib import Path

from experiment import file_sha256


def save_profile(folder, which, result, state, benchmark, update_top=True):
    folder = Path(folder)
    raw = Path(result["raw_directory"])
    raw.mkdir(parents=True, exist_ok=True)
    source = Path(result.get("profiled_file", ""))
    result.update(
        which=which,
        recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        profiled_source_sha256=file_sha256(source) if source.is_file() else None,
        benchmark_sha256=file_sha256(benchmark) if Path(benchmark).is_file() else None,
        dims=state.get("dims", {}),
        ptr_size=state.get("ptr_size", 0),
        analysis_status="not_reviewed",
    )
    # Invocation-specific records survive later attempts and role changes.
    result["record_path"] = str(raw / "profile.json")
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    (raw / "profile.json").write_text(payload, encoding="utf-8")
    (folder / f"{which}.profile.json").write_text(payload, encoding="utf-8")
    if update_top:
        (folder / "dcu_top.json").write_text(payload, encoding="utf-8")
    return result


def profile_records(run_dir):
    """Prefer all per-attempt records; support role snapshots and legacy runs."""
    for folder in sorted(Path(run_dir).glob("iterv*")):
        attempts = sorted(folder.glob("*.*/*/profile.json"))
        paths = attempts or sorted(folder.glob("*.profile.json"))
        if not paths and (folder / "dcu_top.json").is_file():
            paths = [folder / "dcu_top.json"]
        for path in paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("profile must be an object")
            except (OSError, ValueError) as exc:
                data = {"tool": "unreadable", "reason": str(exc), "collection_status": "unknown"}
            yield folder.name, path, data
