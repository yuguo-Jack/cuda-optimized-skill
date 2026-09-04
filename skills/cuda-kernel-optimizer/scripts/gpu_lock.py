"""Cross-process lock for exclusive GPU validation/profiling windows."""
from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path


@contextlib.contextmanager
def gpu_lock(gpu: int = 0, uuid: str = ""):
    key = uuid or f"index-{gpu}"
    digest = hashlib.sha256(key.encode()).hexdigest()[:24]
    path = Path(os.environ.get("CUDA_KERNEL_OPTIMIZER_LOCK_DIR", "/tmp")) / f"cko-gpu-{digest}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        try:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        try:
            yield
        finally:
            try:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except ImportError:
                pass
