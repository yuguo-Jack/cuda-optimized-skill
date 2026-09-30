#!/usr/bin/env python3
"""CPU-only CUDA/CUTLASS build service used by the optimizer.

The build result is content addressed within a run.  Runtime execution is
intentionally outside this module so callers can compile concurrently and
still validate and time kernels serially on one GPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


_STRIP_INCLUDES = re.compile(r"^\s*#\s*include\s*<__clang_cuda[^>]*>\s*$", re.MULTILINE)


def _json_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _dependency_digest(source: str, include_dirs: list[str]) -> str:
    """Hash reachable local headers so header edits invalidate the run cache."""
    seen: set[str] = set()
    chunks: list[tuple[str, str]] = []
    queue = [os.path.abspath(source)]
    while queue:
        path = queue.pop()
        if path in seen or not os.path.isfile(path):
            continue
        seen.add(path)
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        chunks.append((path, hashlib.sha256(text.encode()).hexdigest()))
        for match in re.finditer(r"^\s*#\s*include\s*[\"<]([^\">]+)[\">]", text, re.MULTILINE):
            name = match.group(1)
            candidates = [os.path.join(os.path.dirname(path), name)]
            candidates.extend(os.path.join(d, name) for d in include_dirs)
            found = next((os.path.abspath(c) for c in candidates if os.path.isfile(c)), None)
            if found:
                queue.append(found)
    return _json_hash(sorted(chunks))


def _compiler_identity(nvcc: str) -> str:
    real = shutil.which(nvcc) or nvcc
    try:
        r = subprocess.run([real, "--version"], capture_output=True, text=True,
                           encoding="utf-8", errors="ignore", timeout=15)
        version = (r.stdout or r.stderr or "").strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        version = f"error:{exc}"
    return f"{os.path.realpath(real)}\n{version}\n{platform.platform()}"


def _cutlass_include() -> str:
    values = []
    root = os.environ.get("CUTLASS_PATH", "").strip()
    include = os.environ.get("CUTLASS_INCLUDE_DIR", "").strip()
    if root:
        values.extend([root, os.path.join(root, "include")])
    if include:
        values.append(include)
    values.extend(sorted(Path("/usr/local").glob("cutlass*/include")))
    values.extend([Path("/usr/local/cutlass/include"), Path("/opt/cutlass/include")])
    for value in values:
        value = os.path.abspath(str(value))
        if os.path.isdir(os.path.join(value, "cutlass")) and os.path.isdir(os.path.join(value, "cute")):
            return value
    return ""


def _clean_source(source: str, work_dir: str) -> str:
    text = Path(source).read_text(encoding="utf-8", errors="ignore")
    cleaned = _STRIP_INCLUDES.sub("", text)
    if cleaned == text:
        return source
    target = os.path.join(work_dir, "kernel.nvcc_clean.cu")
    Path(target).write_text(cleaned, encoding="utf-8")
    return target


def _base_command(source: str, output: str, arch: str, nvcc: str, backend: str) -> list[str]:
    cmd = [nvcc, "-Xcompiler", "-fPIC"]
    if backend == "cutlass":
        include = _cutlass_include()
        if not include:
            raise RuntimeError("CUTLASS headers not found (set CUTLASS_PATH or CUTLASS_INCLUDE_DIR)")
        cmd += ["-I", include]
    raw = Path(source).read_text(encoding="utf-8", errors="ignore")
    if "#include <cublas_v2.h>" in raw or "#include <cublasLt.h>" in raw:
        cmd += ["-lcublas", "-lcublasLt"]
    cmd += ["-shared", "-std=c++17", f"-arch={arch}", "-O3", "-o", output, source]
    return cmd


@dataclass(frozen=True)
class BuildSpec:
    source: str
    backend: str
    arch: str
    nvcc: str

    def command(self, output: str, source_for_compile: str | None = None) -> list[str]:
        return _base_command(source_for_compile or self.source, output, self.arch, self.nvcc, self.backend)

    def fingerprint(self) -> str:
        source_bytes = Path(self.source).read_bytes()
        include = _cutlass_include() if self.backend == "cutlass" else ""
        include_dirs = [include] if include else []
        include_dirs.append(os.path.dirname(os.path.abspath(self.source)))
        payload = {
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "backend": self.backend,
            "arch": self.arch,
            "nvcc_identity": _compiler_identity(self.nvcc),
            "include": include,
            "dependency_digest": _dependency_digest(self.source, include_dirs),
            "flags": ["-Xcompiler", "-fPIC", "-shared", "-std=c++17", f"-arch={self.arch}", "-O3"],
        }
        return _json_hash(payload)


def _manifest_ok(manifest_path: str, spec: BuildSpec) -> bool:
    try:
        data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        return (data.get("fingerprint") == spec.fingerprint()
                and os.path.isfile(data.get("artifact", "")))
    except (OSError, json.JSONDecodeError):
        return False


def _build_unlocked(spec: BuildSpec, cache_dir: str, *, force: bool = False) -> dict:
    fingerprint = spec.fingerprint()
    entry = Path(cache_dir) / fingerprint
    artifact = entry / ("kernel.dll" if os.name == "nt" else "kernel.so")
    manifest = entry / "manifest.json"
    log_path = entry / "compile.log"
    entry.mkdir(parents=True, exist_ok=True)
    if not force and manifest.is_file() and _manifest_ok(str(manifest), spec):
        return {"ok": True, "artifact": str(artifact), "manifest": str(manifest),
                "fingerprint": fingerprint, "cache_hit": True, "compile_wall_ms": 0.0}

    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="cko-build-", dir=str(entry)) as tmp:
        clean = _clean_source(spec.source, tmp)
        temp_artifact = os.path.join(tmp, artifact.name)
        try:
            cmd = spec.command(temp_artifact, clean)
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    encoding="utf-8", errors="ignore")
            log = (result.stdout or "") + "\n---STDERR---\n" + (result.stderr or "")
            log_path.write_text(log, encoding="utf-8")
            if result.returncode != 0 or not os.path.isfile(temp_artifact):
                return {"ok": False, "fingerprint": fingerprint, "cache_hit": False,
                        "returncode": result.returncode, "error": (result.stderr or "")[-4000:],
                        "compile_wall_ms": round((time.monotonic() - start) * 1000, 2),
                        "manifest": str(manifest)}
            os.replace(temp_artifact, artifact)
        except (OSError, RuntimeError) as exc:
            log_path.write_text(str(exc), encoding="utf-8")
            return {"ok": False, "fingerprint": fingerprint, "cache_hit": False,
                    "returncode": -1, "error": str(exc),
                    "compile_wall_ms": round((time.monotonic() - start) * 1000, 2),
                    "manifest": str(manifest)}

    payload = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "source": os.path.abspath(spec.source),
        "source_sha256": hashlib.sha256(Path(spec.source).read_bytes()).hexdigest(),
        "backend": spec.backend,
        "arch": spec.arch,
        "nvcc": os.path.realpath(shutil.which(spec.nvcc) or spec.nvcc),
        "artifact": str(artifact),
        "command_flags": spec.command("<artifact>")[1:-2],
        "created_at": time.time(),
    }
    tmp_manifest = str(manifest) + ".tmp"
    Path(tmp_manifest).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_manifest, manifest)
    return {"ok": True, "artifact": str(artifact), "manifest": str(manifest),
            "fingerprint": fingerprint, "cache_hit": False,
            "compile_wall_ms": round((time.monotonic() - start) * 1000, 2)}


def build(spec: BuildSpec, cache_dir: str, *, force: bool = False) -> dict:
    """Build once per fingerprint even when workers request the same key."""
    fingerprint = spec.fingerprint()
    entry = Path(cache_dir) / fingerprint
    entry.mkdir(parents=True, exist_ok=True)
    lock_path = entry / ".lock"
    try:
        import fcntl
        with open(lock_path, "w", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            return _build_unlocked(spec, cache_dir, force=force)
    except ImportError:
        return _build_unlocked(spec, cache_dir, force=force)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="CPU-only CUDA/CUTLASS build helper")
    p.add_argument("source")
    p.add_argument("--backend", default="cuda", choices=["cuda", "cutlass"])
    p.add_argument("--arch", required=True)
    p.add_argument("--nvcc", default="nvcc")
    p.add_argument("--cache-dir", required=True)
    args = p.parse_args()
    result = build(BuildSpec(os.path.abspath(args.source), args.backend, args.arch, args.nvcc), args.cache_dir)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
