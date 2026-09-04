#!/usr/bin/env python3
"""Generic small/medium/large workload generation and resource fitting.

The benchmark contract is intentionally backend agnostic: a workload entry
contains dimensions and an optional pointer-element override.  Archetype
specific generators can be added later without changing branch selection.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


SCALE_WEIGHTS = {"small": 0.20, "medium": 0.30, "large": 0.50}
DEFAULT_TARGET_MB = 384
DEFAULT_HARD_MB = 512

# These are intentionally conservative launch shapes.  A caller-provided
# dimension remains authoritative; defaults are only used when an archetype
# has no complete shape to scale from.
ARCHETYPE_DEFAULTS: dict[str, dict[str, dict[str, int]]] = {
    "attention": {
        "small": {"B": 1, "H": 2, "S": 128, "D": 64},
        "medium": {"B": 2, "H": 4, "S": 512, "D": 64},
        "large": {"B": 2, "H": 8, "S": 2048, "D": 64},
    },
    "gemm": {
        "small": {"M": 256, "N": 256, "K": 256},
        "medium": {"M": 1024, "N": 1024, "K": 1024},
        "large": {"M": 4096, "N": 4096, "K": 4096},
    },
    "reduction": {
        "small": {"N": 4096, "M": 256},
        "medium": {"N": 16384, "M": 512},
        "large": {"N": 65536, "M": 1024},
    },
    "normalization": {
        "small": {"B": 1, "S": 128, "D": 256},
        "medium": {"B": 8, "S": 512, "D": 768},
        "large": {"B": 16, "S": 2048, "D": 1024},
    },
    "elementwise": {
        "small": {"N": 1 << 16},
        "medium": {"N": 1 << 20},
        "large": {"N": 1 << 24},
    },
    "convolution": {
        "small": {"N": 1, "H": 56, "W": 56, "C": 64},
        "medium": {"N": 4, "H": 112, "W": 112, "C": 128},
        "large": {"N": 8, "H": 224, "W": 224, "C": 256},
    },
    "quantized": {
        "small": {"M": 256, "N": 256, "K": 256},
        "medium": {"M": 1024, "N": 1024, "K": 1024},
        "large": {"M": 4096, "N": 4096, "K": 4096},
    },
    "sparse": {
        "small": {"M": 256, "N": 256, "K": 256},
        "medium": {"M": 1024, "N": 1024, "K": 1024},
        "large": {"M": 4096, "N": 4096, "K": 4096},
    },
    "moe": {
        "small": {"T": 256, "E": 8, "D": 512},
        "medium": {"T": 2048, "E": 16, "D": 1024},
        "large": {"T": 8192, "E": 32, "D": 2048},
    },
}

ARCHETYPE_SCALE_AXES = {
    "attention": ["S", "B", "H"],
    "attention_decode": ["S", "B", "H"],
    "attention_prefill": ["S", "B", "H"],
    "gemm": ["M", "N", "K"],
    "quantized": ["M", "N", "K"],
    "sparse": ["M", "N", "K"],
    "moe": ["T", "E"],
}

ARCHETYPE_MIN_DIMS = {
    "attention": {"B": 1, "H": 1, "S": 32, "D": 16},
    "attention_decode": {"B": 1, "H": 1, "S": 1, "D": 16},
    "attention_prefill": {"B": 1, "H": 1, "S": 32, "D": 16},
    "gemm": {"M": 16, "N": 16, "K": 16},
    "quantized": {"M": 16, "N": 16, "K": 32},
    "sparse": {"M": 16, "N": 16, "K": 32},
    "moe": {"T": 32, "E": 1, "D": 16},
}


def _align_down(value: int, alignment: int) -> int:
    if alignment <= 1:
        return max(1, int(value))
    aligned = (int(value) // alignment) * alignment
    return max(1, aligned)


def _infer_axes(dims: dict[str, int], archetype: str) -> list[str]:
    keys = {str(k).lower(): k for k in dims}
    preferred = tuple(a.lower() for a in ARCHETYPE_SCALE_AXES.get(archetype.lower(), [])) or {
        "batched_small_gemm": ("m", "n", "b"),
        "convolution": ("n", "h", "w", "c"),
        "reduction": ("n", "m", "l"),
    }.get(archetype.lower(), ())
    axes = [keys[name] for name in preferred if name in keys]
    if axes:
        return axes
    numeric = sorted(dims, key=lambda k: int(dims[k]), reverse=True)
    return numeric[: max(1, min(3, len(numeric)))]


def estimate_working_set_bytes(
    dims: dict[str, int],
    ptr_size: int = 0,
    pointer_count: int = 4,
    dtype_bytes: int = 4,
    workspace_bytes: int = 0,
) -> int:
    """Conservative estimate used before backend setup.

    ``ptr_size`` is the per-pointer element count used by the existing CUDA
    ABI.  When absent, the product of dimensions is used as a useful generic
    estimate.  The exact peak allocation remains recorded by benchmark.py.
    """
    elems = int(ptr_size) if ptr_size and ptr_size > 0 else 1
    if not ptr_size:
        for value in dims.values():
            if isinstance(value, int) and value > 0:
                elems *= value
    return int(elems * max(1, pointer_count) * max(1, dtype_bytes) + workspace_bytes)


def _scaled_dims(base: dict[str, int], axes: list[str], factor: float, alignment: int) -> dict[str, int]:
    out = dict(base)
    for axis in axes:
        value = int(base[axis])
        if value <= 1:
            continue
        out[axis] = min(value, _align_down(max(1, math.floor(value * factor)), alignment))
    return out


def fit_workload(
    requested_dims: dict[str, int],
    requested_ptr_size: int,
    *,
    archetype: str = "generic",
    target_mb: int = DEFAULT_TARGET_MB,
    hard_mb: int = DEFAULT_HARD_MB,
    scale_axes: list[str] | None = None,
    alignment: int = 1,
    pointer_count: int = 4,
    dtype_bytes: int = 4,
    workspace_bytes: int = 0,
    enforce_target: bool = False,
    minimum_dims: dict[str, int] | None = None,
    requested_shape_not_executed: bool = False,
) -> dict[str, Any]:
    """Fit a requested case to the hard limit, preserving valid alignment."""
    base = {str(k): int(v) for k, v in requested_dims.items()}
    axes = [a for a in (scale_axes or _infer_axes(base, archetype)) if a in base]
    requested_bytes = estimate_working_set_bytes(
        base, requested_ptr_size, pointer_count, dtype_bytes, workspace_bytes)
    hard_bytes = max(1, int(hard_mb)) * 1024 * 1024
    target_bytes = max(1, int(target_mb)) * 1024 * 1024
    limit = hard_bytes
    factor = 1.0
    target_exceeded = bool(enforce_target and requested_bytes > target_bytes)
    hard_exceeded = requested_bytes > limit
    if hard_exceeded or target_exceeded:
        lo, hi = 0.01, 1.0
        desired = target_bytes if target_exceeded else hard_bytes
        for _ in range(28):
            mid = (lo + hi) / 2.0
            candidate = _scaled_dims(base, axes, mid, alignment)
            candidate_bytes = estimate_working_set_bytes(
                candidate,
                max(1, int(requested_ptr_size * (mid ** max(1, len(axes))))) if requested_ptr_size else 0,
                pointer_count, dtype_bytes, workspace_bytes)
            if candidate_bytes <= desired:
                factor, lo = mid, mid
            else:
                hi = mid
        dims = _scaled_dims(base, axes, factor, alignment)
    else:
        dims = dict(base)

    ratio = 1.0
    if requested_ptr_size:
        old = max(1, math.prod(max(1, int(base[a])) for a in axes))
        new = max(1, math.prod(max(1, int(dims[a])) for a in axes))
        ratio = min(1.0, new / old)
        ptr_size = max(1, int(requested_ptr_size * ratio))
    else:
        ptr_size = 0
    realized_bytes = estimate_working_set_bytes(
        dims, ptr_size, pointer_count, dtype_bytes, workspace_bytes)
    # The pointer ABI may represent a product larger than the named scale
    # axes. Enforce the hard limit on the final estimate as a last guard.
    retries = 0
    while realized_bytes > hard_bytes and axes and retries < 24:
        dims = _scaled_dims(dims, axes, 0.8, alignment)
        old_ptr = ptr_size
        ptr_size = max(1, int(ptr_size * 0.8 ** max(1, len(axes)))) if ptr_size else 0
        realized_bytes = estimate_working_set_bytes(
            dims, ptr_size, pointer_count, dtype_bytes, workspace_bytes)
        retries += 1
    if realized_bytes > hard_bytes:
        # No scalable axis or an irreducible workspace dominates the request.
        realized_bytes = estimate_working_set_bytes(
            dims, ptr_size, pointer_count, dtype_bytes, workspace_bytes)
    resource_unavailable = realized_bytes > hard_bytes
    downscaled = dims != base or (requested_ptr_size and ptr_size != requested_ptr_size)
    min_dims = minimum_dims or ARCHETYPE_MIN_DIMS.get(archetype.lower(), {})
    below_minimum = any(int(dims.get(k, v)) < int(v) for k, v in min_dims.items())
    reason = None
    if downscaled:
        reason = "working_set_target" if target_exceeded else "hard_working_set_limit"
    if requested_shape_not_executed:
        reason = "requested_shape_not_executed"
    return {
        "requested_dims": base,
        "realized_dims": dims,
        "requested_ptr_size": int(requested_ptr_size or 0),
        "realized_ptr_size": ptr_size,
        "requested_working_set_bytes": requested_bytes,
        "realized_working_set_bytes": realized_bytes,
        "downscaled": bool(downscaled),
        "downscale_reason": reason,
        "requested_shape_not_executed": bool(requested_shape_not_executed),
        "below_minimum": below_minimum,
        "minimum_dims": min_dims,
        "resource_unavailable": resource_unavailable,
        "archetype": archetype,
        "scale_axes": axes,
        "alignment": alignment,
    }


def generate_workload_matrix(
    dims: dict[str, int],
    ptr_size: int = 0,
    *,
    archetype: str = "generic",
    target_mb: int = DEFAULT_TARGET_MB,
    hard_mb: int = DEFAULT_HARD_MB,
    scale_axes: list[str] | None = None,
    alignment: int = 1,
    pointer_count: int = 4,
    dtype_bytes: int = 4,
    workspace_bytes: int = 0,
    profile: str | None = None,
    minimum_dims: dict[str, int] | None = None,
    adaptive_downscale: bool = True,
) -> list[dict[str, Any]]:
    """Return ordered small/medium/large entries.

    A JSON profile may contain ``scales`` with explicit dimension dictionaries;
    those are still fitted to the same resource limits.
    """
    if profile:
        payload = json.loads(Path(profile).read_text(encoding="utf-8"))
        explicit = payload.get("scales", payload) if isinstance(payload, dict) else {}
        if isinstance(payload, dict):
            scale_axes = payload.get("scale_axes", scale_axes)
            alignment = int(payload.get("alignment", alignment))
            minimum_dims = payload.get("minimum_dims", minimum_dims)
    else:
        explicit = {}
    archetype_key = archetype.lower()
    defaults = ARCHETYPE_DEFAULTS.get(archetype_key, {})
    # A complete user shape is the large reference.  For an empty/partial
    # shape, fill missing values from the archetype's canonical matrix.
    supplied = {str(k): int(v) for k, v in dims.items()}
    complete_user_shape = bool(supplied) and all(v > 0 for v in supplied.values())
    factors = {"small": 0.5, "medium": 0.75, "large": 1.0}
    entries = []
    for scale in ("small", "medium", "large"):
        if scale in explicit:
            requested = {str(k): int(v) for k, v in explicit[scale].items()}
            auto_generated = False
        elif defaults and not complete_user_shape:
            requested = dict(defaults.get(scale, defaults.get("large", {})))
            requested.update(supplied)
            auto_generated = True
        else:
            source = supplied
            requested = _scaled_dims(source, _infer_axes(source, archetype), factors[scale], alignment)
            auto_generated = True
        requested_shape_not_executed = False
        requested_bytes = estimate_working_set_bytes(
            requested, ptr_size, pointer_count, dtype_bytes, workspace_bytes)
        if requested_bytes > int(hard_mb) * 1024 * 1024:
            requested_shape_not_executed = True
        requested_ptr = ptr_size
        if ptr_size and requested != supplied:
            axes = scale_axes or _infer_axes(supplied or requested, archetype)
            old = max(1, math.prod(max(1, int(supplied.get(a, requested.get(a, 1)))) for a in axes if a in supplied or a in requested))
            new = max(1, math.prod(max(1, int(requested.get(a, supplied.get(a, 1)))) for a in axes if a in supplied or a in requested))
            requested_ptr = max(1, int(ptr_size * min(1.0, new / old)))
        fitted = fit_workload(
            requested, requested_ptr, archetype=archetype, target_mb=target_mb,
            hard_mb=hard_mb, scale_axes=scale_axes, alignment=alignment,
            pointer_count=pointer_count, dtype_bytes=dtype_bytes,
            workspace_bytes=workspace_bytes,
            enforce_target=(adaptive_downscale and auto_generated and scale == "large"),
            minimum_dims=minimum_dims,
            requested_shape_not_executed=requested_shape_not_executed)
        fitted["scale"] = scale
        fitted["weight"] = SCALE_WEIGHTS[scale]
        fitted["auto_generated"] = auto_generated
        fitted["effective_scale"] = ("medium" if scale == "large" and fitted.get("below_minimum") else scale)
        fitted["scale_degraded"] = fitted["effective_scale"] != scale
        entries.append(fitted)
    return entries


def shrink_workload(entry: dict[str, Any], ratio: float = 0.8) -> dict[str, Any]:
    """Return a retry entry with scalable axes reduced and realigned."""
    out = dict(entry)
    dims = dict(entry.get("realized_dims", {}))
    axes = [a for a in entry.get("scale_axes", []) if a in dims]
    alignment = int(entry.get("alignment", 1))
    shrunk = _scaled_dims(dims, axes, float(ratio), alignment)
    old_product = max(1, math.prod(max(1, int(dims[a])) for a in axes))
    new_product = max(1, math.prod(max(1, int(shrunk[a])) for a in axes))
    old_ptr = int(entry.get("realized_ptr_size", 0))
    new_ptr = max(1, int(old_ptr * min(1.0, new_product / old_product))) if old_ptr else 0
    old_bytes = int(entry.get("realized_working_set_bytes", 0))
    out.update({
        "realized_dims": shrunk,
        "realized_ptr_size": new_ptr,
        "realized_working_set_bytes": max(1, int(old_bytes * min(1.0, new_product / old_product))),
        "downscaled": True,
        "downscale_reason": "runtime_oom_retry",
    })
    return out
