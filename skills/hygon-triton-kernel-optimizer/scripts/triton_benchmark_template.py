#!/usr/bin/env python3
"""Editable Triton benchmark template for DCU kernel experiments.

Replace the example kernel and input setup with the captured kernel under test.
This file is intentionally standalone so it can be copied into a repro folder.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": block}, num_warps=warps, num_stages=1)
        for block in (128, 256, 512, 1024)
        for warps in (1, 2, 4)
    ],
    key=["n_elements"],
)
@triton.jit
def add_scale_kernel(x, y, out, n_elements, alpha: tl.constexpr, BLOCK: tl.constexpr):
    # Keep the template conservative. Add tl.assume/tl.multiple_of only in
    # copied experiment files after compile-probing the target DCU stack.
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offsets < n_elements
    xv = tl.load(x + offsets, mask=mask, other=0.0)
    yv = tl.load(y + offsets, mask=mask, other=0.0)
    tl.store(out + offsets, xv + yv * alpha, mask=mask)


def run(n: int, dtype: str, warmup: int, repeat: int) -> dict:
    if n <= 0 or warmup < 0 or repeat <= 0:
        raise ValueError("n/repeat must be positive; warmup nonnegative")
    torch_dtype = getattr(torch, dtype)
    x = torch.randn(n, device="cuda", dtype=torch_dtype)
    y = torch.randn(n, device="cuda", dtype=torch_dtype)
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)

    def launch():
        add_scale_kernel[grid](x, y, out, n, alpha=1.0)

    for _ in range(warmup):
        launch()
    launch()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x + y, equal_nan=False)
    ms = triton.testing.do_bench(launch, warmup=0, rep=repeat)
    total_bytes = (x.numel() * x.element_size()) * 3
    gbps = total_bytes / (float(ms) / 1e3) / 1e9
    return {
        "correctness": {"checked": True, "passed": True},
        "timing_scope": "triton.testing.do_bench; repeat is milliseconds, not sample count",
        "n": n,
        "dtype": dtype,
        "amdgcn_use_buffer_ops": os.environ.get("AMDGCN_USE_BUFFER_OPS"),
        "time_ms": float(ms),
        "rough_bandwidth_gbps": gbps,
        "best_config": str(getattr(add_scale_kernel, "best_config", "")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the editable Triton benchmark template")
    parser.add_argument("--n", type=int, default=1 << 20)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50, help="do_bench measurement duration in milliseconds")
    parser.add_argument("--json-out", default="")
    parser.add_argument("--keep-buffer-ops", action="store_true", help="Do not clear AMDGCN_USE_BUFFER_OPS before compiling this raw Triton template")
    args = parser.parse_args()
    if os.environ.get("AMDGCN_USE_BUFFER_OPS") == "1" and not args.keep_buffer_ops:
        os.environ.pop("AMDGCN_USE_BUFFER_OPS", None)
        print("[warn] cleared AMDGCN_USE_BUFFER_OPS for this raw Triton template; use --keep-buffer-ops to probe that lowering")
    result = run(args.n, args.dtype, args.warmup, args.repeat)
    payload = json.dumps(result, indent=2)
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
