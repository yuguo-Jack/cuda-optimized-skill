# Triton Investigation Workflow

Use this reference after loading the skill when the task is a TorchInductor Triton performance investigation on Hygon DCU.

## 1. Profile and Capture

Goal: identify the hot Triton kernel and collect enough artifacts to reproduce and inspect it.

Required outputs:

- kernel name and profile rank/time share;
- generated Graph IR or log snippet connecting Graph IR to the kernel name;
- current timing and shape data;
- whether autotune ran across multiple configs;
- captured folder containing `log_profile.txt`, `autotune.log`, captured kernels, input `.pt` files, generated Python code, and Triton cache dumps.

Commands:

```bash
rm -rf /tmp/torchinductor_root/*
export TORCH_LOGS="+inductor"
export TORCHINDUCTOR_TRACE=1
export TRITON_CAPTURE_DIR=./autotune_kernels
python repro.py 2>&1 | tee log_profile.txt
```

Do not force `AMDGCN_USE_BUFFER_OPS=1` for the first capture unless the target stack has already passed a smoke compile with it. If a run fails with `LLVM Translation failed` or `builtin.unrealized_conversion_cast`, rerun with `AMDGCN_USE_BUFFER_OPS` unset and record the failed buffer-op probe separately.

If no autotune happened for a kernel that should have multiple configs, first investigate the TorchInductor path before tuning the kernel.

## 2. Check Hints and AMDGCN Instructions

Goal: determine whether Triton emitted DCU-friendly memory operations.

Required outputs:

- pointer arguments from `triton_meta['signature']`;
- whether pointer arguments have `tt.divisibility` and `tt.pointer_range`;
- whether AMDGCN contains `buffer_load/store_dwordx4`, `dwordx2`, or only scalar/global/flat operations;
- whether `AMDGCN_USE_BUFFER_OPS` was enabled.

Use:

```bash
python <skill>/scripts/inspect_triton_meta.py ./autotune_kernels/<kernel>.py --json-out meta.json
python <skill>/scripts/scan_amdgcn.py ./triton_artifacts --kernel <kernel> --json-out isa_scan.json
```

Prefer final assembly evidence over source assumptions.

## 3. Tune or Reject the Triton Kernel

Goal: decide whether source-level Triton changes can improve performance.

Check:

- Are offset inputs provably non-negative? Add `tl.assume(...)`.
- Are shape or stride parameters provably divisible by 16? Add `tl.multiple_of(...)`.
- Is access contiguous enough for wide loads/stores?
- Are atomics, index-dependent gathers, or repeated loads dominating?
- Is the kernel too simple to justify a launch?
- Did Inductor scalarize a library operation such as GEMM or reduction into a poor fused pointwise/reduction kernel?

If tuning is plausible, create variants and compare standalone timing, end-to-end timing, and AMDGCN instruction families.

## 4. Avoid Low-Quality Generated Kernels

Goal: stop TorchInductor from generating the bad Triton kernel for a small model region.

Required outputs:

- model function or layer that triggers the kernel;
- minimal eager fallback or compile boundary;
- end-to-end performance before/after.

Use `@torch._dynamo.disable` only on the smallest region that removes the bad kernel. Confirm the rest of the model still benefits from `torch.compile`.

## 5. Rewrite Model Code

Goal: remove the algorithmic cause of the poor kernel.

Common rewrites:

- avoid tiny tensor initialization launches;
- fuse initialization into downstream work;
- reduce scatter/atomic operations;
- make indexing more contiguous;
- preserve library-backed matmul/convolution/reduction paths;
- change shapes or layout so Inductor specializes a better kernel.

Report whether the rewrite hits the performance target and whether it changes numerical behavior or model semantics.
