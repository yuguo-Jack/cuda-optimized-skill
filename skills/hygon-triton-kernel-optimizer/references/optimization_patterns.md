# Triton Optimization Patterns for Hygon DCU

## Buffer Operations

For memory-bound Triton kernels on DCU, first try to enable high-quality buffer operations.

Checklist:

- Probe `AMDGCN_USE_BUFFER_OPS=1`; keep it only if the target stack compiles the kernel family under investigation.
- Inspect `triton_meta['signature']` and identify pointer arguments (`*fp32`, `*bf16`, `*i64`, etc.).
- Check `tt.pointer_range` and `tt.divisibility` metadata.
- Add `tl.assume(ptr.to(tl.int64) >= 0)` when a pointer participates in offset calculation and non-negativity is guaranteed.
- Add `tl.assume(param >= 0)` for scalar shape/stride parameters used in offset calculation when non-negativity is guaranteed.
- Add `tl.multiple_of(param, 16)` only when the real value is known to be divisible by 16.

Do not add assumptions unless the contract is true for all model inputs under the compiled shape specialization.

On `torch 2.9.0` / `triton 3.3.0` / `gfx938`, pointer-to-int assumptions such as `tl.assume(ptr.to(tl.int64) >= 0)`, and even some scalar `tl.assume(...)` forms under `AMDGCN_USE_BUFFER_OPS=1`, have been observed to fail LLVM translation in a simple standalone kernel. Treat every new assumption form as experimental: compile-probe it on the target stack before using it in a real patch. If it fails, keep `tl.multiple_of(...)`, metadata fixes, or alternative address expressions instead of forcing the assumption.

Also separate Inductor-generated kernels from hand-written raw Triton templates. On the validated stack, a simple raw `@triton.autotune` vector kernel failed under `AMDGCN_USE_BUFFER_OPS=1` even after removing assumptions. Use `scripts/triton_benchmark_template.py` without buffer ops for a stable timing harness, and use generated Inductor kernels plus AMDGCN dumps when validating buffer-op conversion.

## Wider Loads and Stores

Prefer patterns that allow `buffer_load_dwordx4` or `buffer_store_dwordx4`.

Signals that wide memory operations are unlikely:

- offsets use `//`, `%`, indirect index loads, or data-dependent gathers;
- masks fragment what should be contiguous lanes;
- stores are atomic;
- block shape is too small or launch geometry creates poor coalescing;
- alignment or divisibility cannot be proven.

## Example Hint Patch

```python
@triton.jit
def kernel(value, stride_m, stride_n, out, dim: tl.constexpr, BLOCK: tl.constexpr):
    tl.assume(value.to(tl.int64) >= 0)
    tl.assume(out.to(tl.int64) >= 0)
    tl.assume(stride_m >= 0)
    stride_n = tl.multiple_of(stride_n, 16)
    # original body...
```

Then rerun standalone and inspect AMDGCN. A faster runtime without expected instruction changes may still be useful, but do not claim the hint worked through buffer-op conversion unless assembly confirms it.

## Low-Value Triton Kernels

Reject or avoid generated Triton kernels when:

- the body only zeros or copies a small tensor;
- launch overhead dominates;
- `tl.atomic_add` or scattered stores dominate and cannot be reorganized;
- index formulas make most lanes read repeated or non-contiguous addresses;
- a library operation was scalarized into a fused pointwise/reduction form;
- autotune configs are nearly identical and all far below bandwidth roofline for structural reasons.

## Model-Level Fixes

Use model changes when the generated code is algorithmically poor:

- move zero initialization into the consumer;
- batch several tiny operations together;
- replace scatter-heavy code with gather or segmented reductions when semantics allow;
- keep matrix multiplication and normalization on library-backed paths;
- disable `torch.compile` only around the specific region that triggers the bad kernel.
