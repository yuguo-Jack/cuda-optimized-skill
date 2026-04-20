# NCU Metrics Guide

How to read the top-K metrics in `ncu_top.json` and map bottleneck signatures to optimization axes.

> **Rule of thumb**: a kernel rarely fits neatly into one bucket. Expect two of the three axes to light up simultaneously. Use the metric *values* (not just the list) to decide which axis is actually dominant; the profiler's ranking is a hint, not a verdict.

---

## Table of contents

- [Compute axis](#compute-axis)
- [Memory axis](#memory-axis)
- [Latency axis](#latency-axis)
- [Signature → optimization lookup](#signature--optimization-lookup)
- [Sanity checks](#sanity-checks)

---

## Compute axis

Does the kernel use the math pipelines the architecture actually provides?

| Metric | What it means | Good value | Bad value → try |
|---|---|---|---|
| `sm__pipe_tensor_op_hmma_cycles_active...pct_of_peak` | Tensor core (HMMA) utilization on FP16/BF16/TF32 matmul. | > 60% for GEMM-like | < 20% on a GEMM → you're not hitting tensor cores. Move to `mma.sync`/`wgmma` / CUTLASS. |
| `sm__pipe_fp32_cycles_active...pct_of_peak` | FP32 CUDA-core utilization. | > 50% for FP32-bound kernels | < 10% → either you're tensor-core bound (good) or memory bound (check mem axis). |
| `sm__inst_executed.avg.per_cycle_active` (IPC) | Warp instruction throughput. | > 2 for compute-bound | < 1 → latency bound; look at stall metrics. |
| `sm__warps_active...pct_of_peak_sustained_active` | Occupancy. | ≥ 25% is often enough | < 10% → launch more blocks, cut register/smem per block, or raise `maxregcount`. Past ~50% rarely helps. |
| `sm__cycles_active...pct_of_peak` | Fraction of SMs doing any work. | > 80% | < 50% → launch too small or serialization across SMs. |
| `launch__occupancy_limit_registers` / `launch__occupancy_limit_shared_mem` | Which resource is capping occupancy. | —   | If the limit is registers → `__launch_bounds__`, split block, spill to smem; if shared mem → shrink tile/stages. |

### Diagnostic fast path
- **Tensor op % very low on what should be a matmul** → the primary optimization *is* "use tensor cores" (via CUTLASS / Triton `dot`, or hand-rolled `mma.sync`).
- **IPC low AND tensor op % low AND occupancy high** → memory or latency-bound, not compute.
- **Occupancy already high but IPC low** → stalls dominate; look at latency axis.

---

## Memory axis

Does data move efficiently through dram → L2 → L1/smem → regs?

| Metric | What it means | Good value | Bad value → try |
|---|---|---|---|
| `dram__throughput...pct_of_peak` | HBM bandwidth used. | > 70% if memory-bound and that's unavoidable | >90% + kernel is "compute-bound in theory" → you're thrashing; improve reuse. |
| `lts__t_sector_hit_rate.pct` (L2 hit rate) | L2 cache effectiveness. | > 70% | < 30% → access pattern thrashes L2. Blocking / swizzling / data layout changes. |
| `l1tex__t_sector_hit_rate.pct` | L1 hit rate. | > 70% | < 30% → uncoalesced loads, poor reuse, or non-128B-aligned access. |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum` | Shared memory bank conflicts. | 0 is achievable | Non-zero → swizzle smem layout (XOR padding for GEMM, `+1` padding for 2D tiles). |
| `smsp__inst_executed_op_shared_{ld,st}.sum` | Smem traffic volume. | — | Very high AND bank conflicts high → review vector widths / conflicts. High with low conflicts → probably fine. |
| `dram__bytes.sum` vs theoretical minimum | Redundant dram traffic. | near minimum | 2× minimum → re-read patterns; use smem staging, cp.async, TMA. |
| `l1tex__throughput...pct_of_peak` | L1 / tex pipeline busy. | — | Saturated + bandwidth-bound → check coalescing; vectorize loads. |

### Diagnostic fast path
- **High dram % AND low L2 hit rate** → poor reuse. Remedy: tile & stage through shared memory, re-order access to increase locality.
- **Low dram % AND low L1 hit rate AND high stalls** → uncoalesced or scattered smem access. Remedy: vector loads (`float4`), coalesced global access, smem swizzling.
- **Bank conflicts dominate** → swizzle smem address mapping (the GEMM `xor` trick / `padded` row-major for 2D stencils).

---

## Latency axis

Why are warps not issuing?

`smsp__warp_issue_stalled_*` (or the `average_warp_latency_issue_stalled_*_per_warp_active.pct` family) enumerates stall reasons. The dominant one tells you the fix:

| Stall reason | Meaning | Primary remedy |
|---|---|---|
| `long_scoreboard` | Waiting for a global/LD.E/MIO op to return. | Prefetch (`cp.async`), double/triple buffer, software pipeline, increase ILP so there's independent work to overlap. |
| `short_scoreboard` | Waiting for shared memory / fast math. | More ILP; larger tiles; unroll inner loop; interleave loads/stores. |
| `barrier` | At `__syncthreads()` too much. | Reduce sync points, split block, use async `__pipeline`/`cp.async.commit_group`. |
| `mio_throttle` | LSU / tex pipe saturated. | Vectorize loads (`float4`); reduce per-warp memory ops; switch patterns to smem. |
| `lg_throttle` | L/S scheduler busy (too many in-flight). | Similar — raise granularity per instruction; vectorize. |
| `wait` | Short fixed-latency stalls between dependent instructions. | Unroll + interleave independent ops; increase register-level parallelism. |
| `drain` / `dispatch_stall` | Pipeline bubbles. | Usually downstream of other stalls; fix the bigger one first. |
| `membar` | Memory fence. | Reduce global atomics / volatile traffic. |
| `no_instruction` | Scheduler starved. | Often means too many branches / scalar work; unroll. |
| `tex_throttle` | Texture pipeline bound. | Reduce tex/ldg frequency or rebalance. |

### Diagnostic fast path
- **`long_scoreboard` dominant** → almost always wants *software pipelining + `cp.async`* (sm_80+) or *TMA + multi-stage* (sm_90+). This is the single most common big win.
- **`barrier` dominant** → async copies + fewer barriers; split-K if applicable.
- **`mio_throttle` dominant** → LSU is saturated. Widen loads (`float4`/`uint4`), move reuse to registers, batch smem ops.
- **Multiple stalls ~equal** → occupancy may be too low. Check compute axis's `launch__occupancy_limit_*`.

---

## Signature → optimization lookup

Use this as a short decision table when picking the 3 methods for the next iteration. Each row lists a plausible (axis, method) combination; pick **one per axis**.

| Signature | Compute pick | Memory pick | Latency pick |
|---|---|---|---|
| GEMM-ish, low tensor-op %, high dram % | Introduce `mma.sync` / CUTLASS tiles | Smem tiling with swizzle | `cp.async` + pipeline stages |
| Low occupancy + register-limited | `__launch_bounds__` / spill to smem | Reduce per-thread tile | Reorder to increase ILP |
| High bank conflicts | Vectorize loads | Swizzle smem layout | Unroll inner loop |
| Memory-bound elementwise | Fuse ops into one kernel | Coalesce + `float4` | Grid-stride loop w/ prefetch |
| Reduction / histogram | Warp-shuffle reduction | Smem bank-free tree | `atomicAdd` → warp-aggregated |
| Scan-ish | Decoupled lookback / Brent-Kung | Smem tile w/ padding | Two-pass to avoid global atomics |

---

## Sanity checks before you decide

1. **Is the kernel time stable?** If `bench.json` says `average_ms < 50µs`, ranking by ncu is noisy — collect more launches or increase dimensions first.
2. **Is the metric you're trusting > 0?** If ncu couldn't attach perf counters (`can_read_counters: false` in `env.json`), many pct metrics will be 0 or absent; don't conclude "compute bound" from zeros.
3. **Cross-check with the `speedup_vs_reference` field.** If you're already ~1× the reference and the reference *is* PyTorch cuBLAS, you're competing with a heavily tuned library — meaningful wins require correctness-preserving algorithmic changes, not just low-level tweaks.
