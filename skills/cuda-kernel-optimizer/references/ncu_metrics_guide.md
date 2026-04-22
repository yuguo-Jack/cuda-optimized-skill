# NCU Metrics Guide (v4 — 匹配重排后优先级)

> **Rule of thumb**: a kernel rarely fits neatly into one bucket. Expect two of the three axes to light up simultaneously.
>
> **v4 变更（2026-04-21）**: 本指南内所有"推荐方法"都指向 v4 重排后的优先级编号。新增 12 条方法触发指引。

---

## Compute axis

| Metric | What it means | Good | Bad → try |
|---|---|---|---|
| `sm__pipe_tensor_op_hmma_cycles_active...pct_of_peak` | Tensor core utilization | > 60% GEMM-like | < 20% → `compute.tensor_core` (P1) |
| `sm__pipe_fp32_cycles_active...pct_of_peak` | FP32 CUDA-core utilization | > 50% FP32-bound | High + TC low → `compute.mixed_precision` (P2) or `compute.tf32_emulation` (P12) |
| `sm__inst_executed.avg.per_cycle_active` (IPC) | Warp instruction throughput | > 2 compute-bound | < 1 → latency-bound or `compute.thread_coarsening` (P5) |
| `sm__warps_active...pct_of_peak` | Occupancy | ≥ 25% often enough | < 10% → `compute.launch_config` (P4) |
| `sm__cycles_active...pct_of_peak` | SM utilization | > 80% | < 50% → grid too small or `latency.persistent_kernel` (P5) |
| `launch__occupancy_limit_registers` / `_shared_mem` | Occupancy limiter | — | registers → `memory.register_pressure` (P10); smem → shrink tile |
| `smsp__warp_issue_stalled_wait.pct` | Wait stall (fixed-latency dep) | < 5% | > 20% + attention → `compute.gemm_softmax_interleave` (P8) |
| `smsp__inst_executed_pipe_tensor.sum` | Total TC instructions issued | high for GEMM | near 0 → TC not engaged → P1 |
| `smsp__inst_executed_pipe_xu.sum` | SFU/transcendental use | moderate | high + softmax → `compute.mufu_ex2_softmax_replacement` (P13) |

### Diagnostic fast path
- **TC % very low on GEMM** → `compute.tensor_core` (P1)
- **FP32 pipe high + TC low + precision tolerant** → `compute.mixed_precision` (P2)
- **FP32 pipe high + TC low + precision strict** → `compute.tf32_emulation_3xtf32_bf16x6` (P12)
- **barrier + long_scoreboard both high** → `compute.warp_specialization` (P3)
- **IPC low + TC low + occupancy high** → memory or latency bound
- **IPC low + no dominant stall** → per-thread work too small → `compute.thread_coarsening` (P5)
- **IPC < 2 after thread_coarsening + FP32 SGEMM** → `compute.ilp_unroll_register_tile` (P6)
- **Occupancy high but IPC low** → stalls dominate → latency axis
- **sm_100+ available + FP8 viable** → `compute.block_scaled_precision` (P11)
- **FP8 accum + high relative error** → `compute.two_level_accumulation_promotion` (P9)
- **FP8 inference + accuracy tolerant** → `compute.fp8_fast_accumulation_mode` (P10)
- **softmax/exp dominates SFU** → `compute.mufu_ex2_softmax_replacement` (P13)

---

## Memory axis

| Metric | What it means | Good | Bad → try |
|---|---|---|---|
| `dram__throughput...pct_of_peak` | HBM bandwidth used | > 70% if unavoidable | >90% + should-be-compute → improve reuse |
| `lts__t_sector_hit_rate.pct` (L2 hit) | L2 cache effectiveness | > 70% | < 30% → blocking/swizzle → `latency.tile_scheduler_swizzle` (P4) |
| `l1tex__t_sector_hit_rate.pct` | L1 hit rate | > 70% | < 30% → uncoalesced → `memory.coalesced_access` (P2) |
| `l1tex__data_bank_conflicts_*` | Smem bank conflicts | 0 | > 0 → `memory.bank_conflict` (P8) or `memory.tma_descriptor_tuning` (P12) |
| `dram__bytes.sum` vs theoretical min | Redundant DRAM | near min | 2× → staging/pipeline; N× with cross-CTA sharing → `memory.tma_multicast` (P11) |
| `dram__bytes_write.sum` | Write-back traffic | ≈ output size | ≫ output → intermediate matrices → `memory.epilogue_visitor_tree_fusion` (P13) or `memory.epilogue_fusion` (P7) |
| `dram__bytes_read.sum.per_second` / `.peak_sustained` | HBM read bandwidth rate | near peak if mem-bound | far below → `memory.vectorized_access` (P4) |
| `smsp__warp_issue_stalled_mio_throttle.pct` | LSU throttle | < 5% | > 20% → `memory.vectorized_access` (P4) or `memory.ldmatrix_stmatrix` (P9) |
| LDG.E.32 count in SASS | Non-vectorized loads | 0 | > 0 → `memory.vectorized_access` (P4) |

### Diagnostic fast path
- **High DRAM % + low L2 hit** → `memory.tiling_smem` (P3) + `latency.tile_scheduler_swizzle` (P4)
- **Low DRAM % + low L1 hit + high stalls** → `memory.coalesced_access` (P2) + `memory.vectorized_access` (P4)
- **dram__bytes.sum ≫ min + cross-CTA sharing** → `memory.tma_multicast` (P11, sm_90+)
- **Write-back includes intermediate matrices + simple chain** → `memory.epilogue_fusion` (P7)
- **Write-back includes complex DAG (scale+bias+act+reduce)** → `memory.epilogue_visitor_tree_fusion` (P13, sm_90+)
- **bank_conflict > 0 + TMA in use** → `memory.tma_descriptor_tuning` (P12, sm_90+)
- **bank_conflict > 0 + no TMA** → `memory.bank_conflict` (P8)
- **mio_throttle high + TC kernel** → `memory.ldmatrix_stmatrix` (P9, sm_75+)
- **mio_throttle high + non-TC** → `memory.vectorized_access` (P4)
- **Data > 228KB per CTA but < Cluster total** → `memory.distributed_smem` (P21, sm_90+)
- **Small hot data (KV cache) + low L2 hit** → `memory.l2_persistence_window` (P16, sm_80+)
- **Streaming read polluting L1** → `memory.ldg_cache_modifier_hint` (P15)
- **Tall-skinny GEMM (M×N small, K large) + SM underutilized** → `memory.split_k_parallel_reduce` (P17)
- **Conv kernel + explicit im2col step** → `memory.tma_im2col_implicit_conv` (P20, sm_90+)
- **Register spill > 0 + smem has headroom + CUDA≥13** → `memory.smem_register_spilling` (P19, sm_80+)

---

## Latency axis

| Stall reason | Meaning | Primary remedy |
|---|---|---|
| `long_scoreboard` | Waiting on global/MIO load | `latency.async_pipeline` (P1) + prefetch |
| `short_scoreboard` | Waiting on smem/fast math | ILP / unroll / `compute.thread_coarsening` (P5) |
| `barrier` | At `__syncthreads()` | `latency.reduce_sync_count` (P2) or `latency.asymmetric_mbarrier_sync` (P9, sm_90+) |
| `mio_throttle` | LSU saturated | `memory.vectorized_access` (P4) |
| `lg_throttle` | L/S scheduler busy | Vectorize; batch smem ops |
| `wait` | Short fixed-latency deps | Unroll + interleave, or `latency.intra_wg_gemm_softmax_pipeline` (P12) |
| `membar` | Memory fence | Reduce atomics → `latency.warp_aggregated_atomics` (P8) or `latency.atomic_optimize` (P19) |
| `no_instruction` | Scheduler starved | Unroll; reduce branches |

### Diagnostic fast path
- **`long_scoreboard` dominant** → `latency.async_pipeline` (P1) or TMA + multi-stage (sm_90+)
- **`barrier` dominant + sm < 90** → `latency.reduce_sync_count` (P2) + async copies
- **`barrier` dominant + asymmetric producer/consumer + sm_90+** → `latency.asymmetric_mbarrier_sync` (P9)
- **`mio_throttle` dominant** → `memory.vectorized_access` (P4)
- **`membar` dominant + atomic-heavy** → `latency.warp_aggregated_atomics` (P8)
- **Nsight Systems inter-kernel gaps** → `latency.pdl_overlap` (P13, sm_90+)
- **L2 hit rate low on GEMM + large grid** → `latency.tile_scheduler_swizzle` (P4)
- **Wave quantization > 10%** → `latency.persistent_kernel` (P5) or `latency.stream_k_load_balancing` (P6)
- **dram__bytes includes O(N²) attention scores** → `latency.online_recomputation` (P7)
- **Attention GEMM-softmax ping-pong candidate + sm_90+** → `latency.pingpong_warpgroup_schedule` (P10) + `latency.intra_wg_gemm_softmax_pipeline` (P12)
- **Short kernel in tight loop + fixed shape + launch overhead** → `latency.cuda_graphs` (P16)
- **Short kernel in tight loop + variable shape + launch overhead** → `latency.static_launch_grid_graph` (P15)
- **sm_100+ + persistent kernel + atomic contention on work stealing** → `latency.cluster_launch_control_scheduler` (P17)
- **WS kernel + consumer register spill** → `latency.producer_regdealloc_setmaxnreg` (P11)

---

## Quick-Reference: Kernel Archetype → Top-3 Methods (v4)

| Archetype | Compute | Memory | Latency |
|---|---|---|---|
| GEMM, FP16/BF16 | `compute.tensor_core` (P1) | `memory.tiling_smem` (P3) | `latency.async_pipeline` (P1) |
| FP32 GEMM, precision-tolerant | `compute.mixed_precision` (P2) | `memory.vectorized_access` (P4) | `latency.async_pipeline` (P1) |
| FP32 GEMM, precision-strict | `compute.tf32_emulation` (P12) | `memory.tiling_smem` (P3) | `latency.async_pipeline` (P1) |
| Attention / Softmax + GEMM (FA3 style) | `compute.gemm_softmax_interleave` (P8) + `compute.mufu_ex2` (P13) | `memory.kernel_fusion` (P1) | `latency.online_recomputation` (P7) + `latency.pingpong_warpgroup` (P10) + `latency.intra_wg_pipeline` (P12) |
| FP8 GEMM (training) | `compute.two_level_accumulation_promotion` (P9) | `memory.tma_multicast` (P11) | `latency.pingpong_warpgroup_schedule` (P10) |
| FP8 GEMM (inference) | `compute.fp8_fast_accumulation_mode` (P10) | `memory.tma_multicast` (P11) | `latency.pingpong_warpgroup_schedule` (P10) |
| GEMM + bias + activation chain (simple) | `compute.tensor_core` (P1) | `memory.epilogue_fusion` (P7) | `latency.tile_scheduler_swizzle` (P4) |
| GEMM + complex epilogue DAG | `compute.tensor_core` (P1) | `memory.epilogue_visitor_tree_fusion` (P13) | `latency.tile_scheduler_swizzle` (P4) |
| Memory-bound elementwise | `compute.thread_coarsening` (P5) | `memory.kernel_fusion` (P1) + `memory.vectorized_access` (P4) | `latency.tile_scheduler_swizzle` (P4) |
| Large GEMM, low L2 hit, Hopper+ | `compute.warp_specialization` (P3) | `memory.tma_multicast` (P11) | `latency.stream_k_load_balancing` (P6) |
| Tall-skinny GEMM (large K, small M×N) | `compute.tensor_core` (P1) | `memory.split_k_parallel_reduce` (P17) | `latency.async_pipeline` (P1) |
| Multi-kernel pipeline with gaps | `compute.overlap_compute_memory` (P7) | `memory.epilogue_fusion` (P7) | `latency.pdl_overlap` (P13) + `latency.cuda_graphs` (P16) |
| W4A16 quantized inference | `compute.lop3_bit_manipulation` (P18) | `memory.kernel_fusion` (P1) | `latency.persistent_kernel` (P5) |
| Blackwell FP4/FP8 GEMM | `compute.block_scaled_precision` (P11) | `memory.tma_descriptor_tuning` (P12) | `latency.cluster_launch_control_scheduler` (P17) |
| Histogram / filter / counting | `compute.warp_shuffle` (P15) | `memory.kernel_fusion` (P1) | `latency.warp_aggregated_atomics` (P8) |
| Convolution (Hopper) | `compute.tensor_core` (P1) | `memory.tma_im2col_implicit_conv` (P20) | `latency.tile_scheduler_swizzle` (P4) |
| Iterative app with many small kernels | `compute.overlap_compute_memory` (P7) | `memory.kernel_fusion` (P1) | `latency.cuda_graphs` (P16) or `latency.static_launch_grid_graph` (P15) |

---

## Per-Method Trigger Summary (v4 全量，按轴)

### Compute axis
| Method ID | Priority | Trigger condition |
|---|---|---|
| `compute.tensor_core` | P1 | TC utilization < 20% on GEMM |
| `compute.mixed_precision` | P2 | FP32 high + TC low + precision tolerant |
| `compute.warp_specialization` | P3 | barrier + long_scoreboard both high; sm_80+ |
| `compute.launch_config` | P4 | occupancy < 25% or SM util < 50% |
| `compute.thread_coarsening` | P5 | 1 elem/thread + IPC low |
| `compute.ilp_unroll_register_tile` | P6 | IPC < 2 after coarsening; no TC |
| `compute.overlap_compute_memory` | P7 | IPC < 1 + low eligible warps + no multi-stage |
| `compute.gemm_softmax_interleave` | P8 | Attention + wait stall + sm_90+ |
| `compute.two_level_accumulation_promotion` | P9 | FP8 + rel error > 0.5% |
| `compute.fp8_fast_accumulation_mode` | P10 | FP8 + accuracy tolerant |
| `compute.block_scaled_precision` | P11 | sm_100+; FP16 TC underutilized; precision allows |
| `compute.tf32_emulation_3xtf32_bf16x6` | P12 | FP32 precision + TC < 15% |
| `compute.mufu_ex2_softmax_replacement` | P13 | softmax/exp dominant |
| `compute.reduction` | P14 | Reduction > 30% of runtime |
| `compute.warp_shuffle` | P15 | Warp-scope data exchange via smem |
| `compute.loop_unroll` | P16 | short_scoreboard / wait stall |
| `compute.fma_and_fast_math` | P17 | MUL+ADD as separate ops + fast_math OK |
| `compute.lop3_bit_manipulation` | P18 | W4A16/W8A16 inference |
| `compute.pragma_no_alias` | P19 | No __restrict__ + NVCC blocks vectorize |
| `compute.strength_reduce` | P20 | IDIV/IMOD in hot loop |
| `compute.branch_eliminate` | P21 | Warp Exec Efficiency low |
| `compute.handwritten_ptx_inline` | P22 | Last resort, all else exhausted |

### Memory axis
| Method ID | Priority | Trigger condition |
|---|---|---|
| `memory.kernel_fusion` | P1 | dram__bytes.sum ≫ theoretical min |
| `memory.coalesced_access` | P2 | Global Load/Store Eff < 50% |
| `memory.tiling_smem` | P3 | L2 hit < 30% with reuse pattern |
| `memory.vectorized_access` | P4 | mio_throttle high or LDG.E.32 in SASS |
| `memory.async_copy` | P5 | Stall Long Scoreboard dominant; sm_80+ |
| `memory.multi_stage_pipeline` | P6 | Stall Long Scoreboard; single buffer |
| `memory.epilogue_fusion` | P7 | dram_bytes_write > output size |
| `memory.bank_conflict` | P8 | bank conflict counter > 0 |
| `memory.ldmatrix_stmatrix` | P9 | TC kernel + mio_throttle high; sm_75+ |
| `memory.register_pressure` | P10 | occupancy_limit_registers + spill > 0 |
| `memory.tma_multicast` | P11 | dram_bytes ≫ min + cross-CTA sharing; sm_90+ |
| `memory.tma_descriptor_tuning` | P12 | TMA + bank conflict > 0 or wrong swizzle |
| `memory.epilogue_visitor_tree_fusion` | P13 | Complex epilogue DAG + sm_90+ |
| `memory.data_layout` | P14 | AoS / misaligned |
| `memory.ldg_cache_modifier_hint` | P15 | Streaming read polluting L1 |
| `memory.l2_persistence_window` | P16 | Small hot data + low L2 hit; sm_80+ |
| `memory.split_k_parallel_reduce` | P17 | Tall-skinny GEMM + SM util < 50% |
| `memory.cache_and_readonly` | P18 | Small hot data |
| `memory.smem_register_spilling` | P19 | Spill > 0 + smem headroom + CUDA≥13 |
| `memory.tma_im2col_implicit_conv` | P20 | Convolution + sm_90+ |
| `memory.distributed_smem` | P21 | Data > 228KB per CTA; sm_90+ |
| `memory.tensor_map_rebuild_avoid` | P22 | TMA descriptor rebuild in tight loop; sm_90+ |

### Latency axis
| Method ID | Priority | Trigger condition |
|---|---|---|
| `latency.async_pipeline` | P1 | Stall Long Scoreboard dominant |
| `latency.reduce_sync_count` | P2 | Stall Barrier + multiple __syncthreads |
| `latency.warp_shuffle_sync` | P3 | Warp-scope data exchange via smem |
| `latency.tile_scheduler_swizzle` | P4 | L2 hit < 50% on GEMM, grid > 100 |
| `latency.persistent_kernel` | P5 | SM cycles < 50% + wave quantization |
| `latency.stream_k_load_balancing` | P6 | Tail effect > 10% |
| `latency.online_recomputation` | P7 | O(N²) materialization + global reduce |
| `latency.warp_aggregated_atomics` | P8 | atomic throughput limited |
| `latency.asymmetric_mbarrier_sync` | P9 | Barrier stall + asymmetric P/C; sm_90+ |
| `latency.pingpong_warpgroup_schedule` | P10 | sm_90+ + attention/GEMM+softmax |
| `latency.producer_regdealloc_setmaxnreg` | P11 | WS + consumer spill; sm_90+ |
| `latency.intra_wg_gemm_softmax_pipeline` | P12 | Softmax→GEMM2 dep exposed; sm_90+ |
| `latency.pdl_overlap` | P13 | Inter-kernel gaps; sm_90+ |
| `latency.cooperative_cluster` | P14 | Fine-grain sync needs |
| `latency.static_launch_grid_graph` | P15 | Short kernel + variable shape |
| `latency.cuda_graphs` | P16 | Launch overhead > 10% + fixed shape |
| `latency.cluster_launch_control_scheduler` | P17 | sm_100+ persistent + atomic contention |
| `latency.stream_event` | P18 | Multi-independent kernels serialized |
| `latency.atomic_optimize` | P19 | High-contention atomics (fallback) |

---

## Notes

1. **Metric names** are for Nsight Compute 2024.3+. Older NCU versions may use slightly different names.
2. **sm_90+**: includes H100/H200 (sm_90), Blackwell (sm_100). GeForce RTX 50xx is **sm_120** and **does not have TMA/TMEM/WGMMA** despite being newer.
3. **sm_100+**: includes B100/B200 (sm_100). **Not** RTX 5090 (sm_120) — different feature set.
4. **mixed precision** → always re-validate atol/rtol after switching.
5. **FP8 fast_accum vs two_level_accum** → mutually exclusive; check application tolerance first.
6. **Stream-K vs Split-K** → mutually exclusive; pick based on whether you want work-stealing (Stream-K) or simple K-parallel (Split-K).
7. **Pingpong WG + Intra-WG pipeline** → complementary (FA3 uses both); but count as one latency slot when combined.
8. When in doubt, check `sm_arch` explicitly: H200=sm_90, B200=sm_100, RTX 5090=sm_120.
