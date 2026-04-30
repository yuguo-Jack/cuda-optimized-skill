# Hygon HIP Optimization Catalog

Use this catalog with `dcu_top.json`, `roofline.json`, `state.json`, `isa_check.json`, and the current best kernel. Scan each axis from P1 downward. Do not skip a higher-priority method unless its skip condition applies, the method was already tried, the architecture is incompatible, the required feature is unavailable, or profiler/ISA evidence does not trigger it.

Important boundary: AMD ROCm and CK Tile guidance is useful because Hygon DCU is close to AMD-style HIP/GPU programming, but Hygon tensorcore paths have diverged. For matrix-core methods, treat Hygon `MMOP` / `v_mmac_*` and compiled HCU CK Tile examples as authoritative. AMD `MFMA` names are only analogies unless the Hygon toolchain emits the matching HCU/MMAC ISA in `dccobjdump`.

Evidence levels:

- Final proof is target ISA from `dccobjdump`: `v_mmac_*`, `MMOP`, `ds_read_m32x*`, `matrix_load*`, `buffer/global ... lds`, wait counters, and resource usage.
- Source-backed builtins from the DCU KB are valid implementation candidates only when copied with their exact source-level signature and target guard. A generic `__has_builtin` probe is not enough to reject them unless it compiles the exact source-backed call shape for the target architecture.
- Do not invent builtin names from AMD or spreadsheet rows. Start from the cited Hygon example, compile the minimal kernel, then inspect the emitted ISA.

Also: do not add FP4 strategies for Hygon DCU unless future hardware/toolchain evidence appears. Existing Hygon material covers FP8/BF8/TF32/FP16/BF16/INT8/INT4-style paths, not FP4 hardware acceleration.

## Global Selection Rules

1. Prefer an existing CK Tile or HCU example path before writing a custom low-level kernel.
2. Change one tuning dimension per branch: geometry, vector width, LDS layout, pipeline, scheduler, epilogue, or precision.
3. For `gfx936/gfx938`, assume wavefront size 64 and re-check every CUDA warp32 heuristic.
4. For HCU/MMAC paths, validate the whole data path: global load -> LDS/tile staging -> matrix read -> MMAC -> epilogue.
5. A source builtin is not proof. Confirm final ISA with `dccobjdump`, then confirm speed with benchmark and `hipprof`.
6. A method with no relevant ISA because the dump is incomplete should be marked inconclusive, not failed, if correctness and timing still support the branch.

## Compute Axis

### P1: `compute.mmac_tensor_core` - HCU MMOP / MMAC Matrix Core

- Trigger: GEMM, convolution, attention, MoE, or batched matmul semantics exist but `SQ_INSTS_MMOP` is low, runtime is compute-bound, or `dccobjdump` shows only VALU/FMA instead of `v_mmac_*`.
- Skip: elementwise/reduction-only kernels, unsupported datatype, or the current best already clearly uses the right HCU MMAC path.
- Implement:
  - Prefer CK Tile/HCU examples for GEMM, conv, grouped GEMM, fused conv, and MoE.
  - For custom code, use verified CK Tile/HCU examples first; only use hand-written asm after a minimal compile probe.
  - Source-backed MMAC candidates include AMD-named gfx936 microbench builtins (`__builtin_amdgcn_mmac_f32_16x16x8f32`, `__builtin_amdgcn_mmac_f32_16x16x8tf32`, `__builtin_amdgcn_mmac_f32_16x16x16f16`, `__builtin_amdgcn_mmac_i32_16x16x32i8`) and gfx938 HCU FP8/BF8 builtins in `paged_attention_938.cu`.
  - Do not accept AMD MFMA-oriented builtin or mnemonic names as proof; verify Hygon lowering instead of assuming equivalence.
- Verify:
  - `dccobjdump` shows `v_mmac_*` or the expected HCU MMOP form.
  - LDS/matrix-read layout matches the MMAC operand contract.
  - Timing and compute utilization improve.

### P2: `compute.mixed_precision_fp8_bf8` - FP8/BF8/TF32 Hygon Low-Precision Path

- Trigger: `gfx938`, tolerant accuracy, FP32/VALU dominates, or memory bandwidth is dominated by FP16/BF16 tensors that could be stored as FP8/BF8.
- Skip: strict FP32/FP64 accuracy, unsupported target, absent scale metadata, or no reference tolerance update.
- Implement:
  - Prefer a compiled CK Tile low-precision path when available.
  - For hand-coded gfx938 FP8/BF8, copy the exact `paged_attention_938.cu` forms: `__builtin_hcu_cvt_f32_fp8(val, false, 0, lane)`, `__builtin_hcu_cvt_f32_bf8(val, false, 0, lane)`, `__builtin_hcu_cvt_pk_fp8_f32(v1, v2, val, high)`, `__builtin_hcu_cvt_pk_bf8_f32(v1, v2, val, high)`, `__builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(reg_a, reg_b, reg_c, false, false)`, and `__builtin_hcu_mmac_f32_16x16x32_bf8_bf8_lit_lts(...)`.
  - Validate with a source-backed compile probe and `dccobjdump`; the generic `__has_builtin` result is not decisive.
  - Use TF32 conversion modes only when the numerical policy is explicit.
- Verify:
  - `dccobjdump` shows `fp8`, `bf8`, `tf32`, conversion, pack, or low-precision MMAC forms.
  - `atol/rtol` is justified in `analysis.md`.
  - Do not use FP4: Hygon DCU has no confirmed FP4 hardware path in the current target environment.

### P3: `compute.launch_config_wave64` - Wave64 Launch Geometry

- Trigger: low `SQ_WAVES`, low CU activity, small grid, block size not a multiple of 64, or uneven tail work.
- Skip: occupancy is adequate and the kernel is limited by memory traffic or a specialized pipeline.
- Implement: tune block sizes such as 64/128/256/512, keep full waves active, and use `__launch_bounds__` when it helps register allocation.
- Verify: benchmark improves and profiler shows better waves/CU activity without VGPR/LDS pressure regression.

### P4: `compute.thread_coarsening` - Thread Coarsening / Register Tile

- Trigger: one element per thread, repeated address arithmetic, low arithmetic per byte, or small per-thread work.
- Skip: VGPR spills, occupancy collapse, or memory access becomes non-coalesced.
- Implement: process 2-8 elements per thread, use register tiles, manually scalarize tiny arrays, and unroll fixed loops.
- Verify: instruction count or global requests drop, no spill/regression appears, benchmark improves.

### P5: `compute.register_pressure_control` - VGPR/SGPR Pressure Control

- Trigger: low occupancy caused by registers, `hipcc --resource-usage` or `dccobjdump` resource dump shows high VGPR/SGPR use, or local arrays spill.
- Skip: kernel is already pipeline-saturated at low occupancy, especially a healthy MMAC kernel.
- Implement: scalarize arrays, reduce live ranges, split long expressions, move rarely reused temporary storage to LDS, tune `__launch_bounds__`.
- Verify: resource usage drops or occupancy improves without extra memory traffic.

### P6: `compute.fast_math_intrinsics` - Fast Math / Special Function Replacement

- Trigger: transcendental/division/sqrt operations dominate and tolerance allows approximate forms.
- Skip: strict numerical requirements, integer-exact code, or SFU is not bottleneck.
- Implement: use HIP fast math intrinsics, reciprocal/multiply replacements, or precomputed constants.
- Verify: correctness tolerance is explicit and profiler shows lower special-function pressure or runtime.

### P7: `compute.inline_asm_builtin` - Inline ASM / Low-Level Escape Hatch

- Trigger: source-level HIP/CK Tile change cannot express the intended path, and `dccobjdump` proves the compiler emits a weak sequence.
- Skip: an existing CK Tile/HCU path or already-proven low-level path can express the same optimization.
- Implement: use a small `asm volatile` block only after compiling a minimal probe. Keep constraints and clobbers minimal and documented.
- Verify: `dccobjdump` shows the intended mnemonic, wait counters are correct, and a small correctness test passes.

## Memory Axis

### P1: `memory.coalesced_access` - Coalesced Global Memory Access

- Trigger: adjacent lanes access strided/scattered memory, TCC/TCP request counters are high, or effective bandwidth is low.
- Skip: lane-to-address mapping is already contiguous and aligned.
- Implement: remap thread indices, use SoA or packed layouts for hot fields, and align fastest-changing dimension with adjacent wave lanes.
- Verify: lower request pressure and faster benchmark.

### P2: `memory.vectorized_global_access` - Vectorized Global Load/Store

- Trigger: bandwidth-bound streaming kernel emits scalar loads/stores for contiguous aligned data.
- Skip: alignment is unsafe, vector tails dominate, or compiler already emits packed memory ops.
- Implement: use `float2`/`float4`, packed integer vector types, or CK Tile vector load traits. Keep scalar tail path correct.
- Verify: `dccobjdump` shows `global_load_dwordx2/x4`, `buffer_load_dwordx*`, or equivalent packed forms.

### P3: `memory.aligned_layout_transform` - Layout / Stride Transform for Access Locality

- Trigger: logical layout forces poor memory order, KV/cache/conv tensor layout does not match the hot kernel, or extra transpose kernels dominate.
- Skip: transformed tensor is single-use and transform cost exceeds savings.
- Implement: choose cache/tensor strides that make hot dimensions contiguous, use aligned allocation/strides, and prefer direct target layout writes.
- Verify: fewer layout conversions, better coalescing, and end-to-end timing improves.

### P4: `memory.lds_tiling` - LDS Tiling and Data Reuse

- Trigger: reusable data is repeatedly loaded from global memory, or arithmetic intensity can rise by staging tiles.
- Skip: pure streaming elementwise kernels with little reuse.
- Implement: stage tiles into `__shared__`/LDS, use cooperative loads, synchronize only where needed, compute from LDS/registers.
- Verify: `ds_read`/`ds_write` appears and global request count drops.

### P5: `memory.global_to_lds_async` - Direct Global-to-LDS / Buffer-Load-LDS Path

- Trigger: matrix/tile kernels stage global data through VGPRs before LDS, causing register pressure or instruction overhead.
- Skip: simple scalar kernel, unsupported target, or CK Tile loader already emits the intended path.
- Implement: prefer Hygon/CK Tile loader paths such as TLS/MLS/WASP. For custom code, `ds_read_m32x16_b16_buffer_load_dword.cpp` provides a source-backed `__builtin_amdgcn_raw_buffer_load_lds(...)` wrapper with an address-space(3) LDS destination and descriptor metadata.
- Validate that exact wrapper with target compilation before using it in a new kernel; do not rely on `__has_builtin`.
- Verify: `dccobjdump` shows `buffer_load_* ... lds` or another compiled direct-to-LDS staging form, and VGPR pressure does not increase.

### P6: `memory.matrix_load_mls` - CK Tile MLS / Tile Staging

- Trigger: GEMM/attention/conv tile wants matrix-formatted data and scalar LDS/global staging is expensive.
- Skip: non-matrix data layout or unsupported header/toolchain path.
- Implement:
  - Prefer CK Tile/HCU MLS examples that compile in the target DTK.
  - Source-backed standalone examples include `__builtin_hcu_matrix_load_32x16_b16(rscr, address_space(3) short*, offset, t, r, sw, flags)` plus `__builtin_hcu_ds_read_matrix_trans_format_u16(...)`, and `__builtin_hcu_matrix_load_b8(addr, 128, 0, 1, 0, 0, 0, 0)`.
  - Use descriptor filter/zero-pad modes for boundary tiles only when the selected compiled path exposes them.
  - Treat flags such as `t`, `r`, `sw`, `glc`, `slc` as ISA fields when they appear in compiled code or tool output.
- Verify: `matrix_load*`, MLS, or an equivalent compiled staging pattern appears and correctness holds on edge tiles.

### P7: `memory.ds_read_matrix_layout` - DS Matrix Read Layout Contract

- Trigger: MMAC operands are in LDS but loaded with scalar LDS operations or wrong matrix layout.
- Skip: no matrix core use, or CK Tile already emits the right matrix-read path.
- Implement: use the compiled inline-asm forms `ds_read_m32x16_b16`, `ds_read_m32x16_b16_alt`, or `ds_read_m32x32_b8`. For int4 or b32 paths, the gfx936 manual also documents `DS_READ_M32X64_B4` and `DS_READ_M32X8_B32`; treat them as architecture patterns that still need a minimal compile probe before use.
- Verify: `ds_read_m*` or HCU matrix-format read appears, and numerical mapping validates on small unique-value tensors.

### P8: `memory.lds_bank_conflict` - LDS Bank Conflict Reduction / Swizzle

- Trigger: `SQ_LDS_BANK_CONFLICT`, SQTT stalls, or matrix-read layout is conflict-heavy.
- Skip: no LDS or conflicts are already low.
- Implement: pad or swizzle LDS layout, use XOR/Morton-style mapping, and align the swizzle with the selected matrix-read form.
- Verify: bank-conflict counters drop, `ds_read*`/`ds_write*` pattern remains valid, benchmark improves.

### P9: `memory.cache_policy_glc_slc` - Cache Policy / Coherency Modifier Tuning

- Trigger: streaming traffic pollutes cache, synchronization flags require visibility, or L2-oriented path is intentional.
- Skip: cache reuse is high and default policy already works.
- Implement: use `glc`/`slc` flags or CK/HCU cache modifier hooks only when profiler and correctness need them.
- Verify: `dccobjdump` shows `glc`/`slc` on the intended memory ops and timing/correctness improve.

### P10: `memory.ck_tile_named_pipeline` - CK Tile Named Pipeline Selection

- Trigger: operator maps to CK Tile/HCU examples and named paths exist: TLS, MLS, WASP, cshuffle, wavelet, persistent, split-k, preshuffle, dsreadm.
- Skip: CK Tile headers unavailable, operator shape unsupported, or one named path is already proven best.
- Implement: instantiate one named variant per branch and keep the rest unchanged.
- Verify: `IsSupportedArgument` passes, benchmark selects a winner, and ISA evidence matches the path.

### P11: `memory.epilogue_fusion` - Epilogue / Post-Op Fusion

- Trigger: GEMM/conv/norm output is immediately followed by bias, add, activation, quantization, residual, or store transform.
- Skip: fusion increases register pressure enough to regress, or output is reused by multiple consumers.
- Implement: fuse post-op into CK Tile epilogue or custom store path; avoid extra global round trip.
- Verify: one fewer kernel or less global traffic, same final result, faster end-to-end time.

## Latency Axis

### P1: `latency.waitcnt_pipeline` - Waitcnt-Aware Software Pipeline

- Trigger: load/compute serialization, SQTT wait bubbles, or visible `s_waitcnt` placement before every small operation.
- Skip: no independent work exists to overlap.
- Implement: double-buffer LDS/register tiles, move waits closer to consumers, and use `s_waitcnt vmcnt(0)` for global consumers and `s_waitcnt lgkmcnt(0)` for LDS/scalar/matrix consumers.
- Verify: `s_waitcnt` is present and better placed; runtime improves.

### P2: `latency.reduce_barrier` - Reduce Barriers and Sync Scope

- Trigger: many `__syncthreads()`/`s_barrier` instructions or barrier stalls.
- Skip: each barrier protects true cross-wave LDS dependency.
- Implement: remove redundant block-wide barriers, use wave-local paths, split phases, or rely on waitcnt where legal.
- Verify: barrier count/stall share drops and correctness remains stable.

### P3: `latency.wavefront_shuffle_ds_bpermute` - Wavefront Exchange / Reduction

- Trigger: wave-local exchange or reduction uses LDS plus full-block sync.
- Skip: data crosses waves or needs workgroup visibility.
- Implement: use HIP shuffle/ballot/cooperative groups, `ds_permute_b32`, `ds_bpermute_b32`, or explicit wave64 reduction.
- Verify: DS permute/shuffle pattern or fewer barriers, and wave64 correctness is tested.

### P4: `latency.ilp_unroll` - ILP, Loop Unrolling, and Schedule Fill

- Trigger: loops are short/fixed, instruction pipeline has idle gaps, or address arithmetic dominates.
- Skip: unroll increases VGPR pressure, code size, or cache pressure too much.
- Implement: `#pragma unroll`, manual unroll for fixed sizes, interleave independent arithmetic/address operations with loads.
- Verify: runtime improves and resource usage remains acceptable.

### P5: `latency.persistent_scheduler` - Persistent Scheduler / Work Queue

- Trigger: small or irregular grids, grouped GEMM/MoE tail effects, or underfilled CU occupancy.
- Skip: grid already saturates CUs evenly or persistent loop harms fairness.
- Implement: CK Tile persistent/grouped GEMM scheduler or simple work-queue loop.
- Verify: CU activity and timing improve; no atomic/work-queue overhead dominates.

### P6: `latency.split_k_streamk` - Split-K / Stream-K Parallelism

- Trigger: K dimension is large, M/N is skinny, blocks are too few, or decode/prefill workload underutilizes CUs.
- Skip: atomic/reduction cost dominates or numerical accumulation order is too sensitive.
- Implement: split K into partial sums, use tree or counting-based reduction, and tune split factor against CU count.
- Verify: more CUs active and total time improves including combine/reduction.

### P7: `latency.salu_valu_phase_balance` - SALU/VALU Phase Balance

- Trigger: scalar address/control work creates bubbles around vector/MMAC phases.
- Skip: address work is tiny or compiler already schedules well.
- Implement: precompute scalar indices, hoist invariant address math, fill empty scheduling phases with useful SALU/VALU work.
- Verify: SQTT/PMC stall mix improves or instruction schedule is visibly better.

### P8: `latency.sqtt_stall_triage` - SQTT Stall Triage

- Trigger: PMC/timing cannot explain a regression or `dcu_top.json` is degraded.
- Skip: simple benchmark and PMC evidence already identify the bottleneck.
- Run: `hipprof --sqtt --sqtt-type stat_stall --kernel-name <kernel> ...` for narrow stall data, or `--sqtt-type 1` for `stat,wave,issue,stat_stall,stat_valu`. Then summarize `thread_trace_*.json` with `scripts/analyze_sqtt.py`.
- Verify: SQTT/Perfetto artifacts exist, `sqtt_analysis.json` identifies waitcnt/branch/instruction-family evidence, and the next real method is chosen from the observed stall type.

## Operator-Specific Shortcuts

- GEMM / batched GEMM / grouped GEMM: start with CK Tile or HCU GEMM examples, then tune tile geometry, cshuffle/wavelet, persistent, split-k, preshuffle, and epilogue fusion.
- Attention / paged attention: prioritize KV/cache layout, vectorized cache operations, split-K for decode, online softmax reduction, LDS/matrix-load staging, and FP8/BF8 only with explicit scaling.
- Convolution: use CK Tile/HCU conv descriptors; tune layout, filter/spatial mapping, MLS/WASP/TLS loader, cshuffle/wavelet, and fused bias/add/activation.
- Norm/reduction: tune vector width, wave64 reductions, one-pass vs multi-pass, hidden-size specialization, and dynamic quant epilogue.
- MoE: tune sorting/routing, block_m, grouped GEMM shape, local-token filtering, quant path, split-k or atomic accumulation, and epilogue fusion.
