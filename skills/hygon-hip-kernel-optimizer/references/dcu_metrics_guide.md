# DCU Metrics Guide

`profile_hipprof.py` parses `hipprof --pmc`, `--pmc-read`, and `--pmc-write` CSV files when available and writes `dcu_top.json`. It can also run `hipprof --codeobj-analyze <elf>` and optional SQTT JSON analysis. Counter availability varies by DTK version, kernel filtering, and concurrency. When `dcu_top.json` has `degraded: true`, treat roofline budgets as a starting heuristic and rely more heavily on benchmark timing, source inspection, SQTT, code-object analysis, and `dccobjdump`.

This guide maps profiler evidence to method IDs from `method_registry.json`. AMD ROCm performance guidance is useful for general GPU behavior: profile first, maximize coalescing, use LDS for reuse, balance registers/LDS/occupancy, and minimize divergence. Hygon-specific HCU/MMAC decisions must still be verified with Hygon ISA and target-compiled code, not AMD MFMA naming alone.

Builtin evidence note: a generic `__has_builtin` probe may not recognize HCU names even when DCU KB source projects contain real call sites. Treat source-backed builtins as implementation candidates only when copied with their exact signature and target guard, then confirm by target compilation and `dccobjdump`.

## First Read: Bound Type

| Evidence | Meaning | First methods to consider |
| --- | --- | --- |
| High arithmetic intensity, low memory pressure, low `SQ_INSTS_MMOP` on matrix work | compute path is weak or matrix core missing | `compute.mmac_tensor_core`, `compute.mixed_precision_fp8_bf8` |
| Low arithmetic intensity, high global/cache request pressure | memory bandwidth or access pattern bound | `memory.coalesced_access`, `memory.vectorized_global_access`, `memory.aligned_layout_transform` |
| Stalls/waits dominate despite reasonable compute and memory metrics | latency/scheduling bound | `latency.waitcnt_pipeline`, `latency.reduce_barrier`, `latency.ilp_unroll` |
| CU activity or wave count is low | launch geometry, occupancy, or scheduling underfills hardware | `compute.launch_config_wave64`, `latency.persistent_scheduler`, `latency.split_k_streamk` |
| Metrics are sparse or degraded | profiler evidence is insufficient | `latency.sqtt_stall_triage`, then benchmark and ISA inspection |

## Compute Signals

| Signal | Interpretation | Try |
| --- | --- | --- |
| `SQ_INSTS_MMOP` near zero on GEMM/conv/attention/MoE | HCU MMAC path is missing | `compute.mmac_tensor_core` |
| `SQ_INSTS_VALU_F32` or VALU arithmetic dominates matrix-like code | scalar FP32 path is doing matrix work | `compute.mmac_tensor_core`, `compute.mixed_precision_fp8_bf8` |
| `SQ_INSTS_VALU_F16/BF16` high but no matrix-core instructions | low precision is not reaching MMOP | `compute.mmac_tensor_core` |
| FP32 storage or conversion dominates and `gfx938` target is available | explicit TF32/FP8/BF8 path may help | `compute.mixed_precision_fp8_bf8` |
| Low `SQ_WAVES`, low CU activity, or tiny grid | not enough wave64 work | `compute.launch_config_wave64` |
| High register use in resource dump, low occupancy | register pressure limits wave residency | `compute.register_pressure_control` |
| Many local arrays, spills, or long live ranges in source | compiler may spill or over-allocate VGPRs | `compute.register_pressure_control`, `compute.thread_coarsening` |
| Division, sqrt, transcendental functions dominate | special function path may be slow | `compute.fast_math_intrinsics` |
| Source uses builtin/asm but ISA is not intended | compiler lowering is wrong | `compute.inline_asm_builtin` |

Hygon tensorcore note: for `gfx936/gfx938`, search `dccobjdump` for `v_mmac_*` or `MMOP`. Do not require AMD `mfma` strings as proof. HCU or AMD-named builtins are source-level routes, not final proof by themselves.

## Memory Signals

| Signal | Interpretation | Try |
| --- | --- | --- |
| High `TCC_EA_RDREQ*`, `TCC_EA_WRREQ*`, or `TCP_TCC_READ_REQ*` | too many global/cache requests | `memory.coalesced_access`, `memory.vectorized_global_access` |
| Effective bandwidth low while memory requests are high | uncoalesced, strided, or too many scalar ops | `memory.coalesced_access`, `memory.aligned_layout_transform` |
| Scalar contiguous loads/stores in ISA | vectorization did not survive lowering | `memory.vectorized_global_access` |
| Layout conversion kernels appear around hot op | layout does not match consumer | `memory.aligned_layout_transform`, `memory.epilogue_fusion` |
| Reuse exists but no `ds_read` / `ds_write` in ISA | data is not staged through LDS | `memory.lds_tiling` |
| Matrix tile data staged through VGPRs before LDS | global-to-LDS or MLS path may be better | `memory.global_to_lds_async`, `memory.matrix_load_mls` |
| `buffer_load_* ... lds` absent in a CK/HCU tile loader | async/direct LDS load path may be missing | `memory.global_to_lds_async` |
| Matrix core uses scalar LDS reads | matrix-read layout contract missing | `memory.ds_read_matrix_layout` |
| `SQ_LDS_BANK_CONFLICT` high or SQTT shows LDS conflict stalls | LDS layout conflict | `memory.lds_bank_conflict` |
| Streaming loads evict useful data or flag synchronization is fragile | cache policy/coherency issue | `memory.cache_policy_glc_slc` |
| CK Tile family has TLS/MLS/WASP/cshuffle/wavelet/persistent variants | named path may encode a better data movement policy | `memory.ck_tile_named_pipeline` |
| Bias/add/activation/quant follows immediately after store | avoid extra global round trip | `memory.epilogue_fusion` |

## Latency and Scheduling Signals

| Signal | Interpretation | Try |
| --- | --- | --- |
| Many `s_waitcnt` close to each load | pipeline is serialized | `latency.waitcnt_pipeline` |
| Wait stalls high in SQTT | memory/compute overlap is poor | `latency.waitcnt_pipeline`, `latency.ilp_unroll` |
| Many `s_barrier` or barrier stalls | synchronization scope is too broad | `latency.reduce_barrier` |
| Wave-local exchange uses LDS plus block barrier | use wave/DS exchange | `latency.wavefront_shuffle_ds_bpermute` |
| Fixed small loops or repeated address arithmetic | unroll/fill pipeline gaps | `latency.ilp_unroll`, `latency.salu_valu_phase_balance` |
| Small/irregular grids or grouped GEMM tails | work distribution underfills CUs | `latency.persistent_scheduler` |
| Skinny GEMM, decode attention, or large K with few M/N blocks | more K-parallelism may help | `latency.split_k_streamk` |
| Scalar pipe/address math stalls around vector/MMAC phases | SALU/VALU imbalance | `latency.salu_valu_phase_balance` |
| PMC cannot explain regression | need instruction stream view | `latency.sqtt_stall_triage` |

## Tooling Notes

- Use `hipprof --pmc --pmc-type 3` for regular PMC collection.
- Also collect `hipprof --pmc-read --pmc-type 3` and `hipprof --pmc-write --pmc-type 3` when memory behavior matters. The DTK hipprof guide separates general PMC, read-side PMC, and write-side PMC families, and the optimizer merges all CSV metrics when `profile_hipprof.py --pmc-mode all` is used.
- Use `hipprof --sqtt --sqtt-type stat_stall` when the stall source is unclear.
- Use `hipprof --codeobj-analyze <elf file>` after compilation to inspect VGPR/SGPR pressure. Treat high VGPR/SGPR pressure as evidence for `compute.register_pressure_control`, lower occupancy, and possible latency amplification.
- Use `dccobjdump --inputs=<binary> --show-sass --show-instruction-encoding --separate-functions` plus resource/symbol dumps after every low-level, builtin, or CK Tile path change.
- For SQTT JSON traces, Perfetto can open generated JSON. If the remote Python environment lacks the `perfetto` package, pull the `thread_trace_*.json` files to local `hygon_tmp/` and run `analyze_perfetto_trace.py` locally.
- If `hipprof --sqtt` emits only `.sqtt.csv` and no `thread_trace_*.json`, check the `*.hipprof.log` for hipprof-internal `llvm-objdump` errors. Supplying DTK `llvm-objdump` in the hipprof subprocess `PATH` can be required for SQTT export, but final ISA verification still uses `dccobjdump`.
- Avoid concurrent kernels during PMC/SQTT collection; Hygon docs warn that concurrent kernels can make PMC/SQTT results misleading.
- Use `hipcc --resource-usage` or `dccobjdump --show-resource-usage` to cross-check VGPR/SGPR/LDS pressure.

## Recommended Collection Commands

Use the bundled profiler for the normal loop:

```bash
python <skill>/scripts/profile_hipprof.py \
  --state ./run_*/state.json \
  --iter 1 \
  --which best_input \
  --pmc-mode all
```

`--pmc-mode all` runs three DTK PMC passes and merges their CSV outputs:

```bash
hipprof -o <prefix>           --pmc       --pmc-type 3 [--kernel-name <name>] <app>
hipprof -o <prefix>.pmc_read  --pmc-read  --pmc-type 3 [--kernel-name <name>] <app>
hipprof -o <prefix>.pmc_write --pmc-write --pmc-type 3 [--kernel-name <name>] <app>
```

Use `--pmc-mode pmc`, `read`, `write`, or `none` when a specific pass is needed. Use `hipprof --list-basic` and `hipprof --list-derived` on the remote target to discover counters for the installed DTK. Use `hipprof -i <config.txt> -o <prefix> --pmc-type 2 <app>` when the target needs a hand-written PMC configuration.

For register pressure, `profile_hipprof.py` runs this automatically when it can find the compiled ELF or shared object:

```bash
hipprof --codeobj-analyze <elf-or-so-file>
```

The summary is stored in `dcu_top.json` under `codeobj_analyze` and includes parsed `max_vgpr`, `max_sgpr`, `max_lds`, and `pressure_flags` when the output format exposes them.

For SQTT, run it only when PMC/timing cannot explain a regression or when waitcnt, LDS, branch, or issue-stream behavior is the question:

```bash
python <skill>/scripts/profile_hipprof.py \
  --state ./run_*/state.json \
  --iter 1 \
  --which kernel \
  --sqtt-type 1 \
  --sqtt-output-type 0 \
  --sqtt-data-dir ./run_*/iterv1/sqtt_json/
```

Useful raw forms from the DTK SQTT guide:

```bash
hipprof -o <prefix>.sqtt -d <trace-dir>/ --output-type 0 --sqtt --sqtt-type stat_stall [--kernel-name <name>] <app>
hipprof -o <prefix>.sqtt -d <trace-dir>/ --output-type 0 --sqtt --sqtt-type stat_valu  [--kernel-name <name>] <app>
hipprof -o <prefix>.sqtt -d <trace-dir>/ --output-type 0 --sqtt --sqtt-type 1          [--kernel-name <name>] <app>
hipprof -o <prefix>.sqtt -d <trace-dir>/ --output-type 0 --sqtt --sqtt-type all        [--kernel-name <name>] <app>
```

DTK documents these SQTT type groups:

- default or `0`: `stat,wave,issue`
- `1`: `stat,wave,issue,stat_stall,stat_valu`
- `all`: `stat,issue,event,wave,all_wave,stat_stall,stat_valu`

For large traces, set `-d <tmp-dir>` to avoid filling `/tmp`. For fine capture windows, start with `--pmc-off` and bracket the interesting region with `hipProfilerStart` / `hipProfilerStop` in the application.
Use a trailing slash on the `-d` directory. In DTK 25.04.4, a path without the trailing slash was observed to concatenate with the generated `rpl_data_*` directory name.

## SQTT JSON Analysis

SQTT can generate files such as:

```text
thread_trace_(kernel_index)__kernel_name_se(SE)_pid.json
thread_trace_(kernel_index)__kernel_name_se(SE)_pid.html
thread_trace_(kernel_index)__kernel_name_se(SE)_pid.stat.html
```

Use the bundled analyzer on the JSON directory:

```bash
python <skill>/scripts/analyze_sqtt.py <sqtt-output-dir> --out ./sqtt_analysis.json
```

If the local or remote Python environment has the `perfetto` package, also run:

```bash
python <skill>/scripts/analyze_perfetto_trace.py <sqtt-output-dir> --max-files 4 --out ./perfetto_analysis.json
```

`analyze_sqtt.py` is a lightweight JSON/CSV walker. It also summarizes `.sqtt.csv` when trace JSON is unavailable, and it ignores unrelated run JSON files such as `methods.json` and `bench.json`. `analyze_perfetto_trace.py` uses Perfetto Trace Processor on Chrome-trace JSON and reports top `slice` names, counts, durations, and stall/wait slices.

Read the resulting fields as follows:

| Field | Meaning | Optimization signal |
| --- | --- | --- |
| `top_mnemonics` | Most frequent instruction-like mnemonics found in the trace JSON | Confirm hot `s_waitcnt`, `ds_*`, `buffer_*`, `v_mmac`, branch, or VALU families |
| `category_counts.vmem` | Global/buffer/flat memory instruction presence | High count plus stalls points to memory path or coalescing work |
| `category_counts.lds` | LDS/DS instruction presence | High count plus stalls or bank metrics points to LDS layout/bank work |
| `category_counts.matrix` | Matrix/MMAC-related instruction presence | Low count on matrix work points back to `compute.mmac_tensor_core` |
| `waitcnt_count` | Count of `s_waitcnt` mnemonics seen in the JSON | High or clustered waits point to `latency.waitcnt_pipeline` |
| `branch_count` | Branch/jump instruction count | High branch activity points to divergence or scheduler imbalance |
| `stall_like_hits` | Keys or values containing stall/bubble/idle/wait terms | Use to select `latency.waitcnt_pipeline`, `latency.ilp_unroll`, or `latency.sqtt_stall_triage` |
| `numeric_cycle_fields` | Cycle/clock/duration-like numeric fields | Compare slowest/fastest waves or identify instruction classes with high cumulative cycles |

If SQTT contradicts PMC, trust neither blindly: verify kernel filtering, disable concurrent kernels, inspect Perfetto/HTML views, and repeat with a smaller `--kernel-name` or runtime capture window.

## ISA Evidence Cheat Sheet

| Expected evidence | What it supports |
| --- | --- |
| `v_mmac_*` or `MMOP` in target ISA | `compute.mmac_tensor_core` |
| `fp8`, `bf8`, `tf32`, `cvt_pk`, `cvt_sr` | `compute.mixed_precision_fp8_bf8` |
| `global_load_dwordx2/x4`, `buffer_load_dwordx*`, `flat_load_*` | `memory.vectorized_global_access` |
| `buffer_load_* ... lds` or another compiled direct-to-LDS form | `memory.global_to_lds_async` |
| `matrix_load*`, MLS, or compiled CK Tile MLS-like staging | `memory.matrix_load_mls` |
| `ds_read_m32x16_b16`, `ds_read_m32x16_b16_alt`, `ds_read_m32x32_b8`; optionally source-probed `ds_read_m32x64_b4` / `ds_read_m32x8_b32` | `memory.ds_read_matrix_layout` |
| `ds_read`, `ds_write` plus lower bank conflicts | `memory.lds_tiling`, `memory.lds_bank_conflict` |
| `glc`, `slc` on intended memory ops | `memory.cache_policy_glc_slc` |
| fewer `s_barrier` | `latency.reduce_barrier` |
| `ds_bpermute`, `ds_permute`, HIP shuffle lowering | `latency.wavefront_shuffle_ds_bpermute` |
| better-placed `s_waitcnt vmcnt(0)` / `lgkmcnt(0)` | `latency.waitcnt_pipeline` |

## Generic GPU Strategies Worth Importing

These are CUDA/AMD-generic and safe to consider on Hygon after wave64 adjustment:

- coalesced global memory access;
- vectorized load/store when alignment is guaranteed;
- LDS/shared-memory tiling for reuse;
- bank-conflict padding or swizzle;
- thread coarsening and register tiling;
- loop unrolling for fixed trip counts;
- launch/block-size tuning;
- avoiding divergent control flow inside a wave;
- epilogue fusion to remove extra global memory passes;
- split-K/persistent scheduling when the grid underfills the GPU.

These are not automatically safe:

- CUDA warp32-specific reductions without wave64 rewrite;
- NVIDIA tensorcore / WMMA / MMA instruction assumptions;
- FP4 tensorcore strategies;
- AMD MFMA string matching as a Hygon MMAC verification rule.
