# HCU 优化策略目录

本目录是候选策略，不是兼容或收益清单。以当前工程与目标工具链的证据为准，需要参考时可查询 HCU-Knowledge。优先读 [共同契约](hcu-workflow-contract.md)；每轮 1..3 项，每轴最多 2 项，不为预算凑数。

## 选择方法

先确定热点、数值/布局契约和目标；找到当前 HCU 实现/例子，最小编译与正确性验证后才改低层路径。gfx 不能按大小推导兼容，少伯新特性限定 gfx946。FP4/FP8/INT4等分别按硬件和编译器证据决定，不能用存储位数证明原生矩阵指令。

## compute

### P1 · `compute.mmac_tensor_core`

HCU MMOP / MMAC matrix core utilization。CK Tile HCU GEMM/conv/attention path, source-backed HCU/AMD-named MMAC builtin with exact signature, or target-compiled inline asm; check actual v_mmac lowering, operand/layout semantics and scoped measurements

- 调查入口：`SQ_INSTS_MMOP` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。
- 要求 `target_evidence`：精确 gfx、编译器/库版本、头文件或固定源码、最小探针及结果。

### P2 · `compute.mixed_precision_fp8_bf8`

gfx938 FP8/BF8/TF32 mixed precision。compiled CK Tile low-precision path or source-backed gfx938 FP8/BF8 conversion/MMAC forms from paged_attention_938.cu when tolerance permits

- 调查入口：`SQ_INSTS_VALU_F32` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。
- 要求 `target_evidence`：精确 gfx、编译器/库版本、头文件或固定源码、最小探针及结果。

### P3 · `compute.launch_config_wave64`

Launch geometry and target wave width (legacy method ID)。query target wave width and allocation granularity; tune grid coverage and launch geometry with LDS/VGPR pressure

- **Trigger**: measured grid coverage, launch geometry, target occupancy and resource constraints; cumulative SQ_WAVES is not resident occupancy.
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。
- 要求 `target_evidence`：精确 gfx、编译器/库版本、头文件或固定源码、最小探针及结果。

### P4 · `compute.thread_coarsening`

Thread coarsening / register tile。per-thread multiple elements, register tiles, fixed-loop unroll

- 调查入口：`SQ_BUSY_CYCLES` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。

### P5 · `compute.register_pressure_control`

VGPR/SGPR pressure control。scalarize local arrays, reduce live ranges, tune __launch_bounds__, avoid spills

- 调查入口：`VGPR|SGPR|OCCUPANCY` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。

### P6 · `compute.fast_math_intrinsics`

Fast math / intrinsic replacement。HIP fast math intrinsics or reciprocal/multiply transforms with explicit tolerance

- 调查入口：`SFU|DIV|SQRT|TRANS` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。

### P7 · `compute.inline_asm_builtin`

Inline asm / low-level escape hatch。small asm volatile or source-backed builtin only after higher-level paths fail and a minimal target compile probe passes

- 调查入口：`dccobjdump` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。
- 要求 `target_evidence`：精确 gfx、编译器/库版本、头文件或固定源码、最小探针及结果。

## memory

### P1 · `memory.coalesced_access`

Coalesced global memory access。contiguous wavefront lanes; SoA or hot-dimension-contiguous layout

- 调查入口：`TD_COALESCABLE_WAVEFRONT_sum` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。

### P2 · `memory.vectorized_global_access`

Vectorized global load/store。aligned float2/float4 or packed global load/store operations

- 调查入口：`TCC_EA_RDREQ_sum` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。

### P3 · `memory.aligned_layout_transform`

Layout / stride transform for locality。hot-dimension-contiguous layout, aligned strides, direct target-layout writes

- 调查入口：`TCC_EA_RDREQ_sum` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。

### P4 · `memory.lds_tiling`

LDS tiling and data reuse。global -> LDS -> register tiling

- 调查入口：`TCC_EA_RDREQ_sum` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。
- 要求 `target_evidence`：精确 gfx、编译器/库版本、头文件或固定源码、最小探针及结果。

### P5 · `memory.global_to_lds_async`

Direct global-to-LDS / async buffer load。CK Tile loader, buffer_load_* lds, or source-backed __builtin_amdgcn_raw_buffer_load_lds wrapper with address_space(3) LDS destination

- 调查入口：`VGPR|TCC_EA_RDREQ|dccobjdump` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。
- 要求 `target_evidence`：精确 gfx、编译器/库版本、头文件或固定源码、最小探针及结果。

### P6 · `memory.matrix_load_mls`

CK Tile MLS / tile staging。Use the exact-target matrix-load/MLS implementation and descriptor contract. gfx936/gfx938 sources have MLS forms; Shaobo gfx946 extensions are a separate feature scope.

- 调查入口：`TCP_TCC_READ_REQ_LATENCY_sum` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。
- 要求 `target_evidence`：精确 gfx、编译器/库版本、头文件或固定源码、最小探针及结果。

### P7 · `memory.ds_read_matrix_layout`

DS matrix-read layout contract。compiled inline asm ds_read_m32x16_b16, ds_read_m32x16_b16_alt, or ds_read_m32x32_b8; source-probe ds_read_m32x64_b4 / ds_read_m32x8_b32 before use

- 调查入口：`dccobjdump` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。
- 要求 `target_evidence`：精确 gfx、编译器/库版本、头文件或固定源码、最小探针及结果。

### P8 · `memory.lds_bank_conflict`

LDS bank-conflict reduction / swizzle。padding, XOR/Morton swizzle, matrix-read-aware LDS layout

- 调查入口：`SQ_LDS_BANK_CONFLICT` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。
- 要求 `target_evidence`：精确 gfx、编译器/库版本、头文件或固定源码、最小探针及结果。

### P9 · `memory.cache_policy_glc_slc`

Cache policy / coherency modifiers。glc/slc flags in Hygon/AMD buffer/global memory operations when justified

- 调查入口：`CACHE|TCC|SYNC` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。
- 要求 `target_evidence`：精确 gfx、编译器/库版本、头文件或固定源码、最小探针及结果。

### P10 · `memory.ck_tile_named_pipeline`

CK Tile named pipeline selection。Choose only pipelines present and guarded for this exact HCU target; names alone do not prove feature availability

- 调查入口：`pipeline` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。
- 要求 `target_evidence`：精确 gfx、编译器/库版本、头文件或固定源码、最小探针及结果。

### P11 · `memory.epilogue_fusion`

Epilogue / post-op fusion。fuse bias/add/activation/quant/store transform into epilogue

- 调查入口：`KERNEL_LAUNCH|TCC_EA_WRREQ` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。

## latency

### P1 · `latency.waitcnt_pipeline`

Waitcnt-aware software pipeline。overlap global/LDS loads with compute; s_waitcnt vmcnt/lgkmcnt near consumers

- 调查入口：`STALL|WAIT|LATENCY` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。

### P2 · `latency.reduce_barrier`

Reduce barriers and sync scope。remove unnecessary __syncthreads; use wave-level patterns when legal

- 调查入口：`BARRIER|S_BARRIER` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。

### P3 · `latency.wavefront_shuffle_ds_bpermute`

Wavefront exchange/reduction via shuffle or DS permute。ds_permute_b32 / ds_bpermute_b32 or HIP shuffle/cooperative groups

- 调查入口：`LDS|BARRIER|REDUCE` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。
- 要求 `target_evidence`：精确 gfx、编译器/库版本、头文件或固定源码、最小探针及结果。

### P4 · `latency.ilp_unroll`

ILP, loop unrolling, and schedule fill。pragma/manual unroll, interleave independent arithmetic and address work

- 调查入口：`STALL|ISSUE|LOOP` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。

### P5 · `latency.persistent_scheduler`

Persistent or work-queue scheduler。CK Tile persistent/grouped GEMM or custom work queue

- 调查入口：`SQ_WAVES` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。
- 要求 `target_evidence`：精确 gfx、编译器/库版本、头文件或固定源码、最小探针及结果。

### P6 · `latency.split_k_streamk`

Split-K / Stream-K parallelism。split K into partial sums, tune combine/reduction, stream-K for skinny/irregular shapes

- 调查入口：`SQ_WAVES|CU_ACTIVITY` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。

### P7 · `latency.salu_valu_phase_balance`

SALU/VALU phase balance。hoist scalar address math, precompute invariants, fill empty scheduling phases

- 调查入口：`SALU|VALU|STALL` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。

### P8 · `latency.sqtt_stall_triage`

SQTT stall triage。hipprof --sqtt --sqtt-type stat_stall/stat_valu

- 调查入口：`hipprof --sqtt` 是候选名称，需核对当前工具的定义、单位和 dispatch；不是自动触发阈值。
- 验证：独立 oracle、多用例未采集测速；按方法查看源码、准确 kernel 的 ISA/资源、定义明确的计数器或 timeline。缺少证据记未验证。
- 要求 `target_evidence`：精确 gfx、编译器/库版本、头文件或固定源码、最小探针及结果。

## 典型算子路线

具体前提、反例、数值变换与工程验收见 [实验操作指南](optimization-playbook.md)。以下路线用于定位方向，不是无条件方法组合；尤其 SQTT 调查本身只提供证据，不算代码优化收益。

- GEMM：访问布局→padding/tail→寄存器预取/LDS复用→MMAC与epilogue；检查stage增加的VGPR/LDS成本。
- Attention/稀疏attention：prefill/decode、KV布局、mask/softmax数值、block稀疏调度、reduction与split策略分开验证。
- MoE：router/sort、token分布、grouped GEMM、quant/scale、combine、EP通信分开计时；空expert、skew、capacity是关键边界。
- Reduction/norm：小网格、跨wave合并、累加精度、同步和tail；不要盲目追求更高occupancy。
- 通算融合：共享内存生命周期、生产/消费同步、跨rank顺序、通信完成语义，必须保留真实多进程测试。

方法有效性只对应已测形状/环境；一次变快不能替整个方法族作通用结论。

## 按准确架构查证的专项候选

北美洲/南美洲资料和 CK 代码已有 MLS 命名/实现，不能误称 MLS 仅在少伯存在。少伯新资料里的 store、BPS、descriptor 编码等扩展须单独查证。参考知识页 `mls-wdra-generation-boundaries` 与 CK 工程指南。

### `compute.shaobo_mx_low_precision` — 少伯 MX 低精度路径

gfx946 instructions describe FP4/FP6 and scale-aware conversion/matrix paths; verify exact format, scale packing, accumulation and compiler support; never infer this for gfx936/gfx938.

限定 gfx946 新特性资料范围，必须提供 target_evidence。来源文档描述支持不等于当前编译器或本机硬件已验证。

### `compute.shaobo_wdra` — HCU producer/consumer VGPR 分配（保留旧 ID）

WDRA redistributes a fixed thread-group register budget; investigate spills and role balance, not an assumed increase in initial occupancy. DCC declarations cover gfx92a/gfx946, while the currently inspected Triton WASP automatic path is gated to gfx946. Read synchronization, initialization and descriptor restrictions before probing; declaration alone does not verify a working initialization sequence.

候选范围 gfx92a/gfx946，必须提供 target_evidence；不能把 Triton 自动路径扩展到月英，也不能从数值编号推断 gfx948 可用。

### `latency.hcu_ebarrier` — Ebarrier 基础同步

gfx92a/gfx946 的基础 arrive/sync/count/slot 声明与 gfx946 reduction 变体、Abarrier 分开。核验参与者、count=0 的默认语义、slot 重用及循环进度；必须最小编译并在准确目标验证。与下述组合 barrier 方法同时选择时解释独立改动，同一同步变更不能重复记收益。

### `memory.shaobo_tls` — 少伯 Tensor Load/Store 描述符路径

Read gfx946 tensor descriptor, im2col/layout and synchronization contracts; compare a correct ordinary loader and preserve exact descriptor/shape bounds.

限定 gfx946 新特性资料范围，必须提供 target_evidence。来源文档描述支持不等于当前编译器或本机硬件已验证。

### `latency.shaobo_abarrier_ebarrier` — 少伯异步/扩展 barrier

Check arrive/wait, phase/transaction count, barrier identity and participant lifetime against exact gfx946 compiler/runtime implementation; add race and progress tests.

限定 gfx946 新特性资料范围，必须提供 target_evidence。来源文档描述支持不等于当前编译器或本机硬件已验证。

## 指令与耦合方法

DCU/HCU 同义；具体 MLS/DS/MMAC、WDRA、低精度、barrier 和 TLS 的适用范围读 [HCU 指令指南](hcu-isa-guide.md)。历史 wave64 方法 ID 保留，实际 wave width 以目标模式为准。耦合项并非一概禁止同用，但需在 methods.json 的 coupling_reviews 解释独立差异和验证计划；同一代码变化不能重复记收益。
