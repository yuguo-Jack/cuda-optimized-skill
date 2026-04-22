# Optimization Catalog — Priority-Ordered 

> **核心原则**：每个轴（compute / memory / latency）内的方法按**真实重要性严格递降**排列。重要性 = **典型收益** × **触发概率** × **不可替代性**。
>
> **选择方法时必须从 P1 开始向下扫描**，选第一个同时满足以下三个条件的方法：
> 1. ncu 指标显示该方法对应的瓶颈存在（"触发条件"栏）
> 2. 方法 id 不在 `state.selected_methods` 中（未被选过）
> 3. 方法的 arch 与 `required_features` 均满足检测到的 `sm_*`
>
> **禁止跳过高优先级方法去选低优先级方法**，除非高优先级方法的"跳过条件"栏明确被满足。跳过时必须在 `analysis.md` 的"排除候选"段落中记录原因。
>

---

## Table of Contents

- [Selection Decision Tree](#selection-decision-tree)
- [Compute Axis (P1-P22)](#compute-axis-按优先级)
- [Memory Axis (P1-P22)](#memory-axis-按优先级)
- [Latency Axis (P1-P19)](#latency-axis-按优先级)
- [NCU Verification Checklist](#ncu-verification-checklist)
- [Combining Rules](#combining-rules)

---

## Selection Decision Tree

```
对于 axis ∈ {compute, memory, latency}:
  for priority = P1, P2, P3, ...:
    method = catalog[axis][priority]
    if method.id ∈ selected_methods:              → 跳过(已试过)
    if detected_sm < method.min_sm:                → 跳过(架构不够)
    if not required_features ⊆ arch_feature_map:   → 跳过(特性不满足)
    if method.skip_condition 成立:                  → 跳过(记录原因)
    if method.trigger_condition 不匹配 ncu:        → 跳过(瓶颈不在这)
    else:                                          → 选中,停止扫描
```

---

## Compute Axis (按优先级)

### P1: `compute.tensor_core` — Tensor Core / 专用硬件利用
- **典型收益**: 8-16× on GEMM kernels
- **触发条件**: `sm__pipe_tensor_op_hmma_cycles_active.pct_of_peak < 20%`（应为 GEMM/conv/attention 类算子）
- **跳过条件**: 算子不包含矩阵乘加语义（纯 elementwise / reduction / scan），或 Tensor Core 利用率已 > 60%
- **CUDA**: 将标量 `a[i]*b[j]` 累加替换为 `mma.sync` PTX 或 `nvcuda::wmma` fragment。sm_70/75: HMMA m16n8k8；sm_80/89: HMMA m16n8k16；sm_90: WGMMA m64nNk16；sm_100: tcgen05.mma + TMEM
- **CUTLASS**: `OpClassTensorOp` MMA Atom。Ampere `SM80_16x8x16`；Hopper `WGMMA`；Blackwell `tcgen05.mma` + TMEM
- **Triton**: 确保 `tl.dot` 输入为 FP16/BF16/FP8；`input_precision="tf32"` 折中
- **验证**: SASS 中出现 `HMMA`/`WGMMA`/`HGMMA`/`UMMA`；tensor op utilization 上升

### P2: `compute.mixed_precision` — 混合精度 / 低位宽计算
- **典型收益**: 2-4×
- **触发条件**: kernel 使用 FP32 但精度要求允许 FP16/BF16/TF32/FP8；FP32 pipe 利用率高而 tensor op 利用率低；或内存带宽受限
- **跳过条件**: 精度要求严格（FP64 科学计算）；已在 P1 中切换了精度；kernel 已是 FP16/BF16
- **CUDA**: FP32→TF32（Ampere+）；FP32→FP16/BF16 存储 + FP32 累加；Hopper FP8（E4M3/E5M2）+ per-tensor 缩放；Blackwell NVFP4 两级缩放。**累加器始终保持 FP32**
- **CUTLASS**: `ElementA=half_t/bfloat16_t/float_e4m3_t`, `ElementAccumulator=float`
- **Triton**: `tl.dot(a.to(tl.float16), b.to(tl.float16))`；FP8 通过 `.to(tl.float8e4nv)` + 缩放
- **验证**: tensor op utilization 上升；`dram__bytes.sum` 下降；**必须重新验证 atol/rtol**

### P3: `compute.warp_specialization` — Warp Specialization (producer-consumer)
- **典型收益**: 1.5-3× on Hopper attention
- **触发条件**: barrier + long_scoreboard 都非平凡；GEMM/attention 类；所有 warp 既搬数据又算
- **跳过条件**: 非 GEMM 类；sm < 80
- **CUDA**: producer warp 发 TMA load + mbarrier arrive；consumer warp 做 WGMMA/tcgen05。producer `setmaxnreg.dec 40`，consumer `setmaxnreg.inc 232`。Ping-pong: 2 个 consumer 交替执行不同 tile
- **CUTLASS (Hopper)**: `TmaWarpSpecialized` / `TmaWarpSpecializedCooperative` / `TmaWarpSpecializedPingpong`
- **Triton**: `num_consumer_groups=2, num_buffers_warp_spec=3`；编译器自动 task partition
- **验证**: producer barrier stall 极低；consumer TC 利用率提升；SASS 出现 `SETMAXREG`

### P4: `compute.launch_config` — Launch Configuration / Tile Shape *(MOVED UP from P6)*
- **典型收益**: 1.2-2× 基础 sanity check
- **触发条件**: `sm__warps_active.pct < 25%` 或 `sm__cycles_active.pct < 50%`
- **跳过条件**: occupancy > 50% 且 IPC > 2
- **CUDA**: 调 block size（128/256/512）；`cudaOccupancyMaxPotentialBlockSize`；`__launch_bounds__`
- **CUTLASS**: 调 CTA Tile Shape（128×128×32 Ampere / 128×128×64 Hopper）；调 Cluster Shape
- **Triton**: `BLOCK_M/N/K` + `num_warps`；`triton.autotune` 列 4-8 config
- **验证**: occupancy 上升；kernel latency 下降

### P5: `compute.thread_coarsening` — 线程粗化 / 寄存器级 Tiling *(MOVED UP from P7)*
- **典型收益**: 2-5× on memory-bound
- **触发条件**: 每线程只处理 1 个元素；IPC 低但 occupancy 充足
- **跳过条件**: 每线程已处理多个元素（如 4×4 register tile）；spill > 0
- **CUDA**: 每线程处理 2-8 元素（grid-stride loop）；register tiling 维护 M×N 累加器（如 8×8）
- **CUTLASS**: CuTe Thread-Value Layout 映射 per-thread tile 到寄存器
- **Triton**: 增大 `BLOCK_M`/`BLOCK_N`
- **验证**: IPC 上升；grid size 减少

### P6: `compute.ilp_unroll_register_tile` — 8×8 寄存器瓦片 outer-product ILP *(MOVED UP from P8)*
- **典型收益**: 2-3× when TC unavailable
- **触发条件**: thread_coarsening 已应用但 IPC 仍 < 2；SASS 中 FMA 依赖链过长；TC 未使用（FP32 SGEMM）
- **跳过条件**: 已在 P5 中达到 8×8 tile；使用 Tensor Core（TC 自带 ILP）
- **CUDA**: 每线程维护 8×8 register fragment，从 smem 加载 `A_reg[8]`、`B_reg[8]`，做 64 次 FMA outer-product。double-buffer smem→register 再进一步隐藏延迟
- **CUTLASS**: CuTe TiledMMA atom 自动构建 register tile
- **Triton**: 编译器自动；增大 tile 尺寸即可
- **验证**: IPC 上升至 > 2；per-thread FMA 数显著增加

### P7: `compute.overlap_compute_memory` — 计算与访存重叠（延迟隐藏）*(MOVED DOWN from P4)*
- **典型收益**: 1.3-2× when multi_stage unavailable
- **触发条件**: IPC < 1 且计算单元不饱和；`Eligible Warps Per Cycle` 偏低；未做 multi-stage pipeline
- **跳过条件**: 已在 memory.multi_stage_pipeline (P6) 中实现；纯 elementwise 无重叠空间
- **CUDA**: 提升 occupancy / 增大 per-thread ILP / 软件流水线 load-compute-store 三阶段交错
- **CUTLASS**: 增加 Pipeline `Stages`（2→4）
- **Triton**: 调大 `num_stages`（2→4）；调整 `num_warps`
- **验证**: `Eligible Warps/Cycle` 上升；`Stall Long Scoreboard` 下降

### P8: `compute.gemm_softmax_interleave` — GEMM-Softmax 两阶段乒乓流水
- **典型收益**: 1.5-2× on attention FP16
- **触发条件**: Attention 类 kernel 含 GEMM→softmax→GEMM 串行链；softmax 的 `MUFU.EX2` 使 TC 空闲；sm_90+
- **跳过条件**: 非 attention 类；sm < 90；已用 FlashAttention 库
- **CUDA**: warpgroup 0 做 softmax rescaling 时 warpgroup 1 做下一 tile 的 GEMM；online softmax running max/sum 增量更新
- **CUTLASS**: `sm90_gemm_tma_warpspecialized_pingpong.hpp`；example 77 (blackwell_fmha)
- **Triton**: 手写 outer loop 维护 `m_i`/`l_i`/`acc`
- **互补**: `latency.pingpong_warpgroup_schedule` (P10)、`latency.intra_wg_gemm_softmax_pipeline` (P12)
- **验证**: TC utilization 上升；`stall_wait` 下降；H100 BF16 attention 达 740+ TFLOPs

### P9: `compute.two_level_accumulation_promotion` — FP8 两级累加精度提升 *(NEW, from DeepGEMM)*
- **典型收益**: 0.85× 速度代价换取 FP32 级精度；FP8 训练/推理的**正确性前提**
- **触发条件**: FP8 kernel 发现相对误差 > 0.5%（QDQ baseline 对比）；Hopper WGMMA FP8 累加器为 14-bit
- **跳过条件**: 精度宽松场景（已用 P10 fast accumulation）；非 FP8 kernel
- **CUDA**: 每 N 条 WGMMA FP8 累加到 FP16/FP32 registers → CUDA core 提升到 FP32 累加 → 写回。DeepGEMM 典型 N=4
- **CUTLASS**: 手动实现（CUTLASS 3.x fast-accum 路径为相反方向，需禁用）
- **Triton**: 需 `tl.dot(..., out_dtype=tl.float32, acc_promote_cycles=N)` 扩展
- **冲突**: `compute.fp8_fast_accumulation_mode` (P10)
- **验证**: FP8 kernel 相对误差 < 0.5%；WGMMA 利用率略降

### P10: `compute.fp8_fast_accumulation_mode` — FP8 Fast Accumulation *(NEW)*
- **典型收益**: 2× vs two-level accum（代价是 14-bit 累加精度）
- **触发条件**: FP8 推理场景允许低精度；WGMMA 吞吐未饱和
- **跳过条件**: 精度要求严格（训练）；非 FP8 kernel
- **CUDA**: 保持所有 WGMMA 直接累加在 FP8 register，不升格
- **CUTLASS**: `KernelTmaWarpSpecializedCooperativeFP8FastAccum` schedule
- **Triton**: 默认（`tl.dot` FP8 路径不升格）
- **冲突**: `compute.two_level_accumulation_promotion` (P9)
- **验证**: WGMMA 吞吐上升；但误差累加可能达 1-2%

### P11: `compute.block_scaled_precision` — NVFP4 / MXFP8 / MXFP6 块缩放精度
- **典型收益**: 2× vs FP8 on Blackwell
- **触发条件**: sm_100+；kernel 使用 FP16/BF16 但精度允许 FP8/FP6/FP4 + 块缩放；内存带宽受限
- **跳过条件**: sm < 100；精度要求严格；已在 P2 mixed_precision 中切换到 FP8
- **CUDA**: `tcgen05.mma kind::mxf8f6f4` / `kind::mxf4nvf4`；每 16-32 个元素共享 UE8M0/UE4M3 缩放因子；CUDA-core 二级累加（promotion）
- **CUTLASS**: `OpClassBlockScaledTensorOp`；example 67 (FP8 block-scaled), 72 (Blackwell narrow-prec), 79a-c (GeForce NVFP4/MXFP8)
- **Triton**: Blackwell Triton 3.x block-scaled 支持
- **验证**: SASS 出现 `UMMA`/`QGMMA`；`dram__bytes.sum` 大幅下降；**必须验证 atol/rtol**

### P12: `compute.tf32_emulation_3xtf32_bf16x6` — FP32 via TF32/BF16 TC 仿真 *(NEW)*
- **典型收益**: 3-5× vs FP32 CUDA core，精度接近 IEEE FP32
- **触发条件**: FP32 精度要求但 Tensor Core 利用率 < 15%；FP32 pipe 饱和
- **跳过条件**: 已使用 P2 mixed_precision（更低精度）；纯标量计算
- **CUDA**: 将 FP32 操作数拆分为 3 个 TF32 或 6 个 BF16 的和：\(a = a_{hi} + a_{lo}\)，做 3×/6× TC MMA 再合并
- **CUTLASS**: 通过 `ElementA=tfloat32_t` + 自定义 accum 合并
- **Triton**: `tl.dot(..., input_precision="tf32x3")` 或 `"bf16x6"`
- **验证**: Tensor Core 利用率上升；误差 < 1 ULP

### P13: `compute.mufu_ex2_softmax_replacement` — SFU ex2 替代 exp *(NEW)*
- **典型收益**: 1.4× on softmax/attention SFU 吞吐
- **触发条件**: kernel 含 softmax/exp；`smsp__inst_executed_pipe_xu.sum` 高
- **跳过条件**: 无指数运算
- **CUDA**: 用 `exp2f(x * 1.4426950408889634f)` 替代 `expf(x)`，直接映射到 SFU `ex2.approx.f32`；FP16 路径用 PTX `ex2.approx.f16`
- **CUTLASS**: epilogue `fast_exp` functor
- **Triton**: `tl.exp2(x * 1.4427)` 替代 `tl.exp(x)`
- **验证**: SFU 利用率上升；SASS 出现 `MUFU.EX2`

### P14: `compute.reduction` — 归约优化
- **典型收益**: 2-4× on reduction-heavy
- **触发条件**: kernel 含 sum/max/softmax/layernorm；归约占总时间 > 30%
- **跳过条件**: 无归约
- **CUDA**: warp shuffle 归约；block 间 atomic 或二次 kernel。Split-K / Stream-K 用于 GEMM
- **CUTLASS**: Split-K；Stream-K `DecompositionMode::StreamK`
- **Triton**: `tl.sum`/`tl.max`/`tl.reduce`；跨 program `tl.atomic_add`
- **验证**: barrier stall 下降

### P15: `compute.warp_shuffle` — Warp Shuffle 用于计算
- **典型收益**: 1.2-2× on warp-scope reductions
- **触发条件**: warp 内数据交换用 shared memory + sync 实现
- **跳过条件**: 已在归约中用 shuffle；CUTLASS/Triton 自动处理
- **CUDA**: `__shfl_xor_sync` / `__shfl_down_sync`，约 1 cycle，无 bank conflict
- **验证**: smem load/store 下降；barrier stall 下降

### P16: `compute.loop_unroll` — 循环展开
- **典型收益**: 1.1-1.3×
- **触发条件**: `short_scoreboard`/`wait` stall 显著；SASS 热循环有大量标量依赖
- **跳过条件**: 编译器已完全展开
- **CUDA**: `#pragma unroll` / `#pragma unroll N`
- **CUTLASS**: `CUTLASS_PRAGMA_UNROLL`；CuTe `Int<N>`
- **Triton**: `tl.static_range`；`BLOCK_K` 作为 `tl.constexpr`
- **验证**: 指令数减少；`Issue Slot Utilization` 上升

### P17: `compute.fma_and_fast_math` — FMA + 编译选项
- **典型收益**: 1.1-1.4× on transcendental-heavy
- **触发条件**: SASS 中 `a*b` 和 `+c` 是两条独立指令；`--use_fast_math` 可接受
- **跳过条件**: 精度要求严格
- **CUDA**: `__fmaf_rn()`；`--use_fast_math`；除法 → `a * __frcp_rn(b)` (4×-16×)
- **Triton**: 编译器自动 FMA；`allow_tf32=True`
- **验证**: 指令数减少；latency 下降

### P18: `compute.lop3_bit_manipulation` — lop3 三目逻辑反量化
- **典型收益**: 2-4× on W4A16 dequant
- **触发条件**: W4A16 / W8A16 量化推理；反量化 kernel 占比高；dequant + matmul 未融合
- **跳过条件**: 非量化推理；已用 Marlin/Machete 等优化库
- **CUDA**: PTX `lop3.b32` 一条指令完成 AND+OR 位操作，将 int4 直接构造为 fp16 mantissa pattern，再减偏移得到 signed fp16 值。同时反量化 2 个 int4（packed in int32）。融合 dequant + GEMM 为单一 kernel
- **Triton**: 位运算 `(packed >> shift) & 0xF`；编译器自动 FMA
- **验证**: SASS 出现 `LOP3`；dequant latency 下降；A10 达 ~3.87× 理论加速

### P19: `compute.pragma_no_alias` — __restrict__ + #pragma unroll + 别名消除
- **典型收益**: 1.2-2× when aliasing blocks vectorization
- **触发条件**: NVCC 未能自动向量化或 ILP；SASS 中出现多余 LD/ST barrier；函数签名无 `__restrict__`
- **跳过条件**: 已在 P20 strength_reduce 中添加 `__restrict__`；编译器已优化
- **CUDA**: 所有指针参数加 `__restrict__`；关键循环加 `#pragma unroll`；`#pragma nv_diag_suppress` 消除别名警告
- **验证**: SASS 指令数减少；IPC 上升

### P20: `compute.strength_reduce` — 强度削减
- **典型收益**: 1.1-1.3× if idx math in hot loop
- **触发条件**: 热循环整数除法/取模（变量除数）；指针 alias 阻碍优化
- **跳过条件**: 编译器已自动优化
- **CUDA**: 移位代替 2 的幂除；`rsqrtf()`
- **CUTLASS**: `FastDivmod` 内建
- **Triton**: 编译器自动
- **验证**: ALU 指令减少

### P21: `compute.branch_eliminate` — 分支消除 + 谓词化
- **典型收益**: 1.1-1.5× when branch divergence severe
- **触发条件**: `Warp Execution Efficiency` 低；SASS 大量 BRA/SSY
- **跳过条件**: 无明显分支发散
- **CUDA**: select 指令代替 if-else；warp vote early exit
- **CUTLASS**: `CpAsyncPredicated`
- **Triton**: `tl.where`；`tl.load(mask=mask)`
- **验证**: `Warp Execution Efficiency` 上升

### P22: `compute.handwritten_ptx_inline` — 手写 inline PTX (后防线)
- **典型收益**: 1.05-1.15×；仅作最后手段
- **触发条件**: 所有高优先级方法均已尝试；NVCC 生成的 SASS 序列已确认次优（通过 NCU source correlation）；需精确控制指令排列
- **跳过条件**: 未穷尽高优先级方法；kernel 不足够性能关键；代码可移植性要求高
- **CUDA**: `asm volatile("...")`；控制 yield/reuse bit（DeepGEMM 获 10%+）；插入 prefetch/预取指令；精确排列 FMA 与 load 交错序列
- **注意**: 后防线方法，仅在其它方法均无效时使用；破坏可移植性；需逐架构维护
- **验证**: SASS 中出现目标 PTX 指令；kernel latency 下降

---

## Memory Axis (按优先级)

### P1: `memory.kernel_fusion` — Kernel Fusion
- **典型收益**: 2-10× when eliminating intermediate materialization
- **触发条件**: 相邻 producer-consumer kernel 可合并；中间数据经 Global Memory 往返。**最高收益单项优化**
- **跳过条件**: 已融合；后续操作有跨 tile 全局依赖
- **CUDA**: 合并相邻 kernel，中间结果留在 reg/smem。常见融合：GEMM+bias+activation (SwiGLU/GELU)；matmul+dequant；attention QKV projection+softmax+output
- **CUTLASS**: EVT 实现 GEMM + Bias + Activation fusion
- **Triton**: 同一 `@triton.jit` 中编写融合逻辑
- **验证**: `DRAM Read Bytes / 理论数据量` ≈ 1.0；kernel launch 次数减少

### P2: `memory.coalesced_access` — 合并访问
- **典型收益**: 4-8× on uncoalesced access
- **触发条件**: `Global Load/Store Efficiency < 50%`；`Sectors/Request` ≫ 1
- **跳过条件**: 最内维 stride=1；CUTLASS TiledCopy / Triton `tl.make_block_ptr` 已保证
- **CUDA**: warp 内 32 线程访问连续 128B；转置先 coalesced load→smem→转置读
- **CUTLASS**: Layout 代数最内维 stride=1
- **Triton**: `tl.make_block_ptr` 的 `order` 声明连续维
- **验证**: `Sectors/Request` ≈ 1

### P3: `memory.tiling_smem` — Tiling / Shared Memory 分块
- **典型收益**: 3-5× on GEMM-like
- **触发条件**: L2 hit rate < 30%；`dram__bytes.sum` ≫ 理论最小值。**数据复用率 > 1 时关键**
- **跳过条件**: 纯 elementwise 无复用
- **CUDA**: 按 tile 搬到 smem 在片上多次复用
- **CUTLASS**: CTA Tile Shape；Cluster Tile
- **Triton**: `BLOCK_M/N/K`；编译器自动提升到 smem
- **验证**: L2 hit rate 上升；`dram__bytes.sum` 下降

### P4: `memory.vectorized_access` — 向量化访存 *(MOVED UP from P7)*
- **典型收益**: 1.5-2× on memory-bound，**最佳 ROI 方法之一**（零重构成本）
- **触发条件**: `mio_throttle` 显著；SASS 中大量 `LDG.E.32`（非 `LDG.E.128`）
- **跳过条件**: CUTLASS Copy Atom / Triton 已自动向量化
- **CUDA**: `float4`/`int4` 读写 128-bit；地址对齐到向量宽度
- **验证**: `mio_throttle` 下降；SASS 出现 `LDG.E.128`

### P5: `memory.async_copy` — 异步拷贝 (cp.async / TMA) *(MOVED UP from P6)*
- **典型收益**: 1.5-3×（multi_stage 的前提）
- **触发条件**: `Stall Long Scoreboard` 占主导；Global→Shared 走 reg 中转
- **跳过条件**: sm < 80；已在 P6 multi_stage 中配合使用
- **CUDA (sm_80+)**: `cp.async.cg.shared.global` / `cp.async.ca`
- **CUDA (sm_90+)**: TMA `cp.async.bulk.tensor` 硬件 DMA + `mbarrier`
- **CUTLASS**: `MainloopSm80CpAsync`；`MainloopSm90Tma*`
- **Triton**: 编译器自动；`tl.make_block_ptr` 触发 TMA
- **验证**: SASS 出现 `LDGSTS`/`CP.ASYNC`/`TMA_LOAD`；`Stall Long Scoreboard` 下降

### P6: `memory.multi_stage_pipeline` — 双缓冲 / 多级流水线
- **典型收益**: 1.3-2×
- **触发条件**: `Stall Long Scoreboard` 占主导；只有 single buffer。公式：`smem_per_stage × num_stages + epilogue_smem ≤ dev_smem_per_CTA`
- **跳过条件**: 已有多级 pipeline
- **CUDA**: N 组 smem buffer ping-pong；`cp.async.commit_group` + `wait_group<N-1>`。推荐 4-6 stages (sm_90+ TMA)
- **CUTLASS**: `Stages`（2→4）；`StageCountAutoCarveout`
- **Triton**: `num_stages=3~5`（超过 5 通常退化）
- **验证**: `Stall Long Scoreboard` 下降

### P7: `memory.epilogue_fusion` — Epilogue Fusion (bias+activation+cast)
- **典型收益**: 1.3-2× on GEMM+activation chains
- **触发条件**: GEMM 后紧跟 bias/activation/type-cast/reduction 需额外 DRAM 往返
- **跳过条件**: 已在 P1 kernel_fusion 中完成；kernel 输出后无后处理；已在 P13 EVT 中用更高级版本
- **CUDA**: 累加器仍在 register 时直接做后处理（bias+activation+cast）一次性写回
- **CUTLASS**: `LinearCombinationGeneric<ActivationFn, ...>`
- **Triton**: 天然支持——`tl.dot` 后直接 `acc + bias[None, :]`、`tl.sigmoid(acc)`
- **验证**: kernel launch 数减少；`dram__bytes_write.sum` 下降

### P8: `memory.bank_conflict` — Bank Conflict 消除 *(MOVED UP from P10)*
- **典型收益**: 1.2-2×
- **触发条件**: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum` > 0
- **跳过条件**: 已为 0；kernel 不用 smem
- **CUDA**: Padding（`float s[32][33]`）或 Swizzle/XOR 索引。注意 LDS.128 按 phase 判定：8 线程/phase × 16B = 128B，对齐时无冲突（即使 naive 分析显示有冲突）
- **CUTLASS**: `Swizzle<B,M,S>`
- **Triton**: 编译器自动 swizzle；调整 `BLOCK_K`
- **验证**: bank conflict → 0

### P9: `memory.ldmatrix_stmatrix` — ldmatrix / stmatrix 合并 SMEM→register *(MOVED UP from P11)*
- **典型收益**: 1.2-1.5× on TC kernels
- **触发条件**: Tensor Core kernel 中 smem→register 搬运使用多条标量 LDS；`mio_throttle` 显著；sm_75+
- **跳过条件**: sm < 75；CUTLASS/Triton 已自动使用 ldmatrix；非 TC kernel
- **CUDA**: PTX `ldmatrix.sync.aligned.m8n8.x4` 一条指令加载 4 个 8×8 matrix fragment（替代 8 条 LDS.32）；`stmatrix` 逆操作；`movmatrix` 跨寄存器搬运
- **CUTLASS**: CuTe Copy_Traits `SM75_U32x4_LDSM_N` 自动使用
- **验证**: SASS 出现 `LDSM.16.M88.x4`；`mio_throttle` 下降

### P10: `memory.register_pressure` — 寄存器压力控制 *(MOVED UP from P12)*
- **典型收益**: 1.2-2× when spilling
- **触发条件**: `launch__occupancy_limit_registers` 是主要限制；spill > 0
- **跳过条件**: spill = 0；限制因素是 smem
- **CUDA**: `__launch_bounds__(256, 2)`；缩小 per-thread tile；大数组放 smem
- **CUTLASS**: 缩小 CTA Tile / 减 Pipeline stage
- **Triton**: 缩小 `BLOCK_M/N`；增大 `num_warps`
- **验证**: spill → 0；occupancy 上升

### P11: `memory.tma_multicast` — TMA Multicast 跨 CTA 数据复用
- **典型收益**: 1.5-2×（bytes reduction 1/cluster_size）
- **触发条件**: 多个 CTA 读取相同 K-tile；`dram__bytes.sum` ≫ 理论最小值；sm_90+
- **跳过条件**: sm < 90；无跨 CTA 数据共享；grid 太小
- **CUDA (sm_90+)**: TMA `cp.async.bulk.tensor` + multicast mask
- **CUTLASS**: Cooperative schedule + Cluster Shape（如 2×1×1）自动 multicast
- **验证**: `dram__bytes.sum` 下降（理想 1/Cluster_size）

### P12: `memory.tma_descriptor_tuning` — TMA descriptor swizzle / multicast 精调
- **典型收益**: 1.1-1.3× on TMA-using kernels
- **触发条件**: 已用 TMA (P5/P11) 但 bank conflict > 0；swizzle mode 选择不当（128B/64B/32B/None）；L2 promotion 未启用
- **跳过条件**: sm < 90；未用 TMA；bank conflict 已为 0
- **CUDA**: `cuTensorMapEncodeTiled` 的 `swizzle` 参数（`CU_TENSOR_MAP_SWIZZLE_128B` 最常用）；`l2Promotion` 参数；multicast mask 精确设置
- **CUTLASS**: CuTe `Swizzle<B,M,S>` + `SM90_TMA_LOAD_MULTICAST`
- **验证**: bank conflict → 0；L2 hit rate 上升

### P13: `memory.epilogue_visitor_tree_fusion` — Epilogue Visitor Tree (EVT) *(NEW)*
- **典型收益**: 1.3-2× beyond basic epilogue_fusion
- **触发条件**: sm_90+；GEMM 后的后处理需要任意 DAG（scale + bias + activation + reduction 并行路径）；basic epilogue_fusion 不足
- **跳过条件**: sm < 90；后处理已简单（单 activation）；已用 basic epilogue
- **CUDA**: 手写 register-level DAG 访问器
- **CUTLASS**: `Sm90EVT<Sm90Compute<ActivationFn>, Sm90AccFetch, Sm90ColBroadcast, ...>`；example 49
- **Triton**: 原生 tensor program 已可表达任意 DAG
- **冲突**: `memory.epilogue_fusion` (P7)
- **验证**: `dram__bytes_write.sum` 进一步下降；多操作融合到单 store

### P14: `memory.data_layout` — SoA / 对齐 / Padding / 数据重排
- **典型收益**: 1.3-2× when AoS→SoA
- **触发条件**: AoS 导致 warp 地址不连续；偏移破坏对齐。NHWC 比 NCHW 更适合 Tensor Core
- **跳过条件**: 已是 SoA；cudaMalloc 天然 256B 对齐
- **CUDA**: AoS→SoA；NCHW→NHWC（卷积）；`cudaMallocPitch`；稀疏数据预排序
- **验证**: `Global Load Efficiency` 上升

### P15: `memory.ldg_cache_modifier_hint` — 显式 Cache Modifier *(NEW)*
- **典型收益**: 1.1-1.3× for streaming reads
- **触发条件**: kernel 有一次性流式读（权重、input activations），但 L1 被污染；`l1tex__t_sector_hit_rate.pct` 虚高但命中无用
- **跳过条件**: 数据需要复用；读取已通过 TMA
- **CUDA**: `__ldcg(ptr)`（cache global, bypass L1）、`__ldcs(ptr)`（streaming, 不污染 L1）、`__ldca(ptr)`（cache all）
- **PTX**: `ld.global.cg` / `ld.global.cs` / `ld.global.ca`
- **验证**: L1 hit rate 更准确反映复用数据；总带宽稳定

### P16: `memory.l2_persistence_window` — L2 persistent cache window
- **典型收益**: 1.2-1.5× for KV cache / embedding table
- **触发条件**: 小热点数据（KV cache / embedding table / lookup table）被反复访问；L2 hit rate 低；sm_80+
- **跳过条件**: sm < 80；数据量超出 L2 容量；无重复访问模式
- **CUDA**: `cudaAccessPolicyWindow` 设置 `num_bytes` + `hitRatio` + `hitProp=persisting`；配合 `cudaCtxResetPersistingL2Cache()` 管理生命周期
- **验证**: L2 hit rate 上升；`dram__bytes.sum` 下降

### P17: `memory.split_k_parallel_reduce` — Split-K 并行规约 *(NEW)*
- **典型收益**: 1.5-3× on tall-skinny M×N small K large
- **触发条件**: GEMM M×N 远小于 SM 数×tile 面积，但 K 很大导致单 CTA 工作过多；SM 利用率 < 50%
- **跳过条件**: GEMM 形状正常；已用 stream-K（P6 latency）；K 维度小
- **CUDA**: launch `split_k × M×N / tile` 个 CTA，各自累加一段 K 的部分和；中间结果写入 workspace；第二阶段 reduction kernel 合并
- **CUTLASS**: `GemmUniversal` 的 `split_k_slices`
- **Triton**: `tl.atomic_add` 跨 program 合并
- **冲突**: `latency.stream_k_load_balancing` (P6) — 两者是同一问题的不同方案
- **验证**: SM 利用率上升；添加的 reduction kernel 开销 < 总增益

### P18: `memory.cache_and_readonly` — L2/只读路径 + smem 容量配置
- **典型收益**: 1.1-1.3×
- **触发条件**: 小热点数据反复访问；smem 容量不足放大 tile
- **跳过条件**: 编译器已自动优化只读路径
- **CUDA**: `const __restrict__` + `__ldg()`；`cudaFuncSetAttribute` 调大 smem（Hopper 228KB）
- **Triton**: `eviction_policy="evict_first"/"evict_last"` 控制 L2
- **验证**: L2 hit rate 上升

### P19: `memory.smem_register_spilling` — Shared-memory register spilling (CUDA 13.0)
- **典型收益**: 1.05-1.1x
- **触发条件**: register spill > 0 且 spill 目标是 local memory（DRAM）；smem 有剩余容量；CUDA toolkit ≥ 13.0；sm_80+
- **跳过条件**: sm < 80；CUDA < 13.0；spill = 0；smem 已满
- **CUDA**: `#pragma nv_enable_smem_spilling` 让编译器将 spill 数据放到 smem 而非 local memory；实测 QUDA 获 5-10% 加速
- **验证**: local memory traffic 下降；kernel latency 下降

### P20: `memory.tma_im2col_implicit_conv` — TMA im2col 隐式卷积 *(NEW)*
- **典型收益**: 1.5-2× vs explicit im2col for convolution
- **触发条件**: 卷积 kernel；当前用显式 im2col 预处理；sm_90+
- **跳过条件**: sm < 90；非卷积；已用 cuDNN
- **CUDA**: TMA im2col mode 通过 `cuTensorMapEncodeIm2col`，让 TMA 单元自动做 patch 提取并写入 SMEM 为 (R×S×C, M) 矩阵
- **CUTLASS**: `Sm90ImplicitGemmTma` kernel 族
- **验证**: 消除 im2col kernel；`dram__bytes.sum` 下降

### P21: `memory.distributed_smem` — 分布式共享内存 (DSMEM)
- **典型收益**: 1.2-1.5× on cluster-scoped workloads
- **触发条件**: 数据超出单 CTA smem（228KB）但适合 Cluster 级总 smem；sm_90+
- **跳过条件**: sm < 90；数据放得进单 CTA smem
- **CUDA (sm_90+)**: `cluster.map_shared_rank(smem_ptr, dst_rank)` 获取远程 smem 指针
- **CUTLASS**: Cooperative schedule 自动利用 DSMEM
- **验证**: `dram__bytes.sum` 下降

### P22: `memory.tensor_map_rebuild_avoid` — __grid_constant__ tensor map 复用
- **典型收益**: 1.02-1.05× launch overhead reduction
- **触发条件**: TMA kernel 在 tight loop 中被反复 launch；每次重建 TMA descriptor 开销显著；sm_90+
- **跳过条件**: sm < 90；非 TMA kernel；只 launch 一次
- **CUDA**: tensor map 作为 `const __grid_constant__ CUtensorMap` 传入，在 kernel 内 `prefetch.tensormap`；若只变 base address 则用 `cuTensorMapReplaceAddress()` 避免完整重建
- **CUTLASS**: `make_tma_copy()` 复用 descriptor
- **验证**: host-side launch overhead 下降；`prefetch.tensormap` 出现在 SASS prologue

---

## Latency Axis (按优先级)

### P1: `latency.async_pipeline` — 异步流水线消除同步等待
- **典型收益**: 1.5-2.5×
- **触发条件**: `Stall Long Scoreboard` 占主导；无异步搬运
- **跳过条件**: 已在 memory P5/P6 中实现（标记"已覆盖"）
- **CUDA (sm_80+)**: `cp.async` + `commit_group/wait_group`
- **CUDA (sm_90+)**: TMA + 硬件 barrier transaction-based 语义
- **CUTLASS**: `PipelineAsync`（Ampere）/ `PipelineTmaAsync`（Hopper）
- **Triton**: `num_stages` 参数
- **验证**: `Stall Long Scoreboard` 下降

### P2: `latency.reduce_sync_count` — 减少 __syncthreads() 次数
- **典型收益**: 1.2-1.5×
- **触发条件**: `smsp__warps_issue_stalled_barrier` 占主导；多处 `__syncthreads()`
- **跳过条件**: 仅 1-2 个 sync 且有真实跨 warp 依赖
- **CUDA**: 审查每处 sync；双缓冲合并；`__syncwarp(mask)` 替代。**必须 `compute-sanitizer --tool racecheck`**
- **CUTLASS**: Pipeline arrive/wait 替代全 block sync
- **验证**: `Stall Barrier` 下降；`Eligible Warps/Cycle` 上升

### P3: `latency.warp_shuffle_sync` — Warp Shuffle 替代 Shared Memory 同步
- **典型收益**: 1.3-2× on warp reductions
- **触发条件**: warp 内数据交换经"写 shared→sync→读 shared"三步
- **跳过条件**: CUTLASS/Triton 已自动处理；无 warp 内通信需求
- **CUDA**: `__shfl_sync` 系列，≈1 cycle，无 bank conflict
- **验证**: `Stall Barrier` 下降

### P4: `latency.tile_scheduler_swizzle` — Tile Scheduler Swizzle / 光栅化顺序优化
- **典型收益**: 1.3-1.8× on large GEMM
- **触发条件**: GEMM/Conv kernel `lts__t_sector_hit_rate.pct < 50%` 且 grid > 100 blocks
- **跳过条件**: grid < 数十 block；L2 hit rate > 70%
- **CUDA**: `GROUP_SIZE_M`（4-8）分组，组内先遍历 N 再遍历 M
- **CUTLASS**: Tile Scheduler `RasterOrder::AlongM`/`AlongN`/`Heuristic`；`Swizzle<log2_M, log2_N>`
- **Triton**: `GROUP_SIZE_M` 参数（见 Triton matmul tutorial）
- **验证**: L2 hit rate 上升；kernel latency 下降

### P5: `latency.persistent_kernel` — 持久化 Kernel *(MOVED UP from P6)*
- **典型收益**: 1.2-1.5×
- **触发条件**: `sm__cycles_active.pct < 50%`；wave quantization（最后一波 CTA 只占部分 SM）
- **跳过条件**: grid 已填满 SM；工作均匀
- **CUDA**: launch `num_sms × waves` 个 block，循环处理工作队列
- **CUTLASS**: `PersistentTileScheduler`
- **Triton**: persistent kernel + `tl.atomic_add` 工作窃取
- **验证**: `sm__cycles_active.pct` 上升

### P6: `latency.stream_k_load_balancing` — Stream-K 跨 K 动态负载均衡 *(NEW)*
- **典型收益**: 1.5-2.5× on wave-quantized GEMM（比 persistent 更激进）
- **触发条件**: GEMM `(NumTiles % NumSMs) / NumSMs > 0.1`（尾效应严重）；persistent 不够
- **跳过条件**: grid 规则；已用 split-K (P17 memory)
- **CUDA**: 每个 persistent CTA 同时沿 K 维做原子式任务窃取；固定块 + 流式块混合调度
- **CUTLASS**: `DecompositionMode::StreamK` / example 47 (Ampere), 74 (Blackwell)
- **Triton**: 手写 persistent + work-stealing loop
- **冲突**: `memory.split_k_parallel_reduce` (P17)
- **验证**: wave quantization 消除；`sm__cycles_active.pct` 进一步上升

### P7: `latency.online_recomputation` — 在线重计算 (FlashAttention 范式) *(MOVED DOWN from P5)*
- **典型收益**: 5-10× on attention（完全消除 O(N²) 物化）
- **触发条件**: 需全局归约（softmax）才能继续后续计算，必须物化 O(N²) 中间矩阵到 DRAM
- **跳过条件**: 无全局归约操作；中间矩阵放得进 smem/register；FLOPs 已是瓶颈
- **CUDA**: Online Softmax: running max m_i + running sum l_i，每 tile 增量 rescaling。N×N attention score 矩阵永不物化到 DRAM
- **CUTLASS**: CUTLASS example 77 (`blackwell_fmha`)
- **Triton**: outer loop 遍历 K/V tiles，inner logic 维护 `m_i`/`l_i`/`acc`
- **互补**: `compute.gemm_softmax_interleave` (P8)
- **验证**: `dram__bytes.sum` 显著下降；kernel 数减少

### P8: `latency.warp_aggregated_atomics` — Warp-Aggregated Atomics *(NEW)*
- **典型收益**: up to 32× on atomic-heavy filter/histogram
- **触发条件**: atomic 吞吐受限（`membar` stall 高）；warp 内多线程 atomic 到同地址
- **跳过条件**: atomic 已稀疏；CUDA ≥ 7.5 编译器已自动聚合
- **CUDA**: `__activemask() + __popc + __ffs + __shfl_sync` 让 warp leader 执行单次 atomic，写回后 broadcast
- **CUB**: `BlockReduce` + `atomicAdd` 替代每线程 atomic
- **验证**: `Stall Membar` 下降；atomic op count / warp 降低

### P9: `latency.asymmetric_mbarrier_sync` — mbarrier 非对称同步
- **典型收益**: 1.2-1.5×
- **触发条件**: 多处 `__syncthreads()` 但 producer/consumer 线程数不对称；barrier stall 高；sm_90+
- **跳过条件**: sm < 90；仅 1 处 sync；线程对称
- **CUDA**: `mbarrier.init` + `mbarrier.arrive.expect_tx` (producer) + `mbarrier.try_wait.parity` (consumer)；只有 consumer 需要等待，producer arrive 后立即继续工作
- **CUTLASS**: CUTLASS Pipeline 类基于 mbarrier 实现 arrive/wait
- **验证**: `stall_barrier` 下降；SASS 出现 `MBARRIER.ARRIVE` / `MBARRIER.TRY_WAIT`

### P10: `latency.pingpong_warpgroup_schedule` — FA3 Pingpong Warpgroup *(NEW)*
- **典型收益**: 1.3-1.5× on attention forward
- **触发条件**: sm_90+；attention/GEMM+softmax 交错场景；有 2 个 consumer warpgroup 可用
- **跳过条件**: sm < 90；单 warpgroup；非 softmax 路径
- **CUDA**: warpgroup1 做 tile_k GEMM，同时 warpgroup2 做 tile_{k-1} 的 softmax rescaling；通过 named barrier 交替
- **CUTLASS**: `TmaWarpSpecializedPingpong` schedule（FA3 的实现参考）
- **Triton**: Triton 自动 WS 3.2+ 可以表达
- **互补**: `latency.intra_wg_gemm_softmax_pipeline` (P12)
- **验证**: `stall_wait` 下降；TC 利用率更稳定

### P11: `latency.producer_regdealloc_setmaxnreg` — setmaxnreg.dec/inc 寄存器再分配
- **典型收益**: 1.2-1.4× eliminating consumer spill
- **触发条件**: warp-specialized kernel 中 producer 持有大量寄存器但只做 TMA load；consumer 寄存器不够导致 spill；sm_90+
- **跳过条件**: sm < 90；非 warp-specialized kernel；无 spill
- **CUDA**: PTX `setmaxnreg.dec.sync.aligned.u32 40` (producer 释放寄存器)；`setmaxnreg.inc.sync.aligned.u32 232` (consumer 获取更多寄存器)
- **CUTLASS**: `TmaWarpSpecializedPingpong` 自动插入
- **Triton**: `reg_dec_producer` / `reg_inc_consumer` 编译器选项
- **验证**: SASS 出现 `SETMAXREG.INC`/`SETMAXREG.DEC`；spill → 0；TC utilization 上升

### P12: `latency.intra_wg_gemm_softmax_pipeline` — FA3 单 WG 内 GEMM-Softmax 2-stage 流水 *(NEW)*
- **典型收益**: 1.15-1.3× 在 pingpong 基础上再提升
- **触发条件**: 已应用 pingpong (P10) 或单 warpgroup；softmax→GEMM2 的依赖暴露
- **跳过条件**: 非 attention；未用 WGMMA
- **CUDA**: 在 softmax 的 max/exp 循环体内提前发起 tile_k+1 的 V load（基于 register fragment ping-pong）；把 softmax 开销和 GEMM 继续重叠
- **CUTLASS**: FA3 csrc 中的 `intra_WG_overlap_softmax_and_GEMM` 模板
- **Triton**: 通过 `num_stages=2` 在 dot→softmax→dot 链上自动做
- **互补**: `latency.pingpong_warpgroup_schedule` (P10)
- **验证**: `stall_wait` 进一步下降；attention 吞吐接近 840 TFLOPs (H100 BF16)

### P13: `latency.pdl_overlap` — Programmatic Dependent Launch (PDL)
- **典型收益**: 1.05-1.15× on multi-kernel pipeline
- **触发条件**: Nsight Systems 连续 kernel 间 5-50μs 空闲间隙；多层 GEMM/attention 串行
- **跳过条件**: sm < 90；单 kernel 或已用 CUDA Graphs
- **CUDA (sm_90+)**: `cudaTriggerProgrammaticLaunchCompletion()` + `cudaGridDependencySynchronize()`
- **CUTLASS**: `-DCUTLASS_ENABLE_GDC_FOR_SM90=1`；Blackwell 默认启用
- **验证**: Nsight Systems 连续 kernel 出现执行重叠

### P14: `latency.cooperative_cluster` — Cooperative Groups / Cluster 同步
- **典型收益**: varies
- **触发条件**: 需跨 warp/block 灵活同步粒度；全 block sync 粒度过粗
- **跳过条件**: warp shuffle 已足够
- **CUDA**: `tiled_partition<8>`；`grid_group.sync()`
- **CUTLASS (Hopper)**: Thread Block Cluster（1-16 CTA）+ DSMEM + `cluster_arrive/wait`
- **验证**: `Stall Barrier` 下降

### P15: `latency.static_launch_grid_graph` — 静态最大 grid + CUDA Graph + 内部早退
- **典型收益**: 1.2-2× on iterative small-kernel apps
- **触发条件**: kernel 在 tight loop 中被反复 launch；shape 每次变化导致无法 graph capture；launch overhead 显著
- **跳过条件**: 已用 CUDA Graph (P16)；shape 固定；launch overhead < 5%
- **CUDA**: 以最大可能的 grid 静态 launch + CUDA Graph capture；kernel 内部通过 `if (blockIdx.x >= actual_blocks) return;` 早退。配合 grid-stride loop 处理动态数据量
- **Triton**: 类似模式：固定 grid + 内部 mask
- **冲突**: `latency.cuda_graphs` (P16)
- **验证**: Nsight Systems CPU-GPU 间隙消失；launch overhead → 0

### P16: `latency.cuda_graphs` — CUDA Graphs 减少调度开销
- **典型收益**: 1.1-2× depending on launch overhead
- **触发条件**: 短 kernel 密集管线；launch 开销占 10-30%
- **跳过条件**: launch 开销 < 5%；动态 shape
- **CUDA**: stream capture 录制 kernel 序列一次提交；`cudaGraphExecUpdate` 更新参数
- **验证**: Nsight Systems CPU-GPU 间隙消失

### P17: `latency.cluster_launch_control_scheduler` — Cluster Launch Control 动态 tile 调度
- **典型收益**: 1.1-1.3× eliminating wave quantization
- **触发条件**: Blackwell sm_100+；persistent kernel 工作窃取导致 atomic 竞争；需要硬件级动态 tile 分配
- **跳过条件**: sm < 100；非 persistent kernel；grid 小
- **CUDA**: `clusterlaunchcontrol.try_cancel` PTX 查询可取消的 CTA → 获取其 blockIdx → 作为新工作 tile。Preferred + fallback cluster size 支持异构 SM 分配
- **CUTLASS**: `PersistentTileSchedulerSm100StreamK` + CLC pipeline depth=3
- **验证**: `sm__cycles_active.pct` 上升；wave quantization 消除；SASS 出现 `CLC.TRY_CANCEL`

### P18: `latency.stream_event` — Stream / Event 细粒度依赖
- **典型收益**: varies on multi-kernel workloads
- **触发条件**: 多个无依赖 kernel 串行执行；`cudaDeviceSynchronize()` 过多
- **跳过条件**: 单 kernel；已在 Graph 中
- **所有后端**: 无依赖 kernel 放不同 stream；`cudaStreamWaitEvent` 精确依赖
- **验证**: 多 kernel 并行；GPU 空闲时间减少

### P19: `latency.atomic_optimize` — Atomic 操作优化（局部归约再 atomic）
- **典型收益**: 1.2-2× on low-contention atomics
- **触发条件**: 高竞争 atomicAdd；`Stall Membar` 显著
- **跳过条件**: 无 atomic；竞争度低；已在 P8 warp_aggregated_atomics 中处理
- **CUDA**: warp/block 内局部归约再 atomic。**禁止 atomic flag spin-wait**。Triton SplitK 用 `tl.atomic_add` + `sem="relaxed"` 降低开销
- **冲突**: `latency.warp_aggregated_atomics` (P8)
- **验证**: `Stall Membar` 下降

---

## NCU Verification Checklist

| 轴 | 关键指标 | 期望变化 |
|---|---|---|
| Compute | `sm__pipe_tensor_op_*_cycles_active.pct_of_peak` | Tensor Core 路径：上升 |
| Compute | `Issue Slot Utilization` / `Eligible Warps/Cycle` | 上升 |
| Compute | `Warp Execution Efficiency` | 分支优化：上升 |
| Compute | `--ptxas-options=-v` spill count | 下降至 0 |
| Compute | `sm__inst_executed.avg.per_cycle_active` (IPC) | thread coarsening/ILP：上升 |
| Compute | `smsp__inst_executed_pipe_xu.sum` | MUFU ex2 softmax 替换后上升 |
| Memory | `Memory SOL %` / `DRAM Throughput` | 带宽受限：接近峰值 |
| Memory | `Global Load/Store Efficiency` / `Sectors/Request` | coalescing：效率上升 |
| Memory | `L1/L2 Hit Rate` | 上升 |
| Memory | `Shared Memory Efficiency` / bank conflict | 降至 0 |
| Memory | `dram__bytes.sum` / 理论最小值 | TMA multicast / DSMEM / fusion：下降 |
| Memory | `dram__bytes_write.sum` | epilogue fusion (EVT)：下降 |
| Memory | LDG.E.128 SASS instructions count | vectorized_access：上升 |
| Latency | `smsp__warps_issue_stalled_barrier` | sync 优化：下降 |
| Latency | `smsp__warps_issue_stalled_long_scoreboard` | pipeline：下降 |
| Latency | `smsp__warps_issue_stalled_membar` | warp-aggregated atomics：下降 |
| Latency | `Eligible Warps Per Cycle` | 上升 |
| Latency | Nsight Systems kernel overlap | PDL：出现重叠 |
| Latency | `lts__t_sector_hit_rate.pct` | tile swizzle：上升 |
| 综合 | **kernel latency (median_ms)** | **最终判据，必须下降** |
| 综合 | **FP8 相对误差** | two_level_accum 启用后 < 0.5% |

**常见误判**: 子指标改善但总 latency 上升 → 瓶颈转移；occupancy 上升但 latency 不降 → 非 occupancy 瓶颈；混合精度后数值通过但应用级精度下降 → 必须验证。

---

## Quick-Reference: Kernel Archetype → Top-3 Methods (v4 重排后)

| Archetype | Compute | Memory | Latency |
|---|---|---|---|
| GEMM FP16/BF16 | `tensor_core` (P1) | `tiling_smem` (P3) + `vectorized_access` (P4) | `async_pipeline` (P1) |
| FP32 GEMM, precision-tolerant | `mixed_precision` (P2) or `tf32_emulation` (P12) | `vectorized_access` (P4) | `async_pipeline` (P1) |
| Attention (FA3 style) | `gemm_softmax_interleave` (P8) + `mufu_ex2` (P13) | `kernel_fusion` (P1) | `online_recomputation` (P7) + `pingpong_warpgroup` (P10) + `intra_wg_pipeline` (P12) |
| FP8 GEMM (training, high accuracy) | `two_level_accumulation_promotion` (P9) | `epilogue_visitor_tree_fusion` (P13) | `async_pipeline` (P1) |
| FP8 GEMM (inference, fast) | `fp8_fast_accumulation_mode` (P10) | `tma_multicast` (P11) | `pingpong_warpgroup` (P10) |
| GEMM + bias + activation chain | `tensor_core` (P1) | `epilogue_visitor_tree_fusion` (P13) or `epilogue_fusion` (P7) | `tile_scheduler_swizzle` (P4) |
| Memory-bound elementwise | `thread_coarsening` (P5) | `kernel_fusion` (P1) + `vectorized_access` (P4) | `tile_scheduler_swizzle` (P4) |
| Large GEMM, low L2 hit, Hopper+ | `warp_specialization` (P3) | `tma_multicast` (P11) | `tile_scheduler_swizzle` (P4) + `stream_k_load_balancing` (P6) |
| Multi-kernel pipeline with gaps | `overlap_compute_memory` (P7) | `epilogue_fusion` (P7) | `pdl_overlap` (P13) + `cuda_graphs` (P16) |
| W4A16 quantized inference | `lop3_bit_manipulation` (P18) | `kernel_fusion` (P1) | `persistent_kernel` (P5) |
| Blackwell FP4/FP8 GEMM | `block_scaled_precision` (P11) | `tma_descriptor_tuning` (P12) | `cluster_launch_control_scheduler` (P17) |
| Histogram / filter / counting | `warp_shuffle` (P15) | `kernel_fusion` (P1) | `warp_aggregated_atomics` (P8) |
| Convolution (Hopper) | `tensor_core` (P1) | `tma_im2col_implicit_conv` (P20) | `tile_scheduler_swizzle` (P4) |
| Tall-skinny GEMM (small M×N, large K) | `tensor_core` (P1) | `split_k_parallel_reduce` (P17) | `async_pipeline` (P1) |

---

## Per-Method Trigger Summary (v4 新增/移动方法)

| Method ID | Priority v4 | Priority v3 | Trigger 核心 |
|---|---|---|---|
| `compute.launch_config` | P4 | P6 | occupancy < 25% — baseline sanity |
| `compute.thread_coarsening` | P5 | P7 | 每线程 1 元素 + IPC 低 |
| `compute.ilp_unroll_register_tile` | P6 | P8 | thread_coarsening 已应用但 IPC < 2 |
| `compute.overlap_compute_memory` | P7 | P4 | IPC < 1 且未做 multi-stage |
| `compute.two_level_accumulation_promotion` | P9 (NEW) | — | FP8 相对误差 > 0.5% |
| `compute.fp8_fast_accumulation_mode` | P10 (NEW) | — | FP8 推理精度宽松 |
| `compute.tf32_emulation_3xtf32_bf16x6` | P12 (NEW) | — | FP32 精度但需 TC 吞吐 |
| `compute.mufu_ex2_softmax_replacement` | P13 (NEW) | — | softmax/exp 密集 |
| `memory.vectorized_access` | P4 | P7 | mio_throttle 高 — 最佳 ROI |
| `memory.async_copy` | P5 | P6 | Stall Long Scoreboard 主导 |
| `memory.bank_conflict` | P8 | P10 | bank conflict > 0 |
| `memory.ldmatrix_stmatrix` | P9 | P11 | TC + mio_throttle 高 |
| `memory.register_pressure` | P10 | P12 | spill > 0 |
| `memory.epilogue_visitor_tree_fusion` | P13 (NEW) | — | 复杂 epilogue DAG |
| `memory.ldg_cache_modifier_hint` | P15 (NEW) | — | 流式读污染 L1 |
| `memory.split_k_parallel_reduce` | P17 (NEW) | — | tall-skinny GEMM |
| `memory.tma_im2col_implicit_conv` | P20 (NEW) | — | 卷积 + sm_90+ |
| `latency.persistent_kernel` | P5 | P6 | wave quantization |
| `latency.stream_k_load_balancing` | P6 (NEW) | — | 尾效应 > 10% |
| `latency.online_recomputation` | P7 | P5 | O(N²) 物化但 softmax |
| `latency.warp_aggregated_atomics` | P8 (NEW) | — | atomic 吞吐受限 |
| `latency.pingpong_warpgroup_schedule` | P10 (NEW) | — | FA3 风格 attention |
| `latency.intra_wg_gemm_softmax_pipeline` | P12 (NEW) | — | attention 单 WG 内流水 |

---

## Combining Rules

1. **同方法 id 不跨轴重复**: `memory.multi_stage_pipeline`（P6）与 `latency.async_pipeline`（P1）本质相同
2. **三方法正交**: `analysis.md` "Orthogonality check" 显式验证
3. **arch 兜底 (v4 精细化)**:
   - sm<70 → 无 TC
   - sm<75 → 无 ldmatrix
   - sm<80 → 无 cp.async / L2 persistence
   - sm<89 → 无 FP8
   - sm<90 → 无 TMA/WGMMA/Cluster/DSMEM/PDL/mbarrier
   - sm<100 → 无 tcgen05/TMEM/CLC/block-scaling
   - sm=120 (RTX 5090) → 有 FP8 但**无 TMA/WGMMA**（特殊例外）
4. **有效方法不重选**: 可选其升级版（如 `memory.epilogue_fusion` → `memory.epilogue_visitor_tree_fusion`）
5. **无效方法禁选**: 除非 ncu 根本性变化
6. **耦合规则** (v4 扩充):
   - `memory.multi_stage_pipeline`（P6）⟷ `latency.async_pipeline`（P1）
   - `memory.async_copy`（P5）⟷ `latency.async_pipeline`（P1）
   - `memory.tma_multicast`（P11）⟷ `memory.distributed_smem`（P21）
   - `compute.gemm_softmax_interleave`（P8）⟷ `latency.online_recomputation`（P7）
   - **`compute.two_level_accumulation_promotion`（P9）⟷ `compute.fp8_fast_accumulation_mode`（P10）**：互斥
   - **`memory.split_k_parallel_reduce`（P17）⟷ `latency.stream_k_load_balancing`（P6）**：互斥
   - **`latency.pingpong_warpgroup_schedule`（P10）⟷ `latency.intra_wg_gemm_softmax_pipeline`（P12）**：FA3 并用（算一个 latency slot）
   - **`memory.epilogue_fusion`（P7）⟷ `memory.epilogue_visitor_tree_fusion`（P13）**：EVT 是超集
   - **`latency.atomic_optimize`（P19）⟷ `latency.warp_aggregated_atomics`（P8）**：后者是前者的具体实现
   - **`latency.cuda_graphs`（P16）⟷ `latency.static_launch_grid_graph`（P15）**：后者扩展动态 shape
