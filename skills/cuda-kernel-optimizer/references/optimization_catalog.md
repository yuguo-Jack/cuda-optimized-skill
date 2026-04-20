# Optimization Catalog — Priority-Ordered

> **核心原则**: 每个轴（compute / memory / latency）内的方法按**重要性严格递降排列**。选择方法时**必须从 P1 开始向下扫描**,选第一个同时满足以下三个条件的方法:
> 1. ncu 指标显示该方法对应的瓶颈存在（"触发条件"栏）
> 2. 方法 id 不在 `state.selected_methods` 中（未被选过）
> 3. 方法的 arch 要求不超过检测到的 `sm_*`
>
> **禁止跳过高优先级方法去选低优先级方法**,除非高优先级方法的"跳过条件"栏明确被满足。跳过时必须在 `analysis.md` 的"排除候选"段落中记录原因。

---

## Table of Contents

- [Selection Decision Tree](#selection-decision-tree)
- [Compute Axis](#compute-axis-按优先级)
- [Memory Axis](#memory-axis-按优先级)
- [Latency Axis](#latency-axis-按优先级)
- [NCU Verification Checklist](#ncu-verification-checklist)
- [Combining Rules](#combining-rules)

---

## Selection Decision Tree

每次迭代选 3 个方法（每轴 1 个）时,按此流程:

```
对于 axis ∈ {compute, memory, latency}:
  for priority = P1, P2, P3, ...:
    method = catalog[axis][priority]
    if method.id ∈ selected_methods:        → 跳过(已试过)
    if method.arch > detected sm_arch:       → 跳过(架构不够)
    if method.skip_condition 成立:            → 跳过(记录原因)
    if method.trigger_condition 不匹配 ncu:  → 跳过(瓶颈不在这)
    else:                                    → 选中,停止扫描
```

如果某轴全部方法要么已选过、要么不适用,则该轴空选,在 `analysis.md` 中声明"当前轮该轴无可用新方法"。

---

## Compute Axis (按优先级)

### P1: `compute.tensor_core` — Tensor Core / 专用硬件利用
- **触发条件**: `sm__pipe_tensor_op_hmma_cycles_active.pct_of_peak < 20%`（应为 GEMM/conv/attention 类算子）
- **跳过条件**: 算子不包含矩阵乘加语义（纯 elementwise / reduction / scan）,或 Tensor Core 利用率已 > 60%
- **CUDA**: 将标量 `a[i]*b[j]` 累加替换为 `mma.sync` PTX 或 `nvcuda::wmma` fragment；操作数布局需匹配架构 fragment layout
- **CUTLASS**: 选择 `OpClassTensorOp` 的 MMA Atom。Ampere 用 `SM80_16x8x16_F32F16F16F32`；Hopper 用 `SM90_64x128x16_F32F16F16F32_SS`（WGMMA）。不用 Tensor Core 做矩阵类运算基本等于浪费一半以上芯片算力
- **Triton**: 确保 `tl.dot` 输入为 FP16/BF16/FP8（不是 FP32）；检查是否有意外 `.to(tl.float32)` 提升。`input_precision="tf32"` 是 FP32 输入折中
- **验证**: SASS 中出现 `HMMA`（Ampere）或 `WGMMA`（Hopper）；`sm__pipe_tensor_op_*_cycles_active` 显著上升

### P2: `compute.overlap_compute_memory` — 计算与访存重叠（延迟隐藏）
- **触发条件**: IPC (`sm__inst_executed.avg.per_cycle_active`) < 1 且计算单元不饱和；或 `Eligible Warps Per Cycle` 偏低
- **跳过条件**: kernel 是纯 elementwise（无复用）,无重叠空间
- **CUDA**: 提升 occupancy（更多 warp 可切换）；增大 per-thread ILP（每线程处理多个独立元素）；软件流水线拆 load-compute-store 三阶段交错执行
- **CUTLASS**: 增加 Pipeline `Stages` 数（2→4）；选 Warp Specialization schedule（Hopper）使 producer/consumer 物理并行
- **Triton**: 调大 `num_stages`（2→4）；调整 `num_warps` 使 SM 有更多可切换 warp
- **验证**: `Eligible Warps Per Cycle` 上升；`Stall Long Scoreboard` 下降

### P3: `compute.launch_config` — Launch Configuration / Tile Shape
- **触发条件**: `sm__warps_active.pct_of_peak < 25%` 或 `sm__cycles_active.pct_of_peak < 50%`；`launch__occupancy_limit_*` 指标指出限制因素
- **跳过条件**: occupancy 已 > 50% 且 IPC 已 > 2（此时 occupancy 不是瓶颈）
- **CUDA**: 调 block size（128/256/512）；用 `cudaOccupancyMaxPotentialBlockSize` 辅助；加 `__launch_bounds__(maxThreads, minBlocks)`
- **CUTLASS**: 调 CTA Tile Shape（常用起点：128×128×32 Ampere / 128×128×64 Hopper）；调 Cluster Shape（Hopper）；对 2-4 个候选做 benchmark 选最优
- **Triton**: 调 `BLOCK_M/N/K` + `num_warps`；用 `triton.autotune` 列 4-8 个 config 自动选最优
- **验证**: occupancy 上升；总 kernel latency 下降（occupancy 上升但 latency 不降说明不是真正瓶颈）

### P4: `compute.reduction` — 归约优化
- **触发条件**: kernel 含归约语义（sum/max/softmax/layernorm）；归约部分占总时间 > 30%
- **跳过条件**: 算子无归约
- **CUDA**: sequential addressing → warp unrolling → warp shuffle 归约；block 间用 atomic 或二次 kernel。Blelloch/Hillis-Steele 用于 scan
- **CUTLASS**: Split-K（K 大 M×N 小时）或 Stream-K（CTA 数不能填满 SM 时）；Split-K 支持 `SplitKSerial` 和 `SplitKAtomic`
- **Triton**: `tl.sum` / `tl.max` / `tl.reduce`（编译器自动 warp shuffle）；跨 program 归约用 `tl.atomic_add` 或独立 reduction kernel；`tl.associative_scan` 用于 scan
- **验证**: 归约路径 barrier stall 下降；整体 latency 下降

### P5: `compute.warp_shuffle` — Warp Shuffle 用于计算
- **触发条件**: kernel 有 warp 内数据交换需求（reduction 尾部、scan、broadcast）；当前用 shared memory + sync 实现
- **跳过条件**: 已在 P4 中用 shuffle 做了归约；或 CUTLASS/Triton 已自动处理
- **CUDA**: `__shfl_xor_sync` / `__shfl_down_sync` 替代 shared memory tree reduction,约 1 个时钟周期,无 bank conflict
- **CUTLASS**: MMA Atom 内建；自定义 epilogue 中手动使用
- **Triton**: `tl.sum`/`tl.reduce` 自动使用；无直接 `__shfl` API,需 inline PTX 或重构
- **验证**: shared memory load/store 指令数下降；barrier stall 下降

### P6: `compute.loop_unroll` — 循环展开
- **触发条件**: `short_scoreboard`/`wait` stall 显著；SASS 中热循环有大量标量依赖指令
- **跳过条件**: 编译器已完全展开（检查 SASS 无循环控制指令）
- **CUDA**: `#pragma unroll` / `#pragma unroll N`
- **CUTLASS**: `CUTLASS_PRAGMA_UNROLL`；CuTe 的 `Int<N>` 静态整数使循环编译期消除
- **Triton**: `tl.static_range`；`BLOCK_K` 作为 `tl.constexpr` 时编译器自动展开
- **验证**: 指令数减少；`Issue Slot Utilization` 上升

### P7: `compute.fma_and_fast_math` — FMA + 编译选项
- **触发条件**: SASS 中 `a*b` 和 `+c` 是两条独立指令而非 FMA；或启用 `--use_fast_math` 可接受精度
- **跳过条件**: 精度要求严格（如 FP64 科学计算）
- **CUDA**: `__fmaf_rn()` 显式 FMA；`--use_fast_math` / `--ftz=true --prec-div=false`。除法 → `a * __frcp_rn(b)`,吞吐差距 4×-16×
- **CUTLASS**: Tile 内计算由 MMA 指令覆盖；epilogue 中注意 FMA 合并
- **Triton**: 编译器自动 FMA 融合；`allow_tf32=True`（默认）走 TF32 Tensor Core
- **验证**: 指令数减少；kernel latency 下降

### P8: `compute.strength_reduce` — 强度削减 + __restrict__
- **触发条件**: 热循环中有整数除法/取模（对变量）；指针 alias 阻碍编译器优化
- **跳过条件**: 编译器已自动优化（常量除数自动转乘移位）
- **CUDA**: 移位替代 2 的幂乘除；`rsqrtf()` 替代 `1.0f/sqrtf()`；`__restrict__` 修饰所有指针参数（零代价,应为默认）
- **CUTLASS**: `FastDivmod` 已内建；自定义 epilogue 中手动应用
- **Triton**: 编译器自动强度削减；`tl.math.fast_expf` 等快速近似
- **验证**: ALU 指令减少

### P9: `compute.warp_specialization` — Warp Specialization
- **触发条件**: barrier + long_scoreboard 都非平凡；kernel 是 GEMM/attention 类；当前所有 warp 既搬数据又算
- **跳过条件**: 非 GEMM 类；sm < 80；已在 P2 中处理
- **CUDA**: 手动 producer-consumer warp 分组,producer 做 cp.async/TMA,consumer 做 MMA,通过 mbarrier 协调
- **CUTLASS (Hopper)**: `TmaWarpSpecialized` / `TmaWarpSpecializedCooperative` / `TmaWarpSpecializedPingpong`——CUTLASS 3.x Hopper 核心架构
- **Triton**: Hopper 后端实验性支持；编译器自动决策
- **验证**: producer warp 的 barrier stall 极低；consumer warp 的 Tensor Core 利用率提升

### P10: `compute.branch_eliminate` — 分支消除 + 谓词化
- **触发条件**: `Warp Execution Efficiency` 低；SASS 中大量 BRA/SSY 指令
- **跳过条件**: 无明显分支发散
- **CUDA**: `condition ? a : b` select 指令替代 if-else；预排序让同类数据聚集在同一 warp；`__all_sync()` / `__any_sync()` warp vote 做 early exit
- **CUTLASS**: `CpAsyncPredicated` 用谓词遮罩处理边界,避免分支；2:4 稀疏自动跳零
- **Triton**: `tl.where(cond, x, y)` 编译为 select 指令；`tl.load(mask=mask)` 谓词化加载
- **验证**: `Warp Execution Efficiency` 上升；分支相关 stall 下降

---

## Memory Axis (按优先级)

### P1: `memory.kernel_fusion` — Kernel Fusion
- **触发条件**: reference 中相邻的 producer-consumer kernel 可合并；中间数据经 Global Memory 往返。**这往往是最高收益的单项优化**
- **跳过条件**: kernel 已经是融合后的形态；后续操作有跨 tile 全局依赖（如 LayerNorm 的全局均值）
- **CUDA**: 合并相邻 kernel,中间结果留在寄存器或 shared memory,消除一次完整 Global Memory 往返
- **CUTLASS**: Epilogue Visitor Tree（EVT）实现 GEMM + Bias + Activation fusion。支持线性组合、逐元素激活、Bias 加法、行列广播、多输出 `AuxStore`
- **Triton**: 直接在同一 `@triton.jit` 中编写融合逻辑——`tl.dot` → bias → activation → `tl.store`,灵活性远超 CUTLASS EVT。Flash Attention 范式（GEMM + Softmax + GEMM）可节省多次 Global Memory 读写
- **验证**: `DRAM Read Bytes / 理论数据量` 接近 1.0；kernel launch 次数减少

### P2: `memory.coalesced_access` — 合并访问
- **触发条件**: `Global Load/Store Efficiency < 50%`；`Sectors/Request` 远大于 1；stride 访问或随机访问导致事务数暴增（最坏 32 倍带宽浪费）
- **跳过条件**: 最内维 stride 为 1（天然合并）；CUTLASS TiledCopy / Triton `tl.make_block_ptr` 已保证
- **CUDA**: 确保 warp 内 32 线程访问连续 128B 地址；转置场景先 coalesced load → shared memory → 转置读。SoA 布局天然合并,AoS 布局需转换
- **CUTLASS**: Layout 代数的最内维 stride=1 即保证合并；`RowMajor`/`ColumnMajor` 已内建。**不合并的 Layout 在编译期就能通过 stride 值发现**
- **Triton**: `tl.make_block_ptr` 的 `order` 参数声明连续维；手动指针算术需确保最内维 stride=1；编译器从 tile 指针 block 自动推导映射
- **验证**: `Sectors/Request` 接近 1；`Global Load Efficiency` 上升

### P3: `memory.tiling_smem` — Tiling / Shared Memory 分块
- **触发条件**: L2 hit rate < 30%；`dram__bytes.sum` 远超算法理论最小值（数据被重复读取）。对任何**数据复用率 > 1** 的算法都是关键手段
- **跳过条件**: 算子无数据复用（纯 elementwise）
- **CUDA**: 按 tile 将 Global Memory 数据搬到 Shared Memory,在片上多次复用；矩阵乘法经典——每个 tile 从 Global 读一次,在 Shared 复用 O(N) 次
- **CUTLASS**: CTA Tile Shape（`TileShape_MNK`）定义分块大小；Collective Mainloop 自动管理搬运。Cluster Tile（Hopper）通过 TMA multicast 扩大复用范围而不增单 CTA 的 smem 用量
- **Triton**: `BLOCK_M/N/K` 定义 tile；编译器自动将 `tl.load` 数据提升到 Shared Memory。`BLOCK_K` 越大 K 方向复用越深但 smem 用量增加
- **验证**: L2 hit rate 上升；`dram__bytes.sum` 下降

### P4: `memory.register_pressure` — 寄存器压力控制
- **触发条件**: `launch__occupancy_limit_registers` 是主要限制；`--ptxas-options=-v` 显示 spill > 0。过度 spill 到 Local Memory 退化为全局内存访问,极其昂贵
- **跳过条件**: 寄存器用量适中（spill = 0）；occupancy 限制因素是 shared memory
- **CUDA**: `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` 引导编译器；缩小 per-thread tile；将大数组显式放 shared memory
- **CUTLASS**: 缩小 CTA Tile Shape 或选更大 MMA Atom（减少 warp 级重复数）；减少 Pipeline stage 数。累加器大小 ≈ (CTA_M/WarpTile_M)×(CTA_N/WarpTile_N)×WarpTile_M×WarpTile_N×sizeof(Accum)
- **Triton**: 缩小 `BLOCK_M/N`；增大 `num_warps`（将工作分摊到更多 warp）；减少同时活跃的 tile 变量数。Triton 无 `__launch_bounds__` 等价物,只能间接控制
- **验证**: spill load/store 降至 0；occupancy 上升

### P5: `memory.multi_stage_pipeline` — 双缓冲 / 多级流水线
- **触发条件**: `Stall Long Scoreboard` 占主导；当前 shared memory 只有 single buffer。所有高性能 GEMM（cuBLAS、CUTLASS）的核心技术
- **跳过条件**: 已有多级 pipeline（检查 shared memory 分配是否已 N× tile size）
- **CUDA**: 分配 N 组 shared memory buffer,ping-pong 交替；`cp.async.commit_group` + `cp.async.wait_group<N-1>` 实现多级流水
- **CUTLASS**: `Stages` 模板参数（2→4）；`StageCountAutoCarveout` 根据 smem 容量自动选最大可用 stage。Ampere 3-4 stage 通常最优；Hopper 2-4 stage 即可
- **Triton**: `num_stages=3~5`；编译器自动生成 prolog/steady-state/epilog 流水线代码。Raising past 5 usually regresses
- **验证**: `Stall Long Scoreboard` 下降；kernel latency 下降

### P6: `memory.vectorized_access` — 向量化访存
- **触发条件**: `mio_throttle` 显著；大量标量 load 指令（SASS 中 `LDG.E.32` 而非 `LDG.E.128`）
- **跳过条件**: 数据类型或对齐不支持 128-bit 加载；CUTLASS Copy Atom / Triton 编译器已自动向量化
- **CUDA**: `float4`/`int4` 等宽类型单条指令读写 128-bit；需保证地址对齐到向量宽度
- **CUTLASS**: Copy Atom 指定 128-bit（`uint128_t`）；`DefaultCopy` 根据类型和对齐自动选
- **Triton**: 编译器根据数据类型+对齐自动向量化；`tl.make_block_ptr` 帮助推导；确保 tensor 指针 16 字节对齐
- **验证**: `Sectors/Request` 接近 1；`mio_throttle` 下降

### P7: `memory.async_copy` — 异步拷贝 (cp.async / TMA)
- **触发条件**: `Stall Long Scoreboard` 占主导；当前 Global→Shared 走 reg 中转（非 DMA）
- **跳过条件**: sm < 80（无 `cp.async`）；已在 P5 中配合 pipeline 使用（合并计入 P5）
- **CUDA (sm_80+)**: `cp.async.cg.shared.global`（bypass L1）或 `cp.async.ca`（keep L1）；配合 `commit_group/wait_group`
- **CUDA (sm_90+)**: TMA（`cp.async.bulk.tensor`）由硬件 DMA 引擎执行,不占线程资源；配合 `mbarrier`。TMA multicast 一次搬运写入 Cluster 内多个 CTA 的 smem
- **CUTLASS**: `MainloopSm80CpAsync`（Ampere）；`MainloopSm90Tma*`（Hopper）；框架自动管理
- **Triton**: 编译器自动选择；`tl.make_block_ptr` 更容易触发 TMA（Hopper）；手动指针算术可能阻碍 TMA
- **验证**: SASS 中出现 `cp.async`/TMA 指令；`Stall Long Scoreboard` 下降

### P8: `memory.bank_conflict` — Bank Conflict 消除
- **触发条件**: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{ld,st}.sum` > 0
- **跳过条件**: 指标已为 0；kernel 不使用 shared memory
- **CUDA**: Padding（`__shared__ float s[32][33]`）或 Swizzle/XOR 索引（`col ^ ((row & mask) * shift)`）。Swizzle 零空间浪费但需编译期计算
- **CUTLASS**: `Swizzle<B,M,S>` 参数；预定义 SmemLayout 已包含正确 Swizzle。验证后 `l1tex__data_bank_conflicts_*` 应降至 0
- **Triton**: 编译器自动 swizzle；如仍有 conflict,调整 `BLOCK_K` 使其对 bank 宽度友好（避免 32 的倍数）
- **验证**: bank conflict 计数器降至 0

### P9: `memory.data_layout` — SoA / 对齐 / Padding / 数据重排
- **触发条件**: AoS 布局导致 warp 同字段地址不连续；子矩阵偏移破坏对齐；不规则访问模式（稀疏/图）
- **跳过条件**: 数据已是 SoA；`cudaMalloc` 分配天然 256B 对齐
- **CUDA**: AoS → SoA；`cudaMallocPitch` 保证行对齐；稀疏/图数据做 Z-order/Hilbert 预排序或 CSR 按行长分桶
- **CUTLASS**: 独立矩阵天然 SoA；leading dimension 编码 pitch；2:4 结构化稀疏（Ampere+）
- **Triton**: 独立 tensor 指针天然 SoA；stride 参数支持任意 pitch；`tl.load(ptr + index_tensor)` 做 gather
- **验证**: `Global Load Efficiency` 上升；L1 hit rate 改善

### P10: `memory.cache_and_readonly` — L2/只读路径 + smem 容量配置
- **触发条件**: 反复访问的小热点数据；或当前 smem 容量不足以放大 tile
- **跳过条件**: 新架构编译器已自动优化只读路径
- **CUDA**: `const __restrict__` + `__ldg()`；`cudaAccessPolicyWindow` L2 Persistence；`cudaFuncSetAttribute` 调大 smem 比例（Ampere 最大 164KB）
- **CUTLASS**: Copy Atom 缓存策略 `CACHEALWAYS`/`CACHEGLOBAL`；kernel launch 自动调 smem
- **Triton**: `eviction_policy` 参数；编译器自动管理 smem 容量和 `cudaFuncSetAttribute`
- **验证**: L2 hit rate 上升；smem 扩容后 tile 可增大

---

## Latency Axis (按优先级)

### P1: `latency.warp_shuffle_sync` — Warp Shuffle 替代 Shared Memory 同步
- **触发条件**: warp 内数据交换当前经"写 shared → `__syncthreads()` → 读 shared"三步；`smsp__warps_issue_stalled_barrier` 显著
- **跳过条件**: CUTLASS MMA Atom 已内建；Triton 编译器已自动处理；无 warp 内通信需求。**不是"减少"同步,而是"消除"同步**
- **CUDA**: `__shfl_sync` 系列替代 shared memory 中转,延迟约 1 个时钟周期,无 bank conflict,不占 smem 容量
- **CUTLASS**: MMA Atom 内建；自定义 epilogue 中手动使用
- **Triton**: `tl.sum`/`tl.max`/`tl.reduce` 自动使用 shuffle；不暴露直接 API
- **验证**: `Stall Barrier` 下降；shared memory load/store 指令数下降

### P2: `latency.reduce_sync_count` — 减少 __syncthreads() 次数
- **触发条件**: `smsp__warps_issue_stalled_barrier` 占主导 stall；kernel 中多处 `__syncthreads()`
- **跳过条件**: 仅有 1-2 个 sync 且均为必要（有真实跨 warp 依赖）
- **CUDA**: 审查每处 sync 是否有真实跨 warp 依赖——无依赖直接删除；双缓冲合并两次 sync 为一次；`__syncwarp(mask)` 替代全 block sync（同步粒度从上千线程→32 线程）。**每次删除或替换同步点后,必须用 `compute-sanitizer --tool racecheck` 验证无数据竞争**
- **CUTLASS**: Pipeline 的 arrive/wait 语义替代全 block sync——从"全员到齐"变为"特定 stage 数据就绪"
- **Triton**: 编译器自动管理；`num_stages` 间接控制；**禁止使用 `tl.debug_barrier()`**（会破坏编译器 pipeline 优化）
- **验证**: `Stall Barrier` 下降；`Eligible Warps Per Cycle` 上升

### P3: `latency.async_pipeline` — 异步流水线消除同步等待
- **触发条件**: `Stall Long Scoreboard` 占主导；全局加载等待时间长；当前无异步搬运
- **跳过条件**: 已在 memory P5/P7 中实现（标记为"已覆盖",换不同方法）；kernel 无 smem staging
- **CUDA (sm_80+)**: `cp.async` + `commit_group/wait_group` 实现"等数据"替代"等线程"
- **CUDA (sm_90+)**: TMA + 硬件 barrier 的 transaction-based 语义——等待指定字节数到达,精确到字节级
- **CUTLASS**: `PipelineAsync`（Ampere）/ `PipelineTmaAsync`（Hopper）/ `PipelineTransactionAsync`（Hopper,支持字节级"数据到达"语义）
- **Triton**: `num_stages` 是唯一需要调的参数；编译器自动生成 cp.async/TMA pipeline
- **验证**: `Stall Long Scoreboard` 下降；Nsight Systems 中观察到搬运与计算时间重叠

### P4: `latency.cooperative_cluster` — Cooperative Groups / Cluster 同步
- **触发条件**: 需要跨 warp 或跨 block 的灵活同步粒度；全 block `__syncthreads()` 粒度过粗
- **跳过条件**: 无跨 warp 协作需求；warp shuffle 已足够
- **CUDA**: `tiled_partition<8>(this_thread_block())` 创建 8 线程子组,`.sync()` 仅同步 8 线程；`grid_group` 提供 grid 级同步（替代 atomic flag 轮询）
- **CUTLASS (Hopper)**: Thread Block Cluster（1-16 CTA）替代 Cooperative Groups 跨 block 协作；DSMEM 直接读写彼此 smem；`cluster_arrive/wait` 比 `grid_group.sync()` 更轻量
- **Triton (Hopper)**: `num_ctas` 参数定义 Cluster；编译器自动生成 cluster 级 barrier
- **验证**: `Stall Barrier` 下降；跨 block 通信延迟降低

### P5: `latency.cuda_graphs` — CUDA Graphs 减少调度开销
- **触发条件**: 短 kernel 密集管线（推理 pipeline）；kernel 本身只有几十 μs 但 launch 开销占 10-30%
- **跳过条件**: kernel launch 开销 < 总时间 5%；动态 shape 频繁变化
- **CUDA**: stream capture 模式录制 kernel 序列,一次提交；`cudaGraphExecUpdate` 支持参数更新
- **CUTLASS**: `GemmUniversal::run()` 支持 graph capture
- **Triton**: `torch.cuda.graph()` 上下文中调用；`torch.compile` + Inductor 自动组合进 Graph
- **验证**: Nsight Systems 中 CPU-GPU 间隙消失；总 pipeline latency 下降

### P6: `latency.stream_event` — Stream / Event 细粒度依赖
- **触发条件**: 多个无依赖的 kernel/GEMM 串行执行；`cudaDeviceSynchronize()` 过多
- **跳过条件**: 只有单个 kernel；已在 Graph 中管理
- **所有后端**: 无依赖 kernel 放不同 stream；`cudaStreamWaitEvent` 精确依赖。**`cudaDeviceSynchronize()` 是全设备 barrier,应仅在调试或最终结果回传时使用**
- **验证**: Nsight Systems 中多 kernel 并行执行；GPU 空闲时间减少

### P7: `latency.persistent_kernel` — 持久化 / Stream-K 调度
- **触发条件**: grid 太小无法填满 SM（`sm__cycles_active.pct < 50%`）；或不均匀 workload 导致尾部拖慢
- **跳过条件**: grid 已能充分填满 SM；工作均匀
- **CUDA**: launch `num_sms × waves` 个 block,每个循环处理工作队列
- **CUTLASS**: Stream-K 调度将工作均匀分配到所有 SM
- **Triton**: 手动 persistent kernel + `tl.atomic_add` 工作窃取；或 split-K 增加 grid 第三维
- **验证**: `sm__cycles_active.pct` 上升；尾部空闲时间消失

### P8: `latency.atomic_optimize` — Atomic 操作优化
- **触发条件**: 大量线程 atomic 同一地址（高竞争 atomicAdd）；`Stall Membar` 显著
- **跳过条件**: kernel 无 atomic；竞争度低
- **CUDA**: 先做 warp/block 内局部归约再做一次 atomic；**禁止用 atomic flag 轮询实现 spin-wait——GPU 上可能死锁**（占住被等待 block 的 SM 资源）。改用 `grid_group.sync()` 或拆分 kernel
- **CUTLASS**: Split-K epilogue 已做"warp 局部归约 + 一次 atomic"
- **Triton**: tile 内 `tl.sum` 局部归约后再 `tl.atomic_add`；调整 grid 使竞争分散
- **验证**: `Stall Membar` 下降

---

## NCU Verification Checklist

优化后**必须**检查以下指标,确认优化方向与实际效果一致:

| 轴 | 关键指标 | 期望变化 |
|---|---|---|
| Compute | `sm__pipe_tensor_op_*_cycles_active.pct_of_peak` | Tensor Core 路径：上升 |
| Compute | `Issue Slot Utilization` / `Eligible Warps/Cycle` | 上升 |
| Compute | `Warp Execution Efficiency` | 分支优化：上升 |
| Compute | `--ptxas-options=-v` spill count | 下降至 0 |
| Memory | `Memory SOL %` / `DRAM Throughput` | 带宽受限场景：接近峰值 |
| Memory | `Global Load/Store Efficiency` / `Sectors/Request` | coalescing：效率上升 |
| Memory | `L1/L2 Hit Rate` | 上升 |
| Memory | `Shared Memory Efficiency` / bank conflict count | conflict：降至 0 |
| Latency | `smsp__warps_issue_stalled_barrier` | sync 优化：下降 |
| Latency | `smsp__warps_issue_stalled_long_scoreboard` | pipeline 优化：下降 |
| Latency | `Eligible Warps Per Cycle` | 上升 |
| 综合 | **kernel latency (median_ms)** | **最终判据,必须下降** |

**常见误判**:
- 某个子指标改善但总 latency 上升 → 瓶颈转移到了别处,不算"有效"
- occupancy 上升但 latency 不降 → 不是 occupancy 瓶颈,不算"有效"
- 只看吞吐不看正确性 → 每次必须先过 `compute-sanitizer --tool racecheck`

---

## Combining Rules

1. **同一方法 id 不能同时出现在两个轴的选择中**。如 `memory.multi_stage_pipeline`（P5）与 `latency.async_pipeline`（P3）本质相同,选了一个另一个自动标为"已覆盖",换一个不同的方法。

2. **三个方法必须正交**——不能一对是同一优化的两个名字。在 `analysis.md` 的 "Orthogonality check" 中显式验证。

3. **arch 兜底**:
   - sm < 70：无 Tensor Core,compute P1 跳过
   - sm < 80：无 `cp.async`,memory P7 和 latency P3 跳过
   - sm < 90：无 TMA / WGMMA / Cluster,相关 Hopper 变体跳过

4. **有效方法的增量复用**：如果 `state.effective_methods` 中已有某方法,本轮不能重选同 id,但可以选其**升级版**（如 `compute.tensor_core` 有效后下一轮选 `compute.warp_specialization` 进一步压榨）。

5. **无效方法的规避**：`state.ineffective_methods` 中的方法本轮禁选,除非 ncu 指标结构已根本性改变（在 `analysis.md` 中说明变化依据）。
