# Triton Kernel 优化完整策略

> 50+ 项优化 → 12 个 Pack → 7 个 Phase → 7-10 次 profile 收敛
> 本文档将原始 CUDA 优化框架映射到 Triton 编译器框架层面。核心方法论（NCU 驱动、按冲突域分组、瓶颈转移检测）。

---

## 一、核心方法论

与原始策略完全一致：

1. **指标驱动，不凭直觉**：每一步优化都由 NCU 指标的异常值触发
2. **按冲突域分组**：不争夺资源的优化可以打包执行，争夺资源的必须隔离验证
3. **瓶颈会转移**：每轮优化后必须重新 Roofline 分类
4. **算子运算速度是最终裁决**：benchmark 实测的 speedup 为主判据；差距 \(< 3\%\) 时优选 NCU 子指标更优的版本

**Triton 特有原则**：

5. **数据类型决定指令路径**：`tl.dot` 的输入精度决定走 Tensor Core 还是 CUDA Core。选错精度等于方向性错误，后续优化无意义
6. **Block Shape 是调优的中心维度**：`BLOCK_M/N/K` 同时影响 Shared Memory 用量、寄存器压力、Occupancy、数据复用率、Tensor Core 利用率——它是内存/计算/同步三个维度的交汇点
7. **编译器是隐含的优化者**：Triton 编译器自动处理 swizzle、pipeline、barrier 插入、向量化等。程序员的代码结构（如 `tl.make_block_ptr` vs 手动指针算术）会显著影响编译器的优化能力
8. **`num_stages` 是延迟隐藏的调节阀**：在 Shared Memory 预算内最大化 stage 数，但不能牺牲 Occupancy
9. **`triton.autotune` 是系统化搜索工具**：对 Block Shape × `num_warps` × `num_stages` 的笛卡尔积做 benchmark，选择最优配置

---

## 二、Step 0 — Baseline 采集

### 2.1 Benchmark 基准速度

使用 `triton.testing.do_bench` 或自定义 benchmark 框架：

关注 `median_ms`（算子运算速度）、`speedup_vs_reference`（相对 PyTorch/cuBLAS 的加速比）、`bandwidth_gbps`（是否接近硬件上界）。

### 2.2 NCU 全量 Profile

对编译后的 Triton kernel 做 NCU profile。Triton kernel 编译后是标准 CUDA kernel，NCU 完全适用。

关注 Speed Of Light (SOL) 面板：

- **Memory SOL%**：`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`
- **Compute SOL%**：`sm__throughput.avg.pct_of_peak_sustained_elapsed`

---

## 三、Step 1 — Roofline 分类

| 条件 | 瓶颈类型 | 含义 |
|---|---|---|
| Memory SOL% > Compute SOL% | Memory-bound | 带宽是瓶颈 |
| Compute SOL% > Memory SOL% | Compute-bound | 算力是瓶颈 |
| 两者均 \(< 30\%\) | Latency-bound | 线程在等待 |

---

## 四、Step 2 — NCU 子指标精准定位

### 4.1 Memory-bound 诊断表（15 项）

| 诊断维度 | NCU 计数器 | 异常阈值 | Triton 对应优化（按优先级） |
|---|---|---|---|
| 顶层判定 | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | > Compute SOL% | 确认 memory-bound |
| DRAM 带宽利用率 | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 接近 SOL | [Mem] ① Kernel Fusion 减少总访存 ② 已接近极限，转 compute 优化 |
| Global Load Efficiency | `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` | \(< 80\%\) | [Mem] ① 检查 `tl.load` 的指针偏移是否连续（内维 stride=1） ② 使用 `tl.make_block_ptr` 改善编译器推导 ③ 检查对齐 |
| Global Store Efficiency | `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct` | \(< 80\%\) | 同上，额外检查 `tl.store` 的指针模式 |
| Sectors/Request | `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` | > 4 | [Mem] ① 内维 stride ≠ 1（转置访问 / AoS 问题） ② 需要数据预处理 SoA 重构 ③ 对齐降级 |
| L1 Hit Rate | `l1tex__t_sector_hit_rate.pct` | \(< 50\%\) | [Mem] ① 增大 Block Shape → 提升 Shared Memory 中数据复用 ② 检查 `eviction_policy` |
| L2 Hit Rate | `lts__t_sector_hit_rate.pct` | \(< 40\%\) | [Mem] ① L2 Persistence ② 减小 `BLOCK_K`（缩小 working set） ③ Kernel Fusion |
| Shared Mem Efficiency | `smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct` | \(< 90\%\) | [Mem] ① **编译器 swizzle 失效** → 调整 Block Shape 使其对 bank 宽度友好 ② 检查编译器 IR |
| Shared Bank Conflict | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | > 0 | [Mem] ① 调整 `BLOCK_K` 避免 bank 对齐冲突 ② 检查编译器自动 swizzle 是否生效 |
| DRAM Read Bytes | `dram__bytes_read.sum` | 远大于理论数据量 | [Mem] 带宽放大系数 > 2× → coalescing 或 tile 复用率问题 |

### 4.2 Compute-bound 诊断表（13 项）

| 诊断维度 | NCU 计数器 | 异常阈值 | Triton 对应优化（按优先级） |
|---|---|---|---|
| 顶层判定 | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | > Memory SOL% | 确认 compute-bound |
| SM FMA 管线 | `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | FMA 高但无 Tensor | [Comp] ① **`tl.dot` 输入精度错误** → 确保输入为 FP16/BF16/FP8 ② 检查是否有意外的 `.to(tl.float32)` |
| Tensor Pipe | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | \(< 30\%\)（已用 TC） | [Comp] ① Block Shape 不匹配 → 调整 `BLOCK_M/N` ② Data feeding 不足 → 增加 `num_stages` ③ `num_warps` 过大 |
| ALU Pipe | `sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active` | > 80% | [Comp] ① 检查 epilogue 中的逐元素运算是否过重 ② 将运行时计算转为 `tl.constexpr` 编译期常量 ③ 减少 mask 计算开销 |
| Issue Slot Utilization | `smsp__issue_active.avg.pct_of_peak_sustained_active` | \(< 60\%\) | [Comp] ① `num_stages` 不足 → 增加 stage 暴露 ILP ② Block Shape 调优 |
| Warp Execution Efficiency | `smsp__thread_inst_executed_per_inst_executed.ratio` | \(< 85\%\) | [Comp] ① 检查 `mask` 使用是否导致分支发散 ② 使用 `tl.where` 替代 `if-else` |
| Register Spill | NCU register 统计 | spill > 0 | [Comp] ① 缩小 `BLOCK_M/N` ② 增大 `num_warps` ③ 减少 `num_stages` |
| Achieved Occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` | \(< 70\%\) 理论值 | [Comp] ① 缩小 Block Shape ② 减少 `num_stages` ③ 增大 `num_warps` |

### 4.3 Latency-bound 诊断表（14 项）

| 诊断维度 | NCU 计数器 | 异常阈值 | Triton 对应优化（按优先级） |
|---|---|---|---|
| 顶层判定 | Memory/Compute SOL% | 两者均 \(< 30\%\) | 确认 latency-bound |
| Stall Barrier | `smsp__warps_issue_stalled_barrier_per_issue_active.pct` | > 30% | [Sync] ① 增加 `num_stages` ② 检查编译器 pipeline 是否正确生成 ③ 使用 `tl.make_block_ptr` 改善 pipeline 质量 |
| Stall Long Scoreboard | `smsp__warps_issue_stalled_long_scoreboard_per_issue_active.pct` | > 25% | [Mem] 等 L2/DRAM → ① 增加 `num_stages` ② 使用 `tl.make_block_ptr` 触发 TMA（Hopper） |
| Stall Short Scoreboard | `smsp__warps_issue_stalled_short_scoreboard_per_issue_active.pct` | > 15% | [Mem] 等 L1/shared → ① **检查 bank conflict**（调整 `BLOCK_K`） ② 增大 `BLOCK_M/N` 暴露 ILP |
| Stall Memory Throttle | `smsp__warps_issue_stalled_lg_throttle_per_issue_active.pct` | > 20% | [Mem] ① 增加 `num_stages` ② 使用 `tl.make_block_ptr` 触发异步路径 |
| Stall MIO Throttle | `smsp__warps_issue_stalled_mio_throttle_per_issue_active.pct` | > 10% | [Sync] shared/atomic 过载 → ① 减少 shared memory 访问密度 ② 减少 `tl.atomic_add` 调用 |
| Stall Math Pipe Throttle | `smsp__warps_issue_stalled_math_pipe_throttle_per_issue_active.pct` | > 15% | [Comp] ① `tl.dot` 的 Block Shape 不匹配 Tensor Core → 调整 `BLOCK_M/N` ② 混合精度 |
| Stall Not Selected | `smsp__warps_issue_stalled_not_selected_per_issue_active.pct` | 高值 | [Sync] ① `num_warps` 过大 → 减少 ② 增加 `num_stages` 分散同步点 |
| Stall Sleeping | `smsp__warps_issue_stalled_sleeping_per_issue_active.pct` | > 5% | [Sync] Pipeline barrier phase 问题 → 检查 `num_stages` 和 K-loop 结构 |
| Eligible Warps/Cycle | `smsp__warps_eligible.avg.per_cycle_active` | \(< 2\) | [Sync/Comp] ① 提升 Occupancy（缩小 Block Shape） ② 增大 `BLOCK_M/N` 暴露 ILP |

---

## 五、Step 3 — Phase 1-3 无条件执行

### Phase 1：Pack A — 数据类型与精度对齐（可打包，1 次 profile）

确保 `tl.dot` 走正确的指令路径，所有输入/输出的精度配置正确。

| 编号 | 优化项 | Triton 实现方式 |
|---|---|---|
| C1 | Tensor Core 路径 | 确保 `tl.dot` 输入为 FP16/BF16/FP8，不要意外提升为 FP32 |
| C7 | FMA 路径 | 逐元素运算由编译器自动融合为 FMA |
| C9 | `allow_tf32` | FP32 输入时确认 `allow_tf32=True`（默认） |
| C11 | `input_precision` | 根据精度需求选择 `"tf32"` / `"tf32x3"` / `"ieee"` |
| C8 | 除法优化 | 编译器自动将常量除数转换为乘法+移位 |
| C15 | 强度削减 | 编译器自动执行 |
| C18 | Select 替代分支 | 使用 `tl.where` |
| C10 | 谓词化 | `tl.load` 的 `mask` 参数自动生成谓词化指令 |
| M16 | 只读缓存 | 编译器自动选择缓存策略 |

### Phase 2：Pack B — Coalescing 与对齐（可打包，1 次 profile）

全部服务于同一目标：让 warp 访问连续对齐的地址。

| 编号 | 优化项 | Triton 实现方式 |
|---|---|---|
| M3 | SoA 替代 AoS | 每个 tensor 独立传入，独立 stride |
| M2 | 合并访问 | `tl.load` 指针偏移最内维 stride=1；优先使用 `tl.make_block_ptr` |
| M7 | 向量化 128-bit | 编译器根据对齐自动选择最宽向量宽度 |
| M12 | 对齐 | 确保 tensor 指针和 `BLOCK_K × sizeof(element)` 满足 16 字节对齐 |
| M13 | Stride / Pitch | 通过 stride 参数支持非紧密排列矩阵 |

### Phase 3：Pack C — 编译器 Pipeline 基础配置（可打包，1 次 profile）

确保编译器能生成正确的 pipeline 和归约代码。

| 编号 | 优化项 | Triton 实现方式 |
|---|---|---|
| S1 | Warp Shuffle | 编译器自动选择 shuffle vs shared 归约路径 |
| S2 | Pipeline 同步 | `num_stages` 参数 + 编译器自动 pipeline 生成 |
| C4 | 归约策略 | Split-K 通过 grid 第三维 + `tl.atomic_add` 实现 |
| C5 | `tl.reduce` | 使用 `tl.sum` / `tl.max` 等高层归约 API |
| M10 | 数据交换 | `tl.dot` 内部编译器自动处理 |
| C6 | 循环展开 | `BLOCK_K` 为 `tl.constexpr` 时编译器自动展开 |

**Phase 1-3 小结：3 次 profile 覆盖约 25 项优化。**

---

## 六、Step 4 — Phase 4-6 按 NCU 指标选择 Pack

> **关键规则：Phase 4-6 中每个 Pack 必须单独验证。跨 Pack 绝不可打包。**

### 6.1 NCU 指标 → Pack 选择映射

| NCU 指标异常 | 选择 Pack | Phase |
|---|---|---|
| L1/L2 Hit Rate 低 | Pack D1（Block Shape 初调 + bank conflict） | 4 |
| Shared Mem Eff \(< 90\%\)、Bank Conflict > 0 | Pack D1（`BLOCK_K` 调整） | 4 |
| Achieved Occ \(< 70\%\)、Register Spill > 0 | Pack E（Block Shape / `num_warps` 系统调优） | 5 |
| DRAM Throughput 接近 SOL | Pack F1（Kernel Fusion） | 6 |
| Stall Mem Throttle > 20%、Long Scoreboard > 25% | Pack F2（`num_stages` / `tl.make_block_ptr`） | 6 |
| FMA 高但无 Tensor pipe | Pack F3（精度修复） | 6 |
| Warp Exec Eff \(< 85\%\) | Pack F4（数据预处理重排） | 6 |

### 6.2 Phase 4：Shared Memory 与 Bank Conflict 调整

**Pack D1 — Block Shape 初调 + Bank Conflict（隔离验证）**

| 编号 | 优化项 | Triton 实现方式 |
|---|---|---|
| M4 | Tiling | 调整 `BLOCK_M/N/K` 平衡复用率和 Shared Memory 用量 |
| M9 | Bank Conflict | 调整 `BLOCK_K` 使其对 bank 宽度友好（避免 32 的精确倍数在某些配置下产生冲突） |
| M15 | Shared Mem 容量 | 编译器自动管理；通过 Block Shape × `num_stages` 间接控制 |

### 6.3 Phase 5：Block Shape / Occupancy 系统调优

**Pack E — Block Shape + `num_warps` + `num_stages` 联合调优（必须隔离）**

这是 Triton 中最关键的调优 Pack——Block Shape 改变牵动全局。

| 编号 | 优化项 | Triton 实现方式 |
|---|---|---|
| M5 | 寄存器控制 | 通过 `BLOCK_M/N` 和 `num_warps` 间接控制每线程寄存器用量 |
| C3 | Launch Config | **调整 Block Shape × `num_warps` × `num_stages` 组合** |
| M11 | 寄存器级复用 | `tl.dot` 累加器天然复用；增大 `BLOCK_M/N` 提升复用 |
| M6 | Pipeline stage | 调整 `num_stages`（2–7） |

**系统化调优方法（`triton.autotune`）**：

1. 列出候选组合（通常 6–12 个）：
   - `BLOCK_M` ∈ {64, 128, 256}
   - `BLOCK_N` ∈ {64, 128, 256}
   - `BLOCK_K` ∈ {32, 64}
   - `num_warps` ∈ {4, 8}
   - `num_stages` ∈ {2, 3, 4}
2. 使用 `triton.autotune` 的 `configs` 列表指定候选
3. 运行 autotune，编译器自动选择在给定 problem size 下最快的配置
4. 用 NCU 检查最优候选的子指标，确认无异常退化

**手动调优（无 autotune 时）**：
1. 从常用起点开始（如 128×128×32, `num_warps`=4, `num_stages`=3）
2. 逐维度扫描（固定其他维度，单独变化一个维度）
3. 找到性能拐点后，在拐点附近做细粒度搜索
4. NCU 验证无 spill、occupancy 合理、无 bank conflict

### 6.4 Phase 6：结构性变更

**Pack F1 — Kernel Fusion（必须隔离）**

| 编号 | 优化项 | Triton 实现方式 |
|---|---|---|
| M1/S8 | Kernel Fusion | 在 `@triton.jit` kernel 中直接编写 GEMM + 后处理逻辑。将多个 kernel 合并为一个 |

**Pack F2 — Pipeline 升级（隔离验证）**

| 编号 | 优化项 | Triton 实现方式 |
|---|---|---|
| M8 | 异步搬运 | 使用 `tl.make_block_ptr` 引导编译器生成 TMA 指令（Hopper） |
| S5 | Pipeline barrier | 增加 `num_stages` + 编译器自动 pipeline 生成 |
| S6 | TMA 路径 | `tl.make_block_ptr` + Hopper 后端 |
| C2 | 计算访存重叠 | `num_stages` + 编译器自动 pipeline |

**`tl.make_block_ptr` 的关键作用**：
- 相比手动指针算术（`tl.load(ptr + offsets)`），`tl.make_block_ptr` 为编译器提供更完整的内存访问信息
- Hopper 上更容易触发 TMA 指令生成
- 自动处理边界检查（`boundary_check` 参数）
- 是 Hopper 上获得最优性能的推荐方式

**Pack F3 — 精度修复（必须隔离）**

| 编号 | 优化项 | Triton 实现方式 |
|---|---|---|
| C1 | Tensor Core 路径 | 修复 `tl.dot` 输入精度：FP32 → FP16/BF16/FP8 |
| C11 | `input_precision` | 调整 `"tf32"` / `"ieee"` 选择 |

**Pack F4 — 数据预处理重排（必须隔离）**

| 编号 | 优化项 | Triton 实现方式 |
|---|---|---|
| M17 | 数据重排 | Host 端 / PyTorch 预处理（Triton 不内建），重排后通过 stride 参数描述新布局 |
| C14 | Warp 内数据连续 | 编译器自动保证（前提是访问模式正确） |

---

## 七、Phase 7 — 系统级优化（独立并行）

不改变 kernel 代码，可与 Phase 1-6 的任何阶段并行推进。

**Pack G1 — 调度与传输**

| 编号 | 优化项 | Triton 关系 |
|---|---|---|
| S7 | CUDA Graphs | `torch.cuda.graph()` capture Triton kernel |
| S9 | Stream + Event | 通过 PyTorch stream API 管理 |
| M14 | Streams 重叠 | 与 Triton 正交 |
| M18 | Pinned Memory | `torch.empty(..., pin_memory=True)` |

**Pack G2 — 缓存策略**

| 编号 | 优化项 | Triton 关系 |
|---|---|---|
| M19 | L2 Persistence | Host API，与 Triton 正交 |
| M20 | Prefetch | 编译器通过 `num_stages` 自动管理 |
| M15 | `eviction_policy` | `tl.load` 的 eviction 策略参数 |

**Pack G3 — 分析与诊断**

| 编号 | 优化项 | Triton 关系 |
|---|---|---|
| C20 | PTX/SASS 分析 | **`MLIR_ENABLE_DUMP` + NCU SASS 视图** 验证 `HMMA`/`WGMMA` 指令路径 |
| C28 | SFU 意识 | 大量 `tl.exp`/`tl.log` 时注意 SFU 瓶颈 |
| S10 | Atomic 考量 | Split-K 中的 `tl.atomic_add` 策略 |

---

## 八、Step 5 — 验证与迭代

### 8.1 Benchmark 度量体系

| 度量 | 含义 |
|---|---|
| `median_ms` | 算子运算速度（越低越好） |
| `speedup_vs_reference` | 相对 PyTorch/cuBLAS 参考实现 |
| `bandwidth_gbps` | 是否接近硬件带宽上界 |
| `TFLOPS` | 是否接近硬件算力上界 |
| 正确性 | 数值结果在容忍范围内 |

### 8.2 每个 Pack 执行后的判决标准

**主判据**：算子运算速度提升 且 正确性通过。

| 结果 | 决策 |
|---|---|
| 速度提升 + 目标 NCU 改善 + 无退化 | **保留**，高置信度 |
| 速度提升 + NCU 未变 + 意外改善 | **保留**，分析真实原因 |
| 速度相近（\(< 3\%\)）+ NCU 更优 | **保留** |
| 速度相近（\(< 3\%\)）+ NCU 更差 | **不保留** |
| 速度未提升 + NCU 改善 | **不保留** |
| 速度下降 | **立即回退** |

### 8.3 正确性验证

- 数值 diff 在容忍范围内（FP32: \(10^{-5}\)，TF32: \(10^{-3}\)，**FP16 MMA: \(10^{-3}\) 相对误差，BF16: \(10^{-2}\)，FP8: \(10^{-1}\)**）
- 与 PyTorch 参考实现（`torch.matmul` 等）对比
- **Triton 特有**：非 Block Shape 整数倍的 M/N/K 测试（验证 mask 边界处理正确性）；Split-K 模式下验证 atomic 归约结果；混合精度下验证累加器精度

### 8.4 收敛判断

1. `speedup_vs_reference` 达到目标（或超过 cuBLAS）
2. `bandwidth_gbps` 接近硬件带宽上界（Memory-bound kernel）
3. `TFLOPS` 接近硬件算力上界（Compute-bound kernel）
4. 最近一轮改善 \(< 3\%\)
5. NCU 显示瓶颈已在硬件极限上（如 DRAM throughput > 85% SOL 或 Tensor pipe > 80%）

### 8.5 瓶颈转移处理

Triton 场景下的常见转移路径：

- **Memory-bound → Kernel Fusion → Compute-bound**：最常见。Fusion 减少了访存量，计算比例上升。下一步：检查 `tl.dot` 是否走了 Tensor Core，Block Shape 是否最优。
- **Memory-bound → 增 `num_stages` → Stall Barrier 上升 → Latency-bound**：stage 数增加了 Shared Memory 用量但没改善延迟隐藏。下一步：检查 Occupancy 是否下降，考虑缩小 Block Shape。
- **Compute-bound → 切 FP16/BF16 走 Tensor Core → data feeding 跟不上 → Memory-bound**：Tensor Core 太快，搬运跟不上。下一步：增加 `num_stages`、使用 `tl.make_block_ptr` 触发 TMA。
- **Latency-bound → 增 `num_stages` → Shared Memory 超限 → Occupancy=1 → 仍 Latency-bound**：经典死循环。下一步：缩小 Block Shape 释放 Shared Memory，或使用 `tl.make_block_ptr` 触发更高效的 TMA 路径。

每次转移后，切换到对应的诊断表和 Pack 继续优化。

---

## 九、验证清单（NCU）— Triton 汇总

**内存优化**：
1. `Memory SOL %`、`DRAM Throughput` — 带宽利用率
2. `Global Load/Store Efficiency`、`Sectors/Request` — **验证编译器生成的合并访问**
3. `L1/L2 Hit Rate` — 验证 Block Shape 和编译器缓存策略
4. `Shared Memory Efficiency` — **验证编译器自动 swizzle**；bank conflict 计数器应为 0
5. `DRAM Read Bytes` / 理论数据量 — Kernel Fusion 后应接近 1.0

**计算优化**：
1. SASS 确认 `HMMA`/`WGMMA` 指令 — **验证 `tl.dot` 精度路径**；`sm__pipe_tensor_cycles_active` > 0
2. `Issue Slot Utilization`、`Eligible Warps/Cycle` — 编译器编译期优化应减少 ALU 开销
3. `Warp Execution Efficiency` — `mask` 谓词化应消除边界分支
4. NCU register 统计 — Block Shape 变更后必检查 spill

**同步优化**：
1. `Stall Barrier` — **编译器 pipeline 应大幅降低**
2. `Stall Sleeping` — Pipeline 设计问题
3. `Eligible Warps/Cycle` — `num_stages` 增加应提升
4. 正确性验证 — 修改 `num_stages` 或 Block Shape 后必做

**常见误判**：
- `tl.dot` 输入被意外提升为 FP32 → Tensor Core 完全没用上
- Block Shape 过大 → spill + occupancy=1 → 性能反降
- `num_stages` 过多 → Shared Memory 超限 → occupancy 降 → 抵消延迟隐藏收益
- 只看 occupancy 或 NCU 子指标，不跑 benchmark 验速度
- 编译器 swizzle 未生效 → bank conflict 未消除
- Kernel Fusion 后只看 memory 指标改善，不检查计算路径是否退化
- 使用手动指针算术阻碍了编译器生成 TMA 指令（Hopper）

---

## 十、总迭代节奏

| 阶段 | 操作 | Profile 次数 | 覆盖优化项数 |
|---|---|---|---|
| Baseline | benchmark + ncu | 1 | 0 |
| Phase 1-3 | 3 Pack（精度对齐 + Coalescing + Pipeline 基础） | 3 | 25 |
| Phase 4-6 | NCU 驱动选 2-3 Pack（Block Shape / `num_stages` / Fusion / 精度修复） | 2-3 | 8-16 |
| 瓶颈转移 | 重新 Roofline，再选 1-2 Pack | 1-2 | 4-8 |
| **总计** | | **7-10** | **37-49 项（有效子集）** |

**Triton 特有的调优顺序建议**：
1. **先确定精度**（决定 `tl.dot` 指令路径：Tensor Core vs CUDA Core）
2. **再用 `triton.autotune` 搜索 Block Shape**（平衡 Shared Memory / 寄存器 / Occupancy）
3. **然后调 `num_stages`**（在 Shared Memory 预算内最大化延迟隐藏）
4. **最后考虑 `tl.make_block_ptr` 改写**（触发 TMA、改善编译器优化质量）
5. **Kernel Fusion 贯穿始终**（Triton 的最大优势——灵活的算子融合）

判决优先级：

正确性不通过 → 立即回退；
速度明显提升（> 3%）→ 保留；
速度相近（< 3%）+ NCU 更优 → 保留；
速度相近（< 3%）+ NCU 更差 → 不保留；
速度下降 → 立即回退。
