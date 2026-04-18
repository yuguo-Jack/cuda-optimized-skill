# CUTLASS Kernel 优化完整策略

> 66 项优化 → 15 个 Pack → 7 个 Phase → 7-10 次 profile 收敛
> 本文档将原始 CUDA 优化框架映射到 CUTLASS 框架层面。核心方法论（NCU 驱动、按冲突域分组、瓶颈转移检测）完全保留，优化项的实现手段替换为 CUTLASS 配置选择。

---

## 一、核心方法论

与原始策略完全一致：

1. **指标驱动，不凭直觉**：每一步优化都由 NCU 指标的异常值触发
2. **按冲突域分组**：不争夺资源的优化可以打包执行，争夺资源的必须隔离验证
3. **瓶颈会转移**：每轮优化后必须重新 Roofline 分类
4. **算子运算速度是最终裁决**：benchmark 实测的 speedup 为主判据；差距 \(< 3\%\) 时优选 NCU 子指标更优的版本

**CUTLASS 特有原则**：

5. **Atom 选择决定指令路径**：MMA Atom 决定走 Tensor Core 还是 CUDA Core；Copy Atom 决定走 cp.async、TMA 还是普通 load。选错 Atom 等于方向性错误，后续优化无意义
6. **Tile Shape 是调优的中心维度**：CTA Tile Shape 同时影响 Shared Memory 用量、寄存器压力、Occupancy、数据复用率——它是内存/计算/同步三个维度的交汇点
7. **Mainloop 策略决定同步模型**：CpAsync Cooperative（Ampere）vs TMA WarpSpecialized（Hopper）是根本性的架构选择，而非参数调优
8. **Pipeline Stage 数是延迟隐藏的调节阀**：在 Shared Memory 预算内最大化 stage 数，但不能牺牲 Occupancy

---

## 二、Step 0 — Baseline 采集

### 2.1 Benchmark 基准速度

```bash
# 有参考实现时（推荐）
python benchmark.py solution.cu --ref ref.py --N=4096 --M=4096 --json-out baseline_bench.json

# 无参考实现时
python benchmark.py solution.cu --N=4096 --M=4096 --json-out baseline_bench.json
```

关注 `median_ms`（算子运算速度）和 `speedup_vs_reference`（加速比）。

### 2.2 NCU 全量 Profile

```bash
ncu --set full -o baseline ./your_kernel
```

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

| 诊断维度 | NCU 计数器 | 异常阈值 | CUTLASS 对应优化（按优先级） |
|---|---|---|---|
| 顶层判定 | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | > Compute SOL% | 确认 memory-bound |
| DRAM 带宽利用率 | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 接近 SOL | [Mem] ① Epilogue Fusion 减少总访存 ② 已接近极限，转 compute 优化 |
| Global Load Efficiency | `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` | \(< 80\%\) | [Mem] ① 检查 Layout 是否 RowMajor/ColumnMajor（stride-1 最内维） ② Copy Atom 向量宽度是否匹配 alignment ③ 检查自定义 stride |
| Global Store Efficiency | `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct` | \(< 80\%\) | 同上，额外检查 Epilogue 的输出 Layout |
| Sectors/Request | `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` | > 4 | [Mem] ① Layout stride 不连续（AoS 问题） ② 需要 SoA 重构 ③ Alignment 降级 |
| L1 Hit Rate | `l1tex__t_sector_hit_rate.pct` | \(< 50\%\) | [Mem] ① 增大 Tile → 提升 Shared Memory 中数据复用 ② Copy Atom 缓存策略选择 |
| L2 Hit Rate | `lts__t_sector_hit_rate.pct` | \(< 40\%\) | [Mem] ① L2 Persistence ② 减小 Tile 的 K 维度（缩小 working set） ③ Epilogue Fusion |
| Shared Mem Efficiency | `smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct` | \(< 90\%\) | [Mem] ① **Swizzle 参数不正确** ② 检查 MMA Atom 的 shared memory 访问模式 |
| Shared Bank Conflict | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | > 0 | [Mem] ① Swizzle<B,M,S> 参数需要调整 ② 确认 SmemLayout 包含 Swizzle |
| DRAM Read Bytes | `dram__bytes_read.sum` | 远大于理论数据量 | [Mem] 带宽放大系数 > 2× → coalescing 或 Tile 复用率问题 |

### 4.2 Compute-bound 诊断表（13 项）

| 诊断维度 | NCU 计数器 | 异常阈值 | CUTLASS 对应优化（按优先级） |
|---|---|---|---|
| 顶层判定 | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | > Memory SOL% | 确认 compute-bound |
| SM FMA 管线 | `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | FMA 高但无 Tensor | [Comp] ① **MMA Atom 选错** → 切换到 Tensor Core Atom ② 检查数据类型是否匹配 Atom 精度 |
| Tensor Pipe | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | \(< 30\%\)（已用 TC） | [Comp] ① Tile Shape 不匹配 MMA shape → 调整 Atom 重复数 ② Data feeding 不足 → 增加 Pipeline stage 或 Warp Specialization ③ Tile Shape 过小 |
| ALU Pipe | `sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active` | > 80% | [Comp] ① CUTLASS 编译期索引计算应减少 ALU → 检查自定义代码是否引入运行时计算 ② 强度削减 ③ FastDivmod |
| Issue Slot Utilization | `smsp__issue_active.avg.pct_of_peak_sustained_active` | \(< 60\%\) | [Comp] ① Pipeline stage 不足 → 增加 stage 暴露 ILP ② Tile Shape 调优 ③ `__restrict__` |
| Warp Execution Efficiency | `smsp__thread_inst_executed_per_inst_executed.ratio` | \(< 85\%\) | [Comp] ① 使用 Predicated Mainloop 处理边界 ② 数据预排序 |
| Register Spill | `launch__registers_per_thread` | spill > 0 | [Comp] ① 缩小 Tile Shape ② `__launch_bounds__` ③ 减少 Pipeline stage |
| Achieved Occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` | \(< 70\%\) 理论值 | [Comp] ① 缩小 Tile Shape 减少 shared+register ② 减少 stage 数 ③ `__launch_bounds__` |

### 4.3 Latency-bound 诊断表（14 项）

| 诊断维度 | NCU 计数器 | 异常阈值 | CUTLASS 对应优化（按优先级） |
|---|---|---|---|
| 顶层判定 | Memory/Compute SOL% | 两者均 \(< 30\%\) | 确认 latency-bound |
| Stall Barrier | `smsp__warps_issue_stalled_barrier_per_issue_active.pct` | > 30% | [Sync] ① Pipeline arrive/wait 替代 `__syncthreads()` ② 增加 stage 数 ③ Warp Specialization 分离 producer/consumer |
| Stall Long Scoreboard | `smsp__warps_issue_stalled_long_scoreboard_per_issue_active.pct` | > 25% | [Mem] 等 L2/DRAM → ① 切换到 cp.async/TMA Mainloop ② 增加 Pipeline stage ③ Tiling |
| Stall Short Scoreboard | `smsp__warps_issue_stalled_short_scoreboard_per_issue_active.pct` | > 15% | [Mem] 等 L1/shared → ① **检查 Swizzle** ② 增大 MMA tile 暴露 ILP |
| Stall Memory Throttle | `smsp__warps_issue_stalled_lg_throttle_per_issue_active.pct` | > 20% | [Mem] ① cp.async/TMA 异步路径 ② 增加 Pipeline stage |
| Stall MIO Throttle | `smsp__warps_issue_stalled_mio_throttle_per_issue_active.pct` | > 10% | [Sync] shared/atomic 过载 → ① Warp Specialization 减少 shared 争用 ② 减少 shared 访问密度 |
| Stall Math Pipe Throttle | `smsp__warps_issue_stalled_math_pipe_throttle_per_issue_active.pct` | > 15% | [Comp] ① MMA Atom shape 不匹配 → 换更大的 Atom ② 混合精度 |
| Stall Not Selected | `smsp__warps_issue_stalled_not_selected_per_issue_active.pct` | 高值 | [Sync] ① Pipeline 的 arrive/wait 应分散同步点 ② Warp Specialization |
| Stall Sleeping | `smsp__warps_issue_stalled_sleeping_per_issue_active.pct` | > 5% | [Sync] Pipeline barrier phase → 检查 stage 数和 arrive/wait 时机 |
| Eligible Warps/Cycle | `smsp__warps_eligible.avg.per_cycle_active` | \(< 2\) | [Sync/Comp] ① 提升 Occupancy（缩小 Tile Shape） ② 增大 MMA tile 暴露 ILP |

---

## 五、Step 3 — Phase 1-3 无条件执行

### Phase 1：Pack A — 零成本编译器提示（可打包，1 次 profile）

不改 Tile Shape、不改 Mainloop、不改 Pipeline。纯粹帮助编译器生成更好的代码。

| 编号 | 优化项 | CUTLASS 实现方式 |
|---|---|---|
| C10 | `__restrict__` | CUTLASS kernel 指针参数默认添加 |
| C6 | `#pragma unroll` | CUTLASS 静态维度 `Int<N>` 自动展开；`CUTLASS_PRAGMA_UNROLL` 标注 |
| C7 | FMA | MMA Atom 保证 MMA/FMA 指令路径 |
| C9 | `--use_fast_math` | 编译选项，注意 denorm flush 对 FP16 的影响 |
| C8 | 乘法替代除法 | CUTLASS `FastDivmod` 已优化地址计算；自定义 epilogue 中手动应用 |
| C15 | 强度削减 | 同上 |
| C18 | Select 替代短分支 | 自定义 epilogue 中用三目运算符 |
| C17 | 整数位操作 | `__popc()`/`__clz()`/`__ffs()` 直接使用 |
| C25 | `__forceinline__` | CUTLASS 内部已大量使用 |
| M16 | 只读缓存路径 | Copy Atom 缓存策略选择（CACHEALWAYS vs CACHEGLOBAL） |
| M21 | In-place | CUTLASS 支持 D 和 C 指向同一地址 |
| C19 | 谓词化 | Predicated Mainloop 内建边界处理 |

### Phase 2：Pack B — Coalescing 全家桶（可打包，1 次 profile）

全部服务于同一目标：让 warp 访问连续对齐的地址。

| 编号 | 优化项 | CUTLASS 实现方式 |
|---|---|---|
| M3 | SoA 替代 AoS | 每个矩阵独立 tensor，独立 Layout |
| M2 | 合并访问 | Layout 最内维 stride=1（RowMajor/ColumnMajor 天然保证） |
| M7 | 向量化 128-bit | Copy Atom 向量宽度 uint128_t |
| M12 | 对齐 128B | Alignment 模板参数 |
| M13 | Padding pitch | Layout 的 leading dimension 编码 pitch |
| C24 | 循环交换 | Layout stride 顺序天然解决 |

### Phase 3：Pack C + C+ — Warp 级同步替代（原子捆绑 + 可打包，1 次 profile）

**Pack C（原子捆绑）**：

| 编号 | 优化项 | CUTLASS 实现方式 |
|---|---|---|
| S1 | Warp Shuffle 替代 shared+sync | MMA Atom 内建 warp 内数据流 |
| S2 | 删除 `__syncthreads()` | Pipeline arrive/wait 替代 |
| S3 | `__syncwarp` 替代 `__syncthreads` | MMA Atom 内部处理 warp 级同步 |
| M10 | Warp Shuffle 数据交换 | MMA Atom TV layout 隐式处理 |
| C5 | Warp Shuffle 计算 | MMA 累加 + 显式 shuffle 最终归约 |
| C4 | 归约优化 | Split-K / Stream-K 策略选择 |

**Pack C+（可打包）**：

| 编号 | 优化项 | CUTLASS 实现方式 |
|---|---|---|
| C21 | Warp Vote | 直接使用 `__ballot_sync` |
| C20 | Early Exit | `__all_sync` |
| C27 | Warp Match | 直接使用 |
| C13 | Scan（warp 级） | CUB `WarpScan` |

**Phase 1-3 小结：3 次 profile 覆盖 30 项优化。**

---

## 六、Step 4 — Phase 4-6 按 NCU 指标选择 Pack

> **关键规则：Phase 4-6 中每个 Pack 必须单独验证。跨 Pack 绝不可打包。**

### 6.1 NCU 指标 → Pack 选择映射

| NCU 指标异常 | 选择 Pack | Phase |
|---|---|---|
| L1/L2 Hit Rate 低 | Pack D1（Tiling + Swizzle） | 4 |
| Shared Mem Eff \(< 90\%\)、Bank Conflict > 0 | Pack D1（Swizzle 参数调优） | 4 |
| SFU/ALU 饱和 | Pack D2（LUT） | 4 |
| Achieved Occ \(< 70\%\)、Register Spill > 0 | Pack E（Tile Shape / Occupancy） | 5 |
| DRAM Throughput 接近 SOL | Pack F1（Epilogue Fusion） | 6 |
| Stall Mem Throttle > 20%、Long Scoreboard > 25% | Pack F2（Pipeline / Mainloop 升级） | 6 |
| FMA 高但无 Tensor pipe | Pack F3（MMA Atom 切换） | 6 |
| Warp Exec Eff \(< 85\%\) | Pack F4（数据预处理重排） | 6 |

### 6.2 Phase 4：Shared Memory 资源改动

**Pack D1 — Tiling + Swizzle（原子捆绑，隔离验证）**

Tile Shape 确定后，Swizzle 参数和 Shared Memory 容量配置是配套措施。

| 编号 | 优化项 | CUTLASS 实现方式 |
|---|---|---|
| M4 | Tiling | CTA Tile Shape（TileShape_MNK）选择 |
| M9 | Bank Conflict 消除 | **Swizzle<B,M,S> 嵌入 SmemLayout**；CUTLASS 预定义 Layout 已包含 Swizzle |
| M15 | Shared Mem 容量 | 自动配置 `cudaFuncSetAttribute`，大小由 Stages × TileSize 决定 |
| S4 | Cooperative Groups | Hopper 用 Cluster（ClusterShape_MNK）替代 |

**Pack D2 — 查表法（必须隔离）**

| 编号 | 优化项 | CUTLASS 实现方式 |
|---|---|---|
| C16 | LUT | 自定义 epilogue 中放入 Shared Memory LUT；与 mainloop 争夺 Shared Memory |
| M23 | `__constant__` | 小参数表通过 constant memory 传入 |

### 6.3 Phase 5：Tile Shape / Occupancy 调整

**Pack E — Tile Shape 调优（必须隔离）**

这是 CUTLASS 中最关键的调优 Pack——Tile Shape 改变牵动全局。

| 编号 | 优化项 | CUTLASS 实现方式 |
|---|---|---|
| M5 | 寄存器控制 | `__launch_bounds__` 配合 CUTLASS kernel |
| C3 | Launch Config | **调整 CTA Tile Shape**（128×128×32 vs 128×256×32 vs 256×128×32 等） |
| M11 | 寄存器级复用 | MMA 累加器天然复用；调整 Atom 重复数 |
| C23 | Loop Fission | 缩小 K-tile 降低每次迭代的寄存器压力 |
| C22 | Loop Fusion | 增大 K-tile 增加复用 |

**Tile Shape 调优的系统化方法**：
1. 列出 2-4 个候选 Tile Shape（从常用起点开始）
2. 用 `--ptxas-options=-v` 检查每个候选的寄存器用量和 spill
3. 排除 spill > 0 和 Shared Memory 超限的候选
4. 对剩余候选做 benchmark，选择 `median_ms` 最低的
5. 用 NCU 检查最优候选的子指标，确认无异常退化

### 6.4 Phase 6：结构性变更

**Pack F1 — Epilogue Fusion（必须隔离）**

| 编号 | 优化项 | CUTLASS 实现方式 |
|---|---|---|
| M1/S8 | Kernel Fusion | Epilogue Visitor Tree（EVT）：LinCombEltAct、Bias、AuxStore 等节点组合 |

**Pack F2 — Mainloop / Pipeline 升级（原子捆绑，隔离验证）**

Mainloop 策略、Pipeline stage 数、异步搬运模式是一套完整体系。

| 编号 | 优化项 | CUTLASS 实现方式 |
|---|---|---|
| M6 | 双缓冲/多级流水 | Pipeline Stages 模板参数（2–7） |
| M8 | cp.async | `MainloopSm80CpAsync*` Mainloop 选择 |
| S5 | barrier/pipeline | `PipelineAsync`（Ampere）/ `PipelineTmaAsync`（Hopper） |
| S6 | 异步搬运 | TMA Mainloop：`MainloopSm90Tma*` |
| C11 | 软件流水 | Pipeline 状态机自动实现 prolog/steady/epilog |
| C2 | 计算访存重叠 | Pipeline 多 stage + Warp Specialization 天然重叠 |

**Mainloop 选择决策**：
- Ampere：`CpAsyncPredicated`（通用）vs `CpAsyncUnpredicated`（tile 整除时更快）
- Hopper：`TmaWarpSpecialized`（默认） vs `TmaWarpSpecializedCooperative`（计算密集）vs `TmaWarpSpecializedPingpong`（小矩阵 / batch）

**Pack F3 — MMA Atom 切换（必须隔离）**

| 编号 | 优化项 | CUTLASS 实现方式 |
|---|---|---|
| C1 | Tensor Core | 切换 MMA Atom：Ampere `SM80_16x8x16_*` / Hopper `SM90_64x128x16_*` |
| C12 | Warp Specialization | `TmaWarpSpecialized` Mainloop 内建 |

**Pack F4 — 数据预处理重排（必须隔离）**

| 编号 | 优化项 | CUTLASS 实现方式 |
|---|---|---|
| M17 | 数据重排 | Host 端预处理（CUTLASS 不内建），重排后用自定义 Layout 描述 |
| C14 | 按 warp 重组 | CUTLASS 的 TiledMma 保证 warp 内数据连续 |

---

## 七、Phase 7 — 系统级优化（独立并行）

不改变 kernel 代码，可与 Phase 1-6 的任何阶段并行推进。

**Pack G1 — 调度与传输**

| 编号 | 优化项 | CUTLASS 关系 |
|---|---|---|
| S7 | CUDA Graphs | CUTLASS kernel 支持 Graph capture |
| S9 | Stream + Event | `run()` 接受 stream 参数 |
| M14 | Streams 重叠 | 与 CUTLASS 正交 |
| M18 | Pinned Memory | 与 CUTLASS 正交 |
| M24 | Stream-Ordered 分配 | CUTLASS workspace 支持 `cudaMallocAsync` |

**Pack G2 — 缓存策略与特殊内存**

| 编号 | 优化项 | CUTLASS 关系 |
|---|---|---|
| M19 | L2 Persistence | Host API，与 CUTLASS 正交 |
| M20 | Prefetch | 与 CUTLASS 正交 |
| M26 | Texture | CUTLASS 不使用 Texture |
| M27 | Unified Memory | 性能场景不推荐 |
| M28 | Zero-Copy | 仅 data-starved 场景 |

**Pack G3 — 分析与诊断**

| 编号 | 优化项 | CUTLASS 关系 |
|---|---|---|
| C26 | PTX/SASS 分析 | **验证 HMMA/WGMMA 指令路径**——CUTLASS 最重要的验证手段 |
| C28 | SFU 意识 | 不变 |
| M25 | Sector 化理解 | 指导 Swizzle 参数选择 |
| S10 | Atomic 考量 | Split-K / Stream-K 中的 atomic reduction 策略选择 |
| M22 | Cooperative Groups | Hopper Cluster 替代 |

---

## 八、Step 5 — 验证与迭代

### 8.1 Benchmark 度量体系

与原始策略完全一致：

```bash
python benchmark.py solution.cu --ref ref.py --N=4096 --M=4096 --json-out result.json
```

| 度量 | 字段 | 含义 |
|---|---|---|
| 运算速度 | `average_ms` / `median_ms` | 越低越好 |
| 加速比 | `speedup_vs_reference` | 相对参考实现 |
| 粗略带宽 | `bandwidth_gbps_rough` | 是否接近硬件上界 |
| 正确性 | `correctness.passed` | 数值结果在容忍范围内 |

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

- 数值 diff 在容忍范围内（FP32: \(10^{-5}\)，fast_math: \(10^{-3}\)，**FP16 MMA: \(10^{-3}\) 相对误差，BF16: \(10^{-2}\)**）
- `benchmark.py --ref` 通过
- `compute-sanitizer --tool racecheck` 无报错
- **CUTLASS 特有**：非 tile 整数倍的 M/N/K 测试（验证 Predicated Mainloop 正确性）；Split-K 模式下验证归约结果；混合精度下验证 accumulator 精度

### 8.4 收敛判断

1. `speedup_vs_reference` 达到目标
2. `bandwidth_gbps_rough` 接近硬件带宽上界（Memory-bound kernel）
3. 最近一轮改善 \(< 3\%\)
4. NCU 显示瓶颈已在硬件极限上（如 DRAM throughput > 85% SOL 或 Tensor pipe > 80%）

### 8.5 瓶颈转移处理

CUTLASS 场景下的常见转移路径：

- **Memory-bound → Epilogue Fusion → Compute-bound**：最常见。Fusion 减少了访存量，计算比例上升。下一步：检查 MMA Atom 是否走了 Tensor Core，Tile Shape 是否最优。
- **Memory-bound → Tiling + 增 Stage → Stall Barrier 上升 → Latency-bound**：Stage 数增加了 Shared Memory 用量但没改善延迟隐藏。下一步：检查 Occupancy 是否下降，考虑 Warp Specialization。
- **Compute-bound → MMA Atom 切 Tensor Core → data feeding 跟不上 → Memory-bound**：Tensor Core 太快，搬运跟不上。下一步：增加 Pipeline stage、切换到 TMA、Warp Specialization。
- **Latency-bound → 增 Stage → Shared Memory 超限 → Occupancy=1 → 仍 Latency-bound**：经典死循环。下一步：缩小 Tile Shape 释放 Shared Memory，或切换到 TMA（减少每 stage 的 Shared Memory 开销）。

每次转移后，切换到对应的诊断表和 Pack 继续优化。

---

## 九、验证清单（NCU）— CUTLASS 汇总

**内存优化**：
1. `Memory SOL %`、`DRAM Throughput` — 带宽利用率
2. `Global Load/Store Efficiency`、`Sectors/Request` — **验证 Layout 合并性和 Copy Atom 向量宽度**
3. `L1/L2 Hit Rate` — 验证 Tile Shape 和 Copy Atom 缓存策略
4. `Shared Memory Efficiency` — **验证 Swizzle 参数**；bank conflict 计数器应为 0
5. `DRAM Read Bytes` / 理论数据量 — Epilogue Fusion 后应接近 1.0

**计算优化**：
1. SASS 确认 `HMMA`/`WGMMA` 指令 — **验证 MMA Atom 路径**；`sm__pipe_tensor_cycles_active` > 0
2. `Issue Slot Utilization`、`Eligible Warps/Cycle` — 编译期索引计算应减少 ALU 开销
3. `Warp Execution Efficiency` — Predicated Mainloop 应消除边界分支
4. `--ptxas-options=-v` — Tile Shape 变更后必检查 spill

**同步优化**：
1. `Stall Barrier` — **Pipeline arrive/wait 应大幅降低**
2. `Stall Sleeping` — Pipeline barrier phase 设计问题
3. `Eligible Warps/Cycle` — Pipeline 多 stage 和 Warp Specialization 应提升
4. `compute-sanitizer --tool racecheck` — Mainloop/Pipeline 变更后必做

**常见误判**：
- MMA Atom 选错（FP32 Atom 处理 FP16 数据 → Tensor Core 完全没用上）
- Tile Shape 过大 → spill + occupancy=1 → 性能反降
- Pipeline stage 过多 → Shared Memory 超限 → occupancy 降 → 抵消延迟隐藏收益
- 只看 occupancy 或 NCU 子指标，不跑 benchmark 验速度
- Swizzle 参数不匹配 Tile Shape → bank conflict 未消除
- Epilogue Fusion 后只看 memory 指标改善，不检查计算路径是否退化

---

## 十、总迭代节奏

| 阶段 | 操作 | Profile 次数 | 覆盖优化项数 |
|---|---|---|---|
| Baseline | benchmark + ncu | 1 | 0 |
| Phase 1-3 | 3 Pack（编译器提示 + Coalescing + Warp 级替代） | 3 | 30 |
| Phase 4-6 | NCU 驱动选 2-3 Pack（Swizzle / Tile Shape / Mainloop / Pipeline / Atom） | 2-3 | 8-16 |
| 瓶颈转移 | 重新 Roofline，再选 1-2 Pack | 1-2 | 4-8 |
| **总计** | | **7-10** | **42-54 项（有效子集）** |

**CUTLASS 特有的调优顺序建议**：
1. **先确定 MMA Atom**（决定指令路径）
2. **再确定 Mainloop 策略**（决定同步模型）
3. **然后调 Tile Shape**（平衡 Shared Memory / 寄存器 / Occupancy）
4. **最后调 Pipeline Stage 数**（在 Shared Memory 预算内最大化）

判决优先级：

```
正确性不通过 → 立即回退
速度明显提升（> 3%）→ 保留
速度相近（< 3%）+ NCU 更优 → 保留
速度相近（< 3%）+ NCU 更差 → 不保留
速度下降 → 立即回退
```
