# CUDA Kernel 优化完整策略

> 66 项优化 → 15 个 Pack → 7 个 Phase → 7-10 次 profile 收敛

---

## 一、核心方法论

本策略整合 NCU 诊断映射表（42 个指标）和优化批量分类（66 项技术），形成一套 **"NCU 数据驱动 → 批量安全执行 → 迭代收敛"** 的工程化优化框架。

核心原则：

1. **指标驱动，不凭直觉**：每一步优化都由 NCU 指标的异常值触发，而非遍历清单
2. **按冲突域分组**：不争夺资源的优化可以打包执行，争夺资源的必须隔离验证
3. **瓶颈会转移**：每轮优化后必须重新 Roofline 分类，因为瓶颈类别可能已经变化
4. **算子运算速度是最终裁决**：以 benchmark 实测的 speedup / throughput 为主判据；当速度相近（差距 < 3%）时，优选 NCU 子指标更优的版本——更健康的子指标意味着更大的后续优化空间

---

## 二、Step 0 — Baseline 采集

### 2.1 Benchmark 基准速度

用 `benchmark.py` 采集算子当前运算速度，作为后续优化的对比基准：

```bash
# 有参考实现时（推荐）：同时验证正确性 + 采集速度 + 计算 speedup
python benchmark.py solution.cu --ref ref.py --N=4096 --M=4096 --json-out baseline_bench.json

# 无参考实现时：仅采集速度
python benchmark.py solution.cu --N=4096 --M=4096 --json-out baseline_bench.json
```

关注输出中的 `median_ms`（算子运算速度）和 `speedup_vs_reference`（加速比）。

### 2.2 NCU 全量 Profile

对目标 kernel 执行全量采集：

```bash
ncu --set full -o baseline ./your_kernel
```

关注 Speed Of Light (SOL) 面板的两个核心数值：

- **Memory SOL%**：`gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`
- **Compute SOL%**：`sm__throughput.avg.pct_of_peak_sustained_elapsed`

---

## 三、Step 1 — Roofline 分类

| 条件 | 瓶颈类型 | 含义 |
|---|---|---|
| Memory SOL% > Compute SOL% | Memory-bound | 带宽是瓶颈，计算单元吃不饱 |
| Compute SOL% > Memory SOL% | Compute-bound | 算力是瓶颈，数据供给充足 |
| 两者均 < 30% | Latency-bound | 既没吃满带宽也没吃满算力，线程在等待 |

---

## 四、Step 2 — NCU 子指标精准定位

### 4.1 Memory-bound 诊断表（15 项）

| 诊断维度 | NCU 计数器 | 异常阈值 | 对应优化（按优先级） |
|---|---|---|---|
| 顶层判定 | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | > Compute SOL% | 确认 memory-bound |
| DRAM 带宽利用率 | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 接近 SOL 但 latency 高 | [Mem] ① Kernel Fusion 减少总访存量 ② 算法层减少数据量 ③ 已接近极限，转 compute |
| Global Load Efficiency | `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` | < 80% | [Mem] ① SoA 替代 AoS ② float4 向量化 ③ 检查 stride 访问 [Zero] 确认 cudaMalloc 对齐 |
| Global Store Efficiency | `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct` | < 80% | 同上。写入端额外检查无效写 |
| Sectors/Request | `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio` | > 4 | [Mem] ① 重排数据保证 warp 连续 ② padding cudaMallocPitch ③ 2D 数组行对齐 |
| Global Load Transactions | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 远大于理论最小值 | [Mem] 用 (实际 sectors)/(理论最小 sectors) 量化浪费倍数 |
| L1 Hit Rate | `l1tex__t_sector_hit_rate.pct` | < 50% | [Mem] ① Tiling 到 shared memory ② __ldg() / const __restrict__ ③ 数据重排 Z-order |
| L2 Hit Rate | `lts__t_sector_hit_rate.pct` | < 40% | [Mem] ① L2 Persistence (cudaAccessPolicyWindow) ② Tiling 减少 working set ③ Kernel Fusion |
| L2 Throughput | `lts__t_sectors.avg.pct_of_peak_sustained_elapsed` | > 80% SOL | [Mem] L2 瓶颈 → ① 增大 tiling 减少 L2 压力 ② L2 Persistence ③ prefetch |
| L1 Bandwidth | `l1tex__throughput.avg.pct_of_peak_sustained_active` | > 80% SOL | [Mem] L1/TEX 饱和 → ① 向量化 load ② 减小 working set ③ 绕过 L1 直走 shared |
| Shared Mem Efficiency | `smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct` | < 90% | [Mem/Sync] ① Padding [32][33] ② Swizzle/XOR 索引 ③ 检查 warp 内 shared 访问 pattern |
| Shared Bank Conflict | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | > 0 | [Mem] 直接量化 conflict 次数，配合 Source 页定位代码行 |
| Shared Mem Throughput | `l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed` | > 80% SOL | [Mem] Shared 饱和 → ① shuffle 替代 ② 减少 shared 访问 ③ 寄存器级复用 |
| DRAM Read Bytes | `dram__bytes_read.sum` | 远大于理论数据量 | [Mem] 实际/理论 = 带宽放大系数。>2x → coalescing 或 cache 问题 |
| DRAM Write Bytes | `dram__bytes_write.sum` | 远大于理论写入量 | [Mem] ① 检查写入合并 ② in-place 操作 ③ L2 write-back 放大 |

### 4.2 Compute-bound 诊断表（13 项）

| 诊断维度 | NCU 计数器 | 异常阈值 | 对应优化（按优先级） |
|---|---|---|---|
| 顶层判定 | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | > Memory SOL% | 确认 compute-bound |
| SM FMA 管线 | `sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active` | FMA 高但无 Tensor | [Comp] ① WMMA/MMA 走 Tensor Core ② 降精度 FP16/BF16/TF32 |
| Tensor Pipe | `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` | < 30%（已用 TC） | [Comp] TC 没喂饱 → ① 数据 feeding 不足 ② tile size 不匹配 MMA shape ③ warp specialization |
| ALU Pipe | `sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active` | > 80% | [Comp] ALU 饱和 → ① 强度削减 ② 查表替代计算 ③ 消除不必要的地址计算 |
| SFU Utilization | `sm__pipe_shared_cycles_active.avg.pct_of_peak_sustained_active` | 接近饱和 | [Comp] ① 查表法替代超越函数 ② 多项式近似 ③ 减少 sin/cos/exp 调用 |
| Issue Slot Utilization | `smsp__issue_active.avg.pct_of_peak_sustained_active` | < 60% | [Comp] ① #pragma unroll 暴露 ILP ② 软件流水线 ③ 增加每线程工作量 [Zero] __restrict__ |
| IPC (active) | `smsp__inst_executed.avg.per_cycle_active` | < 1 | [Comp] 低 IPC 配合 stall 分布 → 定位依赖链/资源不足/同步阻塞 |
| IPC (elapsed) | `sm__inst_executed.avg.per_cycle_elapsed` | 远低于 active IPC | [Comp] 差距大 = SM 空闲周期多 → occupancy 或 workload 不均衡 |
| Warp Execution Efficiency | `smsp__thread_inst_executed_per_inst_executed.ratio` | < 85% (即 <27/32) | [Comp] ① 数据预排序消除分支发散 ② 三目运算符 ③ warp vote early exit |
| Branch Efficiency | `smsp__sass_branch_targets_threads_divergent.avg.pct_of_peak_sustained_active` | > 10% | [Comp] 精确定位高发散分支 → 配合 Source 视图 |
| Register Spill | `launch__registers_per_thread` + `--ptxas -v` | > 0 bytes spill | [Comp] ① __launch_bounds__ ② 减少临时变量 ③ Loop fission |
| Achieved Occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` | < 理论值 70% | [Comp] ① 调 block size ② __launch_bounds__ ③ 减少 shared/register |

### 4.3 Latency-bound 诊断表（14 项）

| 诊断维度 | NCU 计数器 | 异常阈值 | 对应优化（按优先级） |
|---|---|---|---|
| 顶层判定 | Memory SOL% 和 Compute SOL% | 两者均 < 30% | 确认 latency-bound，按 stall 原因分诊 |
| Stall Barrier | `smsp__warps_issue_stalled_barrier_per_issue_active.pct` | > 30% | [Sync] ① Warp shuffle 替代 shared+sync ② 双缓冲合并同步点 ③ Cooperative Groups |
| Stall Long Scoreboard | `smsp__warps_issue_stalled_long_scoreboard_per_issue_active.pct` | > 25% | [Mem] 等 L2/DRAM → ① tiling ② cp.async ③ 增加 occupancy ④ prefetch |
| Stall Short Scoreboard | `smsp__warps_issue_stalled_short_scoreboard_per_issue_active.pct` | > 15% | [Mem] 等 L1/shared → ① 减少 bank conflict ② 检查 constant 广播 ③ 增加 ILP |
| Stall Memory Throttle | `smsp__warps_issue_stalled_lg_throttle_per_issue_active.pct` | > 20% | [Mem/Sync] ① cp.async ② 多级流水线 ③ 增加 occupancy |
| Stall TEX Throttle | `smsp__warps_issue_stalled_tex_throttle_per_issue_active.pct` | > 15% | [Mem] TEX 过载 → ① 减少 __ldg() 密度 ② 合并 texture 请求 ③ shared mem 替代 |
| Stall MIO Throttle | `smsp__warps_issue_stalled_mio_throttle_per_issue_active.pct` | > 10% | [Sync] shared/atomic MIO 过载 → ① 减少 shared 频率 ② 局部归约 ③ shuffle 替代 |
| Stall Math Pipe Throttle | `smsp__warps_issue_stalled_math_pipe_throttle_per_issue_active.pct` | > 15% | [Comp] 计算管线反压 → ① 检查 FMA/SFU ② 强度削减 ③ 混合精度 |
| Stall Drain | `smsp__warps_issue_stalled_drain_per_issue_active.pct` | > 10% | [Mem] 等待写入 → ① 减少 store 密度 ② 向量化 store ③ 检查写入 coalescing |
| Stall Not Selected | `smsp__warps_issue_stalled_not_selected_per_issue_active.pct` | 高值 | [Sync] warp 扎堆就绪 → 减少 syncthreads 或拆分同步域 |
| Stall Membar | `smsp__warps_issue_stalled_membar_per_issue_active.pct` | 显著存在 | [Sync] ① atomic 竞争→局部归约 ② 替换 flag 轮询为 coop grid sync |
| Stall Sleeping | `smsp__warps_issue_stalled_sleeping_per_issue_active.pct` | > 5% | [Sync] cuda::barrier arrive_and_wait → 检查 barrier phase 设计 |
| Eligible Warps/Cycle | `smsp__warps_eligible.avg.per_cycle_active` | < 2 | [Sync/Comp] ① 提升 occupancy ② 增加 ILP ③ 检查 barrier 卡点 |

---

## 五、Step 3 — Phase 1-3 无条件执行

不管瓶颈在哪一类，Phase 1-3 都应该直接执行，因为它们几乎没有 downside。

### Phase 1：Pack A — 零成本编译器提示（可打包，1 次 profile）

不改数据布局、不改资源占用、不改同步结构。纯粹帮助编译器生成更好的代码。

| 编号 | 优化项 | 说明 |
|---|---|---|
| C10 | `__restrict__` 指针修饰 | 告诉编译器指针无 alias |
| C6 | `#pragma unroll` | 暴露 ILP，减少循环控制指令 |
| C7 | FMA 显式调用 | `__fmaf_rn()` 确保一条指令 |
| C9 | `--use_fast_math` | 全局快速数学，精度不够时单独回退 |
| C8 | 乘法替代除法 | `a * __frcp_rn(b)` 替代 `a / b` |
| C15 | 强度削减 | `rsqrtf()` 替代 `1/sqrtf()`，移位替代乘除 |
| C18 | Select 替代短分支 | 三目运算符帮助无分支代码生成 |
| C17 | 整数位操作优化 | `__popc()` / `__clz()` / `__ffs()` 走硬件指令 |
| C25 | `__forceinline__` 控制 | 消除调用开销或控制 I-cache |
| M16 | `__ldg()` / `const __restrict__` | 走只读数据缓存 |
| M21 | In-place 原地操作 | 减少额外 output buffer |
| C19 | 谓词化 | 编译器通常自动处理 |

### Phase 2：Pack B — Coalescing 全家桶（可打包，1 次 profile）

全部服务于同一目标：让 warp 访问连续对齐的地址。

| 编号 | 优化项 | 说明 |
|---|---|---|
| M3 | SoA 替代 AoS | 天然合并，差距可达数倍 |
| M2 | 合并访问 / Coalesced | 同 warp 访问连续地址 |
| M7 | 向量化 `float4` Load/Store | 单条指令 128-bit |
| M12 | 对齐访问 128B 边界 | 关注手动偏移和子数组场景 |
| M13 | Padding `cudaMallocPitch` | 每行起始地址对齐 |
| C24 | 循环交换 Loop Interchange | 最内层连续访存 |

### Phase 3：Pack C + C+ — Warp 级同步替代（原子捆绑 + 可打包，1 次 profile）

**Pack C（原子捆绑）**：用 shuffle 替代 shared mem 路径时，必须同时删除对应的 `__syncthreads()`。

| 编号 | 优化项 |
|---|---|
| S1 | Warp Shuffle 替代 `shared + sync` |
| S2 | 删除对应 `__syncthreads()` |
| S3 | `__syncwarp` 替代 `__syncthreads` |
| M10 | Warp Shuffle 数据交换 |
| C5 | Warp Shuffle 用于计算 |
| C4 | 归约优化（warp 内 shuffle） |

**Pack C+（可打包，随 Pack C 一起）**：

| 编号 | 优化项 |
|---|---|
| C21 | Warp Vote `__ballot_sync` |
| C20 | Early Exit `__all_sync` |
| C27 | Warp Match 分组去重 |
| C13 | Scan / Prefix Sum（warp 级） |

**Phase 1-3 小结：3 次 profile 覆盖 30 项优化。**

---

## 六、Step 4 — Phase 4-6 按 NCU 指标选择 Pack

> **关键规则：Phase 4-6 中每个 Pack 必须单独验证。跨 Pack 绝不可打包。**

### 6.1 NCU 指标 → Pack 选择映射

| NCU 指标异常 | 选择 Pack | Phase |
|---|---|---|
| Global Load/Store Eff < 80%、Sectors/Req > 4 | 已在 Phase 2 Pack B 解决 | - |
| L1/L2 Hit Rate 低 | Pack D1（Tiling） | 4 |
| Shared Mem Eff < 90%、Bank Conflict > 0 | Pack D1（Bank Conflict 消除） | 4 |
| SFU 饱和、ALU 饱和 | Pack D2（查表法 LUT） | 4 |
| Achieved Occ < 70%、Register Spill > 0 | Pack E（寄存器 / Occupancy） | 5 |
| DRAM Throughput 接近 SOL | Pack F1（Kernel Fusion） | 6 |
| Stall Mem Throttle > 20%、Long Scoreboard > 25% | Pack F2（异步流水线） | 6 |
| FMA 高但无 Tensor pipe | Pack F3（Tensor Core） | 6 |
| Warp Exec Eff < 85%（数据相关发散） | Pack F4（数据预处理重排） | 6 |

### 6.2 Phase 4：Shared Memory 资源改动

**Pack D1 — Tiling + 配套措施（原子捆绑，隔离验证）**

Tiling 引入 shared memory 后，bank conflict 消除和 sync 精简是配套措施，必须一起做。

| 编号 | 优化项 |
|---|---|
| M4 | Tiling 分块到 shared mem |
| M9 | Bank Conflict 消除（Padding / Swizzle） |
| M15 | Shared Mem 容量配置 `cudaFuncSetAttribute` |
| S4 | Cooperative Groups 细粒度同步 |

**Pack D2 — 查表法（必须隔离）**

LUT 占用 shared memory 或 constant memory，与 tiling 争夺同一资源。

| 编号 | 优化项 |
|---|---|
| C16 | 查表法 LUT（shared / constant） |
| M23 | `__constant__` 内存 |

### 6.3 Phase 5：寄存器 / Occupancy 调整

**Pack E — 寄存器与 Occupancy 权衡（必须隔离）**

直接改变寄存器用量和 occupancy 平衡点，牵一发动全身。

| 编号 | 优化项 |
|---|---|
| M5 | `__launch_bounds__` 寄存器控制 |
| C3 | Launch Config 调优 block size |
| M11 | 寄存器级数据复用（增 ILP） |
| C23 | Loop Fission（降寄存器压力） |
| C22 | Loop Fusion（增寄存器复用） |

### 6.4 Phase 6：结构性变更

**Pack F1 — Kernel Fusion（必须隔离）**

改变 kernel 边界，影响内存、计算、同步三个维度。

| 编号 | 优化项 |
|---|---|
| M1 | Kernel Fusion（内存视角：消除 Global Memory 往返） |
| S8 | Kernel Fusion（同步视角：消除 kernel 间隐式 barrier） |

**Pack F2 — 异步流水线体系（原子捆绑，隔离验证）**

双缓冲 + async copy + barrier/pipeline 是一套完整体系，分开做没有意义。

| 编号 | 优化项 |
|---|---|
| M6 | 双缓冲 / 多级流水线 |
| M8 | `cp.async` 异步拷贝 |
| S5 | `cuda::barrier` / `cuda::pipeline` |
| S6 | 异步搬运消除同步等待 |
| C11 | 软件流水 Software Pipelining |
| C2 | 计算与访存重叠 |

**Pack F3 — Tensor Core 重构（必须隔离）**

| 编号 | 优化项 |
|---|---|
| C1 | Tensor Core / WMMA / MMA |
| C12 | Warp Specialization |

**Pack F4 — 数据预处理重排（必须隔离）**

| 编号 | 优化项 |
|---|---|
| M17 | 数据重排 Z-order / Hilbert |
| C14 | 按 warp 重组消除分支发散 |

---

## 七、Phase 7 — 系统级优化（独立并行）

不改变 kernel 代码，可与 Phase 1-6 的任何阶段并行推进。

**Pack G1 — 调度与传输优化**

| 编号 | 优化项 |
|---|---|
| S7 | CUDA Graphs 减少 launch 开销 |
| S9 | Stream + Event 依赖管理 |
| M14 | CUDA Streams 重叠拷贝与计算 |
| M18 | Pinned Memory 锁页内存 |
| M24 | Stream-Ordered 分配与内存池 |

**Pack G2 — 缓存策略与特殊内存**

| 编号 | 优化项 |
|---|---|
| M19 | L2 Persistence 缓存钉住 |
| M20 | Prefetch 预取 |
| M26 | Texture / Surface 对象 |
| M27 | Unified Memory 优化 |
| M28 | Zero-Copy 内存 |

**Pack G3 — 分析与诊断工具**

| 编号 | 优化项 |
|---|---|
| C26 | PTX / SASS 分析 |
| C28 | SFU 使用意识 |
| M25 | Sector 化理解 |
| S10 | Atomic 操作的同步考量 |
| M22 | Cooperative Groups（跨 block） |

---

## 八、Step 5 — 验证与迭代

### 8.1 Benchmark 度量体系

每个 Pack 执行后，用 `benchmark.py` 采集算子运算速度（配合 `--ref` 参考实现）：

```bash
python benchmark.py solution.cu --ref ref.py --N=4096 --M=4096 --json-out result.json
```

核心度量（来自 benchmark.py 输出）：

| 度量 | 字段 | 含义 |
|---|---|---|
| 运算速度 | `average_ms` / `median_ms` | 算子单次执行耗时，越低越好 |
| 加速比 | `speedup_vs_reference` | 相对参考实现的加速倍数 |
| 粗略带宽 | `bandwidth_gbps_rough` | 所有 tensor 的总吞吐，用于判断是否接近硬件上界 |
| 正确性 | `correctness.passed` | 数值结果是否在容忍范围内 |

### 8.2 每个 Pack 执行后的判决标准

**主判据**：算子运算速度（median_ms / speedup）提升 且 正确性通过。

**辅助判据**：当速度相近时（差距 < 3%），NCU 子指标作为选择依据。

| 结果 | 决策 |
|---|---|
| 速度提升 + 目标 NCU 指标改善 + 其他无退化 | **保留**，高置信度 |
| 速度提升 + 目标 NCU 指标未变 + 其他意外改善 | **保留**，但分析真实原因 |
| 速度相近（< 3% 差距）+ NCU 子指标更优 | **保留**——更健康的子指标 = 更大的后续优化空间 |
| 速度相近（< 3% 差距）+ NCU 子指标更差 | **不保留**——速度没赚到，还恶化了后续优化的基础 |
| 速度未提升 + NCU 指标改善 | **不保留**（瓶颈已转移，指标改善未兑现为速度） |
| 速度下降 | **立即回退** |

**为什么"速度相近时选 NCU 更优"？**

假设优化 A 和优化 B 都让算子从 1.00ms 降到 0.97ms（速度相近），但 A 的 `Global Load Efficiency` 从 60% 提升到 92%，B 只提升到 75%。选 A，因为：
- A 的 coalescing 已经接近理想值，后续不需要再在这个方向投入
- B 的 coalescing 仍有 17% 的浪费空间，意味着当前的 0.97ms 里还有被掩盖的低效
- 当后续做 tiling 或 fusion 等结构性优化时，A 的基础更干净，收益更容易兑现

### 8.3 正确性验证

正确性是前提，速度是目标，二者缺一不可。

- 数值 diff 在容忍范围内（FP32: \(10^{-5}\) 相对误差，fast_math: \(10^{-3}\)）
- benchmark.py 的 `--ref` 验证通过（`correctness.passed = true`）
- `compute-sanitizer --tool racecheck` 无报错（同步优化必做）
- 边界 case 测试（\(N=1\)、\(N\) 非 block size 整数倍）

### 8.4 收敛判断

每轮迭代后检查：
1. `speedup_vs_reference` 是否已达到目标倍数（取决于具体算子和硬件）
2. `bandwidth_gbps_rough` 是否接近硬件带宽上界（Memory-bound kernel）
3. 最近一轮改善是否 < 3%（收益递减）
4. NCU 显示瓶颈已在硬件极限上（如 DRAM throughput > 85% SOL）

满足任一条件即可停止。

### 8.5 瓶颈转移处理

优化后重新 Roofline 分类，常见转移路径：

- Memory-bound → 做了 Kernel Fusion → 变 Compute-bound
- Memory-bound → 做了 Tiling → Stall Barrier 上升 → 变 Latency-bound
- Compute-bound → 上了 Tensor Core → data feeding 跟不上 → 变 Memory-bound
- Latency-bound → 提升了 Occupancy → 吃满带宽 → 变 Memory-bound

每次转移后，切换到对应的诊断表和 Pack 继续优化。

---

## 九、验证清单（NCU）

内存优化建议至少配套以下验证：

1. **带宽利用率**：关注 `Memory SOL %`、`DRAM Throughput`，确认是否接近预期上界。
2. **访问质量**：关注 `Global Load/Store Efficiency`、`Sectors/Request`，确认 coalescing/对齐是否改善。
3. **缓存行为**：关注 `L1 Hit Rate`、`L2 Hit Rate`，确认优化方向与局部性变化一致。
4. **Shared 路径健康度**：关注 `Shared Memory Efficiency`，确认 bank conflict 是否下降。
5. **整体收益**：最终以 benchmark.py 的算子运算速度（`median_ms` / `speedup`）判断，不只看单个子指标。速度相近时以 NCU 子指标择优。

计算优化建议至少配套以下检查：

- **Tensor Core 路径**：确认是否出现预期 MMA/WGMMA 指令路径，且相关计算管线利用率提升。
- **指令效率**：关注 `Issue Slot Utilization`、`Eligible Warps Per Cycle` 是否改善。
- **分支质量**：关注 `Warp Execution Efficiency` 与分支相关 stall 是否改善。
- **寄存器与溢出**：用 `--ptxas-options=-v` + NCU 检查 spill 是否下降。

同步优化建议至少配套以下验证：
1. **同步等待是否下降**：关注 `Stall Barrier` 与相关等待 stall 是否降低。
2. **调度可发射性是否改善**：关注 `Eligible Warps Per Cycle` 是否提升。
3. **warp 级替代是否生效**：将 `__syncthreads()` 替换为 `__syncwarp()` / `__shfl_sync` 后，确认延迟与吞吐有正向变化。
4. **实战建议**：同步优化的前提是正确性。每次删除或替换同步点后，必须用 `compute-sanitizer --tool racecheck` 验证无数据竞争。先保正确，再减同步。

**常见误判**：
- 局部 NCU 指标改善但算子运算速度未提升（通常是把瓶颈转移到别处）。
- 只看 occupancy 升高，不跑 benchmark 验速度，可能出现"occupancy 上去但更慢"。
- 只看 speedup 数字，不查 NCU 子指标，导致后续优化在一个"亚健康"的基础上进行，越改越难。
- 只优化吞吐，不检查 correctness 与数值稳定性。
- 只减少了 `__syncthreads()` 次数，但引入了数据可见性错误。

---

## 十、总迭代节奏

每轮迭代的完整流程：**Pack 优化 → benchmark.py 验速度 → NCU profile 查子指标 → 判决保留/回退**。

| 阶段 | 操作 | Profile 次数 | 覆盖优化项数 |
|---|---|---|---|
| Baseline | benchmark.py 采集基准速度 + ncu 采集基准指标 | 1 | 0 |
| Phase 1-3 | 3 个 Pack 依次执行，每次 benchmark 验速 + ncu 验指标 | 3 | 30 项 |
| Phase 4-6 | NCU 指标驱动选 2-3 Pack，每次 benchmark 验速 + ncu 验指标 | 2-3 | 8-16 项 |
| 瓶颈转移 | 重新 Roofline 分类，再选 1-2 Pack | 1-2 | 4-8 项 |
| **总计** | | **7-10** | **42-54 项（有效子集）** |

判决优先级：**正确性 > 运算速度 > NCU 子指标健康度**

```
正确性不通过 → 立即回退，无条件
速度明显提升（> 3%）→ 保留，即使某些 NCU 子指标退化
速度相近（< 3%）+ NCU 子指标更优 → 保留（更好的优化基础）
速度相近（< 3%）+ NCU 子指标更差 → 不保留（速度没赚到，基础还变差了）
速度下降 → 立即回退
```