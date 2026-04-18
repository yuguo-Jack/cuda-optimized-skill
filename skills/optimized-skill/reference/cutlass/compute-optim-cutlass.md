# CUTLASS Kernel 计算优化完整方案：按重要性排序（从高到低）

> 本文档将原始 CUDA kernel 计算优化策略映射到 CUTLASS 框架层面。CUTLASS 的核心计算抽象是 MMA Atom——定义了硬件 MMA 指令的形状、精度和线程映射。计算优化在 CUTLASS 中主要体现为"选择正确的 Atom 和 Tile Shape 组合"。

---

### 1. Tensor Core / 专用硬件单元利用 — MMA Atom 选择

CUTLASS 通过 MMA Atom 决定走 Tensor Core 还是 CUDA Core，**这是最关键的单一配置决策**：

* **Ampere（SM80）MMA Atom**：
  - `SM80_16x8x16_F32F16F16F32`：FP16 输入、FP32 累加，16×8×16 MMA shape。最常用。
  - `SM80_16x8x16_F32BF16BF16F32`：BF16 输入，训练场景常用。
  - `SM80_16x8x8_F32TF32TF32F32`：TF32 输入，FP32 精度的折中方案。

* **Hopper（SM90）WGMMA Atom**：
  - `SM90_64x128x16_F32F16F16F32_SS`：Shared × Shared，两个操作数都在 Shared Memory。
  - `SM90_64x128x16_F32F16F16F32_RS`：Register × Shared，A 在寄存器、B 在 Shared Memory。
  - WGMMA 的 64×128 shape 远大于 Ampere 的 16×8——单条指令处理更大矩阵块，吞吐更高。

* **CUDA Core 回退**：`UniversalFMA<float>` 走 FMA 管线，用于不适合 Tensor Core 的非矩阵运算。

**选择原则**：只要运算可以表达为矩阵乘加（GEMM、卷积、Attention），就应该选择 Tensor Core Atom。不用 Tensor Core 基本等于浪费了一半以上的芯片算力。

---

### 2. 计算与访存重叠 — Pipeline 多级流水

CUTLASS 实现计算-访存重叠的核心机制是 Pipeline 的多 stage 设计：

* **原理**：stage K 的数据在被计算的同时，stage K+1 的数据正在从 Global Memory 加载。多个 stage 允许多批数据"在飞行中"。
* **影响重叠质量的因素**：
  - Stage 数：越多越好隐藏延迟，但受 Shared Memory 容量限制。
  - Occupancy：同一 SM 上的多个 CTA 可以在一个 CTA 等待数据时切换到另一个 CTA 执行计算。
  - ILP：MMA 累加器的分布式存储让每个线程持有多个独立计算，调度器可以在等待一个 MMA 结果时发射另一个。

* **Warp Specialization 的额外重叠**：producer warp 和 consumer warp 可以真正并行——producer 执行 TMA 的同时 consumer 执行 MMA，不是交替执行而是物理并行。

---

### 3. Launch Configuration 调优 — CTA Tile Shape 体系

CUTLASS 中 launch configuration 不是简单的 block size 选择，而是一个多维度的 Tile Shape 体系：

* **CTA Tile Shape（TileShape_MNK）**：决定每个 CTA 处理的输出分块大小。直接影响 Shared Memory 用量、寄存器压力、MMA 利用率。
* **Cluster Tile Shape（ClusterShape_MNK，Hopper）**：决定 cluster 内 CTA 的排列方式。影响 TMA multicast 效率和 DSMEM 通信模式。
* **MMA Atom 重复数**：Atom 在 MNK 方向的重复次数决定了 warp-level tile 大小，间接影响每线程寄存器用量。

**Tile Shape 选择的关键权衡**：
- M × N 越大 → 每个 CTA 的输出越多 → 数据复用率越高 → 寄存器和 Shared Memory 用量越大 → Occupancy 越低
- K tile 越大 → 每次迭代处理更多 K 元素 → 减少 mainloop 循环开销 → 但增加 Shared Memory 用量
- **没有理论最优解**，必须对 2-4 个候选 Tile Shape 做 benchmark 选择最优

**常用 Tile Shape 起点**：
- Ampere：128×128×32、128×256×32、256×128×32
- Hopper：128×128×64、128×256×64、256×128×64

---

### 4. 归约优化 — Split-K 与 Stream-K

CUTLASS 提供两种 GEMM 级归约策略：

* **Data-Parallel（默认）**：每个 CTA 独立处理整个 K 维度，无跨 CTA 归约。适合 K 较小或 M×N 足够大的场景。
* **Split-K**：将 K 维度切分到多个 CTA，每个 CTA 处理一个 K 切片，最后归约。适合 M×N 小但 K 很大的场景（如 Attention 中的 V 投影）。
  - `SplitKSerial`：用独立的 reduction kernel 归约，需要 workspace buffer。
  - `SplitKAtomic`：epilogue 内用 atomicAdd 归约，无额外 kernel 但有竞争。
* **Stream-K（高级）**：将整个 GEMM 的工作均匀分配到所有 SM，消除 wave quantization 导致的尾部浪费。适合 CTA 数 < SM 数 × 波数的场景。

**归约层次**（从内到外）：
1. MMA 累加器内部：warp 内 K 维度累加（硬件级）
2. Mainloop K-loop：跨 K-tile 的累加（寄存器级）
3. Epilogue：warp 级局部归约 → block 级汇总（Shared Memory）
4. 跨 CTA：Split-K 或 Stream-K 的全局归约（atomic 或 reduction kernel）

---

### 5. Warp Shuffle 用于计算

CUTLASS 的 MMA Atom 在内部处理了 warp 级数据交换。程序员需要显式 shuffle 的场景有限：

* **自定义 epilogue**：如 warp 级 softmax 需要 `__shfl_xor_sync` 做行级归约。
* **非 GEMM kernel**：如独立的 reduction/scan kernel，CUTLASS 不提供这类 kernel 的高层抽象（应使用 CUB）。
* **Split-K 归约**：CUTLASS 内部的归约 epilogue 已自动使用 warp shuffle 做局部归约。

---

### 6. 循环展开 / Loop Unrolling

CUTLASS 的循环展开策略：

* **K-loop mainloop**：`CUTLASS_PRAGMA_UNROLL` 标注内层 MMA 循环。K-tile 内的 MMA 调用次数在编译期已知，编译器完全展开。
* **编译期维度**：CUTLASS 大量使用 `cute::Int<N>` 静态整数，不仅循环展开，连索引计算都在编译期完成——消除了运行时地址计算指令。
* **展开的代价**：过度展开导致指令缓存压力（I-cache miss）。CUTLASS 的 Tile Shape 选择间接控制了展开程度——K-tile 越大、展开的 MMA 调用越多。

---

### 7. FMA 显式使用

MMA Atom 的选择直接决定指令路径：

* Tensor Core Atom → 编译为 `mma.sync`（Ampere）或 `wgmma`（Hopper）PTX 指令
* `UniversalFMA<float>` Atom → 编译为 `fma.rn.f32` PTX 指令
* **Epilogue 中的 FMA**：CUTLASS 的线性组合 epilogue（alpha × D + beta × C）由编译器生成 FMA。如果编译器未合并，可在自定义 epilogue 中用 `__fmaf_rn()` 显式调用。

---

### 8. 用乘法替代除法

CUTLASS 框架内部已优化了地址计算中的除法（如 tile 坐标计算用 `FastDivmod`）。用户自定义的 epilogue 或后处理中仍需手动注意：

* `a / b` → `a * __frcp_rn(b)` 或 `a * (1.0f / b)`
* CUTLASS 的 `FastDivmod` 将整数除法转换为乘法 + 移位，用于 problem shape 到 tile 坐标的映射。

---

### 9. `--use_fast_math` 与编译选项调优

CUTLASS 项目通常启用以下编译选项：

* `--use_fast_math`：全局快速数学。注意对 FP16 denorm 的影响——`--ftz=true` 会 flush denorm to zero，Tensor Core 输入中的极小值可能被截断。
* `--expt-relaxed-constexpr`：CUTLASS/CuTe 模板元编程所需。
* `-std=c++17` 或更高：CUTLASS 3.x 要求。
* `--ptxas-options=-v`：输出寄存器用量和 spill 信息，每次 Tile Shape 调整后必看。

---

### 10. `__restrict__` 关键字

应作为 CUTLASS kernel 所有指针参数的默认修饰。CUTLASS 的 `GemmUniversal` 内部的 kernel 函数已在关键路径上使用 `__restrict__`。自定义 kernel 时务必手动添加。

---

### 11. 软件流水 / Software Pipelining — Pipeline 自动实现

CUTLASS 的 Pipeline + Mainloop 就是软件流水线的形式化实现：

* **Prolog**：Pipeline 启动阶段，预填充前 `Stages - 1` 个 buffer。
* **Steady State**：每个迭代同时执行"加载 stage K+S"和"计算 stage K"。
* **Epilog**：最后几个 stage 只有计算没有加载（drain phase）。

CUTLASS 的 `CollectiveMma::load` 和 `CollectiveMma::mma` 分离了加载和计算的逻辑，Pipeline 状态机自动管理交替推进。程序员不需要手写 prolog/epilog。

---

### 12. Warp Specialization — CUTLASS 3.x Hopper 核心架构

CUTLASS 3.x 在 Hopper 上的三种 Mainloop 策略：

* **`TmaWarpSpecialized`**：CTA 内 warpgroup 分为 producer（1 个）和 consumer（N-1 个）。Producer 执行 TMA 加载，consumer 执行 WGMMA。两者物理并行。
* **`TmaWarpSpecializedCooperative`**：所有 warpgroup 都参与计算，在计算间隙协作执行 TMA。适合计算密集但搬运轻量的场景。
* **`TmaWarpSpecializedPingpong`**：consumer warpgroup 交替执行 MMA 和 epilogue store，提高 store 管线利用率。

**选择原则**：
- 大矩阵（M, N ≥ 1024）：WarpSpecialized 或 Cooperative，TMA 搬运与计算完全重叠。
- 小矩阵或 batch：WarpSpecializedPingpong，减少 epilogue 等待时间。
- Ampere：无 Warp Specialization，用 CpAsync + Cooperative 模式。

---

### 13. Scan/Prefix Sum 优化

CUTLASS 不提供独立的 Scan 原语。应使用 CUB 的高度优化实现（`cub::DeviceScan`、`cub::WarpScan`、`cub::BlockScan`）。CUTLASS 内部在部分场景使用 prefix sum（如 Stream-K 的 tile 分配），但不暴露为用户 API。

---

### 14. 按 warp 重组数据消除分支发散

CUTLASS 的 TiledMma 保证同一 warp 处理的数据在内存中连续，天然减少了分支发散：

* **Predicated Copy 替代 if-else**：CUTLASS 的 `CpAsyncPredicated` Mainloop 用谓词遮罩处理不整除的 tile 边界，避免分支指令。
* **数据预排序**：不规则 workload（如稀疏矩阵）仍需外部预处理。CUTLASS 的 Sparse GEMM 支持 2:4 结构化稀疏，硬件自动跳过零元素。

---

### 15–28. 其余计算优化

| 编号 | 优化项 | CUTLASS 关系 |
|---|---|---|
| 15 | 强度削减 | CUTLASS 内部用 `FastDivmod` 优化地址计算；自定义 epilogue 中手动应用 |
| 16 | 查表法 LUT | 可在自定义 epilogue 中实现，LUT 放 Shared Memory；与 mainloop 争夺 Shared Memory 容量 |
| 17 | 整数位操作 | `__popc()`/`__clz()`/`__ffs()` 直接使用 |
| 18 | Select 替代短分支 | 自定义 epilogue 中用三目运算符 |
| 19 | 谓词化 | CUTLASS Predicated Mainloop 内建谓词处理边界 |
| 20 | Early Exit | 不适用于 GEMM mainloop（K-loop 必须完整执行）；可在 epilogue 中跳过全零输出 |
| 21 | Warp Vote | 直接使用 |
| 22 | Loop Fusion | Epilogue Fusion 是其高级形式 |
| 23 | Loop Fission | 调整 K-tile 大小间接实现 |
| 24 | Loop Interchange | Layout stride 顺序天然解决 |
| 25 | 内联控制 | CUTLASS 内部已大量使用 `CUTLASS_HOST_DEVICE` + `__forceinline__` |
| 26 | PTX/SASS 分析 | **最重要的验证手段**——确认 SASS 中出现 `HMMA`/`WGMMA` 指令 |
| 27 | Warp Match | 直接使用，适用面窄 |
| 28 | SFU 意识 | 不变——自定义 epilogue 中大量超越函数时注意 SFU 瓶颈 |

---

## 验证清单（NCU）

计算优化验证方法与原始文档一致，CUTLASS 特有关注点：

- **Tensor Core 路径**：SASS 中确认 `HMMA`（Ampere）或 `WGMMA`（Hopper）指令。`sm__pipe_tensor_cycles_active` 应 > 0 且占比显著。如果此指标为 0，说明 MMA Atom 选择错误（如用了 FP32 atom 处理 FP16 数据）。
- **MMA 利用率**：`sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` — Atom shape 与 Tile Shape 的匹配度。利用率低说明 tile shape 没有完整填满 MMA 管线。
- **指令效率**：`Issue Slot Utilization`、`Eligible Warps/Cycle` — CUTLASS 的编译期索引计算应减少 ALU 指令。如果 ALU 占比异常高，检查是否有运行时地址计算泄露。
- **分支质量**：`Warp Execution Efficiency` — Predicated Mainloop 应消除边界检查的分支发散。
- **寄存器与溢出**：`--ptxas-options=-v` — 累加器 fragment 大小由 Tile Shape 和 Atom 共同决定。Spill > 0 时应缩小 Tile Shape 或减少 Pipeline stage。
- **Warp Specialization 平衡**：producer warpgroup 的利用率应极高（几乎不空转）；如果 producer 大量 stall，说明 TMA 加载量不足以喂饱 consumer。

**常见误判**：
- MMA Atom 选错精度（如 FP32 Atom + FP16 数据），Tensor Core 完全没用上——检查 SASS 是否有 `HMMA`/`WGMMA`
- Tile Shape 过大导致 occupancy = 1 block/SM，延迟隐藏能力丧失——occupancy 升高不一定更快，但 occupancy = 1 几乎一定更慢
- Pipeline stage 过多消耗 Shared Memory，反而降低了 occupancy——增加 stage 后必须同时检查 occupancy 变化
- 只看 Tensor Core utilization 升高，不看 kernel 总耗时——可能 data feeding 成为新瓶颈

统一决策树请参考：`rule-cutlass.md` 的"七、统一优化决策树（SSOT）"。
