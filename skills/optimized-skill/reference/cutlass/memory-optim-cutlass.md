# CUTLASS Kernel 内存优化完整方案：按重要性排序（从高到低）

> 本文档将原始 CUDA kernel 内存优化策略映射到 CUTLASS 框架层面。CUTLASS 通过 Collective Mainloop/Epilogue、TiledCopy、Layout 代数、Pipeline 等抽象，将大量底层内存优化内化为框架配置项。程序员的工作从"手写优化代码"转变为"选择正确的配置组合"。

---

### 1. Kernel Fusion — CUTLASS Epilogue Fusion

CUTLASS 3.x 通过 **Epilogue Visitor Tree（EVT）** 实现算子融合。将 GEMM 后的逐元素操作（Bias、Activation、归一化等）编码为 epilogue 内的 DAG 节点树，中间结果留在寄存器中，不写回 Global Memory。

* **支持的融合模式**：线性组合（alpha·D + beta·C）、逐元素激活（ReLU/GeLU/SiLU）、Bias 加法、行/列广播、自定义逐元素函数。
* **Visitor Tree 的组合性**：多个 visitor 节点可以嵌套组合。例如 `LinCombEltAct` 将线性组合和激活合并为一个 visitor，也可以将 AuxStore（辅助输出）挂在树的分支上实现多输出 fusion。
* **不适合 EVT 的场景**：后续操作涉及跨 tile 的数据依赖（如 Softmax 的全局归一化、LayerNorm 的均值/方差计算），此时需要拆分为多个 kernel 或使用 CUTLASS 的 GroupedGemm + 自定义 epilogue。

**收益量级**：典型的 GEMM + Bias + Activation fusion 可节省一次完整的 Global Memory 往返，对 memory-bound 的小矩阵场景加速可达 1.5×–2×。

---

### 2. 合并访问 / Coalesced Access — Layout 代数自动保证

CUTLASS 的 Layout 代数（Shape + Stride）在构造时就编码了访问连续性：

* **核心原则**：最内层维度的 stride 为 1 即保证合并。CUTLASS 的 `RowMajor`/`ColumnMajor` Layout 已内建此属性。
* **TiledCopy 的 TV Layout**：Thread-Value 映射决定了每个线程访问哪些地址。CUTLASS 预定义的 TiledCopy 配置保证 warp 内线程访问连续地址。
* **自定义 Layout 的陷阱**：如果使用自定义 stride（如带 padding 的 pitch），需确保最内层 stride 仍为 1，否则 TiledCopy 生成的访问模式可能不合并。

**CUTLASS 与手写 CUDA 的区别**：手写时需要程序员计算 `threadIdx.x * stride` 并确保连续性；CUTLASS 将合并性编码在 Layout 的类型系统中，**不合并的 Layout 在编译期就能通过 stride 值发现**。

---

### 3. SoA vs AoS — 独立 Tensor 表达

CUTLASS 天然以独立矩阵（A、B、C、D）为输入，每个矩阵有独立的 Layout 和 stride——这本身就是 SoA 思维。

* **多字段输入**：如果业务数据是 AoS（如粒子的 x/y/z），需要在 CUTLASS 外部做 SoA 变换，将每个字段作为独立 tensor 传入。
* **Epilogue 多输出**：EVT 的 AuxStore 可以将多个输出字段写到不同的 tensor，天然保持 SoA 布局。

---

### 4. Tiling / 分块 — Collective Mainloop 核心机制

CUTLASS 的 tiling 由 **CTA Tile Shape**（`TileShape_MNK`）和 **Collective Mainloop** 共同决定：

* **CTA Tile Shape** 定义了每个 CTA 处理的 M×N×K 分块大小。典型选择如 128×128×64、128×256×32 等。
* **Mainloop 负责搬运**：从 Global Memory 搬运 tile 到 Shared Memory 的逻辑全部封装在 `CollectiveMma` 中，包括地址计算、边界处理、拷贝调度。
* **Tile Shape 的三角权衡**：
  - 越大 → 数据复用率越高（K 维度每个元素被 M×N 个输出复用） → Shared Memory 用量越大 → Occupancy 越低
  - 越小 → Occupancy 越高 → 复用率降低 → DRAM 带宽压力增大
  - 最优值必须通过 benchmark 确定，没有封闭公式

* **补充——Cluster Tile（Hopper）**：Cluster 内的多个 CTA 可以共享 tile 数据（通过 TMA multicast），等效于扩大了 tile 复用范围而不增加单 CTA 的 Shared Memory 用量。

---

### 5. 寄存器压力控制 — Tile Shape 与 MMA Atom 联合决定

CUTLASS 中寄存器用量主要来自 **MMA 累加器 fragment**，其大小由 CTA Tile Shape 和 MMA Atom 的 warp-level tile 共同决定：

* **累加器大小** ≈ (CTA_M / WarpTile_M) × (CTA_N / WarpTile_N) × WarpTile_M × WarpTile_N × sizeof(AccumType)
* **减少寄存器压力的手段**：缩小 CTA Tile Shape、选择更大的 MMA Atom（减少 warp 级重复数）、使用 `__launch_bounds__` 引导编译器。
* **Spill 检测**：`--ptxas-options=-v` 输出的 spill load/store 非零即说明寄存器溢出到 Local Memory。

---

### 6. 双缓冲 / 多级流水线 — Pipeline Stages

CUTLASS 的 Mainloop 通过 `Stages` 模板参数控制流水线级数：

* **Stages = 2**：经典双缓冲，一组加载一组计算。
* **Stages = 3–7**：多级流水线，更多 stage 可以更好地隐藏访存延迟，但每级都消耗一份 Shared Memory buffer。
* **Stage 数的选择原则**：
  - 增加 stage → 更好的延迟隐藏 → 更多 Shared Memory 占用 → Occupancy 可能下降
  - 经验法则：Ampere 上 3–4 stage 通常最优；Hopper 上 TMA + Warp Specialization 模式下 2–4 stage 即可
* **自动 stage 选择**：CUTLASS 的 `cutlass::gemm::collective::StageCountAutoCarveout` 可根据 Shared Memory 容量自动选择最大可用 stage 数。

---

### 7. 向量化访存 — Copy Atom 的向量宽度

CUTLASS 的 Copy Atom 直接指定每次访存的位宽：

* **128-bit（uint128_t）**：单条指令读写 128-bit，等价于 float4 或 8 个 half。**绝大多数场景的默认选择。**
* **64-bit / 32-bit**：仅在对齐不满足 128-bit 或数据类型较窄时回退。
* **CUTLASS 自动选择**：`DefaultCopy` 会根据数据类型和对齐自动选择最宽的 Copy Atom。手动选择仅在自定义 kernel 时需要。

---

### 8. 异步拷贝 — Mainloop 架构决定

CUTLASS 通过 Mainloop 选择异步拷贝策略：

* **Ampere（SM80）**：`MainloopSm80CpAsync` 使用 `cp.async` 指令，Global → Shared 由硬件 DMA 完成，不占计算管线。配合 `commit_group` + `wait_group<N>` 实现多级流水。
* **Hopper（SM90）**：`MainloopSm90Tma*` 使用 TMA（Tensor Memory Accelerator），整个 tile 的搬运由单条 TMA 描述符指令完成，不占任何线程资源。配合硬件 barrier 实现零开销同步。
* **Predicated vs Unpredicated**：`CpAsyncUnpredicated` 不做边界检查，要求输入尺寸是 tile 的整数倍；`CpAsyncPredicated` 处理任意尺寸但有少量额外指令开销。

**补充（TMA Multicast）**：Hopper 的 TMA 支持 multicast——一次搬运同时写入 Cluster 内多个 CTA 的 Shared Memory，进一步减少 DRAM 访问量。

---

### 9. Bank Conflict 消除 — Swizzle 机制

CUTLASS 使用 **Swizzle<B, M, S>** 在 Shared Memory Layout 中消除 bank conflict：

* **原理**：对 Shared Memory 地址的 bit [S, S+B) 与 bit [S+M, S+M+B) 做 XOR，打乱 bank 映射，使原本映射到同一 bank 的线程分散到不同 bank。
* **与 Padding 的对比**：传统 `[32][33]` padding 浪费约 3% 空间且破坏对齐；Swizzle 零空间浪费，但增加了地址计算复杂度（编译期解决）。
* **参数选择**：取决于 tile shape、数据类型宽度和 bank 宽度（32-bit）。**CUTLASS 预定义的 SmemLayout 已包含正确的 Swizzle 参数**，自定义 Layout 时需手动选择。

---

### 10. Warp Shuffle — MMA Atom 内建

CUTLASS 的 MMA Atom 内部已处理 warp 内数据交换。程序员在以下场景仍需显式 shuffle：

* **Epilogue 中的跨线程归约**：如 Split-K 的 warp 级归约。
* **自定义后处理**：如 Top-K、Softmax 等需要 warp 内通信的操作。
* **CUTLASS 不封装 shuffle API**：直接使用 `__shfl_sync` 系列。

---

### 11. 寄存器级数据复用 — MMA 累加器

CUTLASS 的 GEMM mainloop 在 K 维度迭代中持续累加到寄存器 fragment，实现了极致的寄存器级复用：

* 每个 MMA fragment 在整个 K-loop 中不写回 Shared/Global Memory。
* 只有 epilogue 阶段才将累加结果写回。
* 增加每线程持有的输出元素数（增大 CTA_M × CTA_N / thread_count）可进一步提升 ILP。

---

### 12–13. 对齐访问 / Padding

* **对齐**：CUTLASS 要求输入指针按 `AlignmentA`/`AlignmentB` 对齐（通常 16 字节 = 128-bit）。`cudaMalloc` 返回 256B 对齐，天然满足。手动偏移子矩阵时需注意对齐降级。
* **Padding / Pitch**：通过 Layout 的 leading dimension（lda/ldb/ldc）编码 pitch。CUTLASS 的 stride 参数直接接受非紧密排列的矩阵。

---

### 14. CUDA Streams 重叠

与 CUTLASS 正交。CUTLASS 的 `GemmUniversal::run()` 接受 stream 参数，多个 GEMM 可在不同 stream 上并行。

---

### 15. Shared Memory 容量配置

CUTLASS 的 kernel launch 逻辑自动调用 `cudaFuncSetAttribute` 设置最大动态 Shared Memory。所需大小由 Mainloop 的 Shared Storage 类型自动计算（= Stages × TileSize × sizeof(element)）。

* **Ampere**：最大 164 KB（A100），需配合 `cudaFuncSetAttribute` 突破 48 KB 默认限制。
* **Hopper**：最大 228 KB（H100），TMA + Warp Specialization 的 Shared Memory 预算更紧张（需额外空间存 TMA 描述符和 barrier）。

---

### 16. 只读数据路径

* **Copy Atom 缓存策略**：`CACHEALWAYS` 走 L1+L2；`CACHEGLOBAL` 绕过 L1 直接走 L2（适合大 working set）。
* **`const __restrict__`**：在 CUTLASS kernel 的指针参数上标注，让编译器自动选择只读缓存路径。
* **补充**：新架构下编译器通常自动优化，手动选择缓存策略的收益递减，应以 NCU 数据为准。

---

### 17. 数据重排 / Data Reordering

CUTLASS 不内建数据重排功能。对稀疏矩阵、图计算等不规则访问模式，需要在 CUTLASS 外部做预处理：

* **CUTLASS Sparse GEMM**：支持 2:4 结构化稀疏（Ampere+），自动处理压缩格式。
* **非结构化稀疏**：需要外部预处理（CSR 排序、分桶等）后，将密集子块交给 CUTLASS 处理。

---

### 18. Pinned Memory / 锁页内存

与 CUTLASS 正交。CUTLASS 不管理 host 端内存分配。推荐在数据搬运频繁的流水线中使用 `cudaHostAlloc`。

---

### 19. L2 缓存优化

* **L2 Persistence**：`cudaAccessPolicyWindow` 是 host 端 API，与 CUTLASS 正交。对反复访问的小矩阵（如 embedding lookup）有效。
* **CUTLASS 中的间接控制**：Tile Shape 越大 → working set 越大 → L2 命中率可能下降。缩小 K-tile 可以减小每次迭代的 working set。

---

### 20–28. 其余优化

| 编号 | 优化项 | CUTLASS 关系 |
|---|---|---|
| 20 | Prefetch | Host API，与 CUTLASS 正交 |
| 21 | In-place | CUTLASS 支持 D 和 C 指向同一地址（in-place epilogue） |
| 22 | Cooperative Groups | CUTLASS Hopper kernel 使用 Cluster 替代跨 block 协作 |
| 23 | `__constant__` | 不常用于 CUTLASS GEMM；可在自定义 epilogue 中通过 constant 传入小参数 |
| 24 | Stream-Ordered 分配 | CUTLASS workspace 支持 `cudaMallocAsync` |
| 25 | Sector 化理解 | 指导 Swizzle 参数选择 |
| 26 | Texture / Surface | CUTLASS 不使用 Texture，走 TiledCopy 路径 |
| 27 | Unified Memory | 性能场景不推荐，CUTLASS 假设显式分配 |
| 28 | Zero-Copy | 仅 data-starved 场景，与 CUTLASS 正交 |

---

## 验证清单（NCU）

内存优化验证方法与原始文档一致，CUTLASS 特有关注点：

1. **带宽利用率**：`Memory SOL %`、`DRAM Throughput`
2. **访问质量**：`Global Load/Store Efficiency`、`Sectors/Request` — **验证 TiledCopy 的 TV Layout 合并性**
3. **缓存行为**：`L1/L2 Hit Rate` — 验证 Tile Shape 和 Copy Atom 缓存策略选择
4. **Shared 路径健康度**：`Shared Memory Efficiency` — **验证 Swizzle 参数是否消除 bank conflict**
5. **整体收益**：以 kernel latency（`median_ms` / `speedup`）为最终判据

**CUTLASS 特有验证**：
- Swizzle 后 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 应降至 0
- `DRAM Read Bytes` / 理论数据量 = 带宽放大系数，Epilogue Fusion 后应接近 1.0
- Pipeline stage 增加后应观察到 `Stall Long Scoreboard` 下降

统一决策树请参考：`rule-cutlass.md` 的"七、统一优化决策树（SSOT）"。
