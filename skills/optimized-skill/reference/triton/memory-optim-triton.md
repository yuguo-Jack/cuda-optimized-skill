# Triton Kernel 内存优化完整方案：按重要性排序（从高到低）

> 本文档将原始 CUDA kernel 内存优化策略映射到 Triton 编译器框架层面。Triton 通过 tile 级编程模型、自动向量化、编译器驱动的 Shared Memory 管理和 Pipeline 抽象，将大量底层内存优化内化为编译器行为。

---

### 1. Kernel Fusion — Triton 的天然优势

Triton 的 kernel fusion 是其最大的架构优势之一——**用户直接在 Python 中编写融合后的 kernel**，无需像 CUTLASS 那样通过 Epilogue Visitor Tree 配置：

* **直接编码融合逻辑**：在同一个 `@triton.jit` kernel 中，`tl.dot` 后紧接 bias 加法、激活函数、归一化等操作，中间结果留在寄存器中，不写回 Global Memory。
* **支持的融合模式**：线性组合、逐元素激活（ReLU/GeLU/SiLU/Swish）、Bias 加法、行/列广播归约（如 online softmax）、Dropout mask 生成与应用、多输出写回。
* **与 CUTLASS EVT 的对比**：CUTLASS 的 EVT 限于预定义的 visitor 节点组合；Triton 可以编写任意 Python 逻辑作为 epilogue，灵活性远超 CUTLASS。
* **不适合单 kernel 融合的场景**：后续操作涉及跨 tile 的全局依赖（如 LayerNorm 的全局均值/方差需要两遍扫描），此时需要拆分为多个 kernel 或使用 Flash Attention 式的 online 算法在 tile 内近似。

**收益量级**：典型的 GEMM + Bias + Activation fusion 可节省一次完整的 Global Memory 往返，对 memory-bound 的小矩阵场景加速可达 1.5×–2×。更复杂的融合（如 GEMM + Softmax + GEMM 的 Flash Attention）可节省多次 Global Memory 读写。

---

### 2. 合并访问 / Coalesced Access — 编译器自动保证（需正确引导）

Triton 的 tile 编程模型天然倾向于生成合并访问，但程序员仍需注意访问模式的正确性：

* **核心原则**：`tl.load` / `tl.store` 的指针 block 中，最内层（连续）维度的地址应在 warp 内线程间连续。Triton 编译器根据指针偏移的 stride 模式自动判断合并性。
* **行优先 vs 列优先**：`tl.make_block_ptr` 的 `order` 参数（如 `(1, 0)` 表示列优先内存布局）告诉编译器哪个维度连续，编译器据此生成合并的加载指令。
* **指针算术模式**：使用 `tl.arange` 构造的偏移如果最内维 stride 为 1，编译器生成合并访问。如果 stride 不为 1（如转置访问），编译器会生成非合并的 scattered load。
* **`tl.make_block_ptr` 的优势**：相比手动指针算术，`tl.make_block_ptr` 显式声明 shape、stride 和 order，编译器可以更好地推导合并性并选择最优指令（如 TMA）。

**Triton 与手写 CUDA 的区别**：手写时需要程序员计算 `threadIdx.x * stride` 并确保连续性；Triton 编译器从 tile 的指针 block 结构自动推导线程到地址的映射。**但如果指针偏移模式错误（如内维 stride ≠ 1），编译器无法修复——合并性仍由程序员的访问模式决定。**

---

### 3. SoA vs AoS — 独立 Tensor 表达

Triton kernel 以独立 tensor 指针为输入，每个 tensor 有独立的 stride 和 layout——天然 SoA 思维：

* **多字段输入**：如果业务数据是 AoS（如粒子的 x/y/z），需要在 Triton kernel 外部做 SoA 变换，将每个字段作为独立 tensor 传入。
* **多输出**：kernel 内可以对不同结果 tensor 分别 `tl.store`，天然保持独立布局。
* **Stride 参数传递**：Triton kernel 通常将 stride 作为参数传入（如 `stride_am`、`stride_ak`），支持灵活的内存布局。

---

### 4. Tiling / 分块 — Block Shape 核心机制

Triton 的 tiling 由 **Block Shape 常量**（`BLOCK_M`、`BLOCK_N`、`BLOCK_K`）和编译器的 Shared Memory 管理共同决定：

* **Block Shape** 定义了每个 program 处理的分块大小。编译器根据 Block Shape 自动分配 Shared Memory、生成加载/存储指令、调度 pipeline。
* **编译器负责搬运**：`tl.load` 的数据在编译器判断需要复用时会自动提升到 Shared Memory；K-loop 中的 `tl.load` 数据会被编译器安排到 Shared Memory buffer 中。程序员不需要显式管理 Shared Memory。
* **Block Shape 的三角权衡**：
  - 越大 → 数据复用率越高（K 维度每个元素被 M×N 个输出复用） → Shared Memory 和寄存器用量越大 → Occupancy 越低
  - 越小 → Occupancy 越高 → 复用率降低 → DRAM 带宽压力增大
  - 最优值必须通过 `triton.autotune` 或手动 benchmark 确定，没有封闭公式

* **Cluster Tiling（Hopper）**：`num_ctas` 参数定义 Cluster 内 CTA 数。Cluster 内的 program 可以通过 DSMEM 共享数据（TMA multicast），等效于扩大 tile 复用范围而不增加单个 program 的 Shared Memory 用量。

---

### 5. 寄存器压力控制 — Block Shape 与 `num_warps` 联合决定

Triton 中寄存器用量主要来自 **`tl.dot` 累加器**和**活跃的 tile 变量**，其大小由 Block Shape 和 `num_warps` 共同决定：

* **累加器大小** ≈ `BLOCK_M × BLOCK_N / (num_warps × 32) × sizeof(AccumType)`（每线程持有的累加器元素数）
* **减少寄存器压力的手段**：缩小 `BLOCK_M/N`、增大 `num_warps`（将工作分摊到更多 warp）、减少同时活跃的 tile 变量数。
* **Spill 检测**：NCU 中的寄存器用量统计。Triton 编译器在寄存器不足时会自动 spill 到 Local Memory，导致性能下降。
* **编译器控制力有限**：Triton 不提供 `__launch_bounds__` 等价物。寄存器分配完全由编译器决定。程序员通过调整 Block Shape 和 `num_warps` 间接控制。

---

### 6. 多级流水线 — `num_stages` 参数

Triton 通过 `num_stages` 控制 K-loop 的软件流水线级数：

* **`num_stages = 2`**：经典双缓冲，一组加载一组计算。
* **`num_stages = 3–7`**：多级流水线，更多 stage 可以更好地隐藏访存延迟，但每级都消耗一份 Shared Memory buffer。
* **Stage 数的选择原则**：
  - 增加 stage → 更好的延迟隐藏 → 更多 Shared Memory 占用 → Occupancy 可能下降
  - 经验法则：Ampere 上 3–4 stage 通常最优；Hopper 上 2–4 stage 即可
* **编译器自动管理**：Triton 编译器根据 `num_stages` 自动生成 prolog（预填充）、steady state（加载-计算交替）、epilog（drain）代码。程序员不需要手写流水线状态机。

---

### 7. 向量化访存 — 编译器自动向量化

Triton 编译器根据数据类型和对齐自动选择最宽的向量化加载/存储指令：

* **128-bit（16 字节）**：FP16 × 8 元素或 FP32 × 4 元素，单条指令。**编译器的默认目标。**
* **降级场景**：对齐不满足 128-bit 或 mask 导致部分元素无效时，编译器回退到更窄的向量宽度。
* **程序员的责任**：确保 tensor 指针按 16 字节对齐（`torch.empty` 分配的 tensor 天然满足）。子矩阵偏移可能破坏对齐——`BLOCK_K` 应选择使偏移保持对齐的值。
* **`tl.make_block_ptr`**：显式声明 block 的 shape 和 stride，编译器可以更好地推导对齐和向量宽度。

---

### 8. 异步拷贝 — 编译器自动选择

Triton 编译器根据目标架构自动选择异步搬运策略：

* **Ampere（SM80）**：编译器自动生成 `cp.async` 指令，Global → Shared 由硬件 DMA 完成。配合 `commit_group` + `wait_group<N>` 实现多级流水。
* **Hopper（SM90）**：编译器自动生成 TMA（Tensor Memory Accelerator）指令（当使用 `tl.make_block_ptr` 时更容易触发 TMA）。TMA 由硬件 DMA 引擎执行，不占任何线程资源。
* **TMA 触发条件**：使用 `tl.make_block_ptr` 且 block shape 和 stride 满足 TMA 描述符要求时，编译器更倾向于生成 TMA 指令。手动指针算术（`tl.load(ptr + offsets)`）可能阻碍 TMA 生成。

**Triton 与 CUTLASS 的区别**：CUTLASS 需要显式选择 Mainloop 类型（`CpAsyncPredicated` vs `TmaWarpSpecialized`）；Triton 由编译器自动决定。程序员通过代码结构（`tl.make_block_ptr` vs 手动指针）间接影响编译器选择。

---

### 9. Bank Conflict 消除 — 编译器自动 Swizzle

Triton 编译器自动在 Shared Memory 布局中插入 swizzle 以消除 bank conflict：

* **自动 swizzle**：编译器在 TTGIR → LLVM IR lowering 阶段分析 Shared Memory 访问模式，自动选择 swizzle 参数。
* **程序员不直接控制**：不同于 CUTLASS 显式指定 `Swizzle<B,M,S>` 参数，Triton 的 swizzle 由编译器决定。
* **验证**：NCU 中 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 应为 0。如果不为 0，通常是编译器的自动 swizzle 失效——可能需要调整 Block Shape 使其对 bank 宽度友好。
* **`BLOCK_K` 对 bank conflict 的影响**：`BLOCK_K` 为 32 的倍数时更容易产生 bank conflict（32 个 bank × 4 字节 = 128 字节对齐）。编译器通常通过 swizzle 解决，但不保证所有情况。

---

### 10. 寄存器级数据复用 — `tl.dot` 累加器

Triton 的 GEMM kernel 在 K-loop 中持续累加到寄存器变量（`acc`），实现极致的寄存器级复用：

* 每个 `acc` tile 在整个 K-loop 中不写回 Shared/Global Memory。
* 只有 K-loop 结束后的 epilogue 阶段（`tl.store`）才将累加结果写回。
* 增大 `BLOCK_M × BLOCK_N`（即每 program 的输出 tile）可进一步提升每线程的计算密集度和数据复用。

---

### 11. 对齐访问

* **Tensor 对齐**：PyTorch 的 `torch.empty` / `torch.zeros` 分配的 tensor 天然满足 256 字节对齐。
* **子矩阵对齐**：当 kernel 处理子矩阵（如 batch GEMM 的 offset）时，偏移量可能破坏对齐。应确保 `BLOCK_K × sizeof(element)` 是 16 字节的整数倍。
* **`tl.make_block_ptr` 的 `boundary_check` 参数**：声明哪些维度需要边界检查，编译器据此决定是否生成谓词化加载。无需边界检查的维度可以生成更高效的非谓词化指令。

---

### 12. Padding / Pitch

Triton 通过 stride 参数天然支持非紧密排列的矩阵：

* **Leading dimension**：作为 stride 参数传入 kernel（如 `stride_am` 表示矩阵 A 在 M 维度的 stride），支持任意 pitch。
* **编译器感知**：编译器根据 stride 值判断是否需要 scatter/gather 访存。stride 为 1 的维度生成合并访问。

---

### 13. CUDA Streams 重叠

与 Triton 正交。Triton kernel 通过 PyTorch 的 stream 管理并行执行：

* 多个 Triton kernel 可在不同 `torch.cuda.Stream` 上并行。
* Triton 的 kernel launch 是异步的，自动参与 stream 的依赖管理。

---

### 14. Shared Memory 容量管理

Triton 编译器自动管理 Shared Memory 分配和容量配置：

* **自动计算**：编译器根据 Block Shape、数据类型和 `num_stages` 自动计算所需 Shared Memory 大小。
* **自动配置**：编译器在 launch 时自动调用 `cudaFuncSetAttribute` 设置最大动态 Shared Memory。
* **容量限制**：Ampere 最大 164 KB（A100）；Hopper 最大 228 KB（H100）。Block Shape × `num_stages` × sizeof(element) 超限时，编译器报错或自动降级。
* **程序员的间接控制**：通过调整 Block Shape 和 `num_stages` 间接控制 Shared Memory 用量。

---

### 15. 只读数据路径

* **`tl.load` 的缓存行为**：编译器根据访问模式自动选择 L1+L2 或 L2-only 缓存策略。
* **`cache_modifier` 参数**：Triton 的部分版本支持 `tl.load` 的 `cache_modifier` 参数（如 `.cg` 绕过 L1），但 API 稳定性视版本而定。
* **`eviction_policy`**：控制缓存驱逐策略（`evict_first`、`evict_last`），影响 L2 缓存行为。适用于流式访问（只读一次的数据用 `evict_first`）。

---

### 16. 数据重排 / Data Reordering

Triton 不内建数据重排功能。不规则访问模式需要在 kernel 外部预处理：

* **Gather/Scatter**：Triton 支持 `tl.load(ptr + index_tensor)`，用于间接寻址。但 scattered load 的带宽利用率远低于合并访问。
* **排序/分桶**：不规则 workload（如稀疏矩阵、图计算）需要在 Triton 外部做数据重排，将密集子块交给 Triton kernel 处理。
* **结构化稀疏**：Triton 不内建 2:4 稀疏支持（不同于 CUTLASS）。需要手动实现压缩格式的加载逻辑。

---

### 17. Pinned Memory / 锁页内存

与 Triton 正交。Triton 不管理 host 端内存分配。推荐在 CPU-GPU 数据搬运频繁的场景使用 `torch.empty(..., pin_memory=True)`。

---

### 18. L2 缓存优化

* **L2 Persistence**：`cudaAccessPolicyWindow` 是 host 端 CUDA API，与 Triton 正交。可在 Triton kernel launch 前设置。
* **Triton 中的间接控制**：Block Shape 越大 → working set 越大 → L2 命中率可能下降。缩小 `BLOCK_K` 可以减小每次迭代的 working set。
* **`eviction_policy`**：`tl.load` 的 eviction policy 可以控制 L2 驱逐行为，间接影响 L2 利用率。

---

### 19–28. 其余优化

| 编号 | 优化项 | Triton 关系 |
|---|---|---|
| 19 | Prefetch | 编译器通过 `num_stages` 自动管理预取 |
| 20 | In-place | 输出 tensor 和输入 tensor 可指向同一地址，需确保无 tile 间写后读冲突 |
| 21 | Cooperative Groups | Hopper 上 `num_ctas` 参数替代跨 block 协作 |
| 22 | `__constant__` | Triton 的 `tl.constexpr` 参数在编译期固化为常量 |
| 23 | Stream-Ordered 分配 | 与 Triton 正交；可通过 PyTorch 的 CUDA caching allocator 间接实现 |
| 24 | Sector 化理解 | 影响编译器 swizzle 选择，程序员不直接控制 |
| 25 | Texture / Surface | Triton 不使用 Texture，走 `tl.load` / `tl.store` 路径 |
| 26 | Unified Memory | 性能场景不推荐，Triton 假设显式 GPU 分配 |
| 27 | Zero-Copy | 仅 data-starved 场景，与 Triton 正交 |

---

## 验证清单（NCU）

内存优化验证方法与原始文档一致，Triton 特有关注点：

1. **带宽利用率**：`Memory SOL %`、`DRAM Throughput`
2. **访问质量**：`Global Load/Store Efficiency`、`Sectors/Request` — **验证编译器生成的访存指令合并性**
3. **缓存行为**：`L1/L2 Hit Rate` — 验证 Block Shape 和编译器缓存策略选择
4. **Shared 路径健康度**：`Shared Memory Efficiency` — **验证编译器自动 swizzle 是否有效**；bank conflict 计数器应为 0
5. **向量化宽度**：`Sectors/Request` 应接近 1（表示完美合并的 128-bit 加载）
6. **整体收益**：以 kernel latency 为最终判据

**Triton 特有验证**：
- 编译器自动 swizzle 后 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` 应降至 0
- `DRAM Read Bytes` / 理论数据量 = 带宽放大系数，Kernel Fusion 后应接近 1.0
- `num_stages` 增加后应观察到 `Stall Long Scoreboard` 下降
- 使用 `tl.make_block_ptr` 时，检查 SASS 中是否出现 TMA 指令（Hopper）

统一决策树请参考：`rule-triton.md` 的"七、统一优化决策树（SSOT）"。
