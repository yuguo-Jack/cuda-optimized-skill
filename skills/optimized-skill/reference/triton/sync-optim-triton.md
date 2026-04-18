# Triton Kernel 同步优化完整方案：按重要性排序（从高到低）

> 本文档将原始 CUDA kernel 同步优化策略映射到 Triton 编译器框架层面。Triton 通过编译器驱动的 Pipeline 生成、自动 barrier 插入和 tile 级编程模型，将同步管理几乎完全从程序员手中转移到编译器。程序员从"手写 barrier"转变为"通过 Block Shape、`num_stages` 和代码结构引导编译器生成最优同步方案"。

---

### 1. Warp Shuffle 替代 Shared Memory 同步路径 — 编译器自动决策

Triton 编译器在多个层面自动选择 warp shuffle 替代 shared memory + barrier 路径：

* **`tl.dot` 内部**：Tensor Core MMA 的 warp 内数据交换完全由编译器处理，无需程序员编码。
* **归约操作**：`tl.sum`、`tl.max` 等操作中，编译器自动决定先做 warp 内 shuffle 归约，再做跨 warp 的 shared memory 归约。
* **`tl.reduce` 自定义归约**：编译器对自定义 combine 函数同样自动选择 shuffle 路径。
* **仍需显式处理的场景**：Triton 不暴露 `__shfl_sync` API。需要特殊 warp 级通信时，只能通过 inline PTX 或重构为 Triton 支持的 tensor 操作。

**排首位的理由不变**：这是同步的"消除"而非"减少"。Triton 编译器的贡献是让程序员完全不需要关心 warp 级数据交换的细节。

---

### 2. 减少 `__syncthreads()` 次数 — 编译器自动 Pipeline 生成

Triton 编译器根据 `num_stages` 参数自动生成 pipeline 同步代码，替代手写的 `__syncthreads()`：

* **传统模式（手写 CUDA）**：每个 tile 需要两次 `__syncthreads()`——加载后一次、计算后一次。K-loop 共 \(2 \times \lceil K / \text{BLOCK\_K} \rceil\) 次全 block 同步。
* **Triton Pipeline 模式**：编译器分析 K-loop 中 `tl.load` 和 `tl.dot` 的依赖关系，自动插入 `cp.async` commit/wait（Ampere）或 TMA barrier arrive/wait（Hopper）。同步粒度从"全 block 全线程"细化到"特定 stage 的数据就绪"。
* **程序员的角色**：程序员只需写简洁的 K-loop（`for k in range(0, K, BLOCK_K): load → dot`），编译器自动完成流水线化和同步插入。`num_stages` 是程序员唯一需要调节的同步相关参数。

**Triton 与 CUTLASS 的区别**：CUTLASS 需要显式选择 Pipeline 类型（`PipelineAsync` / `PipelineTmaAsync`）和管理状态机；Triton 编译器从循环结构自动推导 pipeline 方案。

---

### 3. Warp 级同步 vs Block 级同步 — 编译器自动管理

Triton 编译器根据操作类型自动选择同步粒度：

* **Warp 内操作**：`tl.dot`（MMA 指令）、warp 级 shuffle 归约——无需显式同步。
* **跨 warp 操作**：跨 warp 的 shared memory 归约——编译器自动插入最小范围的 barrier。
* **全 block 同步**：仅在必要时（如 shared memory 的 producer-consumer 切换）插入 `__syncthreads()`。编译器的 pipeline pass 尝试用 `cp.async.wait_group` 或 TMA barrier 替代全 block sync。

**Warp Specialization（Hopper 实验性）**：
* 编译器可自动将 CTA 内的 warp 分为 producer（TMA 加载）和 consumer（WGMMA 计算），通过硬件 barrier 精确协调。
* 启用条件：Hopper 后端 + `tl.make_block_ptr` + 编译器选项启用。
* 与 CUTLASS 的对比：CUTLASS 提供三种显式 WarpSpecialized 模式；Triton 由编译器自动决策，程序员控制力有限但编码成本为零。

---

### 4. Cluster 级同步（Hopper）— `num_ctas` 参数

Triton 通过 `num_ctas` 参数支持 Hopper 的 Thread Block Cluster：

* **`num_ctas`**：定义 Cluster 内 CTA 数量（1–16）。Cluster 内 CTA 共享 Distributed Shared Memory（DSMEM）。
* **Cluster 同步**：编译器自动生成 `cluster_arrive` + `cluster_wait` 指令，提供 cluster 级 barrier。
* **TMA Multicast**：Cluster 内多个 CTA 可以通过 TMA multicast 共享加载的数据，减少 DRAM 访问量。编译器在检测到 block 间数据复用时自动使用。
* **与 CUTLASS 的对比**：CUTLASS 显式指定 `ClusterShape_MNK`；Triton 通过 `num_ctas` 间接控制。

---

### 5. 异步数据搬运消除同步等待 — 编译器自动选择路径

Triton 编译器根据架构自动选择最优的异步搬运策略，消除显式同步等待：

* **Ampere（SM80）**：
  - 编译器自动生成 `cp.async` 指令链。
  - Global → Shared 搬运由硬件 DMA 完成，不占计算管线。
  - 同步从"等线程"（`__syncthreads()`）变为"等数据"（`cp.async.wait_group<N>`）。

* **Hopper（SM90）**：
  - 编译器自动生成 TMA 指令（需使用 `tl.make_block_ptr`）。
  - 搬运由 TMA 硬件引擎完成，不占任何线程资源。
  - 同步通过硬件 barrier 的 transaction-based 语义实现——等待特定字节数的数据到达，精确到字节级。
  - **TMA + Warp Specialization**：producer warp 提交 TMA 指令后立即释放，consumer warp 独立执行 WGMMA。

* **"等线程" vs "等数据"的进化**：
  - `__syncthreads()`：等所有线程到达同步点（等线程）——Triton 编译器尽量避免。
  - `cp.async.wait_group<N>`：等待至多 N 个 group 未完成（等数据）——Ampere 路径。
  - TMA barrier wait：等待指定字节数写入 Shared Memory（等数据，硬件级）——Hopper 路径。

**Triton 让程序员完全不接触同步指令。编译器自动选择架构最优的"等数据"语义。**

---

### 6. CUDA Graphs 减少调度开销

Triton kernel 支持 CUDA Graph capture：

* **PyTorch 集成**：在 `torch.cuda.graph()` 上下文中调用 Triton kernel，自动被 graph capture。
* **`torch.compile` 集成**：PyTorch 的 `torch.compile` + Inductor 后端会将多个 Triton kernel 组合进 CUDA Graph。
* **动态 shape 注意**：Graph capture 时 kernel 参数被固化。动态 shape 的 GEMM 需要 `cudaGraphExecUpdate` 或使用 padding + mask。
* **最大收益场景**：推理 pipeline 中多个小 kernel 串联，launch overhead 占比 10–30%。Graph 将整个序列预录并一次提交。

---

### 7. Kernel Fusion 消除 kernel 间隐式同步

从同步视角看 Kernel Fusion 的价值：

* **每次 kernel launch 之间存在隐式全局同步**——前一个 kernel 所有 block 完成才启动下一个。
* **Triton 的 Fusion 优势**：程序员可以直接在一个 `@triton.jit` kernel 中编写 GEMM + Bias + Activation + 其他后处理，消除所有中间 kernel 的隐式 barrier。
* **Flash Attention 范式**：将 GEMM + Softmax + GEMM 融合为单 kernel，消除了两次全 GPU 级隐式同步和中间矩阵的 Global Memory 读写。
* **`torch.compile` + Triton**：PyTorch Inductor 会自动将多个逐元素操作融合为一个 Triton kernel，减少 kernel launch 次数和隐式同步。

---

### 8. Stream 与 Event 管理

与 Triton 正交，通过 PyTorch 管理：

* **`torch.cuda.Stream`**：无依赖的 Triton kernel 放入不同 stream 并行执行。
* **`torch.cuda.Event`**：跨 stream 的 Triton kernel 通过 event 建立依赖。
* **Workspace 分配**：Split-K 等需要 workspace 的 Triton kernel，workspace 的分配可以通过 `torch.empty` 的 CUDA caching allocator 管理，避免全局同步。

---

### 9. Atomic 操作

Triton 中 atomic 操作主要出现在以下场景：

* **Split-K GEMM**：多个 program 处理同一 output tile 的不同 K 切片，通过 `tl.atomic_add` 归约。这是最常见的 Triton atomic 使用场景。
* **Histogram / Scatter-Add**：不规则输出模式下的原子累加。
* **支持的 atomic 操作**：`tl.atomic_add`、`tl.atomic_max`、`tl.atomic_min`、`tl.atomic_and`、`tl.atomic_or`、`tl.atomic_xor`、`tl.atomic_cas`（compare-and-swap）。
* **高竞争缓解**：
  - 先在 tile 内做局部归约（`tl.sum`），再用一次 `tl.atomic_add` 写出——减少 atomic 次数。
  - 调整 grid 划分使竞争分散到不同地址。

---

### 10. `tl.debug_barrier()` — 显式同步（调试用）

Triton 提供 `tl.debug_barrier()` 作为显式 `__syncthreads()` 的等价物：

* **用途**：主要用于调试和正确性验证。在生产 kernel 中不应该需要。
* **性能影响**：每次调用都是全 block 同步，会打断编译器的 pipeline 优化。
* **正确的做法**：依赖编译器自动插入的 barrier。如果编译器生成的 barrier 不足（race condition），通常是代码结构问题——应重构代码而非手动插入 barrier。

---

## 验证清单（NCU）

| NCU 参数 | 含义 | Triton 特有关注点 |
|---|---|---|
| `smsp__warps_issue_stalled_barrier` | `__syncthreads()` 等待周期 | 编译器 pipeline 应大幅降低此指标；如果仍然很高，检查 `num_stages` 和代码结构 |
| `smsp__warps_issue_stalled_membar` | memory fence 等待周期 | Hopper TMA barrier 应比 cp.async fence 更高效 |
| `smsp__warps_issue_stalled_not_selected` | warp 就绪但未调度 | `num_warps` 过大可能导致 warp 扎堆 |
| `smsp__warps_issue_stalled_sleeping` | barrier arrive_and_wait | `num_stages` 过多或 pipeline 设计问题 |
| `smsp__warps_issue_stalled_long_scoreboard` | 等待 L2/DRAM | `num_stages` 不足，增加 stage 数以隐藏延迟 |
| `gpu__time_duration.sum` | kernel 总耗时 | 最终判据 |

**Triton 特有验证**：
1. **Pipeline overlap 质量**：Nsight Systems 中应观察到 cp.async/TMA 搬运与 MMA 计算的时间重叠
2. **编译器 barrier 插入质量**：dump TTGIR 检查 barrier 位置是否合理——编译器应在 `tl.load` 的 commit 和 `tl.dot` 的 wait 之间插入 pipeline barrier
3. **TMA 使用验证（Hopper）**：SASS 中确认出现 TMA 指令（`UTMALDG`），说明编译器成功使用了异步搬运
4. **正确性**：修改 `num_stages` 或切换 Block Shape 后，必须验证数值正确性（尤其是边界 tile）

**常见误判**：
- `num_stages` 增加了但 `Stall Barrier` 未下降——可能 stage 数已超过延迟隐藏需求，额外 stage 只增加了 Shared Memory 压力
- `num_warps` 过大导致 `Stall Not Selected` 升高——warp 太多但每 warp 工作量不足
- 只看同步 stall 下降，不看 kernel 总耗时——可能 Shared Memory 增加导致 occupancy 降低，抵消了同步优化的收益
- 手动插入 `tl.debug_barrier()` 导致编译器 pipeline 被破坏——移除 debug barrier 后性能可能显著提升

统一决策树请参考：`rule-triton.md` 的"七、统一优化决策树（SSOT）"。
