CUDA kernel memory优化的完整方案：

# CUDA Kernel 内存优化：按重要性重新排序（从高到低）

---

### 1. Kernel Fusion

把相邻的 producer-consumer kernel 合并为一个，中间结果保留在寄存器或 Shared Memory 中，消除一次完整的 Global Memory 往返。**这往往是最高收益的单项优化。**

### 2. 合并访问 / Coalesced Access

同一 warp 的 32 个线程访问连续、对齐的地址，硬件合并为最少的内存事务。理想情况下一次 128B 事务服务整个 warp。反例是 stride 访问或随机访问，会导致事务数暴增（最坏 32 倍带宽浪费）。

### 3. SoA vs AoS

* **AoS（Array of Structures）**：`struct { float x, y, z; } particles[N]`——warp 访问同一字段时地址不连续，无法合并。
* **SoA（Structure of Arrays）**：`float x[N], y[N], z[N]`——天然合并。
* **CUDA 中几乎总是优选 SoA**，差距可达数倍。本质上是为合并访问服务的数据布局决策。

### 4. Tiling / 分块

将全局内存中的数据按 tile 搬到 Shared Memory，在片上多次复用。矩阵乘法经典优化——每个 tile 只从全局内存读一次，在 Shared Memory 中被复用 (O(N)) 次。对任何**数据复用率 > 1** 的算法都是关键手段。

### 5. 寄存器压力控制

* 寄存器是最快存储（零延迟），但总量有限（如每 SM 65536 个 32-bit 寄存器）。
* 用量过高 → occupancy 降低 → 延迟隐藏能力下降。
* 过度 spill 到 Local Memory 会退化为全局内存访问，极其昂贵。
* 用 `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` 引导编译器。

### 6. 双缓冲 / 多级流水线

在 Shared Memory 或寄存器中分配两组 buffer，一组计算、一组加载，交替推进。所有高性能 GEMM 实现（cuBLAS、CUTLASS）的核心技术。

### 7. 向量化访存 / Vectorized Load/Store

使用 `float4`、`int4` 等宽类型，单条指令读写 128-bit，减少指令数、提升每事务有效字节数。需保证地址对齐到向量宽度。

### 8. 异步拷贝 / Async Copy

* `cp.async`（CUDA 11+）：Global → Shared Memory 由硬件 DMA 完成，不占寄存器和计算单元，可与计算完全重叠。

  * **补充（架构限定）**：收益与 SM 架构强相关，建议按目标架构验证。
* 多级流水线配合异步拷贝是现代 kernel 隐藏访存延迟的标准做法。
* **补充（Hopper/Blackwell）**：新架构可结合 TMA（Tensor Memory Accelerator）进一步降低复杂搬运的指令与寄存器开销。

### 9. Bank Conflict 消除

Shared Memory 32 bank，同 warp 多线程命中同 bank 不同地址会串行化。

* **补充（架构限定）**：bank 宽度在不同架构/配置下存在差异，应结合目标 GPU 文档与 profile 结果判断。
* **Padding**：`__shared__ float s[32][33]`。
* **Swizzle/XOR 索引**：更高级且不浪费空间。

### 10. Warp Shuffle

`__shfl_sync` 系列——warp 内线程直接交换寄存器值，约 1 个时钟周期，比 Shared Memory 更快且无 bank conflict。适用于 reduction、scan、broadcast。

### 11. 寄存器级数据复用

每个线程持有多个数据元素（增加 ILP），在寄存器层面完成尽可能多的计算后再写回，减少对 Shared/Global Memory 的依赖。


收益取决于访问模式、数据规模或硬件代际，并非所有 kernel 都需要。

### 12. 对齐访问 / Aligned Access

起始地址对齐到 128B（或 32B sector）边界。未对齐会浪费带宽。实践中 `cudaMalloc` 返回的地址天然 256B 对齐，因此主要关注手动偏移和子数组场景。

### 13. Padding 与对齐

`cudaMallocPitch` / `cudaMalloc3D` 保证每行起始地址对齐。牺牲少量空间换取带宽利用率大幅提升，对二维/三维数组尤为重要。

### 14. CUDA Streams 重叠

不同 stream 间的 kernel 执行、H2D/D2H 拷贝可并行。把大数据分 chunk 做"拷贝-计算-回拷"流水线。属于**系统级**而非单 kernel 级优化。

### 15. Shared Memory 容量配置

`cudaFuncSetAttribute` 调大 Shared Memory 比例（如从 48KB 提升到 100KB+），适合 Shared Memory 需求大的 kernel。

* **补充（架构限定）**：可配置上限因架构而异（Ampere 最大 164KB），按目标 GPU 查文档。

### 16. 只读数据路径

* `__ldg()`：走只读数据缓存，对不规则访问模式可能有更好的缓存行为。

  * **补充（架构限定）**：新架构下编译器常自动优化，`__ldg()` 不再是"无脑必开"，以 Nsight Compute 数据为准。
* `const __restrict__`：让编译器自动选择 `__ldg()` 路径。

### 17. 数据重排 / Data Reordering

对不规则访问模式（稀疏矩阵、图计算），预处理阶段做重排：空间填充曲线（Z-order/Hilbert）、CSR 按行长分桶等。预处理有开销，需全局收益覆盖。

### 18. Pinned Memory / 锁页内存

`cudaHostAlloc` / `cudaMallocHost` 分配锁页内存，H2D/D2H 传输带宽可达 pageable 的 2 倍，且支持异步传输。对数据搬运频繁的流水线是必要条件。

### 19. L2 缓存优化

* **L2 Persistence**：`cudaAccessPolicyWindow` 将热点数据钉在 L2，防止被冷数据驱逐。
* 适合数据量小但反复访问的场景（lookup table、embedding）。

### 20. Prefetch

* Global Memory 上手动或 `__builtin_prefetch` 提前加载下一迭代数据。
* Unified Memory 下用 `cudaMemPrefetchAsync` 避免按需缺页延迟。

### 21. In-place 操作

尽可能原地修改数据，避免额外 output buffer，减少内存占用与访问量。属于良好编程习惯，通常收益温和。

### 22. 协作组 / Cooperative Groups

灵活定义同步组，精确控制同步粒度，避免不必要的 `__syncthreads()`。对 reduction 和跨 block 协作有价值，但大多数 kernel 中影响有限。

### 23. `__constant__` 内存

64KB 专用缓存，warp 内全线程读同一地址时效率最高（广播）。适合卷积核权重、小型查找表。如果线程读不同地址则串行化，反而更差。现代 kernel 中使用频率下降，很多场景被 `__ldg()` + L1 替代。

### 24. Stream-Ordered 分配与内存池

* `cudaMallocAsync` / `cudaFreeAsync`：按 stream 顺序分配释放，减少全局同步干扰。
* `cudaMemPool*`：内存池复用降低频繁分配开销。
* 对动态 shape、分段流水场景有意义，但对单 kernel 性能影响间接。

### 25. Sector 化理解

从 Volta 起 L1 以 32B sector 粒度工作。概念性认知，指导其他优化决策，本身不是独立可操作项。

### 26. Texture / Surface 对象

硬件针对 2D 空间局部性优化（Morton/Z-order 布局），自动边界处理和硬件插值。仅在图像处理、体素采样等 2D 访问模式下有真正优势，一维场景被 `__ldg()` 替代。

### 27. Unified Memory 优化

* `cudaMemPrefetchAsync` 提前迁移。
* `cudaMemAdvise` 提供访问模式提示。
* 主要服务于编程便利性，性能天花板受页迁移机制限制，高性能场景通常直接用显式分配。

### 28. 零拷贝内存 / Zero-Copy

GPU 通过 PCIe 直接访问主机内存。PCIe 带宽（~32 GB/s）远低于显存（~900+ GB/s），仅适合数据量极小或一次性访问的场景。

---

## 十、验证清单（NCU）

内存优化建议至少配套以下验证：

1. **带宽利用率**：关注 `Memory SOL %`、`DRAM Throughput`，确认是否接近预期上界。
2. **访问质量**：关注 `Global Load/Store Efficiency`、`Sectors/Request`，确认 coalescing/对齐是否改善。
3. **缓存行为**：关注 `L1 Hit Rate`、`L2 Hit Rate`，确认优化方向与局部性变化一致。
4. **Shared 路径健康度**：关注 `Shared Memory Efficiency`，确认 bank conflict 是否下降。
5. **整体收益**：最终以 kernel latency（avg/median）判断，不只看单个子指标。

常见误判：
- 局部指标改善但总时延上升（通常是把瓶颈转移到别处）。
- 只优化吞吐，不检查 correctness 与数值稳定性。

统一决策树请参考：`skills/optimized-skill/SKILL.md` 的“七、统一优化决策树（SSOT）”。
