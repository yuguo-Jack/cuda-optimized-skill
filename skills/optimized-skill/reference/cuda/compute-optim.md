# CUDA Kernel compute 优化的完整方案 按重要性重新排序（从高到低）

---

### 1. Tensor Core / 专用硬件单元利用

从 Volta 架构开始，Tensor Core 在一个时钟周期内完成小矩阵乘加运算（如 (16\times16\times16) 的 FP16 MMA），计算吞吐比普通 CUDA Core 高一个数量级。通过 WMMA API、MMA PTX 指令。**不用 Tensor Core 做矩阵类运算基本等于浪费了一半以上的芯片算力。**

### 2. 计算与访存重叠

调度器的核心能力：一个 warp 等待内存返回时，切换到另一个就绪 warp 执行计算指令。提升重叠的手段包括增加 occupancy、增加每线程独立指令数（ILP）、软件流水线。**这是 GPU 编程模型的根基——延迟隐藏**，做不好则计算单元大量空转。

### 3. Launch Configuration 调优

Block size 直接影响 occupancy 和硬件利用率。通常选 128/256/512，最优值取决于 kernel 资源消耗。`cudaOccupancyMaxPotentialBlockSize` API 可辅助决策。**这是每个 kernel 必须做的第一个决策**，选错可能导致 SM 利用率腰斩。

### 4. 归约优化

从朴素交错寻址到 sequential addressing、warp unrolling、warp shuffle 归约，每步优化都有明显收益。最终形态：warp 内用 shuffle，warp 间用 shared memory，block 间用 atomic 或二次 kernel。**归约是最普遍的并行原语之一**，优化与否可导致同一算法数倍性能差异。


### 5. Warp Shuffle 用于计算

`__shfl_sync`、`__shfl_xor_sync`、`__shfl_down_sync` 不仅是数据交换工具，更是计算原语。Warp 内 reduction、scan、broadcast 都可在寄存器层级完成，比 shared memory 快且无 bank conflict。**几乎所有现代 kernel 的 warp 级操作都应以 shuffle 为默认选择。**

### 6. 循环展开 / Loop Unrolling

`#pragma unroll` 减少循环控制指令（比较、跳转），更重要的是暴露更多独立指令给调度器，提升 ILP。对迭代次数已知的小循环效果显著。**代价几乎为零，收益稳定。**

### 7. FMA 显式使用

`a * b + c` 应被编译为一条 FMA 指令，但编译器有时不自动合并。用 `__fmaf_rn()` 或 `fmaf()` 显式调用——一条指令完成、精度更高（中间结果不截断）。**FMA 是 GPU ALU 的原生运算**，未合并意味着指令数翻倍。

### 8. 用乘法替代除法

浮点除法吞吐远低于乘法。`a / b` → `a * __frcp_rn(b)` 或 `a * (1.0f / b)`，`__fdividef(a, b)` 也是快速路径。**除法/乘法吞吐差距在许多架构上达 4×–16×**，热循环中的每一处除法都值得审视。

### 9. `--use_fast_math` 与编译选项调优

开启全部快速数学或按需选择 `--ftz=true`、`--prec-div=false`、`--prec-sqrt=false`。**一个编译开关就可能带来 10%–30% 的整体加速**，前提是精度容忍度允许。

### 10. `__restrict__` 关键字

告诉编译器指针无 alias，编译器可更激进做指令重排和寄存器缓存。**零改动代价、效果有时非常显著**，应作为所有 kernel 指针参数的默认修饰。

### 11. 软件流水 / Software Pipelining

循环体分"加载→计算→存储"三阶段，不同迭代的不同阶段交错执行，最大化功能单元利用率。**对访存密集型循环可接近理论峰值**，但实现复杂度较高。

### 12. Warp Specialization

Block 内 warp 分为"搬运 warp"和"计算 warp"，通过 shared memory 协同。手动 producer-consumer 模型，在矩阵乘法等场景中可获比传统方法更好的资源利用率。**CUTLASS 3.x 的核心架构思路**，但通用性有限。

### 13. Scan/Prefix Sum 优化

Blelloch（work-efficient）和 Hillis-Steele（step-efficient）各有适用场景。大规模 scan 分 block 内 scan、block 间辅助数组 scan、回填三阶段。**scan 是许多并行算法的基本构件**（stream compaction、排序、直方图），优化与否差距明显。

### 14. 按 warp 重组数据消除分支发散

预排序或分组让同类数据聚集在同一 warp。如粒子模拟中活跃/非活跃粒子分开排列，非活跃 warp 整体跳过。**本质是把控制流问题转化为数据布局问题**，对不规则 workload 效果极佳。

### 15. 强度削减 / Strength Reduction

用更低代价的等价运算替换：移位替代 2 的幂乘除、累加替代循环内乘法（`i*stride` → 累加 `stride`）、`rsqrtf()` 替代 `1.0f/sqrtf()`（rsqrt 是硬件原生单条指令）。**收益逐条看不大，但热循环中积少成多。**

### 16. 查表法 / LUT

复杂函数在输入范围有限时预计算结果放 shared memory 或 constant memory 做查表。一次读取替代大量计算。**适合离散映射或分段函数**，但需权衡表的访存开销与计算开销。

### 17. 整数运算与位操作优化

整数除法/取模对常量编译器自动优化，对变量应手动用位运算替代。`__popc()`、`__clz()`、`__ffs()` 直接映射硬件指令。**对整数密集型 kernel（哈希、编解码、图算法）收益显著**，其他场景影响有限。


### 18. Select 指令替代分支

`condition ? a : b` 通常编译为无分支 select 指令。手动把 if-else 改写为三目运算符或算术表达式可帮助编译器生成无分支代码。**对短分支有效**，长分支体则谓词化反而浪费。

### 19. 谓词化 / Predication

很短的分支（一两条指令）编译器自动转为谓词指令——两条路径都执行，仅满足条件的写回。避免 warp divergence 但浪费计算。**编译器通常自动处理**，手动干预的空间有限。

### 20. Early Exit

归约或搜索类 kernel 中，`__all_sync()` 检测 warp 内所有线程满足终止条件后直接 return。**节省的是尾部无效计算**，收益取决于数据分布。

### 21. Warp Vote 函数

`__ballot_sync()`、`__all_sync()`、`__any_sync()` 一条指令收集 warp 条件判断结果，实现 warp 级 early exit 或条件聚合。**单独使用收益有限，但作为组合工具极有价值。**

### 22. 循环合并 / Loop Fusion

多个遍历同一数据的循环合并为一个，数据在寄存器中多次复用。**与 Kernel Fusion 是同一思想在循环层面的体现**，但通常合并的收益已在 Kernel Fusion 阶段兑现。

### 23. 循环分裂 / Loop Fission

含多种不相关计算的循环拆成多个，降低寄存器压力、利于编译器优化。**与 Loop Fusion 方向相反**，需根据寄存器压力 vs 访存复用权衡。

### 24. 循环交换 / Loop Interchange

调整嵌套循环顺序使最内层连续访存。**本质是改善合并访问**，更多属于内存优化范畴，对纯计算的影响间接。

### 25. 内联控制

`__forceinline__` 消除函数调用开销；`__noinline__` 控制代码膨胀以缓解指令缓存压力。**通常编译器自动决策已足够好**，仅在 profile 发现调用开销或 I-cache miss 后手动介入。


### 26. PTX/SASS 分析

`cuobjdump --dump-sass` 确认关键路径使用了 FMA、LDG 等高效指令，排查意外类型转换或寄存器溢出。Nsight Compute 做指令级分析。

* **补充**：异步搬运/新架构优化时应核查 PTX 层是否出现预期的异步/矩阵指令路径。

### 27. Warp Match 函数

`__match_any_sync()` / `__match_all_sync()` 识别 warp 内持有相同值的线程，用于分组和去重。**适用面窄**（哈希聚合、去重场景），但在需要时可替代 shared memory 的 scatter-gather。

### 28. SFU 使用意识

`__sinf`、`__cosf`、`__expf` 等走 SFU 流水线，与 ALU 并行但吞吐更低。**不是主动优化项，而是避坑意识**——大量超越函数调用时需关注 SFU 是否成为瓶颈。


---

## 九、验证清单（NCU）

为避免"只改代码不验指标"，计算优化建议至少配套以下检查：

- **Tensor Core 路径**：确认是否出现预期 MMA/WGMMA 指令路径，且相关计算管线利用率提升。
- **指令效率**：关注 `Issue Slot Utilization`、`Eligible Warps Per Cycle` 是否改善。
- **分支质量**：关注 `Warp Execution Efficiency` 与分支相关 stall 是否改善。
- **寄存器与溢出**：用 `--ptxas-options=-v` + NCU 检查 spill 是否下降。

**常见误判**：

- 只看 occupancy 升高，不看 kernel latency，可能出现"occupancy 上去但性能变差"。
- 只看某个单项指标改善，忽略了同步或访存路径退化导致总体变慢。

统一决策树请参考：`skills/optimized-skill/SKILL.md` 的"七、统一优化决策树（SSOT）"。

