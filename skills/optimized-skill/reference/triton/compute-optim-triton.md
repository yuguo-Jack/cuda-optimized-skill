# Triton Kernel 计算优化完整方案：按重要性排序（从高到低）

> 本文档将原始 CUDA kernel 计算优化策略映射到 Triton 编译器框架层面。Triton 的核心计算抽象是 `tl.dot`——编译器根据输入精度和目标架构自动选择硬件指令（Tensor Core MMA 或 CUDA Core FMA）。计算优化在 Triton 中主要体现为"选择正确的精度、Block Shape 和编译器 hint 组合"。

---

### 1. Tensor Core / 专用硬件单元利用 — `tl.dot` 精度选择

Triton 通过 `tl.dot` 的操作数精度和 `acc` 累加器类型自动选择走 Tensor Core 还是 CUDA Core，**这是最关键的单一配置决策**：

* **FP16 / BF16 输入 + FP32 累加**：`tl.dot(a, b, acc)` 中 `a`/`b` 为 `float16` 或 `bfloat16`，`acc` 为 `float32`。编译器自动生成 `mma.sync`（Ampere）或 `wgmma`（Hopper）指令。**最常用路径。**
* **FP8 输入（Hopper）**：`tl.dot` 支持 `float8e4nv` / `float8e5` 类型，编译器生成 FP8 WGMMA 指令，理论吞吐是 FP16 的 2 倍。
* **TF32 路径**：FP32 输入时，Triton 默认启用 TF32（`allow_tf32=True`），走 Tensor Core 的 TF32 管线；显式设置 `allow_tf32=False` 则回退到 CUDA Core FMA。
* **整数路径**：INT8 输入 + INT32 累加，编译器生成整数 MMA 指令。

* **CUDA Core 回退**：当 `tl.dot` 的输入为 FP32 且 `allow_tf32=False` 时，走 FMA 管线。此外，所有非矩阵乘加的逐元素运算（`tl.exp`、`tl.log`、加减乘除等）走 CUDA Core / SFU。

**选择原则**：只要运算可以表达为矩阵乘加（GEMM、卷积、Attention），就应该使用低精度输入（FP16/BF16/FP8）触发 Tensor Core 路径。FP32 输入 + TF32 是精度折中方案。不用 Tensor Core 等于浪费了一半以上的芯片算力。

**Triton 与 CUTLASS 的关键区别**：CUTLASS 需要显式选择 MMA Atom 类型；Triton 由编译器根据数据类型和目标架构自动推导指令路径。程序员的责任是**确保数据类型正确**——错误的类型（如意外提升为 FP32）会静默走 FMA 路径，性能断崖式下降。

---

### 2. 计算与访存重叠 — `num_stages` 流水线深度

Triton 编译器通过 `num_stages` 参数控制软件流水线级数，实现计算-访存重叠：

* **原理**：编译器自动将 K 维度的 mainloop 拆分为多级流水线。stage K 的数据在被 `tl.dot` 消费的同时，stage K+1 的数据正从 Global Memory 异步加载到 Shared Memory。
* **影响重叠质量的因素**：
  - `num_stages`：越多越好隐藏访存延迟，但受 Shared Memory 容量限制。
  - Block Shape 的 K 维度（`BLOCK_K`）：决定每个 stage 搬运的数据量。
  - Occupancy：同一 SM 上的多个 program 可以在一个 program 等待数据时切换到另一个执行计算。

* **Hopper 上的增强**：Triton 后端在 Hopper 上自动使用 TMA 指令进行异步搬运，配合硬件 barrier 实现更高效的流水线重叠。编译器还可自动生成 Warp Specialization 代码（producer/consumer 分离）。

* **`num_stages` 调优**：通常在 `tl.constexpr` 或 `triton.autotune` 的 config 中指定。Ampere 上 3–4 通常最优；Hopper 上 2–4 即可（TMA 硬件效率更高）。

---

### 3. Launch Configuration 调优 — Block Shape 体系

Triton 中 launch configuration 通过 Block Shape 常量和 `num_warps` 参数体现：

* **Block Shape（`BLOCK_M`、`BLOCK_N`、`BLOCK_K`）**：决定每个 program 处理的分块大小。直接影响 Shared Memory 用量、寄存器压力、Tensor Core 利用率。这是 Triton 调优的中心维度。
* **`num_warps`**：每个 program（thread block）的 warp 数。影响线程级并行度和寄存器/Shared Memory 的分摊方式。
* **`num_ctas`（Hopper Cluster）**：Cluster 内 CTA 数，影响 TMA multicast 效率和 DSMEM 通信模式。

**Block Shape 选择的关键权衡**：
- `BLOCK_M × BLOCK_N` 越大 → 每个 program 的输出越多 → 数据复用率越高 → 寄存器和 Shared Memory 用量越大 → Occupancy 越低
- `BLOCK_K` 越大 → 每次迭代处理更多 K 元素 → 减少 mainloop 循环开销 → 但增加每 stage 的 Shared Memory 用量
- `num_warps` 越大 → 每 warp 分摊的计算越少 → 可能 Tensor Core 利用率不足
- **没有理论最优解**，必须对候选组合做 benchmark

**常用 Block Shape 起点**：
- Ampere：`BLOCK_M/N` = 128, `BLOCK_K` = 32 或 64, `num_warps` = 4 或 8
- Hopper：`BLOCK_M/N` = 128 或 256, `BLOCK_K` = 64, `num_warps` = 4 或 8

**`triton.autotune` 的系统化用法**：列出 4–8 个候选 config（Block Shape × `num_warps` × `num_stages` 组合），编译器自动选择在给定 problem size 下最快的配置。

---

### 4. 归约优化 — Split-K 与 Atomic Reduction

Triton 中实现归约的策略：

* **Data-Parallel（默认）**：每个 program 独立处理整个 K 维度，无跨 program 归约。适合 K 较小或 M×N 足够大的场景。
* **Split-K**：将 K 维度切分到多个 program，每个 program 处理一个 K 切片。最后通过以下方式归约：
  - **独立 reduction kernel**：第一个 kernel 将部分和写入 workspace，第二个 kernel 归约。需要额外显存。
  - **Atomic reduction**：每个 program 在 epilogue 中用 `tl.atomic_add` 归约到输出 tensor。无额外 kernel 但有竞争。
* **Grid 维度编码 Split-K**：通过增加 `grid` 的第三维度表示 K 切片数，在 kernel 内用 `tl.program_id(2)` 区分切片。

**归约层次**（从内到外）：
1. `tl.dot` 累加器内部：warp 内 K 维度累加（硬件级）
2. K-loop 内的累加：跨 `BLOCK_K` tile 的累加（寄存器级）
3. Epilogue 内的 `tl.store`：warp 级输出（寄存器 → Global Memory）
4. 跨 program：Split-K 的全局归约（atomic 或 reduction kernel）

---

### 5. Warp 级操作

Triton 将 warp 级操作抽象为高层 API，程序员无需直接使用 `__shfl_sync`：

* **`tl.dot` 内部**：Tensor Core MMA 的 warp 内数据交换完全由编译器处理。
* **归约操作**：`tl.sum`、`tl.max`、`tl.min` 等沿指定轴的归约，编译器自动生成 warp shuffle + 跨 warp shared memory 归约。
* **自定义 warp 级操作**：Triton 不直接暴露 `__shfl_sync`。需要 warp 级通信时，通常通过 tensor 操作间接实现（如 `tl.reshape` + `tl.sum`），或使用 Triton 的 inline PTX 机制。
* **`tl.reduce`**：通用归约原语，支持自定义 combine 函数。编译器自动选择 warp shuffle 或 shared memory 路径。

---

### 6. 循环展开 / Loop Unrolling

Triton 的循环展开策略：

* **K-loop**：当 `BLOCK_K` 是编译期常量（`tl.constexpr`）时，编译器自动展开内层循环中的 `tl.dot` 和 `tl.load` 调用。
* **`tl.static_range`**：显式标注循环为编译期可展开。等价于 CUTLASS 的 `CUTLASS_PRAGMA_UNROLL`。
* **编译期常量**：Triton 的 `tl.constexpr` 参数在编译期固化为立即数，消除运行时分支和地址计算——等价于 CUTLASS 的 `cute::Int<N>` 静态整数。
* **展开的代价**：过度展开导致指令缓存压力（I-cache miss）。`BLOCK_K` 越大、展开的 `tl.dot` 调用越多、编译时间越长。

---

### 7. FMA 与指令路径控制

Triton 中指令路径由数据类型和编译器自动决定：

* **`tl.dot` + FP16/BF16 输入** → 编译为 `mma.sync` / `wgmma` PTX 指令
* **`tl.dot` + FP32 输入 + `allow_tf32=True`** → 编译为 TF32 MMA 指令
* **`tl.dot` + FP32 输入 + `allow_tf32=False`** → 编译为 FMA 指令
* **逐元素运算**（`+`、`*`、`tl.where` 等）→ 编译器根据精度生成 FMA 或独立指令

**Triton 的自动 FMA 融合**：编译器的后端 pass 会自动将 `a * b + c` 模式融合为 FMA 指令，无需手动调用 intrinsic。

---

### 8. 除法与特殊函数优化

* **整数除法**：Triton 编译器自动将编译期常量除数的整数除法转换为乘法 + 移位（等价于 CUTLASS 的 `FastDivmod`）。
* **浮点除法**：编译器在 `--allow-fp-reorder` 等选项下可能用倒数近似替代除法。
* **特殊函数**：`tl.exp`、`tl.log`、`tl.sqrt` 等走 SFU（Special Function Unit）管线。大量使用时注意 SFU 吞吐瓶颈。
* **`tl.math.fast_expf`** 等快速近似函数：牺牲精度换取 SFU 吞吐。

---

### 9. 编译选项调优

Triton 编译器的关键选项（通过环境变量或 `triton.compile` 参数控制）：

* **`TRITON_PRINT_AUTOTUNING`**：输出 autotune 选择的 config 和耗时。
* **`allow_tf32`**：控制 FP32 输入是否走 TF32 Tensor Core（默认 True）。
* **`MLIR_ENABLE_DUMP`**：输出中间 IR（Triton IR → TTGIR → LLVM IR → PTX），用于验证指令路径。
* **`num_warps`、`num_stages`**：编译期确定的 launch 参数，直接影响生成代码质量。
* **`TRITON_CUDA_ENABLE_WARP_SPECIALIZATION`**：Hopper 上启用 Warp Specialization（实验性）。

**验证指令路径**：通过 dump PTX/SASS 确认出现 `mma.sync` / `wgmma` 指令。如果只有 `fma` 指令，说明精度配置错误。

---

### 10. 掩码与谓词化

Triton 通过 `mask` 参数实现谓词化访存和计算：

* **`tl.load(ptr, mask=mask, other=0.0)`**：对越界地址填充默认值，编译器生成谓词化加载指令，避免分支发散。
* **`tl.store(ptr, val, mask=mask)`**：仅写入有效地址。
* **`tl.where(cond, x, y)`**：编译器生成 select 指令（`selp` PTX），无分支。
* **与 CUTLASS 的对比**：CUTLASS 需要选择 `Predicated` vs `Unpredicated` Mainloop；Triton 的 `mask` 是统一接口，编译器自动判断是否需要谓词。

---

### 11. `tl.dot` 的 `input_precision` 控制

Triton 3.x 引入 `input_precision` 参数，提供更细粒度的精度控制：

* **`"tf32"`**：FP32 输入走 TF32 Tensor Core（默认）。
* **`"tf32x3"`**：三次 TF32 MMA 累加，精度接近 FP32，吞吐约为 TF32 的 1/3。
* **`"ieee"`**：严格 IEEE FP32，走 CUDA Core FMA。
* **用途**：在不改变输入数据类型的前提下，控制精度-性能权衡。

---

### 12. Warp Specialization — Hopper 实验性支持

Triton 在 Hopper 后端实验性支持 Warp Specialization：

* **自动模式**：编译器分析 kernel 的加载-计算模式，自动决定是否将 warp 分为 producer（TMA 加载）和 consumer（WGMMA 计算）。
* **手动 hint**：通过特定编译选项或 kernel 结构暗示编译器启用。
* **与 CUTLASS 的对比**：CUTLASS 提供 `TmaWarpSpecialized` / `Cooperative` / `Pingpong` 三种显式模式；Triton 由编译器自动选择，程序员控制力有限。

---

### 13. Scan / Prefix Sum

Triton 提供 `tl.associative_scan` 原语：

* **用法**：沿指定轴执行 inclusive scan，支持自定义 combine 函数。
* **编译器实现**：自动生成 warp shuffle + shared memory 的多级 scan。
* **限制**：scan 范围限于单个 program 内（block 级），跨 block scan 需要多 kernel 方案。
* **典型应用**：Flash Attention 中的 online softmax 归约、cumsum 等。

---

### 14. 分支发散控制

Triton 的 tile 编程模型天然减少分支发散：

* **Tile 级操作**：所有操作以 tile（2D block of elements）为粒度，同一 tile 内的元素执行相同操作路径。
* **`mask` 替代 `if-else`**：`tl.where(cond, x, y)` 生成 select 指令而非分支。`tl.load` 的 `mask` 参数生成谓词化加载。
* **编译器优化**：Triton 编译器的 TTGIR pass 会尝试将 block 内的条件操作转换为谓词化指令。
* **不规则 workload**：与 CUTLASS 类似，稀疏/不规则数据仍需外部预处理。

---

### 15–20. 其余计算优化

| 编号 | 优化项 | Triton 关系 |
|---|---|---|
| 15 | 强度削减 | 编译器自动执行常见强度削减（乘法替代除法、移位替代乘除 2 的幂等） |
| 16 | 查表法 LUT | 可在 Shared Memory 中手动实现；Triton 不提供高层 LUT API |
| 17 | 整数位操作 | 直接在 Triton 中使用 `&`、`\|`、`^`、`<<`、`>>`；编译器映射到硬件指令 |
| 18 | Select 替代短分支 | `tl.where` 自动生成 select 指令 |
| 19 | Early Exit | 可在 K-loop 中用 `if` 条件跳出，但会破坏编译器的流水线优化 |
| 20 | PTX/SASS 分析 | **最重要的验证手段**——通过 `MLIR_ENABLE_DUMP` 或 Nsight Compute 确认 `mma.sync`/`wgmma` 指令路径 |

---

## 验证清单（NCU）

计算优化验证方法与原始 CUDA 文档一致，Triton 特有关注点：

- **Tensor Core 路径**：SASS 中确认 `HMMA`（Ampere）或 `WGMMA`（Hopper）指令。`sm__pipe_tensor_cycles_active` 应 > 0 且占比显著。如果此指标为 0，说明 `tl.dot` 的输入精度配置错误（如输入被意外提升为 FP32 且 `allow_tf32=False`）。
- **MMA 利用率**：`sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` — Block Shape 与 Tensor Core 管线的匹配度。利用率低说明 `BLOCK_M/N` 太小或 `num_warps` 过多导致每 warp 的计算量不足。
- **指令效率**：`Issue Slot Utilization`、`Eligible Warps/Cycle` — Triton 的编译期索引计算应减少 ALU 指令。如果 ALU 占比异常高，检查是否有运行时动态 shape 计算。
- **寄存器与溢出**：NCU 的 register 统计 — 累加器大小由 `BLOCK_M × BLOCK_N / num_warps` 决定。寄存器溢出（spill）时应缩小 Block Shape 或减少 `num_stages`。
- **编译器生成质量**：dump Triton IR → TTGIR → PTX 逐级检查，确认编译器未引入冗余指令。

**常见误判**：
- `tl.dot` 输入被 `tl.load(...).to(tl.float32)` 意外提升精度，Tensor Core 完全没用上——检查 SASS 中是否有 `HMMA`/`WGMMA`
- `BLOCK_M/N` 过大导致 occupancy = 1 block/SM——occupancy = 1 几乎一定更慢
- `num_stages` 过多消耗 Shared Memory，反而降低 occupancy——增加 stage 后必须同时检查 occupancy 变化
- `num_warps` 过大导致每 warp 分到的 tile 太小，Tensor Core 指令利用率下降

统一决策树请参考：`rule-triton.md` 的"七、统一优化决策树（SSOT）"。
