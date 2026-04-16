CUDA kernel同步优化的完整方案：
---

1. 减少 __syncthreads() 次数：这是最直接的方式。很多时候开发者习惯性地插入同步，但如果warp内的数据访问模式天然不存在跨warp依赖，同步就是多余的。

2. Warp级同步替代Block级同步：一个warp内的32个线程天然是lockstep执行的（Volta之后有independent thread scheduling，但warp级原语仍然有效）。用 __syncwarp() 替代 __syncthreads() 可以把同步粒度从整个block缩小到单个warp，开销大幅降低。
   - 补充（语义边界）：在Volta+独立线程调度下，不应把"天然lockstep"当作隐式同步保证；warp内协作仍应使用正确mask与显式同步点确保可见性与收敛。

3. Warp Shuffle（__shfl_sync 系列）：warp内线程之间交换数据完全不需要经过共享内存，没有bank conflict问题，延迟极低。适用于归约、前缀和、广播等模式。这直接消除了"写shared → sync → 读shared"的三步开销。

4. Cooperative Groups：CUDA 9引入的协作组机制，可以定义任意粒度的线程组并在该组内同步，比如只同步一个tile（比如8个线程），避免不必要的全block等待。

5. cuda::barrier / cuda::pipeline（Ampere+常见）：在异步拷贝和多级流水线中，用显式阶段同步替代"经验式同步"。核心思想是把producer-consumer的提交/等待点写清楚，避免偶现错误和性能抖动。

6. CUDA Graphs减少调度开销：当kernel链路结构稳定且反复执行时，Graph可以降低CPU提交与launch开销，特别是小kernel密集场景。动态图场景要评估capture/update成本。

---

**二、验证清单（NCU）**

同步优化建议至少配套以下验证：

1. **同步等待是否下降**：关注 `Stall Barrier` 与相关等待 stall 是否降低。
2. **调度可发射性是否改善**：关注 `Eligible Warps Per Cycle` 是否提升。
3. **warp 级替代是否生效**：将 `__syncthreads()` 替换为 `__syncwarp()` / `__shfl_sync` 后，确认延迟与吞吐有正向变化。
4. **语义正确性是否保持**：在 Volta+ 独立线程调度下，必须确认 mask、收敛点与可见性语义正确。

常见误判：
- 只减少了 `__syncthreads()` 次数，但引入了数据可见性错误。
- 只看单个 stall 指标下降，没有检查整体 kernel latency 与 correctness。

统一决策树请参考：`skills/optimized-skill/SKILL.md` 的“七、统一优化决策树（SSOT）”。


