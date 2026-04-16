---
name: ncu-rep-analyze
description: Profile a CUDA/CUTLASS/Triton operator with Nsight Compute or analyze an existing .ncu-rep to classify bottlenecks and produce actionable optimization guidance.
---

# NCU Profiling and Analysis

## Overview

这个 skill 用来回答“这个 kernel 为什么慢”。支持两种输入模式：
1. 输入算子源码：优先生成当前版本 fresh `.ncu-rep`
2. 输入现成 `.ncu-rep`：直接分析，但若源码已变更需标记“报告可能过期”

支持后端：`cuda`、`cutlass`、`triton`。

## When to use

- 需要基于 NCU 定位 memory/compute/latency/occupancy 瓶颈
- 需要生成 targeted/full profiling 证据
- 需要把 NCU 指标转成下一轮可执行优化建议

## When not to use

- 只需要 correctness 或 baseline latency（优先 `kernel-benchmark`）
- 无法运行 `ncu` 且不准备先修环境

## Inputs

### Required
- 算子源码路径（`.cu` 或 `.py`）或 `.ncu-rep` 报告路径

### Optional
- `--backend=cuda|cutlass|triton`
- 维度参数：`--M=... --N=... --K=...`
- profiling 参数：`--launch-skip`、`--launch-count`

### Inference rules
- 若输入是源码，报告输出到源码同目录、同 stem
- 若输入是 `.ncu-rep`，直接读报告
- Triton profiling 通过 `ncu --target-processes all ... python benchmark.py <kernel.py>` 抓取 Python 进程中的 kernel

## Workflow

1. 确定输入模式（源码 / 现成报告）
2. 源码模式先做 targeted profiling（快速定位）
3. 必要时再做 full profiling（最终结论依据）
4. 读取 summary + details，按固定顺序诊断
5. 输出主瓶颈类型、判断依据、优先优化建议

诊断顺序：
1. `SpeedOfLight`
2. `Occupancy` + `LaunchStatistics`
3. `MemoryWorkloadAnalysis`
4. `SchedulerStatistics`

## Bottleneck classification

| 类别 | 典型信号 | 第一建议 |
| --- | --- | --- |
| `DRAM_MEMORY_BOUND` | DRAM 高、SM 低、sector/request 差 | 先修 coalescing，再看 vectorization / tiling |
| `L1_PRESSURE_BOUND` | L1/TEX 压力高、shared path 紧张 | shared tiling、transpose、padding/swizzle |
| `LATENCY_BOUND` | SM 低、Memory 也不高、eligible warps 低 | 提升 ILP、减少长依赖、细化同步 |
| `COMPUTE_BOUND` | SM 高、Memory 非主问题 | Tensor Core、低精度、MMA 路径 |
| `OCCUPANCY_BOUND` | achieved occupancy 低且限制因子明确 | 降 registers/smem、调 block size/tile |
| `HOST_OR_LAUNCH_BOUND` | kernel 很短、网格很小、GPU 指标都不高 | 转更上层时序分析 |
| `MIXED_BOUND` | 无单一主症状 | 先做最明确一类验证 |

## Outputs

结构化输出至少包含：
- 报告路径
- backend
- kernel 名称
- 是否 fresh profile
- targeted 命令 + 报告路径
- full 命令 + 报告路径
- 关键指标摘要
- 主瓶颈类型
- 判断依据
- 高优先级优化建议
- 是否需要转上层时序 / correctness 排查

## Failure handling

- `ncu` 不可用、权限不足或环境阻止：明确失败原因并停止
- 不允许用旧 kernel 报告冒充当前版本结论
- 不允许仅凭经验在 profiling 失败后输出高置信瓶颈结论
- 最终交付必须附 winning 版本 full NCU 证据，不能只给 targeted 结果

## Examples

### Targeted profiling (first pass)
```bash
ncu --target-processes all \
    --profile-from-start on \
    --launch-skip 20 \
    --launch-count 1 \
    --section LaunchStatistics \
    --section Occupancy \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section SchedulerStatistics \
    -o {kernel_dir}/{kernel_stem} -f \
    python skills/optimized-skill/kernel-benchmark/scripts/benchmark.py <solution_file> \
    --backend=<cuda|cutlass|triton> [--DIM=VALUE ...] --repeat=22
```

### Read summary
```bash
ncu --import <file.ncu-rep> --print-summary per-kernel
```

### Read details
```bash
ncu --import <file.ncu-rep> --page details
```

## References

### CUDA
- `../SKILL.md`
- `../reference/cuda/memory-optim.md`
- `../reference/cuda/compute-optim.md`
- `../reference/cuda/sync-optim.md`

### CUTLASS
- `../reference/cutlass/cutlass-optim.md`

### Triton
- `../reference/triton/triton-optim.md`
