---
name: kernel-benchmark
description: Compile, validate, and benchmark a CUDA, CUTLASS, or Triton operator via benchmark.py, then return correctness + latency + speedup evidence.
---

# Kernel Benchmark

## Overview

通过 `skills/optimized-skill/kernel-benchmark/scripts/benchmark.py` 统一执行：
- correctness validation（可选 reference）
- baseline benchmark（avg/median）
- 结构化结果输出（可选 JSON）

支持后端：
- `cuda`：`extern "C" void solve(...)` 的 `.cu`
- `cutlass`：同样暴露 `solve(...)` 的 `.cu`
- `triton`：暴露 `setup(...)` + `run_kernel(...)` 的 `.py`

## When to use

- 需要先确认算子是否正确
- 需要拿可复现 baseline latency
- 需要比较 kernel 与 reference 的 speedup
- 进入 NCU 分析或优化 loop 前先拿稳定基线

## When not to use

- 只需要解释 `.ncu-rep`，不需要重跑 benchmark（改用 `ncu-rep-analyze`）
- 无 GPU 或无运行权限且不准备先修环境

## Inputs

### Required
- `solution_file`
  - CUDA / CUTLASS：`.cu`
  - Triton：`.py`

### Optional
- `--backend=cuda|cutlass|triton`
- `--ref=<reference.py>`
- 维度参数：`--M=... --N=... --K=...` 等
- `--warmup`、`--repeat`、`--arch`、`--gpu`、`--ptr-size`
- `--atol`、`--rtol`、`--seed`
- `--nvcc-bin=<path or command>`（CUDA/CUTLASS）
- `--json-out=<file>`

### Inference rules
- 未提供 `--backend`：`.py -> triton`，否则默认 `cuda`
- 未提供 `--ref`：按约定路径尝试推断；找不到则仅做 benchmark，不宣称 correctness 已验证

## Workflow

1. 解析输入并确定 backend
2. 准备输入数据（含维度参数）
3. 若有 reference，先做 correctness
4. 执行 benchmark（warmup + repeat）
5. 汇总 avg/median/speedup
6. 若设置 `--json-out`，落盘结构化结果

## Outputs

返回报告至少包含：
- solution 路径
- backend
- reference 路径（或“未提供”）
- 实际执行命令
- correctness 结果
- kernel latency（avg/median）
- reference latency（若有）
- speedup（若有）
- 是否建议进入 profiling

`--json-out` 时至少包含字段：
- `solution_file`
- `backend`
- `ref_file`
- `has_reference`
- `dims`
- `gpu_name`
- `arch`
- `correctness`
- `kernel`
- `reference`
- `speedup_vs_reference`

## Failure handling

- correctness 失败：停止性能结论，优先报告错误张量、最大误差、首个错误位置
- CUDA/CUTLASS 编译失败：原样返回 `nvcc` 错误，不继续性能判断
- benchmark 噪声过大：统一输入规模与 warmup/repeat 后再比较

## Examples

### Minimal (CUDA)
```bash
python skills/optimized-skill/kernel-benchmark/scripts/benchmark.py <kernel.cu> --backend=cuda
```

### Standard (with dims)
```bash
python skills/optimized-skill/kernel-benchmark/scripts/benchmark.py <solution_file> \
    --backend=<cuda|cutlass|triton> [--DIM=VALUE ...] --warmup=10 --repeat=20
```

### With correctness + JSON output
```bash
python skills/optimized-skill/kernel-benchmark/scripts/benchmark.py <solution_file> \
    --backend=<backend> --ref=<ref_file> [--DIM=VALUE ...] --json-out=benchmark_result.json
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
