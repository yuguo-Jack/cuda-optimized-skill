---
name: operator-optimize-loop
description: Run a multi-iteration optimization loop for CUDA/CUTLASS/Triton operators with correctness, benchmark, targeted/full NCU profiling, strategy memory, and best-version selection.
---

# Operator Optimization Loop

## Overview

该 skill 是 `skills/optimized-skill` 的闭环主入口：
1. correctness validation
2. benchmark
3. backend-aware profiling（targeted + full）
4. 生成 `optimization_proposal.md`
5. 产出下一版算子
6. 多轮迭代并自动选出 best version

支持后端：`cuda`、`cutlass`、`triton`。

## When to use

- 目标是按轮次持续优化算子性能
- 需要保留完整证据链（benchmark + full NCU）
- 需要策略记忆（positive/negative/rejected）指导后续轮次

## When not to use

- 只做单次 correctness/baseline（优先 `kernel-benchmark`）
- 只做单次报告诊断（优先 `ncu-rep-analyze`）

## Inputs

### Required
- 当前算子文件路径：
  - CUDA / CUTLASS：`.cu`
  - Triton：`.py`
- `--max-iterations=<N>`

> 未显式提供 `--max-iterations` 时，不执行，先要求用户给出轮数。

### Strongly recommended
- `--ref=<reference.py>`（否则不能宣称 correctness 已验证）

### Optional
- `--backend=cuda|cutlass|triton`
- `--M=... --N=... --K=...` 等维度参数
- `--warmup`、`--repeat`、`--arch`、`--gpu`、`--ptr-size`、`--seed`
- `--run-dir`
- `--resume-from=best|source|explicit`（默认 `best`）
- `--nvcc-bin=<nvcc>`（CUDA/CUTLASS）
- `--ncu-bin=<ncu>`

### Inference rules
- 未给 `--backend`：`.py -> triton`，否则默认 `cuda`
- `cutlass` 与 `cuda` 共用 `.cu` benchmark 链路，但策略选择不同

## Workflow

### Preflight
运行前必须完成环境检查（由 `scripts/optimize_loop.py` 输出）：
- GPU / Compute Capability / Driver
- `torch.cuda.is_available()`
- 输入文件是否存在
- CUDA/CUTLASS：`nvcc` 可用性
- 全后端：`ncu` 可用性

产物：
- `preflight_check.md`
- `preflight_check.json`

### Iteration loop (v0 -> vN)
每轮强制流程：
1. 对当前版本执行 benchmark + targeted/full profiling
2. 读取本轮证据（`benchmark_result.json`、`iteration_summary.md`、NCU summary/details）
3. 制定本轮优化方向并写 `optimization_proposal.md`
4. 生成下一版算子
5. 进入下一轮，直到 `max_iterations`

策略要求：
- 首轮：广覆盖但必须与算子形态相容
- 后续轮：每轮仅新增 1 个优化方法（控制变量）
- 该方法必须由当前 NCU 症状驱动
- 方法来源仅限：`skills/optimized-skill/reference/`、官方文档，或基于当前 NCU 症状定向检索得到的方法
- 每轮必须显式评估双缓冲/多级流水线是否适配

## Strategy memory

两级记忆自动沉淀：
- 当前 run：`run_manifest.json -> strategy_memory.current_run`
- 全局：`strategy-memory/global_strategy_memory.json`

`optimization_proposal.md` 必须包含：
```md
## Strategy tags
- tag_a

## Optimization method delta
- exactly_one_method

## NCU symptom evidence
- symptom_keyword_from_previous_ncu

## Method sources
- skills/optimized-skill/reference/...
- https://docs.nvidia.com/...
- search: query=<ncu_symptom_based_query>; url=<result_url>; why=<adoption_reason>
```

自动判定：
- proposal 合规失败（单方法/NCU 症状命中/来源约束任一不满足）-> `rejected`
- correctness 失败 -> `rejected`
- benchmark 失败 -> `rejected`
- targeted/full NCU 失败或 full 缺失 -> `rejected`
- 相对上一轮 kernel median 更快 -> `positive`
- 否则 -> `negative`

下一轮约束：
- `blocked`：避免重复踩坑
- `preferred`：优先融合已验证有效策略

## Outputs

### Run-level artifacts
- `run_manifest.json`
- `final_summary.md`
- `preflight_check.md`
- `preflight_check.json`
- `iter_v0/`, `iter_v1/`, ...

### Iteration-level artifacts
- 当前版本文件（`.cu` / `.py`）
- `benchmark_result.json`
- `benchmark.stdout.txt`
- `benchmark.stderr.txt`
- `iteration_summary.md`
- `optimization_proposal.md`
- `targeted.ncu-rep` + `targeted_summary.txt` + `targeted_details.txt`
- `full.ncu-rep` + `full_summary.txt` + `full_details.txt`

### Final response must include
- 最佳版本路径
- baseline vs best benchmark 对比
- best full NCU 报告路径
- best targeted/full 命令
- 主瓶颈判断与关键优化思路
- 被淘汰版本及原因
- 策略记忆结论（positive/negative/rejected）
- 是否避开 blocked、是否融合 preferred

## Failure handling

- correctness 失败：标记 rejected，不参与 best 排名
- profiling 不可用或 full 缺失：明确原因并停止将该版本作为最终答案
- benchmark/环境失败：输出失败证据，不得静默跳过

提前停止条件：
- correctness 问题需先修语义正确性
- `ncu` 无法运行且无法修复
- 连续多轮无可解释性能改善
- 达到用户目标或收益明显递减

## Examples

### CUDA
```bash
python skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.py <kernel.cu> \
    --backend=cuda --max-iterations=<N> [--ref=<ref.py>] [--DIM=VALUE ...] \
    --warmup=10 --repeat=20 [--nvcc-bin=<nvcc>] [--ncu-bin=<ncu>]
```

### CUTLASS
```bash
python skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.py <kernel.cu> \
    --backend=cutlass --max-iterations=<N> [--ref=<ref.py>] [--DIM=VALUE ...] \
    --warmup=10 --repeat=20 [--nvcc-bin=<nvcc>] [--ncu-bin=<ncu>]
```

### Triton
```bash
python skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.py <kernel.py> \
    --backend=triton --max-iterations=<N> [--ref=<ref.py>] [--DIM=VALUE ...] \
    --warmup=10 --repeat=20 [--ncu-bin=<ncu>]
```

### Continue existing run-dir
```bash
python skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.py <next_version_file> \
    --backend=<backend> --run-dir=<existing_run_dir> --resume-from=best --max-iterations=<N> \
    [--ref=<ref.py>] [--DIM=VALUE ...] --warmup=10 --repeat=20
```
- 默认会优先从 `best_kernel_path` 继续；若 best 不存在则回退到当前输入文件。

### Preflight only
```bash
python skills/optimized-skill/operator-optimize-loop/scripts/optimize_loop.py <solution_file> \
    --backend=<backend> --max-iterations=<N> --preflight-only
```

## References

- benchmark convention: `../kernel-benchmark/SKILL.md`
- NCU analysis convention: `../ncu-rep-analyze/SKILL.md`

### CUDA
- `../SKILL.md`
- `../reference/cuda/memory-optim.md`
- `../reference/cuda/compute-optim.md`
- `../reference/cuda/sync-optim.md`

### CUTLASS
- `../reference/cutlass/cutlass-optim.md`

### Triton
- `../reference/triton/triton-optim.md`
