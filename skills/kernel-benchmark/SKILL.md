---
name: kernel-benchmark
description: Compile, validate, and benchmark a CUDA kernel against an optional Python reference by driving `skills/kernel-benchmark/scripts/benchmark.py`. Use when Codex needs to 验证 `.cu` kernel 正确性、做 baseline 性能测试、比较 kernel 与 reference 的 speedup、按 `extern "C" void solve(...)` 签名推断参数，或在进入 NCU 分析前先得到稳定 benchmark 结果。
---

# Kernel Benchmark

通过 `skills/kernel-benchmark/scripts/benchmark.py` 编译、验证、压测 `extern "C" void solve(...)` 风格的 CUDA kernel。
所有命令都从仓库根目录运行。

## 共享文档入口

按需查这些共享文档：
- timing 和 bandwidth: `../cuda_skill/references/best-practices-guide/9-performance-metrics.md`
- 调试和 sanitizer: `../cuda_skill/references/debugging-tools.md`
- 共享 workflow: `../cuda_skill/references/workflow-checklists.md`
- GPU 架构能力: `../cuda_skill/references/cuda-guide/05-appendices/compute-capabilities.md`

## 输入

- 必需：`.cu` 文件
- 可选：reference `.py`
- 可选：维度参数，如 `--M=... --N=... --K=...`
- 可选：`--warmup`、`--repeat`、`--arch`

## reference 推断规则

如果用户没显式给 `--ref`，按这些位置寻找：
- `.cu` 同目录下的 `*_ref.py`
- 算法目录下语义匹配的 reference 文件

找不到 reference 时：
- 仍可 benchmark kernel
- 但不要声称 correctness 已验证

## 维度参数推断规则

脚本会从 `extern "C" void solve(...)` 读取整型参数名。
若用户没给值，需要补一个合理默认值再运行。

常用默认：
- matmul / GEMM: `M=4096, N=4096, K=4096`
- reduction / element-wise: `N=1000000`
- transpose: `M=4096, N=4096`

如果算法明显不是这些类别，就根据签名和上下文给出保守值，不要乱猜极端大尺寸。

## 标准命令

只有 benchmark：

```bash
python skills/kernel-benchmark/scripts/benchmark.py <cu_file> \
    [--DIM=VALUE ...] --warmup=10 --repeat=20
```

验证加 benchmark：

```bash
python skills/kernel-benchmark/scripts/benchmark.py <cu_file> \
    --ref=<ref_file> [--DIM=VALUE ...] --warmup=10 --repeat=20
```

指定架构：

```bash
python skills/kernel-benchmark/scripts/benchmark.py <cu_file> \
    --ref=<ref_file> [--DIM=VALUE ...] --arch=sm_90
```

## benchmark.py 的实际行为

脚本会：
- 从 `solve(...)` 签名推断参数
- 自动检测 GPU 架构，未指定时默认用当前设备 capability
- 用 `nvcc` 编译为共享库
- 有 reference 时先跑 correctness，再 benchmark reference 和 kernel
- 没有 reference 时，先打印输入输出 preview，再 benchmark kernel

注意：
- 输出中的 `~Bandwidth` 是“按所有指针 buffer 总字节数估算”的粗指标，只能看趋势，不能替代精确 effective bandwidth。
- 真正的 effective bandwidth 需要按 kernel 实际读写字节数手算，可参考 `9-performance-metrics.md`。

## 如何解读结果

优先看这些信息：
- correctness 是否 `ALL PASS`
- `Average` 和 `Median` 是否稳定
- `Speedup` 是否真的大于 reference
- `~Bandwidth` 是否与瓶颈判断一致

经验规则：
- `Average`、`Median` 差很多，通常说明 benchmark 不稳定
- 快了但 correctness 失败，不算优化成功
- 规模不同的 benchmark 不能直接比较

## 失败处理

### correctness 失败

立即停止后续性能结论，优先反馈：
- 失败的输出 tensor
- 最大误差和首个错误位置
- 建议先跑：

```bash
compute-sanitizer --tool memcheck <program>
compute-sanitizer --tool racecheck <program>
compute-sanitizer --tool synccheck <program>
```

### 编译失败

直接返回 `nvcc` 错误，不要继续做性能判断。

### benchmark 噪声过大

优先统一：
- 输入规模
- `warmup`
- `repeat`
- GPU 选择

## 输出约定

返回一份简洁报告：
- kernel 路径
- reference 路径或“未提供”
- 维度参数
- GPU / arch
- 完整实际执行命令
- correctness 结果
- kernel latency summary
- reference latency summary
- speedup
- 是否建议进入 `ncu-rep-analyze`

完整实际执行命令必须原样回显，至少包含：
- `.cu` 路径
- `--ref`
- 所有 `--DIM=VALUE`
- `--arch`
- `--warmup`
- `--repeat`
