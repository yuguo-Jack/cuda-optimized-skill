---
name: hygon-hip-kernel-optimizer
description: 在海光 HCU/DCU 上开发和优化 HIP/C++、CK Tile 算子，通过明确的数值/布局契约、正确性与多规模性能回归、XProf/XCompute 或 DTK hipprof 和目标 ISA 证据推进迭代。适用于已有 HIP 基线、CUDA 移植和 HCU kernel 调优；只有 Python/Triton reference 时先生成可验证基线。
---

# Hygon HIP 算子开发与优化

把优化落实为可复现的代码和实验。开始时先读 [共同契约](references/hcu-workflow-contract.md) 的环境/正确性部分；查证、测速与 profiling 时按需读对应章节。不要把 NVIDIA 上游 Skill 的 nvcc、NCU、SASS 指令或硬件门禁直接搬到 HCU。

## 1. 定义任务与查证

- 从用户需求和工程推导算子语义、支持范围、目标设备/工具链、精度和性能指标，记录到任务目录 `contract.md`。缺少的必要事实先检查工程/环境，真正阻塞再问；可独立完成的工作继续。
- 优先查看用户材料、当前源码/测试/文档与目标环境。需要参考代码案例或补充领域事实时，可按需使用 `hcu-knowledge-search`，阅读相关指南/案例及固定源码/原件并记录出处。查询不是必经步骤，未安装或不可用不阻塞已有证据足够的工作。
- 查已存在的 HCU 实现与构建方式：HIP、CK Tile、HCU Cutlass、Triton/TileLang、自有库各有适用点。能直接复用当前工程接口时先复用；不强制某种 DSL，也不拿 AMD/NVIDIA 分支冒充 HCU。
- 没有正确 HIP 基线时先调用 `hygon-hip-baseline-generator`；仅需 Triton/Inductor 调优使用 `hygon-triton-kernel-optimizer`。明确选择原因。

## 2. 建立实际执行环境

Agent 可本地启动。按项目远端配置/远端工作 Skill 选择 SSH、容器、挂载和 SDK 激活；以下命令运行于真正构建/测试的 HCU 环境。

```bash
python <skill>/scripts/check_env.py --out env.json
python <skill>/scripts/preflight.py --baseline kernel.hip --ref ref.py --dims '{"N":1048576}' --out preflight.json
```

确认准确 gfx（936/938/946 等）、编译器、库提交、wave width、设备可用性。AICC 与 DTK 自带编译器分别记录。硬件缺失不妨碍静态开发，但不能写成已经跑过 HCU。

普通 flat ABI 需要 `extern "C" void solve(...)` 和 reference 中同名参数的 `reference(...)`；非 const 指针为输出。它只支持独立连续简单类型；半精度/量化、非连续/alias/in-place、多 stream 和通信应使用项目专用 `--benchmark`，不能悄悄简化契约。专用结果须符合 [实验产物](references/experiment-artifacts.md) 的验证门禁。

## 3. 冻结基线与回归矩阵

```bash
python <skill>/scripts/benchmark.py kernel.hip --ref ref.py --N=1048576 --ptr-size 1048576 --json-out baseline_bench.json
python <skill>/scripts/orchestrate.py setup --baseline kernel.hip --ref ref.py --dims '{"N":1048576}' --ptr-size 1048576 --workloads workloads.json --profiler auto --iterations 3 --branches 4
```

- 默认迭代预算 3、分支最多 4；用户有明确预算则遵从。不要为默认预算反复询问。
- `--workloads` 用 [模板](templates/workloads.example.json) 按真实任务改写；覆盖多规模、多 seed 和边界。没有该选项仅主 shape，不能交付为“全面验证”。
- 静态 preflight、编译、数值、race safety、性能和端到端收益是独立状态。只通过编译或只有时间不算正确。
- 模型 reference、baseline 与环境冻结后不随候选变化。benchmark 改动也要重测基线。

## 4. 定位瓶颈并实验

**遇到 kernel 优化瓶颈时，使用性能分析工具是必选步骤。** 例如收益停滞/落入噪声、出现明显回退、耗时显著偏离预期，或无法解释计算/访存/等待限制时，先采集当前目标 kernel 并分析，再决定下一轮修改；不能只靠继续扫 tile/stages、读源码或 ISA grep 代替。优先 XProf 采集和 XCompute 分析，目标不支持时用能回答同一问题的 hipprof/DTK 工具。

工具、目标机器或采集权限暂不可用时，记录具体阻塞和待采命令，标记瓶颈诊断未完成；可继续独立的代码/正确性准备，不把静态猜测当作已完成诊断。`--profiler none` 只适用于无需瓶颈诊断的准备阶段，不能绕过此步骤。

读 state、原始 bench、dcu_top、roofline 和实际使用的参考证据。`roofline.json` 的 null 是未知；预算是建议，不能据此宣布 compute/memory bound 或 near-peak。

从 [性能分析入口](references/dcu_metrics_guide.md) 选择对应指南：[XProf/XCompute](references/xprof-xcompute-guide.md) 使用 sections/metrics 与 `.perf` 分析；[hipprof](references/hipprof-guide.md) 使用自己的 trace/PMC/SQTT 参数和产物。命令、格式、指标公式与查看器不能互换。自动采集只提供原件与发现信息，命令成功不代表分析完成；必须把观察到的指标/时间线连接到瓶颈假设与下一项实验，并对修改前后做同口径比较。

1. 用 [策略目录](references/optimization_catalog.md) 和 registry 选 1..3 项，不凑数；每轴最多 2 项，跳过更高优先项写具体理由。
2. `methods.json` 按 [schema](templates/methods.schema.json)：方法 ID、改什么、为何可能有效、预期证据、精确目标证据、跳过理由。重试先前方法给 `retry_reason`。
3. 在 `itervN/branches/b1..bK/kernel.<ext>` 写变体，同组方法一致、超参数不同；只修改用户任务范围的源码。
4. 运行 `branch_explore.py --state RUN/state.json --iter N`，先检查正确性/稳定计时/多用例回退；失败的分支修复后再测，保留失败原因。OOM 不缩小原场景冒充成功。
5. 低层路径变化读实际目标 ISA、资源与同步语义；`sass_check.py` 的名字为兼容保留，输出是 HCU ISA 提示。自动 grep 不算语义验证。
6. 有必要做消融时，将只去掉单项方法且仍语义正确的文件放 `ablations/<id中点换成下划线>/kernel.<ext>`。多个方法相互依赖时写清归因限制。
7. 在 close 前可单独运行 profiler/ISA/ablation 做调查，并填写 `mechanism-review.json`。close 会重测，若重新产生的 artifact 哈希改变，复核文件失配则方法保持未验证；可以下一轮再补证据，不能伪造已验证。

```bash
python <skill>/scripts/orchestrate.py close-iter --run-dir RUN --iter 1
```

close 串行跑分支、选 champion、profile、消融、ISA、更新状态，并准备下一轮数据。通过正确性与稳定测量且更快的 kernel 可以晋级；缺少消融/机制证据的方法仍记 `unverified_methods`。已关闭迭代不可覆盖；单 run 只允许一个写入者。

## 5. 交付

```bash
python <skill>/scripts/orchestrate.py finalize --run-dir RUN
```

用 [报告模板](templates/iteration_report.md) 补全程序不能自动推出的解释：代码目录/接口与调用链、热点、修改机制、适用范围、正确性矩阵、真实样本、端到端收益、失败/回退、实际使用的资料与源码引用。未跑项目测试、race 检查或 HCU 硬件验证明确列出。迭代预算耗尽、目标达到或收益落入噪声时总结，不无限试错。

报告与原始证据保留在当前任务工程。
