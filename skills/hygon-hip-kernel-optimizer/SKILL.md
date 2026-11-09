---
name: hygon-hip-kernel-optimizer
description: 在海光 HCU/DCU 上开发和优化 HIP/C++、CK Tile 算子，通过明确的数值/布局契约、正确性与多规模性能回归、XProf/XCompute 或 DTK hipprof 和目标 ISA 证据推进迭代。适用于已有 HIP 基线、CUDA 移植和 HCU kernel 调优；只有 Python/Triton reference 时先生成可验证基线。
---

# Hygon HIP 算子开发与优化

把优化落实为可复现的代码和实验。开始时先读 [共同契约](references/hcu-workflow-contract.md) 的环境/正确性部分；查证、测速与 profiling 时按需读对应章节。不要把 NVIDIA 上游 Skill 的 nvcc、NCU、SASS 指令或硬件门禁直接搬到 HCU。

## 1. 定义任务与查证

- 从用户需求和工程推导算子语义、支持范围、目标设备/工具链、精度和性能指标，记录到任务目录 `contract.md`。缺少的必要事实先检查工程/环境，真正阻塞再问；可独立完成的工作继续。
- 继续旧任务或优化相关算子时，先读任务工程已有的经验摘要及其关联实验，核对适用条件再复用。入口、停滞和恢复时按需读 [架构探索与经验复用](references/architecture-search-and-experience.md)，避免重复试错或沿用未经验证的架构假设。
- 优先查看用户材料、当前源码/测试/文档与目标环境。需要参考代码案例或补充领域事实时，可按需使用 `hcu-knowledge-search`，阅读相关指南/案例及固定源码/原件并记录出处。查询不是必经步骤，未安装或不可用不阻塞已有证据足够的工作。
- 按 [查证路径](references/hcu-workflow-contract.md#1-证据来源与按需知识查询) 把 HCU 案例追到 PR/MR 描述、review、固定源码、调用点和测试；本地缺答案时主动搜索线上 PR。需要优化灵感时可查 NVIDIA `KernelWiki`，提取可迁移机制并核对 HCU 前提，不直接套用 NVIDIA 指令或收益。
- 查已存在的 HCU 实现与构建方式：HIP、CK Tile、HCU Cutlass、Triton/TileLang、自有库各有适用点。能直接复用当前工程接口时先复用；不强制某种 DSL，也不拿 AMD/NVIDIA 分支冒充 HCU。
- 从源码开发而没有正确 HIP 基线时先调用 `hygon-hip-baseline-generator`；仅需 Triton/Inductor 调优使用 `hygon-triton-kernel-optimizer`。只有可运行代码对象时可走下述汇编实验路线，保留原对象作差分参照并注明验证范围，不凭空生成数学 oracle。明确选择原因。

## 2. 建立实际执行环境

Agent 可本地启动。按项目远端配置/远端工作 Skill 选择 SSH、容器、挂载和 SDK 激活；以下命令运行于真正构建/测试的 HCU 环境。

下面的 `kernel.hip` 与编排命令针对源码实验；仅有代码对象或直接改写对象时，按汇编指南使用项目专用入口，共用契约、诊断与验证要求，不强行套源码脚手架。

```bash
python <skill>/scripts/check_env.py --out env.json
python <skill>/scripts/preflight.py --baseline kernel.hip --ref ref.py --dims '{"N":1048576}' --out preflight.json
```

确认准确 gfx（包括完整拼写 gfx92a）、编译器、库提交、wave width、设备可用性。名称映射与指令能力分开，参照指令指南核对月英/少伯/塞班的具体形式。AICC 与 DTK DCC/hipcc 分别记录。硬件缺失不妨碍静态开发，但不能写成已经跑过 HCU。

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

### 按参考质量选择起点

区分数学 oracle、冻结的测速基线、当前已确认 best 与正在探索的实现。冻结语义和测量口径，不冻结候选架构；best 可以成为下一轮实现起点，不能替换 oracle 或重设容差。

- **有成熟专家参考**：先理解并验证其在当前 HCU、输入和数值契约下的结构与性能。适用时复用其机制，优先调查剩余瓶颈；专家参考仍可能受原硬件、shape、库版本或设计取舍限制，不能作为架构上限。
- **生成的正确性基线**：`hygon-hip-baseline-generator` 的保守映射、串行循环和中间写回只服务于建立基线。正确性通过后，在投入大量参数扫描前评估数据流、任务划分及计算路径是否合适；可直接重写结构，不必先把脚手架局部调到极限。生成来源和实际质量分别记录，不因作者身份替代实测。

按算子证据决定探索深度，无须凑固定路线数、先冻结完整设计或逐项尝试策略目录。具体做法见 [架构探索与经验复用](references/architecture-search-and-experience.md)。

## 4. 定位瓶颈并实验

首次深入热点、准备结构变化，或出现停滞/回退/热点转移时，按 [分析深度选择](references/hcu-workflow-contract.md#按难度与进度选择分析深度) 主动选择上限评估、融合分析、汇编流水分析中的相关项。依据难度、复杂度、收益空间、已有证据和进度展开；复杂问题可提前分析，简单且证据充分时直接实验，不要求每轮三项全做。触发后若暂缓，简记原因和重开条件；分析结果回到候选选择与原验证循环。

**遇到 kernel 优化瓶颈时，使用性能分析工具是必选步骤。** 例如收益停滞/落入噪声、出现明显回退、耗时显著偏离预期，或无法解释计算/访存/等待限制时，先采集当前目标 kernel 并分析，再决定下一轮修改；不能只靠继续扫 tile/stages、读源码或 ISA grep 代替。优先 XProf 采集和 XCompute 分析，目标不支持时用能回答同一问题的 hipprof/DTK 工具。

工具、目标机器或采集权限暂不可用时，记录具体阻塞和待采命令，标记瓶颈诊断未完成；可继续独立的代码/正确性准备，不把静态猜测当作已完成诊断。`--profiler none` 只适用于无需瓶颈诊断的准备阶段，不能绕过此步骤。

读 state、原始 bench、best_input/kernel 各自的 profile 记录、dcu_top、roofline 和实际使用的参考证据。`dcu_top.json` 是兼容的最近一次摘要，不能代替前后原件。`roofline.json` 的 null 是未知；预算是建议，不能据此宣布 compute/memory bound 或 near-peak。制定候选、处理数值变换/编译资源/失败恢复时按需读 [实验操作指南](references/optimization-playbook.md)。

从 [性能分析入口](references/dcu_metrics_guide.md) 选择对应指南：[XProf/XCompute](references/xprof-xcompute-guide.md) 使用 sections/metrics 与 `.perf` 分析；[hipprof](references/hipprof-guide.md) 使用自己的 trace/PMC/SQTT 参数和产物。命令、格式、指标公式与查看器不能互换。自动采集只提供原件与发现信息，命令成功不代表分析完成；必须把观察到的指标/时间线连接到瓶颈假设与下一项实验，并对修改前后做同口径比较。

先判断本轮需要验证结构选择还是做局部调参。局部收益停滞、多个参数变体表现相近或与可信实现有无法解释的差距时，结合采集结果重新检查当前架构假设，用能区分候选解释的小探针决定继续、换路或停止；不能把扫参饱和直接当作硬件极限。结构重写允许必要的耦合改动，探索中的实现和已确认 best 分开保存。

上限粗估解释不通或需要评估融合/重算时，按需读 [性能上限、数据流与融合取舍](references/performance-limits-and-fusion.md)：分清必要工作、接口交接与实际执行工作，核对相应资源和关键路径；融合同时估计省下的流量与新增计算、布局、寄存器/LDS、显存和并发代价。模型用于选方向，可被实测修正；不要求每轮全量建模或使用 KRA，也不把未知指标、单项 busy 或局部停滞当成极限。

怀疑编译后的调度、等待、转换或寄存器安排限制性能时，按需读 [汇编流水与编译后优化](references/assembly-guided-optimization.md)：将 profile 热点对应到实际 ISA，检查依赖后可改源码、builtin/inline asm，也可在目标工具链支持时直接编辑并重建代码对象。路线由证据决定，无须先穷尽源码调参；直接改写才需要恢复往返、ABI/资源与实际加载验证。现有源码编排不自动支持二进制候选，使用项目专用实验入口，保留原 oracle 和确认要求。

1. 标准局部调优用 [策略目录](references/optimization_catalog.md) 和 registry 选 1..3 项，不凑数；每轴最多 2 项，跳过更高优先项写具体理由。这些是现有脚本的计划约束，不是结构探索的候选全集或通用先后次序；目录无法表达的路线按 [脚本衔接](references/architecture-search-and-experience.md#5-与现有脚本衔接) 使用项目实验入口。准备替换/融合实现、瓶颈改法不明或迭代停滞且现有依据不足时，按共同契约主动查 HCU 案例/PR/源码，必要时查 KernelWiki；把参考机制变成本轮可验证的 HCU 假设，再选择改动。证据已足够时不重复检索。
2. `methods.json` 按 [schema](templates/methods.schema.json)：方法 ID、改什么、为何可能有效、预期证据、精确目标证据、跳过理由。重试先前方法给 `retry_reason`。
3. 在 `itervN/branches/b1..bK/kernel.<ext>` 写局部变体，同组方法一致、超参数不同。不同结构路线分开记录和归因，可从保留的探索分支继续，不限于每次复制 best；当前脚本不自动调度这些路线。只修改用户任务范围的源码。
4. 运行 `branch_explore.py --state RUN/state.json --iter N`，先检查正确性/稳定计时/多用例回退；失败的分支修复后再测，保留失败原因。OOM 不缩小原场景冒充成功。
5. 低层路径变化按 [指令指南](references/hcu-isa-guide.md) 读实际目标 ISA、资源与同步语义；`sass_check.py` 的名字为兼容保留，输出是 HCU ISA 提示。反汇编失败、缺指令或目标符号不完整时，按指南的“实际工程反汇编失败时”沿用工程真实构建配置保留汇编，测试对应的新构建并记录产物关系；通用重编译回退仅作辅助。自动 grep 不算语义验证。
6. 有必要做消融时，将只去掉单项方法且仍语义正确的文件放 `ablations/<id中点换成下划线>/kernel.<ext>`。多个方法相互依赖时写清归因限制。
7. 在 close 前可单独运行 profiler/ISA/ablation 做调查，并填写 `mechanism-review.json`。close 会重测，若重新产生的 artifact 哈希改变，复核文件失配则方法保持未验证；可以下一轮再补证据，不能伪造已验证。

```bash
python <skill>/scripts/orchestrate.py close-iter --run-dir RUN --iter 1
```

close 串行跑分支、选 champion、对当前 best 做四轮独立交替对照、profile、消融、ISA、更新状态，并准备下一轮数据。四轮正确、输入/设备/测量参数一致且收益持续超过门槛才晋级；矩阵场景逐例守住回退上限。没有新鲜对照记 unconfirmed，重复结果落入噪声记 not_improved。缺少消融/机制证据的方法仍记 `unverified_methods`。已关闭迭代不可覆盖；单 run 只允许一个写入者。

## 5. 交付

```bash
python <skill>/scripts/orchestrate.py finalize --run-dir RUN
```

用 [报告模板](templates/iteration_report.md) 补全程序不能自动推出的解释：代码目录/接口与调用链、热点、修改机制、适用范围、正确性矩阵、真实样本、端到端收益、失败/回退、实际使用的资料与源码引用。未跑项目测试、race 检查或 HCU 硬件验证明确列出。迭代预算耗尽、目标达到或收益落入噪声时总结，不无限试错。

按操作指南回到真实分派验证命中路径、fallback、单测/端到端与显存成本；失败/中止时也可 finalize，列出停止原因和未完成诊断。流程闭合不等于项目验收完成。

报告与原始证据保留在当前任务工程。收尾或切换会话前，更新可供下次读取的经验摘要：实测事实、适用条件、失败反例、未决假设、best 与值得保留的探索分支及原件入口；不只留下最终加速比。按 [经验留存](references/architecture-search-and-experience.md#4-让实测经验跨轮跨会话复用) 维护，下一次先核对再复用。
