---
name: hygon-triton-kernel-optimizer
description: 分析和优化海光 HCU/DCU 上手写 Triton 或 TorchInductor 生成算子，结合热点追踪、autotune 输入捕获、目标 AMDGCN/ISA、数值与多规模回归以及端到端验证。适用于 Triton kernel 调优、编译器 lowering 排查、attention/MoE 与模型局部优化。
---

# Hygon Triton 算子优化

先读兄弟 HIP Skill [共同契约](../hygon-hip-kernel-optimizer/references/hcu-workflow-contract.md) 的环境/正确性部分，检索、测速和分析时再读对应章节。优先使用用户材料、当前源码/测试/文档与目标环境；需要补充证据时可用 `hcu-knowledge-search`，不要求先安装或查询知识库。

## 1. 明确场景与环境

记录 exact gfx、Torch/Triton/DTK/AICC 及源码版本、模型/算子、shape/dtype/stride、数值政策、目标与远端运行方式。

```bash
python <skill>/scripts/check_env.py --out triton_env.json
```

gfx946 少伯新特性不能推广到 936/938。Triton 对某 gfx 的支持、`num_warps` 与 wave、dot lowering、编译开关取决于 HCU 分支与安装版本。先读目标源码/帮助与小编译探针；不能用 NVIDIA/AMD 新文档直接推导 HCU 行为。

月英gfx92a/塞班gfx948应保留完整目标名；工具识别到名称不等于该Triton分支已支持。WASP/WDRA按 [候选优化](references/optimization_patterns.md) 检查真实IR请求、分区wave/寄存器配额、spill反馈与目标门禁，不能只照抄旧总wave参数或用scratch大小替代spill判断。

手写 Triton：直接建立独立 reference 和工作负载矩阵。TorchInductor：先用应用 profile 找到真实热点并连接模型节点→生成代码→kernel/dispatch；用 XProf/XCompute 或已有 DTK 工具，而非只看 autotune 最快数字。

## 2. 捕获并复现 Inductor kernel

本节用于 Inductor；手写 Triton 已有独立 repro 时直接进入第 3 节。捕获细节见 [调查流程](references/investigation_workflow.md)。每次调查使用任务独立的 cache/capture 路径；需重编或重新捕获时隔离新产物，保留原缓存，不执行全局 `/tmp/torchinductor_root` 删除。

```bash
export TORCHINDUCTOR_CACHE_DIR="$PWD/repro/cache/inductor"
export TRITON_CACHE_DIR="$PWD/repro/cache/triton"
export TRITON_CAPTURE_DIR="$PWD/repro/capture"
export TORCH_LOGS="+inductor"
export TORCH_COMPILE_DEBUG=1
# 在首次 compiled-model 执行前，把本 Skill scripts 加入 sys.path 并 import autotune_capture_patch
python model_repro.py
```

该示例按当前 HCU PyTorch 的 `torch._inductor.config.trace.enabled` 使用 `TORCH_COMPILE_DEBUG`；换版本先核对已安装源码的实际消费者。记录日志给出的 debug 目录，检查本次新生成的图与 kernel 源码，不能仅凭环境变量已设置或目录存在认定捕获成功。编译缓存命中也不保证重新导出调试文件；确需重编时只使用本任务独立缓存，不删除共享缓存。调试 trace 与下述 autotune 输入快照分别检查，不能相互替代。

捕获补丁依赖 TorchInductor 私有 API；不兼容时按已安装源码适配或写最小独立 repro，不能假装捕获成功。捕获在执行/autotune 前保存完整 storage、shape、stride、offset 与同 dtype alias，独立目录保存源哈希与元数据，避免同名 kernel 覆盖。混合 dtype storage 或不可序列化 launcher 参数明确失败并留日志。

原生 `.pt`/源码只执行可信的当前任务产物。replay 是性能调查入口，**没有独立 oracle，不能证明正确性**；autotuner 对 mutated args 的恢复需要核对实际版本。修改 kernel 必须另建 reference 回归，所有配置都应正确。

```bash
python <skill>/scripts/summarize_autotune_log.py repro/capture/autotune.log --json-out autotune_summary.json
python <skill>/scripts/collect_inductor_artifacts.py --log log_profile.txt --capture-dir repro/capture --cache-root repro/cache --kernel KERNEL --out repro/artifacts
python <skill>/scripts/run_captured_kernel.py repro/capture/UNIQUE/KERNEL.py --json-out replay.json
```

不默认强制 `AMDGCN_USE_BUFFER_OPS`；分别保存当前配置和一次显式探针。失败时记录编译日志、退回已知可编译配置。捕获会同步/复制数据，捕获运行不能作为端到端性能。

## 3. 定位机制并调优

### 3.1 采集与按需分析

首次深入热点、准备结构变化，或出现停滞/回退/热点转移时，按 [分析深度选择](../hygon-hip-kernel-optimizer/references/hcu-workflow-contract.md#按难度与进度选择分析深度) 主动选择上限评估、融合分析、汇编流水分析中的相关项。结合阶段/布局/依赖复杂度、收益空间、已有证据和进度展开；复杂问题可提前分析，简单且证据充分时直接实验，不要求每轮三项全做。触发后若暂缓，简记原因和重开条件；结果进入当前特化的实现、正确性/性能对照与真实模型验证。

**遇到 kernel 优化瓶颈时必须使用性能分析工具。** 调参收益停滞/落入噪声、明显回退或耗时原因不清时，先用 XProf/XCompute（或目标支持的 hipprof/DTK 工具）分析实际目标 kernel，再提出下一轮修改。autotune 最佳时间、源码阅读和静态 ISA 扫描不能代替性能采集与解读。

按 [性能分析入口](../hygon-hip-kernel-optimizer/references/dcu_metrics_guide.md) 选择 [XProf/XCompute](../hygon-hip-kernel-optimizer/references/xprof-xcompute-guide.md) 或 [hipprof](../hygon-hip-kernel-optimizer/references/hipprof-guide.md) 的独立操作步骤，确认 kernel/dispatch 与输入，检查计算、访存、资源和等待。需要解释流水空泡时按当前工具支持的 SQTT 方式采集并关联 ISA，不互换两套参数/格式。工具或权限受阻则记录待采命令、标记诊断未完成，可继续独立的正确性准备，不把推测写成工具结论。

上限不清或考虑融合/重算时，读 [上限与融合取舍](../hygon-hip-kernel-optimizer/references/performance-limits-and-fusion.md)，估计节省与新增成本；实际 lowering/等待/资源有疑点时，按 [编译后检查](references/investigation_workflow.md#42-按需检查编译后实现) 定位本次 JIT 产物，再用 [汇编流水分析](../hygon-hip-kernel-optimizer/references/assembly-guided-optimization.md) 选择源码或低层改法。两类指南服务于下一项实验，无需按顺序通读。

### 3.2 候选、正确性与独立性能确认

按 [实验操作指南](../hygon-hip-kernel-optimizer/references/optimization-playbook.md) 核对算子特有边界、数值变换前提、JIT/autotune 与计时范围及工程验收。此 Skill 使用捕获/重放/扫描流程，不自动执行 HIP 编排的四轮晋级；实际报告须说明已运行的对照协议。

按 [策略](references/optimization_patterns.md) 检查：布局/合并、对齐、向量化、tile/warps/stages、dot 精度与转换、LDS/寄存器/occupancy、tail mask、同步、原子/归约、pipeline 和融合。

准备替换/融合实现、改法不明或迭代停滞且依据不足时，按 [查证路径](../hygon-hip-kernel-optimizer/references/hcu-workflow-contract.md#1-证据来源与按需知识查询) 主动查相关 HCU 工程案例；本地缺答案继续搜线上 PR/MR，读描述、review、固定源码、分派和测试。可按需用 NVIDIA `KernelWiki` 寻找布局/流水/调度等灵感，核对原架构、shape、数值条件，再验证当前 HCU Triton 分支的 lowering、wave/资源和性能，不能直接沿用 NV 配置。将实际采用的来源、HCU 差异、候选假设及验证结果写入调查报告；两个知识库均为可选参考，已有证据足够不重复查询。

- `tl.assume` / `multiple_of` 只能陈述所有受支持输入都成立的事实；给出数学/调用契约依据，再编译探针与负面边界测试。不要从指针“看起来为正”推导合法转换。
- buffer/global/flat 或宽向量指令只提供机制证据，不能按名字宣判快慢。查看目标 ISA、资源、计数器、实际时间。
- 不擅自改 FP8 scale、累计精度、causal/mask、路由、atomic 顺序或模型输出。低精度/近似必须符合任务精度要求。
- 保存基线与每个变体，单卡串行测、多 seed/shape/stride 回归；原地写/alias 输入每次恢复原状态。样本、计时口径、噪声、OOM/编译失败全部保留。
- 瓶颈修改后，用相同采集口径复核目标指标/时间线的变化，并独立测量未开启 profiler 的性能。新的热点或回退无法解释时重新采集分析。

autotune 最快候选需脱离筛选过程再次独立与基线对照，交替执行顺序，保持相同输入/seed、warmup、repeat、缓存和计时口径；保留逐轮结果，不能只报告最小值。适配 HIP 迭代器时使用其四轮确认与产物契约。

## 4. 从单 kernel 到模型

先比较独立 kernel，再对真实模型验证编译/缓存状态、graph break、launch、数据转换、通信与端到端影响。小 tensor/atomic 热点不自动意味着放弃 Triton；比较重排/融合/归约与库实现，再决定是否缩小 compile 边界或用局部 eager fallback。更改模型层必须保持原语义。

`triton_benchmark_template.py` 仅提供 add-scale 教学基准且先做正确性检查；`--repeat` 是 `do_bench` 的毫秒预算。复制到任务后必须按契约替换输入、oracle和指标，不能当任意算子通用验证器。

## 5. 报告与交付

`make_investigation_report.py` 生成待填写草稿；依 [报告模板](references/report_template.md) 补环境/源码、实际使用的资料出处、调用链/目录、捕获完整性、正确性、实际样本、瓶颈证据、候选失败、单 kernel 与模型结果、适用/回退条件。

同名多次捕获、多个编译产物或解析缺口，按 [调查步骤](references/investigation_workflow.md#5-回到模型) 选择记录并检查报告 Notes，不能把其他输入/配置的证据拼到当前候选。

footprint rate 不是 HBM bandwidth utilization；缺少 ISA/消融/归一化计数时标明机制待证。没有 HCU 运行只报告静态结果。报告与原始证据保留在当前任务工程。
