---
name: hygon-triton-kernel-optimizer
description: 分析和优化海光 HCU/DCU 上手写 Triton 或 TorchInductor 生成算子，结合热点追踪、autotune 输入捕获、目标 AMDGCN/ISA、数值与多规模回归以及端到端验证。适用于 Triton kernel 调优、编译器 lowering 排查、attention/MoE 与模型局部优化。
---

# Hygon Triton 算子优化

读取兄弟 HIP Skill 的 [共同契约](../hygon-hip-kernel-optimizer/references/hcu-workflow-contract.md)。优先使用用户材料、当前源码/测试/文档与目标环境。需要参考代码案例或补充领域事实时，可按需通过 `hcu-knowledge-search` 查询相关工程、专题/案例及固定源码/原件；查询不是必经步骤，也不要求安装知识库。

## 1. 明确场景与环境

记录 exact gfx、Torch/Triton/DTK/AICC 及源码版本、模型/算子、shape/dtype/stride、数值政策、目标与远端运行方式。

```bash
python <skill>/scripts/check_env.py --out triton_env.json
```

gfx946 少伯新特性不能推广到 936/938。Triton 对某 gfx 的支持、`num_warps` 与 wave、dot lowering、编译开关取决于 HCU 分支与安装版本。先读目标源码/帮助与小编译探针；不能用 NVIDIA/AMD 新文档直接推导 HCU 行为。

月英gfx92a/塞班gfx948应保留完整目标名；工具识别到名称不等于该Triton分支已支持。WASP/WDRA按 [候选优化](references/optimization_patterns.md) 检查真实IR请求、分区wave/寄存器配额、spill反馈与目标门禁，不能只照抄旧总wave参数或用scratch大小替代spill判断。

手写 Triton：直接建立独立 reference 和工作负载矩阵。TorchInductor：先用应用 profile 找到真实热点并连接模型节点→生成代码→kernel/dispatch；用 XProf/XCompute 或已有 DTK 工具，而非只看 autotune 最快数字。

## 2. 捕获并复现 Inductor kernel

读取 [调查流程](references/investigation_workflow.md)。每轮使用独立 cache/capture 路径，保留原缓存，不执行全局 `/tmp/torchinductor_root` 删除。

```bash
export TORCHINDUCTOR_CACHE_DIR="$PWD/repro/cache/inductor"
export TRITON_CACHE_DIR="$PWD/repro/cache/triton"
export TRITON_CAPTURE_DIR="$PWD/repro/capture"
export TORCH_LOGS="+inductor"
export TORCHINDUCTOR_TRACE=1
# 在首次 compiled-model 执行前，把本 Skill scripts 加入 sys.path 并 import autotune_capture_patch
python model_repro.py
```

捕获补丁依赖 TorchInductor 私有 API；不兼容时按已安装源码适配或写最小独立 repro，不能假装捕获成功。捕获在执行/autotune 前保存完整 storage、shape、stride、offset 与同 dtype alias，独立目录保存源哈希与元数据，避免同名 kernel 覆盖。混合 dtype storage 或不可序列化 launcher 参数明确失败并留日志。

原生 `.pt`/源码只执行可信的当前任务产物。replay 是性能调查入口，**没有独立 oracle，不能证明正确性**；autotuner 对 mutated args 的恢复需要核对实际版本。修改 kernel 必须另建 reference 回归，所有配置都应正确。

```bash
python <skill>/scripts/summarize_autotune_log.py repro/capture/autotune.log --json-out autotune_summary.json
python <skill>/scripts/collect_inductor_artifacts.py --log log_profile.txt --capture-dir repro/capture --cache-root repro/cache --kernel KERNEL --out repro/artifacts
python <skill>/scripts/run_captured_kernel.py repro/capture/UNIQUE/KERNEL.py --json-out replay.json
```

不默认强制 `AMDGCN_USE_BUFFER_OPS`；分别保存当前配置和一次显式探针。失败时记录编译日志、退回已知可编译配置。捕获会同步/复制数据，捕获运行不能作为端到端性能。

## 3. 定位机制并调优

**遇到 kernel 优化瓶颈时必须使用性能分析工具。** 调参收益停滞/落入噪声、明显回退或耗时原因不清时，先用 XProf/XCompute（或目标支持的 hipprof/DTK 工具）分析实际目标 kernel，再提出下一轮修改。autotune 最佳时间、源码阅读和静态 ISA 扫描不能代替性能采集与解读。

按 [性能分析入口](../hygon-hip-kernel-optimizer/references/dcu_metrics_guide.md) 选择 [XProf/XCompute](../hygon-hip-kernel-optimizer/references/xprof-xcompute-guide.md) 或 [hipprof](../hygon-hip-kernel-optimizer/references/hipprof-guide.md) 的独立操作步骤，确认 kernel/dispatch 与输入，检查计算、访存、资源和等待。需要解释流水空泡时按当前工具支持的 SQTT 方式采集并关联 ISA，不互换两套参数/格式。工具或权限受阻则记录待采命令、标记诊断未完成，可继续独立的正确性准备，不把推测写成工具结论。

```bash
python <skill>/scripts/inspect_triton_meta.py KERNEL.py --json-out meta.json
python <skill>/scripts/scan_amdgcn.py repro/artifacts --kernel KERNEL --json-out isa_scan.json
```

低层路径按 [HCU 指令指南](../hygon-hip-kernel-optimizer/references/hcu-isa-guide.md) 核对精确 gfx 的指令形式、布局和同步。`scan_amdgcn.py --kernel` 指实际汇编符号；匹配缺失或歧义会记录 unresolved，不能用文件名代替。

元数据扫描是启发式，只解析部分格式；返回空项不证明编译器没做优化。ISA 扫描不把 LLVM IR 当最终指令，不把静态次数当动态热点。

按 [策略](references/optimization_patterns.md) 检查：布局/合并、对齐、向量化、tile/warps/stages、dot 精度与转换、LDS/寄存器/occupancy、tail mask、同步、原子/归约、pipeline 和融合。需要参考时，可按 attention、稀疏 attention、MoE 的阶段和输入分布查找已有 HCU 案例。

- `tl.assume` / `multiple_of` 只能陈述所有受支持输入都成立的事实；给出数学/调用契约依据，再编译探针与负面边界测试。不要从指针“看起来为正”推导合法转换。
- buffer/global/flat 或宽向量指令只提供机制证据，不能按名字宣判快慢。查看目标 ISA、资源、计数器、实际时间。
- 不擅自改 FP8 scale、累计精度、causal/mask、路由、atomic 顺序或模型输出。低精度/近似必须符合任务精度要求。
- 保存基线与每个变体，单卡串行测、多 seed/shape/stride 回归；原地写/alias 输入每次恢复原状态。样本、计时口径、噪声、OOM/编译失败全部保留。
- 瓶颈修改后，用相同采集口径复核目标指标/时间线的变化，并独立测量未开启 profiler 的性能。新的热点或回退无法解释时重新采集分析。

## 4. 从单 kernel 到模型

先比较独立 kernel，再对真实模型验证编译/缓存状态、graph break、launch、数据转换、通信与端到端影响。小 tensor/atomic 热点不自动意味着放弃 Triton；比较重排/融合/归约与库实现，再决定是否缩小 compile 边界或用局部 eager fallback。更改模型层必须保持原语义。

`triton_benchmark_template.py` 仅提供 add-scale 教学基准且先做正确性检查；`--repeat` 是 `do_bench` 的毫秒预算。复制到任务后必须按契约替换输入、oracle和指标，不能当任意算子通用验证器。

## 5. 报告与交付

`make_investigation_report.py` 生成待填写草稿；依 [报告模板](references/report_template.md) 补环境/源码、实际使用的资料出处、调用链/目录、捕获完整性、正确性、实际样本、瓶颈证据、候选失败、单 kernel 与模型结果、适用/回退条件。

autotune 最快候选需脱离筛选过程再次独立与基线对照，交替执行顺序，保持相同输入/seed、warmup、repeat、缓存和计时口径；保留逐轮结果，不能只报告最小值。适配 HIP 迭代器时使用其四轮确认与产物契约。

footprint rate 不是 HBM bandwidth utilization；缺少 ISA/消融/归一化计数时标明机制待证。没有 HCU 运行只报告静态结果。报告与原始证据保留在当前任务工程。
