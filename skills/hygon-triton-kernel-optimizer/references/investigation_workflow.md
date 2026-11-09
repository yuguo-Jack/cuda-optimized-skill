# Triton 调查步骤

本页负责 Inductor 捕获、实际 JIT 产物与重放的操作细节；手写 Triton 已有 repro 时直接使用相关编译后检查和验证部分。分析深度、数值/测速约束共用 HIP Skill 的契约，不另建一套优化门禁。

## 1. 模型热点与契约

原始模型→调用图→生成源码→实际 dispatch 建立对应，记录 Torch/Triton/HCU 工具链与实际使用的资料来源。先用未采集运行保留端到端基线，再采系统热点和必要的单 kernel 指标。

## 2. 独立工作目录

任务使用独立 TORCHINDUCTOR_CACHE_DIR、TRITON_CACHE_DIR、TRITON_CAPTURE_DIR；需要重编/重捕获时隔离新产物，不要求每个计时重复建缓存。日志/缓存路径按实际运行用户和版本查，不清全局缓存，不能猜 /tmp/torchinductor_root。

首次 compiled-model 执行前导入捕获补丁。它记录执行前 storage/stride/offset/alias 和源哈希；捕获会增加同步/数据复制，不能拿此轮测端到端收益。无 autotune 不等于异常，先看缓存命中、单配置及实际 dispatch。

## 3. 复现与独立正确性

只重放可信文件。生成 replay 保持布局与同dtype alias，重新取得当前stream handle，但仅做 autotune 时间调查。读取安装版 CachingAutotuner 确认 mutated args 是否在每次试验间恢复；另写独立 oracle，逐配置核对。

混合dtype storage、函数对象launcher参数、不可序列化对象出现 capture-error 时写项目最小repro，不能把一个不完整目录当已捕获。

## 4. 差异与验证

### 4.1 从采集到假设

遇到 kernel 优化瓶颈必须先采集分析，不能只继续 autotune 或扫描 ISA。从 [性能分析入口](../../hygon-hip-kernel-optimizer/references/dcu_metrics_guide.md) 选定工具。以下 2–3 项是 [XProf/XCompute](../../hygon-hip-kernel-optimizer/references/xprof-xcompute-guide.md) 的操作；使用 hipprof 时改按 [hipprof 指南](../../hygon-hip-kernel-optimizer/references/hipprof-guide.md)，不能套用 XCompute 的视图和参数：

1. 将生成文件、源码/代码对象、实际 kernel 名、shape/stride/config 与 dispatch 对应起来。初始化、JIT、warmup、输入捕获产生的 kernel 不作为目标；新运行需重新确认 dispatch 编号。
2. 对代表性瓶颈用例采 speed-of-light/compute/memory/occupancy，再按现象补 scheduler/wave-state；在 XCompute 中确认到底是计算、内存、资源还是等待问题。
3. 指令依赖/流水空泡仍不能解释时，定向采 SQTT，在 Wavefront/Inst/Source 视图连接等待区间、live registers、寄存器依赖与生成 ISA。编译 metadata 中的 stages/warps 不能代替动态观察。
4. 保存原件、命令、指标定义、dispatch、关键观察及下一项实验；工具不可用则记录诊断缺口，不把推测写成实测。

按 [分析深度选择](../../hygon-hip-kernel-optimizer/references/hcu-workflow-contract.md#按难度与进度选择分析深度) 解释实际热点，选有关的上限/融合/流水分析，再提出候选与预期观测；可复用足够的同条件证据，停滞或新反例时重新判断。

### 4.2 按需检查编译后实现

```bash
python <skill>/scripts/inspect_triton_meta.py KERNEL.py --json-out meta.json
python <skill>/scripts/scan_amdgcn.py repro/artifacts --kernel KERNEL --json-out isa_scan.json
```

按 [HCU 指令指南](../../hygon-hip-kernel-optimizer/references/hcu-isa-guide.md) 核对精确 gfx 的指令形式、布局和同步。扫描的 `--kernel` 是实际汇编符号；缺失/歧义保持 unresolved，不用文件名代替。metadata 仅覆盖部分格式，空项不证明没优化；LLVM IR 和静态指令计数也不能替代最终 ISA 或动态热点。

`isa_scan.json` 的 `status` 区分 scanned、partial 和 unresolved；没有解析到指令时 classification 为 null，不能解释为“没有这类指令”。目录可能包含不同配置/构建的同名 kernel，合计值只用于找线索；分析目标性能或生成报告前，选定实际加载的单个汇编文件。源码和 ISA 摘要保留文件哈希，但同名或哈希记录本身不证明两者来自同次构建。

反汇编失败或汇编不完整时，沿该指南的工程导出流程保存本次实际 JIT 特化的汇编/代码对象，核对符号、编译选项及加载关系；若重新编译则重测新产物，不借用旧缓存或旧计时。buffer ops、dot、assume、warps/stages 等前提不明或实现改变时按实际工具链做针对性探针，已有同条件证据可复用；失败不全局强开环境变量。

流水疑点按 [汇编分析指南](../../hygon-hip-kernel-optimizer/references/assembly-guided-optimization.md) 关联热点和依赖。读 ISA 可以只指导 Triton/inline asm/lowering；直接重建对象则走专用 harness，保留 oracle、ABI/资源和多用例验证，并确认分派未被重新编译覆盖。不要直接改共享 JIT 缓存；本仓捕获/replay/扫描工具不是二进制改写或差分验证器。

### 4.3 验证候选并反馈

baseline/variant 同 shape、数据、seed、布局、stream、恢复和测速口径，GPU 串行运行。多用例加性能回退门禁；非连续、alias、空/尾块、量化与路由分布以支持契约为准。

瓶颈修改后做同口径 profile 对照，解释关键指标是否改善、限制是否转移；最终收益取未开启 profiler/捕获的独立测量。

## 5. 回到模型

核对 graph break、额外转换、launch、通信/overlap 与端到端结果，必要时比较最小 eager 区域或模型重写。先证明语义保持和整体收益，再保留方案。报告引用原件、目录/符号、失败与回退，不把 standalone最快配置当整个模型收益。

`make_investigation_report.py` 可用 `--autotune-summary`、`--meta`、`--isa` 生成草稿。日志有多条同名记录时，按摘要中的输入、产物路径及 `record_index` 选择 `--record-index N`（从 0 开始），不默认选第一条。未提供的可选证据留空；显式传入的文件不存在或 JSON 无效则报错。不同目标、多个汇编产物或无法确定范围的字段不自动拼成单 kernel 结论，缺口在 Notes 中保留；确认源码/输入/config/符号与实际 dispatch 的关系后补充解释。
