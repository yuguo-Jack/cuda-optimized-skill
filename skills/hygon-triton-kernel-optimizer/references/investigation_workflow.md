# Triton 调查步骤

## 1. 模型热点与契约

原始模型→调用图→生成源码→实际 dispatch 建立对应，记录 Torch/Triton/HCU 工具链与实际使用的资料来源。先用未采集运行保留端到端基线，再采系统热点和必要的单 kernel 指标。

## 2. 独立工作目录

每次实验指定独立 TORCHINDUCTOR_CACHE_DIR、TRITON_CACHE_DIR、TRITON_CAPTURE_DIR；不要清全局缓存。日志/缓存路径按实际运行用户和版本查，不能猜 /tmp/torchinductor_root。

首次 compiled-model 执行前导入捕获补丁。它记录执行前 storage/stride/offset/alias 和源哈希；捕获会增加同步/数据复制，不能拿此轮测端到端收益。无 autotune 不等于异常，先看缓存命中、单配置及实际 dispatch。

## 3. 复现与独立正确性

只重放可信文件。生成 replay 保持布局与同dtype alias，重新取得当前stream handle，但仅做 autotune 时间调查。读取安装版 CachingAutotuner 确认 mutated args 是否在每次试验间恢复；另写独立 oracle，逐配置核对。

混合dtype storage、函数对象launcher参数、不可序列化对象出现 capture-error 时写项目最小repro，不能把一个不完整目录当已捕获。

## 4. 差异与验证

遇到 kernel 优化瓶颈必须先采集分析，不能只继续 autotune 或扫描 ISA。从 [性能分析入口](../../hygon-hip-kernel-optimizer/references/dcu_metrics_guide.md) 选定工具。以下 2–3 项是 [XProf/XCompute](../../hygon-hip-kernel-optimizer/references/xprof-xcompute-guide.md) 的操作；使用 hipprof 时改按 [hipprof 指南](../../hygon-hip-kernel-optimizer/references/hipprof-guide.md)，不能套用 XCompute 的视图和参数：

1. 将生成文件、源码/代码对象、实际 kernel 名、shape/stride/config 与 dispatch 对应起来。初始化、JIT、warmup、输入捕获产生的 kernel 不作为目标；新运行需重新确认 dispatch 编号。
2. 对代表性瓶颈用例采 speed-of-light/compute/memory/occupancy，再按现象补 scheduler/wave-state；在 XCompute 中确认到底是计算、内存、资源还是等待问题。
3. 指令依赖/流水空泡仍不能解释时，定向采 SQTT，在 Wavefront/Inst/Source 视图连接等待区间、live registers、寄存器依赖与生成 ISA。编译 metadata 中的 stages/warps 不能代替动态观察。
4. 保存原件、命令、指标定义、dispatch、关键观察及下一项实验；工具不可用则记录诊断缺口，不把推测写成实测。

检查 current HCU 分支的 metadata/lowering、最终代码对象及资源，提出一项明确假设。buffer ops、dot、assume、warps/stages 每次均按实际工具链编译探针。失败不全局强开环境变量。

baseline/variant 同 shape、数据、seed、布局、stream、恢复和测速口径，GPU 串行运行。多用例加性能回退门禁；非连续、alias、空/尾块、量化与路由分布以支持契约为准。

瓶颈修改后做同口径 profile 对照，解释关键指标是否改善、限制是否转移；最终收益取未开启 profiler/捕获的独立测量。

## 5. 回到模型

核对 graph break、额外转换、launch、通信/overlap 与端到端结果，必要时比较最小 eager 区域或模型重写。先证明语义保持和整体收益，再保留方案。报告引用原件、目录/符号、失败与回退，不把 standalone最快配置当整个模型收益。
