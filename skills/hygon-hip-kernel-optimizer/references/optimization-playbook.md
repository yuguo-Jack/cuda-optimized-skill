# HCU 优化实验操作指南

用于制定候选、排查失败和交付验收；方法 ID 与精确目标约束仍以 [策略目录](optimization_catalog.md)、[注册表](method_registry.json) 为准。记录可核查的证据、决策及限制，不要求输出隐藏推理过程。

## 1. 先确定值得优化哪一段

先从真实调用链确认耗时占比、调用次数、输入分布及当前分派，再建立最小复现。分别记录单 kernel、完整算子和端到端目标；一个 kernel 快了不保证模型快了，融合还可能增加转换、workspace 或通信开销。

- 检查生产实际走的分支：dtype、shape、stride、对齐、gfx、库版本、环境变量、autotune/JIT 特化及 fallback。先确认改到被调用的代码。
- 小 kernel/小网格先检查启动、同步和网格覆盖；大 GEMM 检查供数、矩阵计算、尾块及 epilogue；多 kernel 链检查中间写回与依赖。用实际 trace/指标确认，不按算子名字宣布瓶颈。
- 在 `analysis.md` 给出：当前输入与 kernel → 观察及原件 → 候选修改 → 预期变化 → 推翻假设的结果。每轮只选择有前提和验证办法的方法；无合适候选可停止，不为填满 methods 预算制造改动。
- 比较 before/after 时保持工具、采集范围、指标组、replay/缓存策略一致；换工具/配置后需重新建立可比较的基线。自动记录原件不等于已完成 XCompute/hipprof 解读。

## 2. 按算子类型展开候选

这些是条件化调查方向，不是自动支持表；只读当前任务相关行。表中的算法改写若超出现有方法 ID，应先明确实验定义和验证方式，不能把整个算法加速强行归因给某条 ISA。

| 场景 | 首先核对 | 有证据后可尝试 | 必须保留的验证 |
| --- | --- | --- | --- |
| GEMM / batched / grouped GEMM | M/N/K、转置/stride、batch/expert 分布、矩阵指令供数、尾块、epilogue | tile/stage、寄存器预取、LDS/MLS 布局、分组调度、split-K、epilogue 融合；每次明确一个变化 | 累加精度、alpha/beta、非整除 K、空任务、workspace/归约成本；tile/stage 增加后的 VGPR/LDS/spill |
| Attention prefill | Q/K/V 布局、causal/padding、softmax 和中间矩阵写回 | 分块在线 softmax/重计算、QK 与 PV 流水、融合、供数与同步调整 | 缩放及 exp 误差、整行被 mask、极端 logits、长序列、前向/反向契约；不凭少量随机输入放松容差 |
| Attention decode / 稀疏 attention | KV 页表/stride、batch 与上下文长度、稀疏块分布、工作不均衡 | KV 访问布局、split/reduction、稀疏块调度、持久化/分组处理 | 空块、重复/乱序索引的定义、页边界、尾块、多 query/KV head 关系；索引构建与合并也计入算子时间 |
| MoE | router → sort/gather → grouped GEMM → combine → EP 的真实占比 | 改善 token 局部性、分组/持久化调度、量化与 epilogue 融合、减少冗余拷贝 | 空 expert、极端 skew、top-k 重复、capacity/drop、scale/路由权重、scatter 冲突、反向；整体不能只测最快 GEMM |
| Reduction / norm / softmax | 归约轴、连续性、行长、小网格、跨 wave/block 合并 | 每线程聚合、wave 交换、LDS 合并、不同大小的分派；按数值契约选择稳定归约算法 | 求和次序、累加精度、方差消减误差、epsilon、NaN/Inf、tail/空维度、确定性；避免用更高 occupancy 掩盖工作量增加 |
| 量化 / 低精度融合 | 输入/累加/输出格式、scale 粒度、packing、饱和/舍入语义 | 减少 unpack/转换与中间写回、融合 dequant-GEMM/epilogue、目标支持的矩阵路径 | zero point、scale 生命周期、极值/零值、非整除打包；更改精度或校准方式必须符合用户既定目标 |
| Elementwise / 布局 / 卷积 | 广播、alias、alignment、访存合并、layout 转换成本 | 向量化、循环合并、直接写目标布局、融合、复用已有 HCU 库实现 | 有效对齐与 tail、in-place 依赖；卷积 padding/stride/dilation/groups 与 workspace；TLS/im2col 需目标探针 |
| 通信与计算重叠 | 真实 rank/stream 依赖图、生产消费顺序、通信完成语义 | 缩小同步范围、分块流水、减少中间拷贝、合法的通信/计算并发 | 原始 rank 数和拓扑、重复运行/进度、buffer 生命周期与可见性、最慢 rank 时间；profiler 可能改变调度 |

从高层到低层推进：先减少无效工作/数据移动，再选 tile/layout/pipeline，最后有明确瓶颈和最小探针时使用 Builtin/汇编。复用当前 HCU 库时保留版本和实际分派证据；不因 NVIDIA 存在 TMA、TMEM、warpgroup 或 graph 优化就推导 HCU 有同一接口。

## 3. 数值变换的前提

在 `contract.md` 固定参考实现、有效 atol/rtol 或误差指标、累加/输出类型、NaN/Inf/舍入/饱和/确定性政策。默认沿用 reference；通用 benchmark 支持 reference 的 `atol/rtol`，其默认值不代表用户允许降精度。

| 变换 | 先确认的前提 | 主要反例 |
| --- | --- | --- |
| reciprocal/近似 exp、fast math、FMA 收缩 | 用户允许的误差与输入值域；精确/逐位要求下不能擅自使用 | 近零、溢出/下溢、特殊值、累计误差 |
| split-K、原子合并、归约重排 | 求和次序和非确定性是否允许，独立稳定 oracle | 大 K、严重抵消、跨 block 累加、反复运行漂移 |
| FP8/BF8/FP4 等路径 | 准确 gfx/工具链、格式与 scale、累加精度、用户既定精度目标 | 饱和、极端 scale、packing tail、输入分布偏移 |
| 融合/在线重计算 | 中间舍入点、数值稳定性、mask/epsilon 与原语义一致 | 全 mask 行、训练反向、alias/in-place、不同输出 dtype |

容差通过不是授权改算法语义。若契约要求 bitwise，使用专用比较器；不要把 `allclose(atol=0, rtol=0)` 自动当逐位相等。专用 benchmark 可记录 `numerical_policy`，配对对照要求两边一致；计算方法允许变化时在方法分析记录，验收政策仍保持相同。

## 4. 构建与运行资源

- 当前 HCU `branch_explore.py` 串行构建/测试，不支持 CUDA Skill 的 `--compile-jobs`。确实受构建耗时限制时，可在工程专用构建系统按主机 RAM/CPU 限制构建并发，所有 GPU 正确性、测速和 profiler 仍按设备串行。不同 run/进程之间没有自动 GPU 锁，需由任务调度保证设备独占。
- 编译 OOM 与设备 OOM 分别记录。编译资源不足可降低构建并发；设备不足可调整实现/workspace，不能缩小原 shape 后报告成功。不无限重试相同命令，记录改变的原因。
- 缓存键至少区分有效源码/头文件/生成文件、编译器与目标、宏/include、优化/设备链接参数和依赖版本。不能只比较入口源码 mtime；构建缓存可以复用，旧计时不能代替新鲜对照。
- 多文件工程使用自己的 benchmark 和真实构建方式；通用独立源码 harness 不保留任意工程的相对 include/链接关系。按 [实际工程汇编处理](hcu-isa-guide.md) 保存设备汇编与本次实际加载代码对象。
- Triton/Inductor 调查保留实际生效的 autotune 配置与特化，比较单个 kernel 时固定配置；JIT/autotune 时间单独报告。最终再用生产分派/缓存模式验证端到端效果。

## 5. 失败、恢复与停止

| 现象 | 下一步 | 不能据此推导 |
| --- | --- | --- |
| 无设备/编译器/权限 | 列实际阻塞和待执行命令，继续独立静态准备 | 已经运行或验证 HCU |
| 编译/ABI/正确性失败 | 看本分支原始日志，修复后在尚未关闭的轮次重试；保留失败记录 | 整个方法族无效或其他 shape 正确 |
| 不稳定计时/收益落入噪声 | 检查进程干扰、热身、温度/时钟、缓存和计时范围，必要时另开更长实验 | 选出的最小值就是收益 |
| 四轮确认未改善 | 保留当前 best，研究回退用例或停止；新假设才开下一轮 | 筛选 champion 可以直接晋级 |
| 采集失败/原件为空/指标不适用 | 修复工具配置，或按实际目标选择另一工具并重采基线；记录未完成诊断 | 命令 exit 0 或有 CSV 就分析完成 |
| ISA 不可读/不完整 | 真实工程构建导出、符号与产物绑定；仍缺失则保持限制 | builtin/IR/正则命中已证明机制 |
| 缺消融或机制复核 | 更快且正确的实现仍可保留，具体方法记 unverified | 每个方法都贡献了总加速 |

`orchestrate.py` 将其调用步骤的退出状态追加到 `stage-results.jsonl`；直接运行的独立脚本以自身产物/日志为准。退出码只反映命令执行，原始采集仍需分析。失败轮次可修复后重试；先复制需要比较的旧 `bench.json/branch_results.json` 与日志到独立 attempt 目录，再重跑，避免覆盖调查材料。已关闭轮次进入新轮，冻结的 reference/benchmark/best 被修改则新建 run。`finalize` 可以为失败/中止的 run 生成总结，写清停止原因、保留 best 和未完成项。

达到目标、没有足够证据的新方法、重复实验证明收益停滞或预算耗尽时收尾。近峰值需要 workload/traffic/可达峰值依据；当前 `roofline.py` 的预算是建议，不自动证明达到硬件极限。

## 6. 最终工程验收

把候选接回真实分派：核对命中优化路径和合法 fallback，跑工程单测、真实 shapes/数据分布及必要的多 rank/stream 检查，再测端到端时间、workspace/峰值显存和 cold/warm/JIT 成本。保留构建/运行/复现命令、依赖、代码差异、产物哈希、回退办法和未验证项。

程序 `summary.md` 汇总实验，不自动证明 race safety、诊断解释或项目验收完成；按 [报告模板](../templates/iteration_report.md) 补这些人工分析。知识库查询仍为可选参考，成果保留在任务工程。
