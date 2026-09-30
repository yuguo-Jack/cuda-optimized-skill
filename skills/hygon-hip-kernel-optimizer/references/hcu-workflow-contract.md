# HCU 算子开发与优化：共同契约

baseline、HIP/CK Tile、Triton 三个 Skill 共用本规范。按问题读取相应章节；Skill 负责开发与实验流程，HCU-Knowledge 是可选参考来源。未安装或未查询知识库也可完成开发优化；不能把本文件中的候选策略当作硬件兼容表。

## 1. 证据来源与按需知识查询

优先读取用户提供的材料、当前工程源码/测试/文档、目标环境的头文件、工具帮助和实验结果。已有证据足够时直接推进，不为完成流程而查询知识库。需要查找可复用代码/优化案例、补充领域事实或解决证据冲突时，可以按需查询 HCU-Knowledge；不要求每次任务或每轮迭代都查询。

以下检索步骤仅在决定使用知识库时适用：

1. 如果已配置 `hcu-knowledge-search`，按其说明查询；未安装、未绑定或不可访问时继续利用其他可用证据。只有任务所必需的事实确实无法查明时才说明缺口。
2. 按精确算子、符号或现象检索。保留搜索状态，不能把在线失败当没有相关资料；认证或访问问题按搜索 Skill 处理。
3. 复杂问题拆开查：算子/调用链、数据布局、硬件特性、编译器、工具指标、类似案例。先工程 `local/overview/engineering-guide.md`，再 topics/cases 和源码。用 `cases` 与案例索引查算子、手段、硬件和输入规模；未命中不等于不支持。
4. `read` 读全文，`original` 看原 PDF/指令表/图，`read-code` 固定源码提交。父仓子模块按 gitlink 固定子提交。记录 ID、SHA、路径/页码、适用条件；pending 仍可查，但兼容性待证。源码、PR review、文档描述和本次实测分开记录。

选择参考源码时，同用途且同步的实现优先 HYGON-AI GitHub；内部 GitLab 有实质领先则用内部活跃 HCU 分支。新 CI 时间不等于新实现；默认分支可能落后。DeepGEMM 通用发版方向与 MegaMoE 方向分别判断。HCU rocBLAS、hipBLASLt、MIOpen、RCCL 的不可见源码不能用 AMD 上游代替。NVIDIA/AMD 官方资料仅作标明厂商的对比参考；同名 API 或相邻 gfx 编号不证明兼容。

可选检索入口：`xprof-xcompute-workflow`、`hcu-performance-workflow`、AICC engineering-guide、CK Tile example/cases、FlashAttention、DeepGEMM、BoltOPs、Triton 的当前工程指南，以及数学库 `gemm-assembly-examples` 原始示例。

## 2. 目标环境与执行位置

- 北美洲=gfx936、南美洲=gfx938、少伯=gfx946。少伯新增 Abarrier/Ebarrier/MLS 扩展/TLS/WDRA 等资料仅用于 gfx946；gfx936/gfx938 本身也有 MLS 指令或源码命名族，不能反过来禁止这些既有路径。逐条核对 descriptor、opcode、同步与数据布局，不能从名称继承少伯新增语义。少伯指令资料还描述 FP4/FP6 与带 scale 的矩阵/转换路径，不再沿用旧 Skill 的全局“无 FP4”判断；实际支持仍需精确工具链探针。
- gfx 是标识符，不是可用 `>=` 比较的能力等级。记录真实设备、CU 数、wave width、驱动、DTK、编译器路径/版本、PyTorch/HIP、Triton、库提交、工具版本、环境变量与运行方式。
- AICC 是独立编译器，区别于 DTK 自带工具链。保留准确二进制与安装来源；同一源码切编译器也要重做正确性、代码对象、资源和性能对照。`--hipcc-bin` 只适配支持相同参数的驱动，不能自动把任意 AICC 二进制当 hipcc。
- Agent 可在本机编辑、检索和读产物，构建/执行/采集在实际 HCU 节点。按项目配置和远端工作 Skill 决定 SSH、容器、挂载与 DTK 激活；不统一要求 Docker，也不统一禁止 Docker。不把本机 NVIDIA GPU 测试当 HCU 测试。
- 探测工具存在只是发现能力，不证明目标架构支持、权限或采集成功。读取已安装 `--help`，对必要接口做小型编译/运行探针。无硬件时仍可写代码与做静态检查，明确标记待硬件验证。

## 3. 问题契约与正确性

先写 `contract.md`：运算数学语义、所有输入输出 shape/dtype/stride、广播、padding、mask、量化尺度、索引范围、alias/in-place、stream、原子与确定性要求，允许的误差和目标场景。维度参数不都是 shape；alpha、负 stride、空张量等不能被通用脚本擅自解释。

- 原 reference 是独立 oracle，不随候选同步“修正”；调整精度/近似策略必须符合用户目标。保留 reference 哈希。整数/布尔精确比对，浮点保留实际精度；NaN/Inf 单独规定，不能用大容差掩盖逻辑错误。
- 覆盖典型、小/大、非整除尾块、退化形状、稀疏/路由不均、边界索引以及多 seed。attention 分 prefill/decode、causal/padding、KV cache 和稀疏模式；MoE 分路由、token sorting、GEMM、合并与通信；并非每个实验都适用全部维度。
- 本仓 flat HIP harness 只支持独立、连续、同一 per-pointer 容量的简单 ABI，非 const 指针视作输出。必须显式 `--ptr-size`，且覆盖每个指针真实索引范围。FP16/BF16/FP8、不同布局、in-place、alias、多个 stream、自定义 scratch/通信必须使用任务自己的 `--benchmark` 适配器；不要为了套 harness 改算子语义。
- Python harness 也只支持独立连续输入；复杂布局用专门适配器。`preflight` 会导入 reference/adapter 并执行顶层代码，只对已审查的当前任务文件使用，它不是沙箱。
- 正确性与 race safety 分开。检查越界、写覆盖、同步、跨 block/设备通信；使用目标实际支持的检查工具并保存命令。不能因一次 allclose 通过宣称无竞争。
- 资源不足是该用例未完成，不偷偷缩小 shape 后宣布原场景成功。需要缩小只作为显式命名的补充用例。

## 4. 测速与多规模比较

- 基线、候选和消融保持相同设备/环境、输入、seed、计时范围、缓存策略及 warmup/repeat。单 GPU 串行测试，避免 profiler、其他候选或训练进程干扰；编译可按资源并行。
- 区分编译/JIT/autotune、kernel device time、host launch、拷贝、完整算子/模型端到端时间。复杂多 stream 用明确同步契约或任务专用计时器。
- 通用 HIP benchmark 每次调用前恢复原状态，恢复在计时间隔外；保留真实 event 样本、均值、中位数、p90、CV。默认至少 5 个样本且 CV≤10% 才可用于晋级；边缘收益需更长测量、交错基线/候选及噪声分析，固定 2% 阈值不是统计显著性证明。
- 逐次同步/恢复会改变缓存状态，短 kernel 的 event/host 调度也会有扰动。若需连续 steady-state 或 graph replay，使用专门 harness，保存该口径，所有版本一致。不要混用两类数据。
- `workloads.json` 是显式回归矩阵。`orchestrate setup --workloads ...` 将内容冻结到 state；候选对每个 case/seed 串行重测 baseline 和 candidate。全部正确且稳定、每例不超过自己的回退上限才参与加权几何平均速度比选择。配置文件中的单一案例只支持单一范围的结论。
- 不传 `--workloads` 仅执行主 shape。交付可复用 kernel 前必须补该功能支持范围的矩阵；报告明确实际覆盖。真实数据分布、量化与路由由专门 harness 构造，而不是随便随机一个 int 张量。
- 分配字节数/时间是 footprint rate，不是 HBM 带宽；实际 traffic 还涉及缓存、复用、读写次数、压缩、写分配。Roofline 需要计算量、流量及相同 dtype/设备/时钟的可达峰值依据。

## 5. 性能分析层次与工具

1. 应用系统时间线：找 kernel/launch 热点、CPU、通信、同步和 overlap。先明确优化单 kernel 还是端到端。
2. 优先按 HCU 专业工具资料使用 XProf 采集、XCompute 分析；现有 hipprof/DTK 路径仍可用。`profile_hcu.py` 默认 auto，XProf 存在时使用其已确认参数；否则进入 hipprof adapter。可以显式选择 xprof/hipprof/none。工具失败应记录降级，不等于性能验证完成。
3. XProf `.perf` 保留原件；在 XCompute 中选择真实 kernel/dispatch/设备再读计算、内存、occupancy、scheduler、wave state。默认自动脚本只保存原件，不臆造 `.perf` 解析器或 JSON 导出接口。手工整理的标准化值还须附单位、指标定义、采样范围、工具版本与原始产物。
4. hipprof CSV 目前仅提供指标发现与原件保留；多个 dispatch 的聚合值不能作为目标 kernel 利用率。采集 reference、分配/恢复、warmup 时可能产生额外 kernel，必须按 kernel 名、dispatch 或应用采集区间定位。
5. `SQ_WAVES` 累计波数≠驻留 wave。理论 occupancy（资源上限）、实测活跃/驻留、网格是否覆盖 CU、issue/依赖等待分别分析。高 occupancy 不自动等于快，低 occupancy 不自动需要降寄存器。
6. 原始 PMC 计数≠百分比，静态 waitcnt/branch 次数≠stall fraction。`roofline.py` 未知项输出 null，只给搜索预算建议；不凭猜测峰值自动宣布 near-peak 或停止。
7. SQTT 用于普通数据不能解释的问题，按具体硬件/工具支持选择 SE/CU 和范围，记录开销与漏采。系统 kernel 时间线、PC sampling、SQTT 内部指令时间线不同；没有确认导出功能时保留原始格式，不许虚构通用 timeline JSON。
8. 通信/跨进程同步 kernel 的 replay 可能改变正确性；查当前工具关闭 replay 的选项，采用真实多进程测量。缓存/UTCL2 等指标也可能受 profiler/replay 干扰。

## 6. 实验与归因

先提出可证伪的瓶颈假设，再选择 1..3 个方法；三条预算是上限，不要凑数。同组 branch 保持方法组合一致，仅改变一项超参数。新组合开新轮；有理由时可重试此前方法，写明输入、实现或瓶颈如何改变。

记录链条：问题与环境 → 热点与瓶颈证据 → 为什么这样改 → 文件/符号与调用链 → 最小差异 → 正确性矩阵 → 未采集性能 → 同口径 profile/ISA → 失败条件/回退。

- ISA grep 只是模式出现提示。定位实际符号/目标代码对象/dispatch，并检查源码与加载二进制一致；编译器 `-save-temps` 是重新编译的辅助证据，不自动等于执行过的二进制。
- 不存在一条普遍适用的 ISA 就能证明 fusion、coalescing、occupancy、流水重叠或消除竞争。按方法选择源码、资源、timeline 和计数器等证据。
- 有效消融必须保持语义、通过正确性和稳定计时；消融失败不能证明某方法“性能必需”。没有消融或机制证据进入 `unverified_methods`，可保留更快且正确的 kernel，但不能给每个方法分配收益。
- `isa_check.json` 保留自动扫描；复核后另写 `mechanism-review.json`，不要篡改原件。包含候选 `source_sha256` 和每个方法的 `id/status/explanation/artifact/artifact_sha256`；status=verified 必须有可读、哈希一致的证据。`state.py` 结合正确消融判断有效性。
- 新一轮源文件、reference、工具/目标变化后重新验证。已关闭迭代不可覆盖，失败迭代可修复后重试。单人串行操作一个 run，state.json 当前不支持多 Agent 并发写。

## 7. 实验留存

工作目录内保留 contract、实际使用的资料引用、env、原 reference、baseline、候选 diff、编译命令/日志/二进制 SHA、工作负载矩阵、原始计时与 profile、机制复核、summary。知识库 ID 仅在实际引用时记录。`hygon_tmp/` 仅为本仓可清理临时区；重要报告与原件应转存到任务工程，再清缓存。

脚本/Skill 更新后旧 run 不自动升级成已验证；需要新结果或标明历史状态。依赖接口不明确时先查目标安装源码/`--help`，仍不清楚再按需查知识库或官方资料。
