# 调查发现

## 初始状态

- 本地 main 跟踪 origin/main，origin 为 yuguo-Jack/cuda-optimized-skill；尚未配置 upstream。
- 基线 018f9f1（Sync Hygon HIP skills with hcu-knowledge-search）；工作区初始干净。
- 项目包含 NVIDIA 通用优化技能、Hygon 扩展和临时产物目录；进一步按 Git 跟踪文件核实范围。
- HCU-Knowledge 绑定 D:/codex/projects/HCU-Knowledge；搜索采用全文词项/规则和飞书联合搜索，知识需按硬件/版本/固定源码判断。
- 三个 Hygon Skill 的仓内与 Codex 已安装副本逐文件一致（5/23/13个有效文件），没有待保留的安装端差异。
- HIP 路径提供环境探测→baseline→profile/roofline→分支选择→消融→ISA→状态/总结；baseline 生成器基于 AST；Triton 独立处理 Inductor捕获、autotune 与 AMDGCN。
- 现有 HIP 入口含固定 wave64、固定 PMC 命令、三轮无收益后 SQTT、全局否定 FP4、gfx936/938 builtin 和仅凭 ISA 模式判定实现失败等规则，需要按硬件/工具实际版本复核，不能简单延续到 gfx946。
- 源码树中 HCU 搜索 Skill 相对链接指向未随仓分发的兄弟目录；安装后存在但仓内使用需明确用户 Skill 查找方式。

## 待重点核查

- 上游来源与新增优化/验证能力；Hygon 分支是否存在漂移或脚本故障。
- gfx936/gfx938/gfx946 范围、DTK 与独立 AICC、HIP/CK/Triton 路由。
- 正确性、benchmark 口径、profiling 与 ISA 证据链，以及本地与远端环境边界。
- 知识检索/原件/固定子模块、权限错误区分与案例回写接口。

## 实质问题

- GitHub已认证API确认上游为KernelFlow-ops/cuda-optimized-skill。失效本机代理127.0.0.1:7893用单次Git直连覆盖恢复，未改全局配置。
- HCU-Knowledge已有其他工作产生的catalog及固定源码证据变更，本任务只检索读取，不提交或覆盖。
- HIP roofline将访问次数/等待计数当百分比，并包含无来源硬件峰值；state对缺失正确性与ISA证据默认通过；benchmark复制同一批次均值冒充样本，整数/双精度验证转float32，空输出可通过。需修正并回归。
- 远端执行说明互相冲突；应由当前环境/远端工作Skill确定容器与DTK/AICC，不硬编码个人节点。

## 上游与方案决定

- 上游两个提交 b6662a3/114a6cb：多规模矩阵、内存预算/OOM跟踪、契约/编译/数值/race/时序状态、统计/CV/置信区间、候选多案例门禁、串行GPU与并行编译、NCU采集降级与硬件停止门禁。
- 以d1fb612合并保留NVIDIA流程。HCU采用显式矩阵，不自动缩小真实shape；保留无profiler时可开发/测速但不能虚构机制的边界。
- 原始计数->百分比、无消融->有效、缺ISA->有效、失败消融->必需等均修正。增加真实样本、CV/源码哈希、旧结果清除与非零退出门禁。
- baseline参数转发修正，未知模板阻止编译；Triton执行前快照保留storage/stride/offset/同dtype alias，runner不再默认强开buffer ops。
- HCU知识依据：xprof-xcompute-workflow（XProf/XCompute为优先路径）、AICC工程指南（独立工具链）、CK GEMM流水与总览（prefetch资源、分派范围）、mls-wdra-generation-boundaries（gfx946扩展与旧MLS区分）、少伯part1-指令.pdf（ID bbd11339f2a4a17c333b1493，原件SHA 2b08b4f2206310a1c3673454b86eb9e5370eba4f04a60f311ec68636bce8cc60）。
- 特别避免新误判：MLS不是少伯独占，gfx936/938已有形式；少伯扩展单独约束。少伯文档已有FP4/FP6与scale-aware路径，不沿用旧全局禁止规则，未在硬件验证。
- 工作流程统一为知识/契约→基线/矩阵→profile假设→候选→回归/机制→端到端→案例交接；同仓总览/专题/案例更新一致性明确。

## 验证边界

- 本机系统PyTorch为2.11.0+cu128（NVIDIA），本轮只运行CPU数据/流程测试和CLI检查，不声称HCU编译或性能验证。
- XProf仅自动保留原始perf，按dispatch的语义提取需Agent/XCompute复核；hipprof CSV仅发现聚合，不直接计算利用率。多stream/量化/非连续复杂ABI用项目专用benchmark。
- HCU注册表删除未经证明的跨代feature map和典型加速比，gfx编号不作数值能力继承；保留历史方法ID。

- 36项CPU回归覆盖旧JSON复用、统计/精度、矩阵回退、准确champion选择、未知归因、布局与alias快照、输入特化、超时、专用benchmark冻结、安装备份及隔离。源码引用文件本身固定哈希；外部依赖/头文件/工具链仍需按共同契约另存版本，本轮没有硬件证据。

## 最终整体 review（2026-09-30，用户授权同步并推送）

- 三个入口、共同契约、工具独立指南、schema/报告、脚本与安装器一并复核。按用户最终要求，查询知识库是可选参考，算子 Skill 不含知识库维护流程；瓶颈必须使用性能工具，hipprof 与 XProf/XCompute 分别解释。
- 发现并修复：历史空 method_names 导致总结失败；独立 open/benchmark/branch/profile/ISA/ablation/roofline 入口可覆盖已关闭迭代；重复 seed 可重置最佳计时并覆盖基线证据；独立 benchmark 按后缀选到旧文件。
- 从 state 创建时固定 baseline/reference/benchmark；验证候选和矩阵用例的 reference SHA，不再接受缺失值。suite 汇总增加源码与用例身份，拒绝不匹配的旧结果。旧 run 缺少新证据不会自动升级，按新流程重新测量。
- 修复 max_regression_pct=0 被错误拒绝；报告分别标出主用例加速与矩阵选择指标，矩阵优胜不代表每个用例都等比例加速。
- 旧 ISA regex 的 verified 不作为语义证明；真正归因仍需有效消融以及绑定源码/产物哈希的 mechanism-review。移除 registry 残留的全局 no_fp4 字段；专门保留 gfx946 低精度路径的适用边界。
- Triton 捕获不再静默丢弃超出 arg_names 的位置参数；产物收集避免同名文件在同毫秒覆盖，提前拒绝输出目录位于源缓存内部。
- 58 项 Hygon CPU 回归和 10 项上游回归通过。实进程模拟覆盖 init→seed→open→多 seed/shape 分支选择→close→finalize、回退候选淘汰、关闭迭代保护和重复 seed 保护。模拟时间只用于流程验证，没有 HCU 性能含义。
- 3 个 Skill 格式、30 个 Python 文件语法、4 个 JSON、3 个 YAML、31 处本地 Markdown 链接、25 个适用 CLI 帮助均通过。硬件编译/采集和不同目标 TorchInductor 私有 API 兼容性仍需目标环境验证。

## 指令集与科学优化专项调查

- DCU/HCU 同义，既有字段/文件名不需要替换。此次只读 HCU-Knowledge，保留其既有未提交变更。
- hybrid 搜索本地和飞书 user 均成功；部分查询有更多分页，本轮定向查证而非全量资料更新。原件内容优先，飞书搜索片段不作为已读正文。
- 当前扫描器全文件 regex 会误计注释、宏或不相关符号；sass_check 还混合 dump 日志/文件，且通过相邻文件名寻找 .so，不能证明它是被测二进制。需分别固定代码对象身份和符号范围。
- 注册表仍有固定 wave64 建议、仅 hipprof 的 SQTT 路由、耦合方法全部禁止和源码接口混入 ISA 线索的问题，需按实际语义纠正。
- 科学实验需进一步避免历史基线漂移与选最快候选偏差；正确性输出初始化、实际输入/测量范围、成对对照及独立复测要检查实际脚本实现。
- MLS/WDRA 原文本含跨代对照表和图示；文本表格对齐不足不能推断逐型号支持格子，后续取原 PDF 关键页核对。

### 专项复核结论

- 对 MLS 第 4–5 页、WDRA 第 6/23 页、TLS 第 15 页做了原页核对，保留指令资料、barrier 指南、GFX938 Builtin 表及 GEMM 汇编的 ID/原件 SHA。新指南区分描述、编译、实际代码对象与硬件验证。
- 修复输出预填零可能掩盖漏写；参考适配器可声明有效输出前缀，整数/布尔双哨兵，检查输入与 padding。仍不替代越界/race 工具。
- 每次 HIP 构建保留独立路径与 binary/source SHA；扫描只认实际指令，排除注释、宏、IR/源码名称与无关符号。真实 gfx936 GEMM 样本识别 1 个符号、16995 个静态指令位置、352 个 MMAC、66 个矩阵 DS 读位置；不解释为动态次数。
- 分支筛选后四轮交替独立重测当前 best/candidate，绑定输入指纹、设备/签名和测量参数；按矩阵逐例回退门禁与逐轮加权速度比判定。state 重读哈希固定的原始记录，历史单次时间不能直接晋级。
- 消融改为同参数配对重测，方向不一致/噪声/失败均不推导有效性；有效归因同时需要原始配对证据与独立机制复核。诊断本身不算代码优化。
- 修复 gfx92a 等字母后缀标识解析；耦合方法有可分离差异和验证计划时允许同选；SQ_WAVES 不再作为低 occupancy 自动触发条件。
- HCU-Knowledge 只读，既有 catalog/evidence 未提交改动保持原样。新指南不携带原始 PDF 或汇编全文，不绑定本机知识库路径。

## 2026-09-30 本轮复核

- 七个名称映射只表示目标身份，不代表指令能力继承；保留完整 gfx92a。DCC 的基础 Ebarrier/s_set_vgpr_size 声明与 Triton gfx946 WASP 门禁分开，初始化接口随版本核验。
- hipprof 26.10 命名 PMC 组与旧 read/write flags 分开；波驻留、L2请求、MMOP计量不能混作理论 occupancy、HBM流量或全部VALU算力。
- 修复旧脚本通过相邻 .so 找 code object 的风险：只接受本次 benchmark 构建收据且源文件/二进制哈希一致；SQTT 返回失败、无可读产物或解析失败都标记 degraded，PMC成功不覆盖它。
- SQTT 解析排除 benchmark receipt；工具 help 只记录参数发现，不再用未核实的 --list-basic 推断设备可采集。
- 回归验证了 stale binary、错误收据类型、命名组/旧组互斥、SQTT失败/缺产物、准确gfx与未知目标；无硬件性能结论。

## 实际工程汇编回退规则

- 现有 sass_check 在未识别到指令时以通用参数重编译并保留中间汇编，不能完整复现任意工程配置，部分反汇编成功也不保证目标 kernel 完整。
- 用户要求沿用现有流程，补充工程级可靠处理方式：保留真实编译与链接配置，导出对应阶段设备汇编，核对实际加载产物并重新验证；Triton 同样绑定实际 JIT 特化。仅补充 Skill 指引，不改变脚本和状态协议。
