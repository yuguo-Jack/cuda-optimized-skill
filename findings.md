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
