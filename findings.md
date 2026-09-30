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
