# Hygon 算子 Skills 系统复核与上游合并

## 目标与范围

详细熟悉现有工程，确认并合并上游代码，解释新增能力；依据当前 HCU-Knowledge 的知识与证据规则系统完善 Hygon baseline/HIP/Triton Skills 及配套脚本，验证后更新 Codex 用户安装副本。

- 用户已授权上游合并、本地修订与系统目录 Skill 更新；本轮进一步要求整体 review 后同步并推送 origin。
- 保留现有 Hygon 扩展和本机配置；源码初始工作区干净，基线 018f9f1。
- 不运行未授权的远端 GPU/集群实验，不把静态验证写成硬件实测。
- 继续使用全文索引和规则排序的 HCU-Knowledge；不引入新的 embedding/rerank 服务。

## 阶段

1. 工程、现有 Skills/脚本、安装副本和上游关系调查 — complete
2. 获取并分析上游差异，合并且检查冲突 — complete
3. 检索当前 HCU 证据，逐项修订 Hygon 工作流、参考和必要脚本 — complete
4. 脚本/模板/Skill 验证与整体复核 — complete
5. 备份并同步 Codex 安装副本，核验安装一致性并总结 — complete

## 错误与恢复

- 不存在的可选 AGENTS.md 已在探查时跳过。
- GitHub匿名API触发403额度限制；使用本机已有GitHub凭据的受控请求，避免反复匿名重试。

## 用户后续修订：简化知识库关系

- 三个 Hygon Skill 专注算子开发优化，HCU-Knowledge 查询按需参考，不是依赖或必经步骤。
- 按用户最新要求，入口、共同规范、模板、示例和默认提示语均移除知识库更新/入库/交接流程；成果留在任务工程。
- 文档复核、格式与链接检查、备份同步安装副本后，以本地提交留存；不修改 HCU-Knowledge。

## 用户后续修订：瓶颈时必须使用性能分析工具 — complete

- HIP/Triton 的 kernel 瓶颈诊断必须使用工具采集并解读，已有同条件有效证据可复用；工具受阻保留诊断缺口，不能由静态猜测替代。
- 从保留的 XProf/XCompute 原手册提炼过滤、sections、指标/时间线/源码视图、replay/SPM 与前后对照流程；无需运行时强制查询知识库。
- 完成格式、相对链接、改动检查及安装副本 SHA256 核对；本轮仅修改流程文档，没有执行 HCU 采集。

## 用户后续修订：分开两套性能工具 — complete

- XProf/XCompute 与 hipprof 分别有独立指南，共同入口只保留工具选择和通用诊断；逐项区分过滤、PMC、SQTT、输出格式、replay/SPM 与查看器。
- 产物记录实际工具，修正 none 误标为 xprof 和报告默认 hipprof 的问题；新增命令路由/标记回归并同步安装副本。

## 最终整体 review、同步与推送

1. 核对入口/规范/模板、运行与状态流、采集隔离和失败路径 — complete
2. 修复实际问题，执行对应回归及整体一致性检查 — complete
3. 备份同步 Codex 安装副本，核对文件 SHA256 — complete
4. 检查 origin 变化，提交并正常推送，核对远端提交 — complete

## 指令集与科学优化流程专项完善（用户追加授权）

- DCU/HCU 同义，保留既有字段与文件名。依据当前 HCU-Knowledge 原件/固定源码修订 Skill，不更新知识库。
- 重点覆盖 gfx936/938/946 指令、builtin/ISA 区分、实验对照、数值正确性、多规模回归、性能与机制归因，以及脚本实际行为。
1. 查阅原件/源码并记录精确适用边界 — complete
2. 修订指令/策略参考与相关脚本，补齐科学实验约束 — complete
3. 回归、整体 review、安装副本核验 — complete
4. 同步 Codex 并提交推送，确认远端一致 — complete

## 2026-09-30：知识库 v0.5.0 后的算子 Skill 同步

1. 核对七个架构名称、Builtin/指令/测试的适用边界 — complete
2. 修订 WDRA/Ebarrier、WASP 与 hipprof 26.10 指南及脚本 — complete
3. CPU 回归 92 项、33 个脚本 AST、三个 Skill 校验与相对链接检查 — complete
4. 备份安装副本、同步哈希核验、提交推送 — complete（主体 f5daf22；远端 main 与本地一致）

保持知识查询可选，不加入知识库更新流程；瓶颈诊断必须采集并解读，hipprof 与 XProf/XCompute 分开。未运行 HCU 构建/设备测试。

## 2026-09-30：实际工程汇编回退规则

1. 核对现有反汇编/通用重编译路径及产物契约 — complete
2. 在共享指令指南和 HIP/Triton 入口补充真实工程构建、汇编留存与重新测试规则 — complete
3. 文档/Skill 校验与安装副本同步 — complete
4. 提交并正常推送，随后核对远端 SHA；发布结果在本轮完成回复中报告。

## CUDA/HCU 优化 Skill 对照 review

用户要求：分析当前 CUDA 优化 Skill 全流程，系统复核并完善 HCU 优化 Skill；更新过时流程图，检查后同步安装副本、提交 push。

1. 对照入口、策略/数值契约、构建、实验门禁、采集、归因、恢复与交付 — complete
2. 修复有证据的流程/实现缺口，补充必要文档与真实失败边界回归 — complete
3. 更新 HCU 流程图与示例，确认图/文/脚本一致 — complete
4. 整体验证、备份同步安装副本、提交 push 并核验 — complete（主体 74f7b36 已确认 origin/main 一致；随后提交本完成记录）

沿用 HCU 原有科学优化流程：查询知识库可选；瓶颈必须性能分析；两套分析工具独立；无硬件可继续静态准备，但不冒充实测。不照搬 CUDA 的硬件名称、硬停止政策或模型工具参数。

检查边界：本机无 HCU Triton 运行时，triton_benchmark_template.py 的 --help 在顶层 import triton 时失败；该硬件模板只做 AST 检查，其执行须在项目要求的 HCU Triton 环境完成，不用普通 PyPI 后端替代。
