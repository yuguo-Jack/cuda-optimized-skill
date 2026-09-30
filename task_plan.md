# Hygon 算子 Skills 系统复核与上游合并

## 目标与范围

详细熟悉现有工程，确认并合并上游代码，解释新增能力；依据当前 HCU-Knowledge 的知识与证据规则系统完善 Hygon baseline/HIP/Triton Skills 及配套脚本，验证后更新 Codex 用户安装副本。

- 用户已授权上游合并、本地修订与系统目录 Skill 更新；未要求推送远端。
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
