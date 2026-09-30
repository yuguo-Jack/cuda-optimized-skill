# 工作进度

## 2026-09-30

- 读取 skill-creator、planning-with-files、当前 hcu-knowledge-search；确认系统 Python、本机 Skill 根与知识库绑定。
- 完成初始 Git/目录检查：main 干净，origin 为用户 fork，无 upstream remote。
- 创建本轮规划文件；进入详细工程调查和上游核实。
- 读取所有跟踪文件清单、README、三个 Hygon 入口和项目远端工作流；仓内与安装副本一致。
- GitHub匿名API返回403 rate limit，改用本机既有凭据的受控API助手核实fork关系；未打印凭据。session-catchup无额外报告。

- 使用已存在的127.0.0.1:7890代理单次配置获取上游；合入b6662a3、114a6cb，合并提交d1fb612。仅.gitignore冲突，保留Hygon与测试规则。
- HCU联合检索成功（本地及飞书user均可用）；阅读XProf/XCompute工作流、AICC及CK流水线证据，进入脚本/工作流修订。

- 完成三个Skill主入口与共同规范、原始产物/矩阵/报告、工具选择及源码证据流程修订。
- 新增HCU实验门禁、显式workload suite、XProf入口、保留stride/alias的捕获辅助器和备份安装器。
- 首轮26项CPU回归通过；扩展到30项后通过。上游pytest最初Windows默认GBK读取失败，修复测试UTF-8后10项通过。unittest发现0项，不作为通过证据。
- 25个适用CLI --help成功，三个Skill quick_validate通过。Git diff检查发现Python文本写入CRLF导致噪声，已将本任务更改文本规范为LF。
- 继续总体复核、架构证据对照、错误路径测试；尚未同步系统Skill。

- 最终复核补齐：自定义benchmark在主流程及单独branch/ablation/profile命令中沿用冻结路径/哈希；hipprof按branch_results选择准确champion；Triton按实例与输入布局去重，超时覆盖旧成功JSON，显式LLVM IR也排除出ISA统计。
- 最终36项Hygon CPU回归通过，修改脚本compileall通过，git diff --check通过。上游10项测试已通过，25项适用CLI帮助已验证；未运行真实HCU GPU实验。
- 安装到 C:/Users/Administrator/.codex/skills：baseline 5、HIP 30、Triton 15个文件，两边独立SHA256清单一致，三个已安装Skill再次quick_validate通过。
- 本轮最初旧版本备份：hygon_tmp/skill-backups/20260930_094922_076069；最终增量修订前的中间版本也保留于20260930_095111_584600。其他Skill不改动。
- 清除上游误跟踪的20个__pycache__条目（仅Git索引），保留忽略规则；本轮变更以本地提交留存，不推送远端。

## 后续修订：知识库查询改为可选

- 按用户最新要求，三个 Skill 专注 HCU 算子开发优化；当前工程证据优先，需要代码案例/领域资料时才查询，未安装知识库也可使用。
- 删除入口、共同规范、README、模板、示例中的知识库维护流程与强制查询表述，默认提示语只描述算子任务。
- 三个 Skill 格式与 YAML 检查、13 份 Markdown 相对链接检查、git diff --check 通过。本轮仅修改说明，不重复执行 GPU/CPU 算法测试。
- 已同步 Codex 安装目录，50 个文件 SHA256 与仓内一致。旧副本备份于 hygon_tmp/skill-backups/20260930_105543_826993；HCU-Knowledge 未修改。

## 后续修订：瓶颈时的必需性能分析

- 读取已保留的 xprof-guide/xcompute-guide 正文，对照其过滤、sections、PMC/SPM/replay、SQTT和分析视图说明；没有刷新或修改知识库。
- HIP/Triton 入口、共同规范和报告模板明确瓶颈时必须工具采集并分析；收益停滞、回退或原因不明时先诊断，再决定修改。无法采集时记录原因/命令和未完成状态。
- 扩充共享指标指南：设备/kernel/dispatch过滤、定向sections、XCompute Summary/Details/Raw/Source/Wavefront/Inst/Baseline、occupancy区别、SQTT范围、replay off下SPM限制、UTCL2采集干扰及前后验证。baseline仅增加进入优化阶段的导航。
- 三个Skill格式与13份Markdown相对链接检查通过，git diff --check通过；已备份同步Codex，50文件SHA256一致。备份目录hygon_tmp/skill-backups/20260930_110205_818241；仅文档变化，未运行硬件采集。

## 后续修订：工具内容与产物分离

- 核对DTK 26.04.1 hipprof手册及25.04.1 SQTT专项资料，与XProf/XCompute指南分开。通用hipprof手册的output-type不能直接证明SQTT格式，更不能推导XProf导出能力。
- 新增独立xprof-xcompute-guide.md、hipprof-guide.md，共用dcu_metrics_guide只保留选型/通用解释；入口和Triton流程按工具导航。
- XProf/hipprof摘要标记tool，none不再误写xprof；编排警告使用实际选择语境，报告逐轮列出工具/原件。旧无标记产物保持not_recorded，不能猜测。
- 41项Hygon CPU回归通过，包含新增路由/参数/工具身份测试；三个Skill格式、15份Markdown链接、compileall和两个采集CLI帮助及diff检查通过。没有HCU硬件采集。
- 同步Codex安装目录，52文件SHA256一致；备份hygon_tmp/skill-backups/20260930_110944_994939。知识库未修改。
