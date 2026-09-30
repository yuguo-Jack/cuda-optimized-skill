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

## 最终整体 review、同步与推送

- 用户明确授权本轮 review 后同步并推送。origin 已获取，本地相对远端 0 落后 / 7 领先，保留此前上游合并与用户逐次修订。
- 完成状态生命周期、候选选择、输入与矩阵证据绑定、历史报告空名、旧 ISA 归因和 Triton 产物收集修复；具体发现写在 findings.md。
- 最终 58 项 Hygon CPU 回归、10 项上游测试通过，3 个 Skill 格式与全部适用静态/CLI 检查通过。未执行真实 HCU 编译或 profiler 采集；HCU-Knowledge 未修改。
- Codex 安装副本已备份至 hygon_tmp/skill-backups/20260930_112556_951678 并同步，baseline 5 / HIP 32 / Triton 15 文件经独立 SHA256 核对一致；无多余旧文件，安装目录格式检查通过。
- 推送前再次获取 origin，确认无远端新增提交；提交本轮修复并正常推送，随后核对远端 SHA。
- 修复提交 321d7b2 已随此前上游合并及规则修订推送至 yuguo-Jack/cuda-optimized-skill 的 main；ls-remote 确认远端 SHA 与本地完全一致。随后仅补齐本段完成记录并同步。

## 指令集与科学优化专项完善

- 新一轮从干净 005c8bc 开始。读取知识库绑定/查询规则，执行两组 hybrid 检索（本地和飞书均成功），查 MLS/WDRA、Abarrier/Ebarrier 与硬件-编译器映射。
- 定位少伯 ISA/MLS/WDRA/TLS、gfx938 builtin 清单及 GEMM 原始汇编证据；只读保留原件，不修改知识库内容。
- 批量原件输出过长，改按页/符号定向摘读；一次跨 cwd 读取脚本路径错误已纠正，使用各仓明确 workdir。

### 指令与科学实验专项：实现及验证

- 完成 3 个入口、共同契约、指令指南、策略/签名、产物/自定义 harness 契约更新。
- 完成 benchmark、baseline adapter、配对测量、消融、状态、ISA 解析/扫描、方法验证、矩阵和报告脚本修订。
- CPU 回归新增失败边界与真实 subprocess 矩阵闭环；截至本阶段 Hygon 81 项通过，上游 CUDA 10 项通过。
- 静态检查：3 个 Skill、32 个 Python 文件、4 个 JSON、3 个 YAML、38 个 Skill 内本地 Markdown 链接、25 个 CLI 帮助通过。
- Git fetch origin 成功，远端未出现待合并提交。尚未执行 HCU 硬件编译/计时/采集。

- 最终验证再次通过 81 + 10 项测试；3 Skill / 32 Python / 4 JSON / 3 YAML / 38 链接 / 25 CLI 检查通过。
- 已同步到 C:\Users\Administrator\.codex\skills，逐文件验证 55 个 SHA256（5/35/15），无额外遗留文件。旧版本备份：hygon_tmp/skill-backups/20260930_120404_801998。

- 实现提交 e52bfd1 已正常推送 origin/main（yuguo-Jack/cuda-optimized-skill）；安装目录的 Triton 扫描 CLI 已验证能加载共享 ISA 模块。此次没有修改 HCU-Knowledge，没有声称硬件实测。

## 2026-09-30

完成三个 HCU Skill 的说明、策略表、ISA 线索及环境/hipprof脚本更新。92 项 CPU 回归通过（5.74 s）；33 个脚本解析、三个 quick_validate、相对链接与 diff whitespace 检查通过。等待最终安装哈希同步及 origin 推送核验。

安装完成：5/36/15 文件逐个 SHA256 匹配，旧版备份位于 hygon_tmp/skill-backups/20260930_204857_127891。主体 f5daf22952a2037c8d259537a2e4dbd84bf63f8c 已正常推送 origin/main，远端核验一致；GitHub 账户 yuguo-Jack。全局代理未改，仅推送命令禁用失效本机代理。
