# 工作进度

## 2026-11-09：全流程按需分析衔接

- 更新 HCU Mermaid 总图、三个 Skill 入口/交接、共享契约、操作/采集/产物指南与报告模板；三类分析回到同一实现与验证循环，不要求每轮全部执行。上限/融合指南也补上遇实际等待、转换或资源疑点时转汇编分析的入口。
- 按用户后续提醒继续检查完整执行路径：简单问题复用、复杂结构提前分析、融合后资源代价、停滞/回退、工具受阻、单 kernel 与端到端反例、跨会话恢复。各路径均保持证据边界与未完成项，不为了补表拖延实验。
- 更新 HIP setup/open/close 的引导及 summary 文案、Triton 自动报告草稿，使实际使用脚本时也提示条件化分析和反馈；修正“必须写满分支”和“必走 fallback/模型改写”的文案倾向，不修改命令执行、数值政策、原晋级协议或自动派发 Agent。
- Mermaid 10.9.3 成功解析/渲染 30 个节点、47 条边，查看分析分流及回到实现的布局；首次 110 项 CPU 回归通过，补脚本提示后最终再次 110 项通过（5.85 s）。34 Python AST、3 Skill 格式、115 处仓内链接/锚点、Triton 报告空/有数据冒烟与 diff whitespace 检查通过。
- 已重新同步 C:/Users/Administrator/.codex/skills 的 baseline 5 / HIP 41 / Triton 15 共 61 份文件，逐份 SHA256 一致、无额外文件，103 处安装后链接/锚点通过。本次没有 HCU 硬件实验，也没有修改 NVIDIA Skill 或共享知识库。
- 已清理本轮 `hygon_tmp/analysis-routing-20261109`、`hygon_tmp/analysis-routing-review-20261109` 中的渲染/测试/校验文件和验证后不再需要的安装备份；两目录均不存在，检查各自 57 个 pytest 链接均位于本任务内。清理期间观察到 D 盘可用空间合计增加约 20.4 MB。共享 Mermaid 渲染工具保留，历史清理受阻目录未重试。
- fetch 后本地起点与 origin/main 均为 16b816c；正常提交推送本轮变更，最终 SHA 与远端一致性在完成回复报告，不新增 tag。

## 2026-11-09：AsmEvo 方法整合

- 完成 PDF 全文阅读、关键图表复核和已有构建/扫描/验证流程对照，新增 `assembly-guided-optimization.md` 并连接三 Skill、ISA/实验/架构指南及报告。README 增加按需汇编支路，KRA 的在线阅读与下载条件写入共同契约和依赖说明。
- 本轮只修改指引与引用，不改源码编排器、不自动下载外部运行系统、不执行远端 HCU 实验。
- 110 项本仓 CPU 回归通过（5.73 s）；3 Skill 格式、34 Python AST、102 处仓内链接/锚点及 diff whitespace 检查通过。复核 README 汇编支路与源码/专用 harness 的边界。
- 安装前 60 份文件与旧 HEAD 一致；安装后 baseline 5 / HIP 41 / Triton 15 共 61 份 SHA256 一致、无额外文件，91 处安装后链接/锚点通过。
- 已删除本次 `hygon_tmp/asmevo-20261109` 中 PDF 提取/渲染、pytest 临时文件和验证后不再需要的安装备份，目录已不存在，操作期间 D 盘可用空间增加约 5.8 MB；用户 E 盘原 PDF 保留。
- 前次 `hygon_tmp/expert-methods-20261109` 普通删除清掉部分文件后遇到只读 Git pack；随后针对已核对的临时 `.git` 的清理被自动审批拒绝（仅返回 blocked by policy），未重试或绕过。该任务目录余 315,930,127 字节，留给用户清理；保留共享工具/缓存和其他历史任务目录。
- 获取 origin 后起点仍为 48f768e，与本地一致；将以 yuguo-Jack 提交并正常推送本轮文档，远端 SHA 与最终工作区状态在完成回复报告。

## 2026-11-09：专家方法论整合

- 完成飞书材料与 KRA 代码/案例核对，确认旧 Skill 已有能力与需要补强的分析环节。
- 新增共享 `performance-limits-and-fusion.md`，连接三个 Skill、性能分析/架构/实验指南、报告模板和 HCU README。KRA 保持可选，没有新增包依赖或自动运行外部脚本。
- 已通过 110 项本仓 CPU 回归、86 处仓内链接/锚点、34 Python AST、三个 Skill 格式检查；KRA 21 项 CPU 测试及六个 CLI 帮助入口正常。
- 安装前核对 59 份旧文件与仓库 HEAD 一致，没有独立安装端修改；已同步 Codex 三个 Skill（5/40/15 文件），60 份哈希及 76 处安装后引用通过。未修改 NVIDIA Skill 或 Python 执行脚本，本轮无 HCU 硬件实验。
- 自动审批拦截本轮临时 clone、pytest scratch 和安装备份的清理（仅返回 blocked by policy）；未执行删除、不采用其他方式绕过。临时材料仍在忽略提交的任务目录中。
- 仓库旧代理 7893 不可用，通过单次 Git 7890 代理覆盖成功 fetch；未改全局或仓库代理设置。实际 Git 凭据的 GitHub 身份核验为 yuguo-Jack，发布前 HEAD 与 origin/main 一致；随后提交本轮文档变更并正常推送，远端结果在完成回复报告。

## 2026-10-12：按需参考 HCU 案例与 KernelWiki

- 已补共同契约、三个 Skill 入口、实验指南、两份报告模板及 Triton 草稿生成提示，并更新 HCU README 的说明与可选查证支路。
- 验证通过：110 项 CPU 回归、三个 Skill 格式检查、62 处文档链接及锚点、Mermaid 解析与渲染；Triton 草稿生成冒烟和 KernelWiki 症状检索均成功。
- 已同步 Codex 系统目录的三个 Skill，5/38/15 个文件逐个 SHA256 一致，无额外遗留文件；旧文件备份在忽略提交的 hygon_tmp/skill-backups 下。最终提交前核对 diff，随后正常推送并检查远端 SHA，结果在完成回复报告。
- 本轮没有 HCU 硬件执行或知识库内容更新；保留 profiler、正确性和原实验晋级规则。

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

## 实际工程汇编回退规则

- 在共享 ISA 指南补充真实工程构建下保留汇编、设备链接/LTO 阶段识别、符号与产物哈希、实际加载检查和重新测试；HIP/Triton 入口及共同契约已关联。
- 仅修改流程文档，沿用现有编排和状态协议。三个 Skill 格式校验、41 处本地 Markdown 链接及 diff whitespace 检查通过；未改脚本，没有执行 HCU 硬件编译。
- 已备份并同步 Codex 安装副本，5/36/15 个文件逐个 SHA256 匹配，无额外遗留文件；备份位于 hygon_tmp/skill-backups/20260930_223643_386437。推送前核对 origin/main 仍为 974d663，与本地起点一致；发布后在完成回复报告提交及远端核验结果。

## CUDA/HCU 全流程对照 review

- 完成入口、策略、实验与状态、采集/ISA/归因、失败及交付对照；详见 findings.md 的逐阶段表。
- 新增每次采集记录和 before/after 角色快照、编排命令尝试日志与收尾缺口汇总；补 XProf 无原件失败判定、明确失败门禁与配对数值/计时政策一致性。
- 修复自定义 benchmark 被通用 flat ABI 预检误拒绝；补算子路线/数值/构建/恢复/工程验收操作指南、可校验的方法计划示例及当前 Mermaid 流程图，移除未引用的旧 HCU PNG/SVG。
- HCU 110 项 CPU 回归、CUDA 10 项回归通过。后续进行最终文档/脚本/入口检查、安装同步与正常推送；无 HCU 设备编译/采集结论。
- 最终静态检查通过：3 Skill、34 Python AST、4 JSON、3 YAML、50 处 Skill 本地链接、README 链接、25 CLI 帮助、示例 JSON schema 和真实方法校验器。Triton benchmark 模板因本机无 HCU Triton 运行时仅做 AST，不以普通后端代替目标执行。
- 安装副本已备份至 hygon_tmp/skill-backups/20260930_225824_049418；baseline 5 / HIP 38 / Triton 15 共 58 文件逐个 SHA256 一致，没有额外遗留文件。准备提交并正常推送，远端起点核对为 8f5ed2c。
- 主体提交 74f7b36204d144c0d0ec312efc3cc926c8330cf3 已正常推送 yuguo-Jack/cuda-optimized-skill 的 main，并通过 ls-remote 核对一致；安装目录 profile_hcu --help 成功加载新增共享模块。此后仅提交本完成记录。

## 2026-11-03：原工程同步与方法论更新

- 找到 D:/git/cuda-optimized-skill 原工程，核对干净 main 和 origin/main 起点均为 f39ba95。安装副本中 baseline 3 文件、HIP 5 文件和新增架构/经验参考共 9 文件已回写；Triton 内容一致。
- 同步 HCU README 的参考质量判断、结构探索、保留探索分支和经验复用流程；新增生成器交接信息保留原数学/测试语义，不修改实验晋级机制。
- 本次同步前的源码快照和身份清单保存在 hygon_tmp/methodology-sync-20261103_033258。先核对源码，再通过原工程安装器备份同步并校验三个已安装 Skill。
- 原全局代理失效；通过当前可用代理进行命令级 fetch 已成功，没有改写全局配置。发布状态以本轮最终核验为准。
- 110 项 HCU CPU 回归通过（10.46 秒）；3 Skill 格式、34 Python AST、4 JSON、19 份 Markdown 的 75 个本地链接及锚点、git diff whitespace 检查均通过，没有执行 HCU 硬件实验。
- 已通过原工程安装器再次备份并同步三个安装副本：5/39/15 共 59 文件 SHA256 一致，无额外文件。备份在 hygon_tmp/skill-backups/20261103_113515_004109；回归 XML、静态验证结果与同步前清单在本次同步目录。接下来提交并正常 push origin/main，核对远端 SHA。
- 主体提交 de6af9df1edb1de5cf25678494dacac544afac39 已正常推送 origin/main；通过独立 ls-remote 核对远端完整 SHA 与本地一致，工作区干净。本条和计划状态仅补发布完成记录，不改变已验证的 Skill 内容；记录提交的最终远端核验在完成回复报告。
