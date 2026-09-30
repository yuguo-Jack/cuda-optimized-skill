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
