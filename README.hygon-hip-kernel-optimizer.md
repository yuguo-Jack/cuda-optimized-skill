# HCU 算子开发与优化 Skills

本工程的 Hygon 扩展包含三个可安装的 Skill：从参考实现建立正确基线、优化 HIP/CK Tile kernel、调查和优化 Triton/Inductor kernel。以当前工程和目标环境为依据，需要参考代码案例或补充领域资料时可查询 HCU-Knowledge，代码和实验在实际目标环境验证。

## 组织

```text
skills/
  cuda-kernel-optimizer/          # NVIDIA 上游流程；独立保留
  hygon-hip-baseline-generator/   # AST 调查、保守 FP32 脚手架、reference adapter
  hygon-hip-kernel-optimizer/     # 主优化循环、实验门禁、profiling、策略与共同契约
  hygon-triton-kernel-optimizer/  # Inductor 捕获、原始布局复现、Triton 调查
    scripts/ references/ agents/
tests/test_hygon_workflows.py     # CPU 上的语义与流程回归，不冒充 HCU 测试
tools/install_hygon_skills.py    # 备份、安装、文件哈希核验
hygon_tmp/                      # 不提交的采集、缓存、安装备份
```

核心规范是 [HCU 工作契约](skills/hygon-hip-kernel-optimizer/references/hcu-workflow-contract.md)，程序产物见 [实验接口](skills/hygon-hip-kernel-optimizer/references/experiment-artifacts.md)。每个 Skill 的 SKILL.md 是给 Agent 的入口。

## 工作原理

1. 用户需求与当前代码形成精度/布局/版本/性能契约。
2. 优先查看当前源码、测试、文档和目标工具链；需要参考时可用 `hcu-knowledge-search` 查相关代码/案例和原始资料，已有证据足够则直接推进。
3. 保留独立 oracle，验证 baseline，多规模、多 seed、tail 和真实数据分布列入矩阵。
4. 先定位热点与瓶颈假设，再做候选修改，串行基准/回归，按明确的噪声与回退条件选择。
5. 遇到 kernel 优化瓶颈时必须使用性能分析工具，优先 XProf/XCompute，也可选适用的 hipprof/DTK 工具；据采集结果决定下一项修改，保留原件、准确 kernel/dispatch、指标定义。未知计数不转成利用率。
6. 结合源码/ISA/资源/timeline 与正确消融解释收益，缺证据的方法记待验证。最后回到真实模型/项目做端到端验收。
7. 报告与实验原件保留在当前任务工程，说明适用范围、实际收益和未验证项。

## 安装

本机管理使用 PATH 中的系统 Python。三个 Hygon Skill 一起安装，基线/Triton 会引用 HIP Skill 的共同规范。

```powershell
cd D:\git\cuda-optimized-skill
python -X utf8 tools/install_hygon_skills.py
```

默认目标为 `$CODEX_HOME/skills`（未设置时是用户目录 `.codex/skills`），可以 `--skills-dir PATH`。先保存旧文件到本仓 `hygon_tmp/skill-backups/TIMESTAMP`，再复制和核对 SHA256。不会修改别的 Skill，也不随安装执行远端代码。当前会话若已经载入旧 Skill，下一任务/新会话再加载新入口。

`hcu-knowledge-search` 是可选参考工具；没有配置知识库也可以使用这三个 Skill。无需安装本工程为 Python 包，不依赖 MCP 或向量检索模型。

### 依赖与环境

- 管理/静态分析：Python 3.10+，Skill 格式检查需要 PyYAML，回归测试需要 pytest 和 PyTorch。缺包按 `python -m pip install PACKAGE` 安装到相同环境。
- HCU 编译运行：目标机器可用的 DTK/HIP、匹配的 HCU PyTorch、hipcc 兼容驱动；CK Tile/Triton 或项目特定库按任务安装。AICC 与 DTK 默认编译器分别固定版本。
- 采集：目标支持的 XProf/XCompute 或 hipprof；ISA工具以实际代码对象/版本为准。工具存在不代表有权限或支持当前 gfx。
- Perfetto 分析是可选依赖；指定已安装 trace_processor_shell 路径，脚本不隐式下载二进制。新工具按本机约定放 D盘工具目录。
- SSH/容器通过项目既有远端流程。Agent 可本地，实际构建、正确性、性能验证在 HCU 节点；不要求所有 Agent 必须远程运行。

## 怎么用

在 Codex 中直接说明算子、源码/参考文件、输入/精度与远端环境，例如：

- “用 `$hygon-hip-baseline-generator` 把这个 reference 建成 HCU 正确性基线。”
- “用 `$hygon-hip-kernel-optimizer` 优化这个 kernel，覆盖这组 shapes，保持 BF16 数值规则。”
- “用 `$hygon-triton-kernel-optimizer` 找这个模型的 Inductor 热点，核对生成 ISA 和端到端收益。”

详细命令位于各 Skill；通用 HIP harness **只支持简单连续 flat ABI**，半精度/量化/复杂布局等用工程专用 benchmark。优化框架接受 `--benchmark`，但结果必须带真实样本、正确性与源码哈希。`--workloads` 接显式矩阵；不传只能得到单形状结论。

## 检查

```powershell
python -X utf8 -m pytest tests/test_hygon_workflows.py -q
$env:PYTHONPATH = "$PWD/skills/cuda-kernel-optimizer/scripts"
python -X utf8 -m pytest skills/cuda-kernel-optimizer/tests -q
```

这些检查覆盖不可信/缺失实验数据、精度、输入捕获、生成器和流程边界。不能代替真实 HCU 编译、采集、race/越界检测、项目单测和模型性能验收。历史 run 的旧结果也不会因脚本升级自动变成已验证。
