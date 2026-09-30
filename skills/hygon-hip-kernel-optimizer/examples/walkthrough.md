# HCU elementwise 调优示例

本例说明流程与目录，不提供虚构性能数字。实际环境、源码与输入须先写 contract，并通过 HCU 知识库查当前实现与工具。参阅 [共同契约](../references/hcu-workflow-contract.md)。

## 参考与基线

reference.py 中定义 `reference(x, y): return x + 2 * y`。执行 baseline Skill 的 inspect/generate 后，检查 FP32 连续张量、输出写入和原 reference 一致。

```bash
python <baseline-skill>/scripts/inspect_ref.py --ref reference.py --dims '{"N":1048576}' --out ref_analysis.json
python <baseline-skill>/scripts/generate_baseline.py --analysis ref_analysis.json --out-dir case
```

复制并修改 `templates/workloads.example.json` 为任务的 tiny/typical/tail、多 seed。目标编译节点可能是容器或裸机，遵从当前项目配置。

```bash
python <hip-skill>/scripts/orchestrate.py setup --baseline case/kernel.hip --ref case/ref.py --dims '{"N":1048576}' --ptr-size 1048576 --workloads workloads.json --profiler auto --iterations 3 --branches 2
```

## 一个可证伪的候选

假设当前访存连续且地址有实际对齐证据，检查宽 load/store 是否值得试。选择 `memory.vectorized_global_access`，在 methods 中记录为何不先改 coalescing（本例已连续）。在 b1/b2 中试不同向量宽度/尾块处理；每个 variant 保持数值语义。

运行 branch_explore 后查看全部 shape/seed 的原始样本与回退门禁。若尾块越界，不允许因为 typical shape 快就采用。针对实际 champion 采集、读正确符号的 ISA；出现宽指令并不自动说明访存带宽饱和。

可构建只去掉向量化且仍正确的消融，保存到 `ablations/memory_vectorized_global_access/kernel.hip`。机制复核记录准确源码/产物哈希。不知道收益来源时保持 unverified。

```bash
python <hip-skill>/scripts/orchestrate.py close-iter --run-dir RUN --iter 1
python <hip-skill>/scripts/orchestrate.py finalize --run-dir RUN
```

## 最终解释

报告 baseline/candidate 的输入/环境、未采集计时与噪声、多案例速度比、失败与未覆盖项、profile/ISA 原件、文件/符号和为何改。没有实际 HCU 测量就写未运行；任务若有模型集成，还须做端到端回归。后续知识入库依据新源码和证据，不复制旧案例性能数字。
