# HCU 性能分析入口与通用判断

遇到瓶颈时必须使用性能分析工具。先选择工具，再按对应指南操作；两条路径只共享诊断思路，不共享命令、格式或指标定义。

## 工具与产物分开

| 项目 | XProf / XCompute | DTK hipprof |
| --- | --- | --- |
| 操作指南 | [XProf/XCompute](xprof-xcompute-guide.md) | [hipprof](hipprof-guide.md) |
| 采集与分析 | XProf 采集，XCompute GUI 或 xcu 分析 | hipprof trace/PMC；匹配版本的查看器/解析器，du-compute 另有自己的流程 |
| kernel 过滤 | `--kernels`，另有 `--devices` / `--dispatches` | `--kernel-name`；其他过滤看当前 hipprof 帮助 |
| 计数器选择 | `--sections` / `--metrics` | `--pmc` / `--pmc-read` / `--pmc-write` |
| 计数器产物 | sections 为 `.perf`；metrics 为 `results.txt`，SPM另有采样数据 | PMC 默认文本；手册中 `--pmc-type 3` 为 CSV |
| SQTT | `--enable-sqtt`，`.perf`，`--enable-raw` 保留 `.out` | 已确认支持的版本用 `--sqtt` / `--sqtt-type`，产物依版本 |
| 指令时间线 | XCompute Wavefront/Inst/Source；未确认通用 Chrome JSON 导出 | 部分 DTK SQTT资料描述 `thread_trace_*.json`/HTML，必须核对实际版本 |
| Replay / SPM | 本工具手册中的 `--replay-mode` / `--metrics-mode spm` | 单独查 hipprof 版本，不能套用 XProf 的选项或默认值 |

`--output-type` 是 hipprof 的导出参数；`--pmc-type` 是其 PMC 格式参数，两者也不能混为一个开关。XCompute 与 du-compute 不视为同一工具，当前没有确认二者可以互读任意产物。

`profile_hcu.py` 只负责按配置调用 XProf 或 hipprof adapter，不转换两套原始格式。`dcu_top.json` 是本工程的摘要文件，不是厂商统一格式；检查 `tool`、原件路径和指标来源。缺少工具标记的历史结果需要回看实际命令，不能按文件名猜工具。

## 何时必须使用

kernel 优化收益停滞/落入噪声、明显回退、耗时偏离预期，或无法解释计算/访存/等待限制时，必须采集并分析，再决定下一项修改。优先 XProf → XCompute，目标版本不支持时使用能回答当前问题的 hipprof/DTK 工具。已有同源码、输入和环境的有效采集可复用；不要求每个小变体都采全套指标，也不要求每次都做 SQTT。

只有计时、源码阅读、ISA grep、空产物或采集命令成功，都不算完成瓶颈诊断。最低交付是：准确目标与输入、原始产物、关键指标/时间线观察、瓶颈假设、据此选择的下一项实验。工具或权限受阻时记录原因和待采命令，诊断保持未完成；可以继续独立的代码/正确性准备。


## 指标怎样转成假设

| 观察 | 应进一步确认 | 可尝试方向 |
| --- | --- | --- |
| VALU/MMAC 路径、计算吞吐 | dtype、理论/可达峰值、转换开销、实际 ISA | 矩阵路径、精度、tile、流水 |
| HBM/VMEM、cache、访存请求 | 实际 traffic、层级、命中与合并的公式 | 布局、向量化、复用、融合 |
| dependency/issue wait | 来自 trace 还是定义明确的派生指标 | 独立指令、预取、同步范围 |
| VGPR/LDS/scratch 与 occupancy | 每 CU 理论上限、实际驻留、网格覆盖、spill | live range、tile/stages、launch |
| SQ_WAVES | 累计波数，不是驻留波数 | 配合网格、CU/wave 资源解释 |
| LDS 活动/bank conflict | 单位、范围、swizzle和矩阵读布局 | LDS padding、分块、正确同步 |
| launch次数/系统时间线 | host/device/通信与overlap、真实端到端热点 | 融合、批处理、调度 |

raw counter 大小不能排序为跨指标“严重程度”。SQTT 中字符串或 waitcnt/branch 出现次数只是发现线索。分配字节/时间不是 HBM 流量。现有 parser 保留这些原件，不自动判定瓶颈。


## 修改后的验证与留存

每项瓶颈修改保留“观察→假设→代码改动→预期指标变化”。先验证正确性与未采集性能，再用同一工具、同一版本/指标定义、设备/输入及该工具适用的采集设置比较基准与候选；耗时下降而预期指标未变时应重查归因，不强行宣布机制成立。出现新的主瓶颈时据新证据调整方向。

任务目录保存工具/驱动/编译器版本、源码/代码对象 SHA、准确过滤条件、命令和返回码，以及当前工具实际输出的原件和指标定义。XProf 的 XML 与 hipprof 的指标文档分别留存。不要只保留截图或“occupancy低”一句话。


## 可选标准化输入

Agent 根据原件和准确公式整理 `dcu_top.json.normalized`；保留 `tool` 和原始指标名称，同名指标跨工具也需重新核对公式、分母与范围：

```json
{"normalized":{"compute":{"value":45,"unit":"percent","definition":"精确公式与dtype对应可达峰值","source":"原始产物/版本","scope":"设备+kernel+dispatch+采样范围"}}}
```

memory 对应已明确定义的带宽比例；latency 对应有分母的 stall fraction。完整且同范围复核后才将顶层 degraded 设为 false。缺少轴为 null；roofline 只给建议，不自动判为 near-peak。人工标准化的值同样需要可追溯复核。
