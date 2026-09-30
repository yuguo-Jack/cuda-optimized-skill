# HCU 指标与采集指南

先读 [共同契约](hcu-workflow-contract.md) 第 5 节，命令以目标版本的 `--help` 和工具手册为准。资料不足时可用 `hcu-knowledge-search` 查 `xprof-xcompute-workflow` 等参考。不互换 XProf、hipprof、rocprof 的选项。

## 首选 XProf → XCompute

已收录手册中的入口（确认当前版本后执行）：

```bash
xprof --sections speed_of_light,instruction_statistics --output-dir out <app command>
xprof --metrics SQ_WAVES,SQ_INSTS_VALU,L2CacheHit --output-dir out <app command>
```

section 输出 `.perf`，单 metric 路径可输出 results.txt。用 XCompute 按 kernel/dispatch/架构查看 compute、memory、occupancy、scheduler、wave state 等；从安装目录 gfx_metrics.xml/metrics.xml 保存定义、分母、单位、工具版本。不要把全应用所有 kernel 的算术平均当热点本身。

`profile_hcu.py` 是统一入口，auto 优先 XProf，缺失则 hipprof；`--profiler none` 为显式无采集模式。XProf adapter 留存 help、command、returncode、原件；还没实现任意版本 .perf 自动语义解析，degraded=true 表示不能直接提取可靠归一化指标，并不表示原件没有价值。

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

## hipprof 适配

```bash
python <skill>/scripts/profile_hipprof.py --state RUN/state.json --iter 1 --which best_input --pmc-mode pmc
```

默认只做普通 PMC；`--pmc-mode read/write/all` 是额外采集，先确认当前版本支持相应 --pmc-read/--pmc-write/--pmc-type。脚本保存当前 help，未确认的必需选项返回降级状态。用目标工具的 kernel 过滤/采集区间排除分配、reference、warmup及恢复 kernel。

CSV 的 `all_raw_metrics` 保留全量发现项，轴内列表按名字排列，不作无量纲严重程度排序。当前多个 dispatch 的发现聚合必须回原 CSV 复核。每次采集有独立目录，失败不能混入上一轮 CSV。

`hipprof --codeobj-analyze` / `dccobjdump` 的 VGPR/SGPR/LDS 是资源证据；固定阈值只能提醒复核，不是跨代 occupancy 公式。

## SQTT 和指令时间线

仅在硬件/当前工具支持、问题需要时采。XProf 手册提供 --enable-sqtt、target-se/cu/tuple、raw；范围不等于全芯片。已读 XCompute 文档没有证明通用 Chrome JSON 导出；保留 .perf/raw，核对实际版本导出能力。

DTK 部分 hipprof 版本支持 `--sqtt --sqtt-type ... --output-type ...`。脚本参数只适用于帮助/手册已经确认的版本；不要因示例存在就批量执行所有类型。通信 kernel 的 replay 要特别核对，跨进程同步不可随便单 kernel 重放。

`analyze_sqtt.py` 是宽松 JSON/CSV walker，识别助读线索；不保证去重后动态指令数或等待周期。`analyze_perfetto_trace.py` 仅适用实际 Chrome trace JSON，不能直接解析 XProf .perf；slice 时间可能重叠，累计 duration 不是 wall time/利用率。需要 trace processor 时指定已安装二进制，避免自动下载到默认缓存。

## 可选标准化输入

Agent 根据原件和准确公式整理 `dcu_top.json.normalized`：

```json
{"normalized":{"compute":{"value":45,"unit":"percent","definition":"精确公式与dtype对应可达峰值","source":"原始产物/版本","scope":"设备+kernel+dispatch+采样范围"}}}
```

memory 对应已明确定义的带宽比例；latency 对应有分母的 stall fraction。完整且同范围复核后才将顶层 degraded 设为 false。缺少轴为 null；roofline 只给建议，不自动判为 near-peak。人工标准化的值同样需要可追溯复核。
