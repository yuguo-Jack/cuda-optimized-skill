# HCU 指标与采集指南

先读 [共同契约](hcu-workflow-contract.md) 第 5 节，命令以目标版本的 `--help` 和工具手册为准。资料不足时可用 `hcu-knowledge-search` 查 `xprof-xcompute-workflow` 等参考。不互换 XProf、hipprof、rocprof 的选项。

## 何时必须使用

kernel 优化收益停滞/落入噪声、明显回退、耗时偏离预期，或无法解释计算/访存/等待限制时，必须采集并分析，再决定下一项修改。优先 XProf → XCompute，目标版本不支持时使用能回答当前问题的 hipprof/DTK 工具。已有同源码、输入和环境的有效采集可复用；不要求每个小变体都采全套指标，也不要求每次都做 SQTT。

只有计时、源码阅读、ISA grep、空产物或采集命令成功，都不算完成瓶颈诊断。最低交付是：准确目标与输入、原始产物、关键指标/时间线观察、瓶颈假设、据此选择的下一项实验。工具或权限受阻时记录原因和待采命令，诊断保持未完成；可以继续独立的代码/正确性准备。

## 1. 固定目标与采集范围

先保留未开启 profiler 的正确性与计时基线，固定 gfx、工具链、源码/二进制、shape/dtype/stride、grid/block、stream、数据、缓存策略和运行方式。应用中先用适用的系统时间线工具确认热点、launch 次数及 CPU/通信等待；不要按单次耗时就判断端到端占比。

读取 `xprof --help`，核对本节选项、sections 和目标支持。手册提供以下过滤器：

| 选项 | 用途 | 复核点 |
| --- | --- | --- |
| `--devices 0` | 限定采集设备 | 使用当前进程实际可见的编号 |
| `--kernels kernel_substring` | 按 kernel 名的子串过滤 | 同名/同前缀可能命中多个变体，回产物核对 |
| `--dispatches 3` 或 `3:9` | 限定 launch 编号或范围 | 不跨运行照搬编号；核对当前工具的区间语义 |
| `--output-dir PATH` | 保存本轮结果 | 每次使用独立目录，保留命令、日志和返回码 |

排除初始化、reference、分配/恢复和 warmup kernel。Triton/Inductor 还要对应生成源码、配置与实际 dispatch；采集输入快照产生的同步/复制不能混入最终性能测试。

## 2. XProf 采集：先概览，再定向补充

以下为手册中已确认的参数组合示例；将 `target_kernel`、设备/dispatch 和 `python repro.py` 换成当前任务实际值，确认目标版本后执行。每组选项只收集回答当前问题所需的数据。

```bash
# 第一轮：计算/内存概览与资源限制
xprof --devices 0 --kernels target_kernel --dispatches 3 --sections speed_of_light,memory_workload,occupancy --output-dir profile/base-overview python repro.py

# 等待与发射限制：根据第一轮现象定向补采
xprof --devices 0 --kernels target_kernel --dispatches 3 --sections compute_workload,scheduler_statistics,wave_state,instruction_statistics --output-dir profile/base-wait python repro.py

# 只查指定指标时，保存原始值和定义
xprof --devices 0 --kernels target_kernel --dispatches 3 --metrics SQ_WAVES,SQ_INSTS_VALU,L2CacheHit --output-dir profile/base-metrics python repro.py
```

sections 输出供 XCompute 读取的 `.perf`，单 metric 路径输出 `results.txt`。后者的 dispatch 信息可包含 kernel 名、grid/workgroup、VGPR/SGPR、LDS/scratch 等。从安装目录保留 `gfx_metrics.xml` / `metrics.xml` 中用到的定义、分母与单位，以及工具版本。

| 疑问 | 优先查看/补采的 sections |
| --- | --- |
| 计算还是内存限制 | `speed_of_light`、`compute_workload`、`memory_workload` |
| 访存层级、请求、命中、LDS | `memory_workload`、`memory_chart` |
| 寄存器/LDS限制还是网格太小 | `occupancy`，结合 launch/grid 与资源数据 |
| 发射不足、依赖或同步等待 | `scheduler_statistics`、`wave_state`、`instruction_statistics` |
| 缓存行为随时间变化 | 目标支持的 SPM metrics 或 `cache_counter` |
| UMC/UTCL2专项 | `umc_statistics`、`utcl2_statistics`，先看下文采集干扰限制 |

`profile_hcu.py` 的 auto 优先 XProf，缺失则 hipprof；自动 XProf 路径当前只采基础 sections，没有封装这里全部过滤/定向选项，进一步诊断按上面的原生命令执行。adapter 留存 help、command、returncode、原件；`degraded=true` 表示尚未自动提取可靠归一化指标，仍需打开原件分析。`--profiler none` 只用于不涉及瓶颈诊断的准备阶段。

## 3. XCompute 中怎么分析

打开对应 `.perf`，按以下顺序选择必要视图，不要求逐页浏览：

| 视图 | 操作与要回答的问题 |
| --- | --- |
| Summary | 核对设备、kernel、dispatch、grid/block 与耗时；选定目标进入 Details，不把全应用各 kernel 的平均值当目标数据 |
| Details：Speed Of Light / Compute | 按真实 dtype 查看 VALU/MMOP 吞吐与分层 Roofline，结合 IPC、指令构成和转换开销判断计算路径；峰值和流量口径要一致 |
| Details：Memory | Memory Chart 连接请求数、传输量和带宽，继续看 cache/LDS 与请求延迟；命中率必须结合实际流量和访问模式解释 |
| Details：Scheduler / Wave State / Occupancy | 区分 issue/依赖等待、资源限制和并发不足；结合 grid 是否覆盖 CU，不能只看一个占用率百分比 |
| Raw | 回查派生指标使用的 PMC；异常比例先核对原始值、公式、分母和范围 |
| Wavefront / Inst（需 SQTT） | 按 waveslot 或指令类别观察执行区间与空泡，选定区间追踪依赖；范围只覆盖实际采集的 SE/CU |
| Source | 将 ISA 与对应源码关联，检查 live VGPR、寄存器读写依赖、hit count/latency；关联源码和行号信息必须匹配实际代码对象 |
| Panorama（有对应数据时） | 用 workload heatmap 查看 CU 工作分布，用指令树定位值得追踪的 issue latency |
| Baseline Analysis | 对同文件或不同 `.perf` 的基准/候选做比较；先确认设备、输入和采集口径可比 |

Occupancy Calculator 可以离线输入架构、VGPR、SGPR、workgroup、LDS 估计资源限制下的驻留上限。该上限、采样时的实际活跃/驻留状态、全 GPU 网格覆盖是不同问题；`SQ_WAVES` 是执行期间累计波数，不能代替它们。

手册还提供 `xcu` 命令行工具：`--list-sections` 列 section，`--list-sections-metrics SECTION` 查看指标说明，`-S/--sections` 选 section，`-F/--output-format` 选择输出。使用前检查 `xcu --help` 的调用方式；已读手册仅确认终端/CSV 输出，不推测其任意格式解析或 JSON 导出参数。可采用 CLI 或 GUI，以实际可用功能完成同一分析目标。

## 4. 指标怎样转成假设

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

## 5. hipprof 适配

```bash
python <skill>/scripts/profile_hipprof.py --state RUN/state.json --iter 1 --which best_input --pmc-mode pmc
```

默认只做普通 PMC；`--pmc-mode read/write/all` 是额外采集，先确认当前版本支持相应 --pmc-read/--pmc-write/--pmc-type。脚本保存当前 help，未确认的必需选项返回降级状态。用目标工具的 kernel 过滤/采集区间排除分配、reference、warmup及恢复 kernel。

CSV 的 `all_raw_metrics` 保留全量发现项，轴内列表按名字排列，不作无量纲严重程度排序。当前多个 dispatch 的发现聚合必须回原 CSV 复核。每次采集有独立目录，失败不能混入上一轮 CSV。

`hipprof --codeobj-analyze` / `dccobjdump` 的 VGPR/SGPR/LDS 是资源证据；固定阈值只能提醒复核，不是跨代 occupancy 公式。

## 6. SQTT 和指令时间线

普通计数器仍无法解释依赖、发射或流水空泡时，按硬件/当前工具支持定向采集。例如（替换实际设备、kernel、dispatch 与 SE/CU）：

```bash
xprof --devices 0 --kernels target_kernel --dispatches 3 --enable-sqtt --target-se 1 --target-cu 2 --enable-raw --output-dir profile/base-sqtt python repro.py
```

手册中 `--target-tuple '{1.2}'` 可替代 SE/CU 组合且优先级更高，花括号在 shell 中应引用。保存 `.perf` 和启用 raw 时的 `.out`；默认目标 CU 与首次 dispatch 分配有关，不假设覆盖全芯片。在 XCompute 中选中等待区间，沿 Wavefront/Inst → Source 对照相关指令与寄存器依赖。

已读 XCompute 文档没有证明通用 Chrome JSON 导出；需要该格式时核对实际版本导出能力，不能用系统 kernel 时间线冒充内部指令时间线。

DTK 部分 hipprof 版本支持 `--sqtt --sqtt-type ... --output-type ...`。脚本参数只适用于帮助/手册已经确认的版本；不要因示例存在就批量执行所有类型。通信 kernel 的 replay 要特别核对，跨进程同步不可随便单 kernel 重放。

`analyze_sqtt.py` 是宽松 JSON/CSV walker，识别助读线索；不保证去重后动态指令数或等待周期。`analyze_perfetto_trace.py` 仅适用实际 Chrome trace JSON，不能直接解析 XProf .perf；slice 时间可能重叠，累计 duration 不是 wall time/利用率。需要 trace processor 时指定已安装二进制，避免自动下载到默认缓存。

## 7. Replay、SPM 与采集干扰

- 已读 XProf 手册中默认 `--replay-mode kernel` 会串行并按指标分组重放。RCCL 等通信或依赖跨 kernel/进程同步的程序需要 `--replay-mode off`，保留实际多进程执行环境；不能为采更多指标破坏原同步关系。
- 此手册规定 replay off 时禁用 SPM，PMC 一次可采数量受硬件限制。需要更多指标时分批选择，并重新确认执行条件一致，不照搬开启 replay 的全量采集方案。
- PMC 累计执行区间内的计数；SPM 采样每个区间的增量。支持时可用 `--metrics TCC_MISS_list --metrics-mode spm --sample-rate 2048 --trace-size 64`，metric 名需查当前 XML。记录采样间隔与 buffer 大小，并检查数据完整性；手册默认分别为 1024 cycles 和 32 MB。
- profiler/replay 内存分配会刷新 UTCL2。UTCL2 敏感用例需单独采集，关闭其他 GC sections，至多开启 `umc_statistics,utcl2_statistics`；不能据受干扰的低命中率直接改 kernel。
- 原地写、atomic、随机状态和数据依赖 kernel 要核对 replay 的输入恢复；保存未采集的独立基线。最终收益用未开启 profiler 的计时，采集中的耗时仅在对应口径内解释。

## 8. 修改后的验证与留存

每项瓶颈修改保留“观察→假设→代码改动→预期指标变化”。先验证正确性与未采集性能，再以相同 sections/metric 定义、设备/输入、replay 和 SE/CU 范围比较基准与候选；耗时下降而预期指标未变时应重查归因，不强行宣布机制成立。出现新的主瓶颈时据新证据调整方向。

任务目录保存工具/驱动/编译器版本、源码/代码对象 SHA、准确过滤条件、命令和返回码、`.perf`/`results.txt`/raw/CSV、指标 XML、分析结论与下一项实验。不要只保留截图或“occupancy低”一句话。

## 可选标准化输入

Agent 根据原件和准确公式整理 `dcu_top.json.normalized`：

```json
{"normalized":{"compute":{"value":45,"unit":"percent","definition":"精确公式与dtype对应可达峰值","source":"原始产物/版本","scope":"设备+kernel+dispatch+采样范围"}}}
```

memory 对应已明确定义的带宽比例；latency 对应有分母的 stall fraction。完整且同范围复核后才将顶层 degraded 设为 false。缺少轴为 null；roofline 只给建议，不自动判为 near-peak。人工标准化的值同样需要可追溯复核。

## 本指南的资料依据

此处提炼实际操作与解释方法，使用时无需先访问知识库。参数和能力取自已保留手册，目标安装版本仍以实际帮助和支持情况为准：

- [xprof-guide.pdf](http://42.228.13.241:18000/ci/tool/DOCs/xprof-guide.pdf)：第 1–2 页参数/sections，第 3–6 页 metrics、SPM、replay、SQTT 与过滤；原件 SHA256 `ccc2f88d95f2c080bb1978dfcc0cc50d41b0cd905960d526fb9097f7769e7501`。
- [xcompute-guide.pdf](http://42.228.13.241:18000/ci/tool/DOCs/xcompute-guide.pdf)：第 7–15 页概览与指标，第 16–25 页 SQTT/ISA/Raw，第 26–27 页基准比较/occupancy，第 29 页 CLI；原件 SHA256 `4bc464a15313da58eb54c8ee91b2561ce2720ce267a92117aff4f395176252f4`。
