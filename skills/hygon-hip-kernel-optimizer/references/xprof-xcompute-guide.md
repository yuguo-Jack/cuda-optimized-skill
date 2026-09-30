# XProf / XCompute 操作指南

本页仅说明 XProf 采集和 XCompute/xcu 分析。hipprof 的参数、PMC 输出格式和 SQTT 导出能力见 [hipprof 指南](hipprof-guide.md)，不能互换。遇到瓶颈必须工具分析的规则及通用判断见 [性能分析入口](dcu_metrics_guide.md)。

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


## 4. SQTT 和指令时间线

普通计数器仍无法解释依赖、发射或流水空泡时，按硬件/当前工具支持定向采集。例如（替换实际设备、kernel、dispatch 与 SE/CU）：

```bash
xprof --devices 0 --kernels target_kernel --dispatches 3 --enable-sqtt --target-se 1 --target-cu 2 --enable-raw --output-dir profile/base-sqtt python repro.py
```

手册中 `--target-tuple '{1.2}'` 可替代 SE/CU 组合且优先级更高，花括号在 shell 中应引用。保存 `.perf` 和启用 raw 时的 `.out`；默认目标 CU 与首次 dispatch 分配有关，不假设覆盖全芯片。在 XCompute 中选中等待区间，沿 Wavefront/Inst → Source 对照相关指令与寄存器依赖。

已读 XCompute 文档没有证明通用 Chrome JSON 导出；需要该格式时核对实际版本导出能力，不能用系统 kernel 时间线冒充内部指令时间线。


## 5. Replay、SPM 与采集干扰

- 已读 XProf 手册中默认 `--replay-mode kernel` 会串行并按指标分组重放。RCCL 等通信或依赖跨 kernel/进程同步的程序需要 `--replay-mode off`，保留实际多进程执行环境；不能为采更多指标破坏原同步关系。
- 此手册规定 replay off 时禁用 SPM，PMC 一次可采数量受硬件限制。需要更多指标时分批选择，并重新确认执行条件一致，不照搬开启 replay 的全量采集方案。
- PMC 累计执行区间内的计数；SPM 采样每个区间的增量。支持时可用 `--metrics TCC_MISS_list --metrics-mode spm --sample-rate 2048 --trace-size 64`，metric 名需查当前 XML。记录采样间隔与 buffer 大小，并检查数据完整性；手册默认分别为 1024 cycles 和 32 MB。
- profiler/replay 内存分配会刷新 UTCL2。UTCL2 敏感用例需单独采集，关闭其他 GC sections，至多开启 `umc_statistics,utcl2_statistics`；不能据受干扰的低命中率直接改 kernel。
- 原地写、atomic、随机状态和数据依赖 kernel 要核对 replay 的输入恢复；保存未采集的独立基线。最终收益用未开启 profiler 的计时，采集中的耗时仅在对应口径内解释。


## 本指南的资料依据

此处提炼实际操作与解释方法，使用时无需先访问知识库。参数和能力取自已保留手册，目标安装版本仍以实际帮助和支持情况为准：

- [xprof-guide.pdf](http://42.228.13.241:18000/ci/tool/DOCs/xprof-guide.pdf)：第 1–2 页参数/sections，第 3–6 页 metrics、SPM、replay、SQTT 与过滤；原件 SHA256 `ccc2f88d95f2c080bb1978dfcc0cc50d41b0cd905960d526fb9097f7769e7501`。
- [xcompute-guide.pdf](http://42.228.13.241:18000/ci/tool/DOCs/xcompute-guide.pdf)：第 7–15 页概览与指标，第 16–25 页 SQTT/ISA/Raw，第 26–27 页基准比较/occupancy，第 29 页 CLI；原件 SHA256 `4bc464a15313da58eb54c8ee91b2561ce2720ce267a92117aff4f395176252f4`。
