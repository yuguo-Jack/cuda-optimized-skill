# DTK hipprof 操作指南

本页仅说明 hipprof。XProf/XCompute 的 `.perf`、sections、`--enable-sqtt`、replay/SPM 参数见 [独立指南](xprof-xcompute-guide.md)，不要套到 hipprof。瓶颈诊断要求和通用判断见 [性能分析入口](dcu_metrics_guide.md)。

## 1. 系统时间线与 PMC

先用目标安装的 `hipprof -h` 核对功能。以下基础选项来自 DTK 26.04.1 hipprof 手册；替换应用和目标 kernel，分别采集并保留未开启 profiler 的基线。

```bash
# HIP API、拷贝和 kernel 时间线；不是 kernel 内部指令轨迹
hipprof --hip-trace -o profile/base-trace/ python repro.py

# 目标 kernel 的通用 PMC，CSV 输出
hipprof --pmc --pmc-type 3 --kernel-name target_kernel -o profile/base-pmc/ python repro.py

# 需要深入缓存读/写时，分别补采对应组
hipprof --pmc-read --pmc-type 3 --kernel-name target_kernel -o profile/base-read/ python repro.py
hipprof --pmc-write --pmc-type 3 --kernel-name target_kernel -o profile/base-write/ python repro.py
```

`-o` 的值以 `/` 结尾时表示输出目录，否则最后一段作为文件名前缀。`-d` 设置临时数据目录。用 trace 核对真实热点，再从 PMC 产物中确定设备、kernel、dispatch、grid/workgroup 和输入；初始化、reference、warmup/恢复 kernel 不计作目标。`--kernel-name` 不等于 XProf 的 `--kernels`，其具体匹配规则以当前版本为准。

PMC 默认文本通常为 `pmc_results_<PID>.txt`；`--pmc-type` 在此手册中：0=全名文本、1=缩写文本、2=原始数据、3=CSV。计数、百分比和速率需分别核对单位/公式，不能沿用 XProf 的 XML 解释同名值。

`--output-type` 则是另一个导出选项：此版本手册列 0=JSON、1=HTML、2=Perfetto（默认）。这不代表所有采集模式都能输出同一种结构，更不代表 XProf 也支持这些参数。

## 2. 本工程 adapter 的能力

```bash
python <skill>/scripts/profile_hipprof.py --state RUN/state.json --iter 1 --which best_input --pmc-mode pmc --kernel-name target_kernel
```

`--pmc-mode read/write/all` 控制分别调用 `--pmc-read` / `--pmc-write`；不是 XProf 的 section 名。adapter 默认 `--pmc-type 3`，读取本次目录的 CSV；`all_raw_metrics` 是发现汇总，必须回原 CSV 按准确 dispatch 解释，不能把多个 kernel 的聚合当目标利用率。

`hipprof --codeobj-analyze <ELF>` 可辅助分析寄存器压力，可能需要交互选择符号。配合目标 `dccobjdump` 看 VGPR/SGPR/LDS 和 ISA；资源数据不是实测驻留/occupancy，也不能直接套用 XCompute 页面的计算口径。

du-compute 是另有 profile/analyze 流程的工具，已有资料描述其基于 hipprof。需要它时读对应版本手册；不能把它叫作 XCompute，也不假设可以直接打开任意 XProf `.perf`。

## 3. SQTT：按已确认的 DTK 版本使用

DTK 25.04.1 的 SQTT 专项资料描述以下方式；26.04.1 通用 hipprof 手册没有完整描述这些专项选项，因此不能仅凭 DTK 版本数字推定支持或删除。以实际安装帮助、对应专项说明和成功采集为准。

```bash
hipprof --sqtt --sqtt-type 1 --kernel-name target_kernel -o profile/base-sqtt/ python repro.py
```

专项资料中，`--sqtt-type` 选择内容（stat、wave、issue、stat_stall、stat_valu 等）；特殊值 `1` 对应上述五项，`all` 还包含更多 event/all_wave 内容。它不是 JSON/HTML/Perfetto 的格式选择器。按问题选择需要的内容，不默认全量抓取。

专项资料描述 `thread_trace_*.json` / HTML 及统计产物；适用的浏览器/Perfetto/解析工具取决于实际导出结构。即使扩展名相同，API/kernel trace 与 SQTT 指令轨迹也不能混用。`--output-type` 能否控制当前 SQTT 导出需另行核对，通用 trace 导出说明不能直接充当 SQTT 证明。

此专项资料中的 CU/SE/SIMD 过滤来自 hipprof 配置项，如 `sqtt: TARGET_CU=... MAX_SE=... SIMD=... MAX_WAVE=...`，合法值取决于架构；不要传入 XProf 的 `--target-se` / `--target-cu` / `--target-tuple`。hipprof 的 `--flush-mode`、启动/停止抓取等语义也不能当成 XProf `--replay-mode` 的同义参数。

`analyze_sqtt.py` 仅辅助阅读已确认来自 hipprof 的 JSON/CSV；宽松遍历结果不保证动态指令去重或真实等待周期。`analyze_perfetto_trace.py` 只处理已确认格式兼容的 Chrome trace JSON，不解析 XProf `.perf` 或任意 Perfetto 二进制。

## 4. 对照与依据

修改前后保持同一 hipprof 版本、模式、kernel/输入、采集范围和指标定义。保存命令、日志、返回码、原件与分析结果；最终性能用未开启 profiler 的独立计时。切换到另一套工具后重新确认定义和测试口径，不直接拼接数值。

- [DTK 26.04.1 hipprof 使用手册](https://download.sourcefind.cn:65024/file/1/DTK-26.04.1/Document/DTK%2026.04.1%20hipprof使用手册.pdf)：参数、trace、PMC、代码对象分析；原件 SHA256 `6d74556313bf2d1f5befc632ff3163b67e3c032fdfffdeb076a2442f40485e54`。
- [SQTT 新增功能介绍 25041](https://r0ddbu55vzx.feishu.cn/slides/PmjyshlAllZ7k6dAdRYcTK8OnHh)：专项参数与产物说明，保留修订 5；原件 SHA256 `240f8ef6690da053a60ca6eda9d1d074b060e66676baa523fe87d8bf02ebdc7e`。图形细节不依据文本抽取猜测。
