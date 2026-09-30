# 实验产物与恢复

## 工作负载

`templates/workloads.example.json` 是示例，需修改为算子真正支持的输入。每例有 id、dims、ptr_size、seeds、weight、max_regression_pct。权重表达实际流量重要性，不能事后调整来隐藏回退；上限是允许回退的百分比。所有参数在 state 内冻结，OOM/错误不自动缩小规模。

## 自定义 benchmark 接口

`orchestrate setup --benchmark CUSTOM.py` 的 preflight 仅核对文件和维度对象，不强制通用 `solve(...)` 或导入 reference；接口、数值与设备正确性由该适配器实际检查并报告。`contract_status=delegated_to_custom_benchmark` 不代表契约已通过。没有指定自定义适配器时，仍执行原 flat ABI 预检。

调用：`python CUSTOM.py SOURCE --ref REF --warmup N --repeat N --ptr-size N --json-out FILE --DIM=VALUE`；workload suite 和四轮确认另传 `--seed`。exit 0 只代表执行完成，结果仍经过门禁。

JSON 必须有：

```json
{
  "source_sha256": "实际SOURCE文件内容SHA256",
  "reference_sha256": "实际REF文件内容SHA256",
  "correctness": {"checked": true, "passed": true},
  "dims": {"N": 1024}, "seed": 42, "warmup": 10, "repeat": 20,
  "ptr_size_override": 1024,
  "gpu_index": 0, "gpu_name": "实际设备名", "arch": "gfx938",
  "inputs_sha256": "输入数据、shape、dtype及标量参数的内容指纹",
  "signature": [{"name": "x", "type": "tensor[float32]", "is_const": true}],
  "kernel": {"average_ms": 1.0, "samples_ms": [1.0, 1.01, 0.99, 1.0, 1.0]},
  "reference": {"average_ms": 2.0},
  "error": null
}
```

samples_ms 必须是真实测量，不可复制均值；目前晋级要求至少 5 个有限正样本且 CV≤10%。所有输出失败应非零退出，结果缺失/异常也被拒绝。`run_json` 每次先清理精确输出文件再启动进程，不能拾取旧成功。自定义适配器必须记录 dtype/layout、数值政策、构建与依赖哈希、设备/stream/测量范围；通用脚本不能静态证明这些语义。

`source_sha256` 只覆盖入口文件；多文件工程另保存 Git SHA、dirty diff 和构建命令/产物哈希。每次源码/头文件/benchmark/reference/环境改变都重测基线，不用入口哈希冒充完整构建身份。

专用 benchmark 可另报顶层 `compile_pass/contract_pass/correctness_pass/race_safe/timing_valid`：任一显式 false 会拒绝该结果，缺失/null 表示未确认，不等于通过。正确性记录中的有效 `atol/rtol` 必须有限且非负；验收政策可通过 `numerical_policy` 记录。配对确认比较双方容差、特殊值政策、numerical_policy 与 `kernel.timing_scope`，不允许一边放宽标准或换计时范围。旧结果两边均缺这些可选字段仍需人工核对契约，程序不据此补造数值或 race 验证。

setup 从创建 state 起固定 baseline/reference/benchmark 的哈希；候选和每个矩阵用例都必须提供对应 reference 哈希。矩阵汇总还绑定 baseline、candidate、reference、benchmark 与 cases/默认 ptr_size，避免旧 suite 被误用。基线一旦 seed 成功不能在原 run 重置，需要修改时新建 run。`max_regression_pct: 0` 表示该用例不允许性能回退。

自动 profiler 另用 `python CUSTOM.py SOURCE --warmup N --repeat N --ptr-size N --DIM=VALUE` 运行目标，不传 `--ref`，避免混入 oracle；hipprof 路径还传 `--json-out` 保存本次构建收据，XProf 当前只采集原始 .perf。专用 benchmark 需支持无 reference 的采集入口及对应参数，或手动对项目专用 repro 采集。reference 缺失的采集运行不能作为正确性证据，采集中的计时不用于晋级。

## 独立确认与输出覆盖

分支结果只用于筛选；`confirmation/comparison.json` 保存四轮 A/B 交替实验、当前 best/candidate/reference/benchmark 与协议哈希，逐份原始计时另带哈希。state 读取原件重新判定，不能只改 summary 的 improved 字段。以上输入/设备/参数字段缺失的旧自定义 harness 仍可用于探索，但没有完整确认不能晋级；按真实契约补字段，不能填伪造值。所有轮次的收益须超过预设门槛。阈值是重复性门禁，不是统计显著性结论。

消融在 `ablations/<method>/paired/` 保存同类原件；混合方向、接近噪声或无效结果保持未知。即使写了 validation_passed，也须通过原始配对证据复核。主用例消融只支持该 shape/seed 的归因。

通用 benchmark 仅支持完全写出的输出，验证时使用哨兵并检查只读输入/输出 padding。独立 reference 可提供：

```python
def output_extents(**dims):
    return {"C": dims["M"] * dims["N"]}
```

默认整个输出分配区有效。映射单位是元素数，仅支持连续前缀；in-place、带洞布局、scratch 等用专门 harness。整数会换哨兵再次执行；输出覆盖检查不能替代设备内存和竞争检查。

HIP 每次构建的 `.hcu-build/<id>/kernel.so` 独立留存；`bench.build` 记录 binary/source SHA、arch、compiler 与 argv。ISA 检查不再猜相邻 `.so`。自定义 HIP harness 若要自动提取 ISA，也必须提供同格式 build；Triton 保存实际 JIT 产物和 kernel 对应关系后扫描。构建缓存属于实验原件，完成转存后才清理。

## 机制复核

### 前后采集记录

每次 XProf/hipprof 尝试在自身时间戳目录保留 `profile.json` 和日志；迭代目录的 `best_input.profile.json`、`kernel.profile.json` 分别指向各角色最近一次结果。旧 `dcu_top.json` 保留为兼容摘要，不能当成完整前后对照。记录源码/benchmark 哈希、输入维度、实际工具、原件目录、采集状态与失败原因；源码哈希不自动证明采集时加载的二进制身份，多文件/JIT 仍按工程规则核验。

`collection_status=collected` 仅说明采到预期原件；XProf 的 `degraded=true` 可表示自动转换尚未完成，不等于一定采集失败。`analysis_status=not_reviewed` 是程序初始状态；在 `analysis.md` 另写准确 dispatch、指标定义、结论及下一项实验，不修改原始采集记录冒充分析。换工具/指标组后需重采可比较的 baseline。失败/重试都会在 summary 展示，已有记录不会因后续成功而消失。

### 方法机制

自动 ISA regex 只生成线索。Agent 读实际产物后，在迭代目录写独立 `mechanism-review.json`：

```json
{
  "source_sha256": "候选kernel内容SHA256",
  "methods": [{
    "id": "memory.vectorized_global_access",
    "status": "verified",
    "explanation": "具体符号、gfx、代码对象、指令位置及该变化如何实现该方法；限制也写在这里",
    "artifact": "isa_dump.txt",
    "artifact_sha256": "引用产物的SHA256"
  }]
}
```

可引用源码差异、资源报告或 timeline，而不强制所有方法都有唯一指令。源/证据哈希不符不采信；即使机制已核实，缺失或无效消融仍为 unverified，正确且更快的 kernel 可独立晋级。机制验证未自动证明性能因果或 race safety。

## 目录与恢复

```text
TASK/
  contract.md / knowledge-evidence.md / workloads.json
  original-reference/ / baseline/
  run_TIMESTAMP/
    state.json                    # 单写入者；冻结环境、矩阵、best 与历史
    stage-results.jsonl           # orchestrate 调用的命令尝试；退出状态不等于科学验收
    baseline/                     # 基线入口、bench/日志
    iterv1/
      methods.json / analysis.md
      branches/b1/kernel.*        # 同方法组合的不同超参数
      branches/b1/bench.json
      branches/b1/workloads/      # case/seed的baseline与candidate
      branch_results.json         # champion的唯一入口
      kernel.* / bench.json
      confirmation/               # 四轮独立对照及原始JSON/日志
      .hcu-build/                 # 实际构建位于各对应源码目录下
      best_input.xprof/或.hipprof/ # 每次采集独立时间目录
        TIMESTAMP/profile.json    # 本次尝试及原始日志
      best_input.profile.json / kernel.profile.json
      dcu_top.json / roofline.json
      ablations/method_id/ / attribution.json
      isa_dump.txt / isa_check.json / mechanism-review.json
    summary.md
```

失败的同轮可修复后重新 close；成功 close 后进入新轮，不能覆盖旧 champion。open/close、独立 benchmark、分支选择、采集、ISA、消融和 roofline 写入入口均拒绝已关闭轮次；最终报告仍可重新生成。一目录只放一个候选入口，多个后缀不代表自动择优；已选择的 winner 以 branch_results 为准。不要并发写同一个 run。跨会话先读 state 和失败日志：当前脚本未提供自动分布式断点调度，Agent 选择尚未完成的阶段，必要时新建 run。发布前手动核对完整矩阵、项目单测、race 与端到端，不把 finalize 输出文件当所有验收已通过。

## 耦合方法

同选 registry 的耦合项须在 methods.json 顶层加 coupling_reviews，每项含 ids、separate_deltas、validation_plan。后两项描述可分离的源码差异与验证/消融计划；不可拆开的组合只选一个主要方法并在分析中解释相互作用。

## 测量成本

四轮确认每个 case/seed 执行 8 次独立 benchmark 进程；每个消融变体在主用例再执行 8 次。编译/JIT 不计入 kernel 时间，但会消耗任务预算。预算不足时减少有理由的候选数量、显式缩小承诺的覆盖范围或保留待确认结果，不能把筛选结果直接标成已确认。
