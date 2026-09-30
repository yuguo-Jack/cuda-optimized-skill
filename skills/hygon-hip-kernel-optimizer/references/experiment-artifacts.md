# 实验产物与恢复

## 工作负载

`templates/workloads.example.json` 是示例，需修改为算子真正支持的输入。每例有 id、dims、ptr_size、seeds、weight、max_regression_pct。权重表达实际流量重要性，不能事后调整来隐藏回退；上限是允许回退的百分比。所有参数在 state 内冻结，OOM/错误不自动缩小规模。

## 自定义 benchmark 接口

调用：`python CUSTOM.py SOURCE --ref REF --warmup N --repeat N --ptr-size N --json-out FILE --DIM=VALUE`；workload suite 另传 `--seed`。exit 0 只代表执行完成，结果仍经过门禁。

JSON 必须有：

```json
{
  "source_sha256": "实际SOURCE文件内容SHA256",
  "reference_sha256": "实际REF文件内容SHA256",
  "correctness": {"checked": true, "passed": true},
  "kernel": {"average_ms": 1.0, "samples_ms": [1.0, 1.01, 0.99, 1.0, 1.0]},
  "reference": {"average_ms": 2.0},
  "error": null
}
```

samples_ms 必须是真实测量，不可复制均值；目前晋级要求至少 5 个有限正样本且 CV≤10%。所有输出失败应非零退出，结果缺失/异常也被拒绝。`run_json` 每次先清理精确输出文件再启动进程，不能拾取旧成功。自定义适配器必须记录 dtype/layout、数值政策、构建与依赖哈希、设备/stream/测量范围；通用脚本不能静态证明这些语义。

`source_sha256` 只覆盖入口文件；多文件工程另保存 Git SHA、dirty diff 和构建命令/产物哈希。每次源码/头文件/benchmark/reference/环境改变都重测基线，不用入口哈希冒充完整构建身份。

setup 从创建 state 起固定 baseline/reference/benchmark 的哈希；候选和每个矩阵用例都必须提供对应 reference 哈希。矩阵汇总还绑定 baseline、candidate、reference、benchmark 与 cases/默认 ptr_size，避免旧 suite 被误用。基线一旦 seed 成功不能在原 run 重置，需要修改时新建 run。`max_regression_pct: 0` 表示该用例不允许性能回退。

自动 profiler 另用 `python CUSTOM.py SOURCE --warmup 1 --repeat 1 --ptr-size N --DIM=VALUE` 运行目标，不传 `--ref`/`--json-out`，避免混入 oracle；专用 benchmark 需支持此采集入口，或手动对项目的专用 repro 采集。reference 缺失的采集运行不能作为正确性证据。

## 机制复核

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
    baseline/                     # 基线入口、bench/日志
    iterv1/
      methods.json / analysis.md
      branches/b1/kernel.*        # 同方法组合的不同超参数
      branches/b1/bench.json
      branches/b1/workloads/      # case/seed的baseline与candidate
      branch_results.json         # champion的唯一入口
      kernel.* / bench.json
      best_input.xprof/或.hipprof/ # 每次采集独立时间目录
      dcu_top.json / roofline.json
      ablations/method_id/ / attribution.json
      isa_dump.txt / isa_check.json / mechanism-review.json
    summary.md
```

失败的同轮可修复后重新 close；成功 close 后进入新轮，不能覆盖旧 champion。open/close、独立 benchmark、分支选择、采集、ISA、消融和 roofline 写入入口均拒绝已关闭轮次；最终报告仍可重新生成。一目录只放一个候选入口，多个后缀不代表自动择优；已选择的 winner 以 branch_results 为准。不要并发写同一个 run。跨会话先读 state 和失败日志：当前脚本未提供自动分布式断点调度，Agent 选择尚未完成的阶段，必要时新建 run。发布前手动核对完整矩阵、项目单测、race 与端到端，不把 finalize 输出文件当所有验收已通过。
