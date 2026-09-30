---
name: hygon-hip-baseline-generator
description: 从 Torch、Triton、TileLang、Python 或 CUDA 参考实现建立海光 HCU/DCU HIP 算子的独立正确性基线，梳理数值、布局和输入输出契约，生成保守脚手架并移交 HIP 优化流程。用于缺少可运行 HIP 基线、CUDA 到 HCU 移植和算子测试准备。
---

# Hygon HIP 基线生成

目标是得到符合原始语义的可验证基线和独立 reference。读取兄弟 HIP Skill 的 [共同契约](../hygon-hip-kernel-optimizer/references/hcu-workflow-contract.md)，重点是环境和数值/布局契约。三个 Hygon Skills 应一同安装。

## 1. 查原工程与输入契约

先查看输入算子的接口、调用点、shape/dtype/stride、输出、alias/in-place、stream、边界及容差，优先使用用户材料、当前源码/测试/文档与目标头文件。需要参考代码案例或补充领域事实时，可按需查询 HCU-Knowledge；查询不是必经步骤，也不要求安装知识库。

CUDA 移植按 API、线程模型、共享内存/同步、数值类型、矩阵指令、库接口分别检查。HIPIFY 可辅助替换，但不是兼容证明。HCU rocBLAS/hipBLASLt/MIOpen/RCCL 等接口资料与 AMD 源码分清；AICC 与 DTK 编译器分清。

选择独立 oracle：优先可读、语义完整的原 reference；Triton/TileLang/CUDA 参考必须保留，必要时另写 Torch/CPU 数学实现并与原实现对照。不要同步修改 oracle 与候选来让测试通过。

## 2. 使用脚手架的边界

```bash
python <skill>/scripts/inspect_ref.py --ref original.py --dims '{"M":128,"N":256,"K":64}' --out ref_analysis.json
python <skill>/scripts/generate_baseline.py --analysis ref_analysis.json --out-dir new_case
```

`inspect_ref` 静态读取 Python AST，不执行源文件。自动识别是提示，需要阅读正文确认。生成目录必须为空，以免覆盖已有基线。

产物：`ref_original.py`、`ref.py` adapter、`ref_analysis.json`、`kernel.hip`、`baseline_manifest.json`。manifest 的 `generated_unvalidated` 只代表生成成功，不是编译/正确性成功。

自动生成仅覆盖独立连续 FP32 的简单 elementwise 和 row-major GEMM。复杂 op、控制流、索引、cast、in-place、量化、半精度/双精度、多输出等必须人工实现/适配。已发现不支持时 manifest 为 `needs_manual_implementation`，kernel 带 `#error` 阻止占位 copy 被误当成正确基线。读懂并解决全部假设后才能去掉该 guard；禁止仅为编译而删除。

检查 adapter 是否按原函数参数名调用、二维 view 是否匹配、输出是否全部写入；AST 没有推断出的 dtype/layout 不视作 FP32 兼容证明。需要复杂分配时使用项目专用 harness，不强塞进通用 flat ABI。

## 3. 编译、正确性和交接

在实际 HCU 节点/项目容器中激活正确工具链。本地 Agent 可以编辑和读结果；没有 HCU 时完成静态准备并标明未运行。

```bash
python <hip-skill>/scripts/preflight.py --baseline new_case/kernel.hip --ref new_case/ref.py --dims '{"M":128,"N":256,"K":64}'
python <hip-skill>/scripts/benchmark.py new_case/kernel.hip --ref new_case/ref.py --M=128 --N=256 --K=64 --ptr-size 32768 --json-out new_case/baseline_bench.json
```

flat ABI 每指针容量至少 max(MK,KN,MN)，上述例子为 32768。benchmark 保留原精度比较，整数精确比较；支持范围、容差、NaN/Inf/空张量策略必须来自 contract，不凭脚手架默认值决定。

补多 seed、尾块、代表规模以及必要项目单测；编译通过、静态检查与硬件正确性分开。记录环境、reference SHA、实际输入、命令、日志与失败。基线正确后交给 `hygon-hip-kernel-optimizer`，默认 3 轮/4 分支，可按任务调整，无需为默认参数停下来询问。

交接包含 contract、原参考与adapter、manifest未解决项、基线源码、编译/正确性/测速状态、工作负载矩阵、实际使用的资料引用和远端运行方式，保存在当前任务工程。参阅 [转换与适配要点](references/ref_to_baseline_patterns.md)。
