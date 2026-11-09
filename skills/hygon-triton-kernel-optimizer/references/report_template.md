# HCU Triton 调查报告

`make_investigation_report.py` 按以下九部分生成草稿，自动填入的只是输入产物内容。Agent 补全实际验证和解释，缺项保持待确认；可合并简短小节，不为填写报告新增无关实验。

## 1. 问题和契约

目标模型/算子、输入输出、dtype/stride/alias/stream、精度、性能目标及独立 oracle。

## 2. 环境和来源

gfx、Torch/Triton/DTK/AICC、库与子模块提交、实际参考的资料；引用知识库时附 ID/SHA。使用 HCU 案例/PR/MR 或 NVIDIA KernelWiki 时，说明固定源码/位置、原架构与条件、review 结论/缺口及 HCU 差异，关联候选和结果；未查询不填表。

## 3. 工程目录与调用链

模型节点→生成代码→kernel 分派→实际 ISA/资源，包含工程路径和调用证据。

## 4. 捕获与复现

source SHA、执行前 snapshot、shape/stride/storage_offset/alias、launcher 参数、捕获错误/缺失。手写 Triton 使用独立 repro 时记录其入口，无须伪造 Inductor 捕获。

## 5. 热点假设与分析选择

准确 dispatch、原始 profile、定义/单位/范围；遇到瓶颈记录实际采集/分析、观察及下一项实验。工具受阻写原因和待采命令，诊断未完成；静态计数不替代动态测量。

按 [分析深度选择](../../hygon-hip-kernel-optimizer/references/hcu-workflow-contract.md#按难度与进度选择分析深度) 接起证据 → 所选分析/假设 → 候选/预期观测。已触发但暂缓时简记原因和重开条件；未触发不填跳过表。用 [上限与融合指南](../../hygon-hip-kernel-optimizer/references/performance-limits-and-fusion.md) 或 KRA 时再附影响决策的工作量/字节口径、可达参照及成本范围，不要求完整模型表。

## 6. 候选与正确性

源码差异、候选和配置、oracle、case/seed/尾块/实际数据、race/越界的实际检查状态。涉及融合、模型重写或 compile 边界时验证原语义；这些是可选路线。诊断性删工作探针与语义正确候选/消融分开。

## 7. 性能与工程验收

筛选/autotune 数据与独立确认分开；记录真实样本、噪声、未采集的 kernel 时间、编译/JIT 成本及完整模型收益。footprint rate 不叫 HBM 利用率。单 kernel 变快但模型未受益时写清反例及下一步。

## 8. 机制与实测反馈

指令/资源/时间线的前后变化是否支持第 5 节假设、消融与归因局限；模型估算和本轮实测分开。只读 ISA 后改 Triton/布局/编译配置时，附本次 JIT 特化、热点依赖和结果即可；静态模式不能推断动态瓶颈。

直接改写汇编/代码对象时，再按 [编译后优化指南](../../hygon-hip-kernel-optimizer/references/assembly-guided-optimization.md) 关联第 4、6 节的原/父/新产物与补丁身份、ABI/资源和正确性覆盖、实际加载且未被重编译覆盖的证据。源码调优无须补这套改写记录。

## 9. 结论、回退与后续

支持条件、失败、未覆盖项、硬件未跑部分、采用或停止原因、报告/原件/保留分支的位置。恢复方式与未解问题的重开条件；直接改写对象时提供恢复原件办法，不把独立 replay 通过当作真实模型通过。
