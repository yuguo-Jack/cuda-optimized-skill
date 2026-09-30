# HCU Triton 调查报告

1. 问题和契约：目标模型/算子、输入输出、dtype/stride/alias/stream、精度、性能目标。
2. 环境和来源：gfx、Torch/Triton/DTK/AICC、库与子模块提交、当前HCU知识ID/SHA。
3. 工程目录与调用链：模型节点→生成代码→kernel分派→实际ISA/资源。
4. 捕获与复现：source SHA、执行前snapshot、shape/stride/storage_offset/alias、launcher参数、捕获错误/缺失。
5. 热点假设：准确dispatch、原始profile与定义/单位/范围；静态计数不替代动态测量。
6. 候选与正确性：源码差异、所有配置的oracle、case/seed/尾块/实际数据、race/越界的实际检查状态。
7. 性能：真实样本和噪声、未采集kernel时间、编译/autotune成本、完整模型收益；footprint rate不叫HBM利用率。
8. 机制和消融：指令/资源/时间线变化与局限；无证据的归因保留待证。
9. 结论和回退：支持条件、失败、未覆盖项、硬件未跑部分、知识案例草稿及关联页更新建议。
