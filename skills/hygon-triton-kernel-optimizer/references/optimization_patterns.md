# HCU Triton 候选优化

所有候选以当前工程和精确目标验证为准，需要补充资料时可按需查询 HCU 知识库；无统一 gfx 数值继承规则。以下是调查路线，不能保证收益。

| 方向 | 为什么可能有用 | 必须核实 |
| --- | --- | --- |
| 连续访问/布局与向量化 | 减少访问和地址开销 | 对齐、mask/tail、真实ISA、traffic/时间 |
| tile/num_warps/num_stages | 改变并行度、寄存器、LDS、流水 | 实际wave width、资源、驻留与网格覆盖 |
| dot/MMAC 路径 | 减少标量矩阵工作 | HCU编译器支持、数据布局、精度/scale、最终指令 |
| pipeline与预取 | 减少等待空泡 | 数据生命周期、同步、stage引入的VGPR/LDS成本 |
| fusion/epilogue | 减少launch和中间存储 | 寄存器压力、重算、端到端与所有消费者 |
| reduction/atomic重组 | 减少冲突与串行 | 顺序/确定性、索引范围、数值、稀疏/偏斜分布 |
| attention分块/分派 | 调整prefill/decode/稀疏/长短序列路径 | causal/mask、KV cache布局、softmax数值、整模型 |
| MoE grouped调度 | 改善小expert或不均衡 | router/sort/scale/GEMM/combine各阶段及EP通信 |

## 编译提示

`tl.assume`、`tl.multiple_of`、metadata 的 divisibility/range 必须由实际输入契约证明，对不满足条件的输入有正确分派/回退。指针整数表示不能随意推导正数约束。metadata parser只覆盖部分格式，缺失解析结果应回生成源码。

不默认设置 AMDGCN_USE_BUFFER_OPS。每个环境对真实 kernel 家族做 unset/显式启用两路探针，保留失败日志。buffer/global/flat 和 dwordx2/x4 名字本身不判快慢：查看实际load/store路径、寄存器、访存层级与测量。

## 模型层调整

小tensor初始化、scatter/atomic、融合后scalarized GEMM都是调查入口。比较初始化融合、批处理、gather/分段归约、库分派和最小compile边界；不能看到某种kernel就强制fallback。模型重写保留语义，并重新验证训练/推理必要的正反向与端到端性能。
