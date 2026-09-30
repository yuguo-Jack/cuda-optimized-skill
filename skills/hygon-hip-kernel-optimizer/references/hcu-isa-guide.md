# HCU 指令与数据通路查证指南

DCU 与 HCU 同义；历史 `dcu_*` 字段、文件名和策略 ID 保留。本文用于提出和验证优化假设，列出的指令来自已核对的资料与源码，并未在当前编写环境完成 HCU 硬件测试。目标编译器、代码对象和实测结果决定最终适用性。

## 1. 先建立四层对应关系

| 层次 | 记录内容 | 能证明的范围 |
|---|---|---|
| 源码 | API、builtin、inline asm、模板实例、指针地址空间 | 实现意图与调用契约 |
| 编译器 | 真实头文件/声明、LLVM IR、lowering、版本与完整参数 | 此工具链如何处理这段代码 |
| 目标代码 | 实际测量二进制 SHA、gfx、kernel 符号、ISA、资源描述 | 此构建生成了什么 |
| 实验 | 正确性/覆盖/同步、非采集测速、目标 dispatch 的 profile | 在给定输入和环境下是否正确、是否更快及原因 |

例如 `__builtin_amdgcn_raw_buffer_load_lds` 是源码入口名称，不是机器指令。不能在日志或 LLVM IR 里搜到它就声称已经看到 direct-to-LDS ISA。`__has_builtin` 为真也不能替代实际参数、地址空间、返回值和目标 lowering 的验证。

GFX938 Builtin 表中存在“所列名称不存在、实际为带 `_alt2` 的名称”的备注。这类表格应提供检索线索，使用时仍要读已安装头文件/当前源码调用点并编译最小例子。[E7]

## 2. MLS、LDS 与 MMAC：先核对布局和代际

| 名称 | target |
|---|---|
| 孔明 / kongming | gfx926 |
| 孔明e / kongming-e | gfx928 |
| 伯温a0 / bowen a0 / 北美洲 | gfx936 |
| 伯温b0 / bowen b0 / 南美洲 | gfx938 |
| 少伯 / shaobo | gfx946 |
| 塞班b1 / saipan b1 | gfx948 |
| 月英 / yueying | gfx92a |

这些名称是身份映射，不是能力矩阵。gfx92a 是完整目标名；某个测试内部用数值宏9210，不可传给 `--offload-arch`。更早架构按实际任务单独查证。gfx 编号不能按数值大小推导功能；同属 `matrix_load` 也不能沿用 descriptor、tile 和同步设置。

以下只列旧少伯资料原表明确支持的部分形式，不能当作当前所有代际的完整兼容矩阵。[E2，PDF 第 4–5 页，已看原表] 新 Builtin 表还区分月英/塞班形式；阅读时检查整行、重载和真实合并单元格，空白不自动继承上一行。支持列与备注不一致时保留差异，再核对目标编译器与用例。

| 指令形式 | 原表涉及的 HCU 目标 | 使用时的注意点 |
|---|---|---|
| `matrix_load_b8` / `matrix_load_128x8_b8`、`matrix_load_b16` / `matrix_load_64x8_b16` | gfx936、gfx938 | 旧形式在 gfx946 列标为删除，不能直接复用 |
| `matrix_load_64x16_b8`、`128x16_b8`、`64x32_b8`，以及 `32x16_b16`、`64x16_b16`、`32x32_b16` | gfx938、gfx946 | 同名形式仍需核对实际 descriptor、地址和布局 |
| `matrix_load_128x16_b4`、`matrix_load_256x16_b4` | 表中仅 gfx946 | 低位宽打包、tile 边界与转换语义需单独验证 |
| `matrix_store_*` | 文档列为 gfx946 新增 | LDS 修饰决定数据来源路径；不是 load 的简单逆操作 |

分析矩阵 kernel 时画清：GLOBAL → LDS 或 VGPR → lane/register 排布 → MMAC 输入 → 累加器 → epilogue/store。每一步标出逻辑元素、实际字节数、对齐、转置/交织、有效 lane、尾块及同步点。

gfx936 的数学库 GEMM 示例同时出现 `ds_read_m32x16_b16` 与 `v_mmac_f32_16x16x16_bf16`，是可以阅读布局与供数过程的源码实例。[E8] 它不证明该汇编可原样用于 gfx938/gfx946，也不证明任意 shape 都受益。

优化 MLS/DS/MMAC 组合时，优先核对：

1. 目标格式是否在编译器支持范围内；descriptor 字段和地址单位是否一致。
2. LDS swizzle/stride 与矩阵读、MMAC lane 映射是否成套匹配。
3. global→LDS 完成、LDS 读完成、跨 wave 可见性分别由什么保证。
4. BPS/bypass 路径是否需要对应等待域；普通 `vmcnt/lgkmcnt` 不能凭习惯替代文档要求。少伯资料列有 `s_waitcnt_vbcnt`。[E1]
5. 检查 MMAC 利用、LDS 冲突、访存量和等待的变化，再判断流水是否改善。指令数量减少并不自动意味着更快。

## 3. 少伯低精度与 scale 数据通路

少伯资料描述 scale-aware MMAC、转换和 scale 数据搬运，例如：

- `v_mmac_scale_f32_16x16x32_fp8_bf8`、`v_mmac_scale_f32_16x16x64_fp4`。
- `v_cvt_scale_*`，包括 FP8/FP4 等格式的转换路径。
- `ds_scale_copy_buf2ds`、`ds_scale_copy_ds2buf`。[E1，PDF 第 7–10 页]

需一起明确输入编码、scale 编码/分组粒度、量化/舍入/饱和策略、累加精度、打包、尾块与 scale 的生命周期。不能只替换矩阵指令就认为完成低精度优化。

验证包含极值、小值、零、分组边界、尾部 mask 和真实分布；按业务契约约定 NaN/Inf。比较精度误差与端到端成本，不能把转换、scale 构建或 epilogue 移到计时外制造收益。编译器支持某个 spelling 与硬件可正确运行分别记录。

## 4. WDRA 与 Co-Issue

文档给出了 `__builtin_hcu_s_set_vgpr_size(short N)` 到 `s_set_vgpr_size N` 的路径；另有 `_prsv` 形式及参数形式差别。[E3 第 6 页；E1 第 5 页] 这些是查证入口，不应自行猜测 intrinsic 原型。

新编译器资料中，基础 Ebarrier 与 `s_set_vgpr_size` 出现 gfx92a/gfx946 的 feature 条件，而部分 Ebarrier reduction 变体只列 gfx946。因此旧少伯文档的范围不应变成“月英没有这些能力”的断言。反过来，编译器声明接受也不证明任意设备/编译分支均能运行。WDRA 初始化接口有版本撤回情况，必须查当前声明和 lowering，不能复制旧示例后只凭名字补 intrinsic。

WDRA 调整 wave 间 VGPR 分配，不会凭空增加整个 threadgroup 的寄存器预算。资料中的波组、分配粒度与 phase/barrier 约束要结合实际 kernel 核对：wave 分工、谁释放/申请、是否仍有存活值、跨 wave 数据与同步是否安全。不能声称一条 `s_set_vgpr_size` 就能提高初始 occupancy。

实测应检查寄存器/驻留限制、spill/scratch、生产者与消费者等待和整体耗时。减少某一角色 VGPR 若导致 spill 或另一角色等待，可能适得其反。

Co-Issue 有明确指令配对限制。资料对 gfx946 的 AI-MMAC pass 数和可搭配 VALU 作了限定；非 AI 的 F32/F64 MMAC、VOP3R、`v_swap` 及部分 LDS/GPR/scale 组合不能被泛化为“任意 MMAC 与 VALU 都并发”。[E3 第 23 页] 静态相邻排布不证明动态重叠，需要调度/时间线证据。

## 5. Abarrier、Ebarrier 与 TLS

### Abarrier

按 arrival count 和 transaction count 两者完成条件、phase、slot、参与者和生命周期检查。init/invalidate、arrive/drop、expect_tx、wait 的相对次序均属于正确性契约。bypass 与非 bypass 的 transaction 计数单位可能不同，应以所用路径的指南为准，不能混用字节数与指令数。[E4]

不要把某种 `try_wait` 名称直接理解成“绝不会阻塞”。测试必须覆盖多次循环/phase 翻转、边界工作量和不均衡参与者；单轮正确不能排除死锁。超时要报告为失败，不能删掉同步后只看速度。

### Ebarrier

区分 arrive 与 sync、显式 wave count 与全 threadgroup 默认行为，以及 AND/OR/POPC 的参与语义。不同 wave 子组复用同一 slot 时要遵循文档规定的整组同步条件，防止上一轮参与者污染下一轮。[E5]

### TLS

`tensor_load` 需要正确的 tensor descriptor、SGPR 对齐、LDS 目标与坐标/偏移单位；原文明确区分元素坐标与 LDS 字节偏移。描述符解释和多维越界处理必须来自目标资料与真实调用点。[E6]

编码表出现 `OP=1 tensor_store` 不足以证明当前工具链已实现可用的 store 路径；资料正文着重描述 load，不能仅凭预留编码承诺支持。[E6 第 15 页]

## 6. 其他容易误读的线索

- `glc/slc` 等修饰含义依具体操作而定；例如 atomic 返回值行为与一般 cache/coherency 策略不能混为一谈。
- `multimem_*` 在指令资料中出现，不代表普通单设备内存可直接套用。先核对多设备映射、归约类型、原子性、可见性与真实通信场景。[E1 第 14 页]
- `SQ_WAVES` 累积波数不等于驻留 wave；理论 occupancy、实际活跃、可发射和数据依赖等待是不同维度。
- 脚本输出是静态指令位置数量，宏未展开、函数混合、循环执行次数、分支路径都可能影响解释。`scan_amdgcn.py --kernel SYMBOL` 必须唯一匹配实际汇编符号，匹配不到不能回退扫描整文件后称作该 kernel。
- builtin、IR、汇编注释、工具报错和其他 kernel 指令不应计入目标 ISA 命中。`sass_check.py` 的无 symbol 扫描仅提供整个代码对象的线索，机制复核必须定位具体符号。

## 7. 最小验证流程

1. 写出待验证能力、精确 gfx、编译器/头文件版本、来源和预期 lowering。
2. 从已知调用点构建最小编译探针；保存命令、诊断、目标代码及资源。
3. 对实际测量的二进制定位 kernel 符号，核对操作数、地址空间、descriptor、layout、wait/barrier。
4. 运行独立 oracle、输出覆盖、边界/多 seed 与同步检查。仅编译通过只记为“可编译”。
5. 对优化前后做同环境新鲜对照；结果需超过重复测量波动，筛选最快候选后再独立确认。
6. 遇到瓶颈，按 [XProf/XCompute](xprof-xcompute-guide.md) 或 [hipprof](hipprof-guide.md) 的独立流程采集，解释指标/资源/时间线与瓶颈的关系。两套工具参数、格式和指标不得混用。
7. 报告支持范围与反例；多个耦合修改的总体收益不能平均分给每条指令或方法。

### 用 ISAtest 查例子时

查宿主注册/兼容规则与黑名单，再读设备 kernel、输入初始化和 golden。注册矩阵是测试选择逻辑，不是硬件完整向后兼容表；0 tests 或 skip 不算通过。存在性环境变量设成字符串 `0` 仍可能开启，检查实际 `getenv` 判断。

编译缓存可能只比较源文件时间与 ELF target。更换编译器、头文件或 flags 后用独立目录或工程提供的清理目标，避免旧 code object 被当成新结果。核对宿主/设备缓冲长度（`sizeof(pointer)` 不是 pointee 大小）、动态 LDS、descriptor、数值比较规则；测试仓本身也可能有缺陷。功能测试总时长包含初始化/编译/拷贝，不能当指令 latency。

### 新接口与多编译器

AICC 的 MMAC 入口可能采用无后缀重载，DCC 的声明/参数编码与后端树需独立核对。省略累加器的重载可能默认零，不能替代原有 `C += A*B`。MLS 旧返回向量接口与 descriptor 搬运接口、VMEM atomic 的 GLC/返回值限制都要逐目标核验。DCC 内有顶层 LLVM 与 GCVM 两套目标实现时，先确认构建开关实际选中了哪套；分支名不能代替 LLVM、DTK 和产品版本。

## 8. 本次查证来源

以下是可选知识库中的稳定资料 ID 与原件 SHA256。未安装知识库时可用同标题、版本的原件或目标工程源码核对，Skill 不依赖固定知识库目录。页码按 PDF 物理页计数。

| 编号 | 资料 | 知识条目 ID | 原件 SHA256 |
|---|---|---|---|
| E1 | part1-指令.pdf | `bbd11339f2a4a17c333b1493` | `2b08b4f2206310a1c3673454b86eb9e5370eba4f04a60f311ec68636bce8cc60` |
| E2 | part2 - MLS.pdf | `b2aeb123c20231b01a2d2cf6` | `50b03859660cd7ab61dc25512b74ae9abeb79b556ae665d7c9c2f1855ea64765` |
| E3 | part5 - WDRA+Co-Issue.pdf | `e6c2f797c0876cd6e230ae57` | `bb75bcb22160d571fc1af5ba3b3d5b872402fa8c316db22eee6dafb2e356231b` |
| E4 | Abarrier编程指南.pdf | `72796d6d4afdec1fca3ab520` | `938331ed6a012936874689369348db55aa2007683fc9bea3e8677a52578abc9a` |
| E5 | Ebarrier编程指南.pdf | `69680ec2e8af6db3cdb43b6f` | `4b252449479358350327b543668a54b12ac255b9cf7d3ed66f2d70a565c4b73b` |
| E6 | patr3 - TLS1.0.pdf | `fe5c04dd00da3fe5af76a5cf` | `8af0cab7e172e80194aaa9503b1d0cf8355d360af3a9ed2aa7f01ce09b7f17d0` |
| E7 | GFX938新增Builtin汇总.xlsx | `828850a3f9d2ec6e9348f9c9` | `6af7d62a4d5f75bb3076a21354c023a1cef835a3bb64f6a5cc98cd84c40a6420` |
| E8 | GEMM MT256x256x16 汇编示例；target 第 8 行、DS 第 1143 行、MMAC 第 1182 行附近 | `90d32e8c6c71861dfeb6f044` | `29c2a123e12181a02b9bcac8bc5a8a5403374c5b102baa9ce4dd5d7e79035153` |
