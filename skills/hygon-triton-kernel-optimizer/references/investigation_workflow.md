# Triton 调查步骤

## 1. 模型热点与契约

原始模型→调用图→生成源码→实际 dispatch 建立对应，记录 Torch/Triton/HCU 工具链与实际使用的资料来源。先用未采集运行保留端到端基线，再采系统热点和必要的单 kernel 指标。

## 2. 独立工作目录

每次实验指定独立 TORCHINDUCTOR_CACHE_DIR、TRITON_CACHE_DIR、TRITON_CAPTURE_DIR；不要清全局缓存。日志/缓存路径按实际运行用户和版本查，不能猜 /tmp/torchinductor_root。

首次 compiled-model 执行前导入捕获补丁。它记录执行前 storage/stride/offset/alias 和源哈希；捕获会增加同步/数据复制，不能拿此轮测端到端收益。无 autotune 不等于异常，先看缓存命中、单配置及实际 dispatch。

## 3. 复现与独立正确性

只重放可信文件。生成 replay 保持布局与同dtype alias，重新取得当前stream handle，但仅做 autotune 时间调查。读取安装版 CachingAutotuner 确认 mutated args 是否在每次试验间恢复；另写独立 oracle，逐配置核对。

混合dtype storage、函数对象launcher参数、不可序列化对象出现 capture-error 时写项目最小repro，不能把一个不完整目录当已捕获。

## 4. 差异与验证

检查 current HCU 分支的 metadata/lowering、最终代码对象及资源，提出一项明确假设。buffer ops、dot、assume、warps/stages 每次均按实际工具链编译探针。失败不全局强开环境变量。

baseline/variant 同 shape、数据、seed、布局、stream、恢复和测速口径，GPU 串行运行。多用例加性能回退门禁；非连续、alias、空/尾块、量化与路由分布以支持契约为准。

## 5. 回到模型

核对 graph break、额外转换、launch、通信/overlap 与端到端结果，必要时比较最小 eager 区域或模型重写。先证明语义保持和整体收益，再保留方案。报告引用原件、目录/符号、失败与回退，不把 standalone最快配置当整个模型收益。
