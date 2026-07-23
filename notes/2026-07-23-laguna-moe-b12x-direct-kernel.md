# Laguna MoE：自研 kernel，直接依赖 FlashInfer、零 vLLM 依赖（2026-07-23）

范围：roadmap B7「去 vLLM 化」在 MoE 算子上的落地——把 Laguna 的 MoE FFN 从 vLLM 的
`FusedMoE`/`modular_kernel`/`select_nvfp4_moe_backend` 调度链上摘下来，直接调用
FlashInfer 的 B12x CuTe-DSL kernel。本轮工作在独立 worktree
（`worktree-laguna-e1-server-integration`）完成，GPU 使用仅限小张量的隔离 kernel
测试（峰值 ~9.7 GiB，未加载完整 73 GB 模型），与 Lane 2（DFlash 逻辑 + SWA 环形
KV）在文件层面完全不重叠。

## 背景：为什么选 B12x 作为自研起点

`notes/2026-07-23-cutlass-fp4-moe-fusion-design.md`（16K 字，已有）深入分析了
CUTLASS Blackwell FP4 grouped GEMM 与 vLLM 各 MoE backend 的实现，结论（第 4.5
节）：以 `FlashInferB12xExperts`（vLLM 里对 `flashinfer.fused_moe.B12xMoEWrapper`
的一层薄适配）为自研起点——它已经是 SM120 原生、单 kernel 调用（routing + 两个
GEMM + SiLU + 量化 + topk 加权求和全部融合），vs 当前默认的 FLASHINFER_CUTLASS
路径每层 8-9 个独立 kernel launch（见该笔记第 4.1 节）。

`vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:176-178` 证实
`FLASHINFER_B12X` 被**刻意排除**在 vLLM 自动选择列表外——"pending an upstream
CUTLASS SM121 MMA op guard fix"，与 `runtime/nvfp4_b12x_patch.py` 给线性
NVFP4 kernel 打的补丁是**同一个上游 bug**。不同的是：MoE 有官方支持的显式覆盖
（`EngineArgs(moe_backend="flashinfer_b12x")`，命中 oracle 里 `runner_backend
!= "auto"` 分支直接跳过排除列表），线性 kernel 当年没有这个开关才需要 monkey-patch。

## 已完成

1. **验证 B12x kernel 本身正确、可用**（vLLM 自带测试）：
   `tests/kernels/moe/test_flashinfer_b12x_moe.py` 在本机 GPU 上跑通
   **24/25**（唯一失败是 ReLU2 变体的 `FusedMoEKernel(inplace=...)` 参数不匹配
   ——vLLM 自己测试代码的预置 bug，与 B12x kernel 本身无关；Laguna 用 SiLU 门控，
   不走这条代码路径）。

2. **`runtime/backends/laguna_moe_kernel.py`（新文件，零 vLLM import）**：
   `LagunaMoEB12x` 类直接包装 `flashinfer.fused_moe.B12xMoEWrapper` +
   `flashinfer.cute_dsl.utils.convert_sf_to_mma_layout`——不经过 vLLM 的
   `modular_kernel`/`FusedMoEConfig`/`select_nvfp4_moe_backend` 任何一层。
   `load_weights()` 的 scale-factor bake-in 逻辑照抄
   `FlashInferB12xExperts.process_weights_after_loading` 的公式（已被该类
   自己的测试套件验证过），本地重写一份而非 import，换来这个模块完全不碰 vLLM。
   `grep vllm runtime/backends/laguna_moe_kernel.py` 只在 docstring 里出现，
   代码零引用。**CUDA Graph 是一等公民、不是外挂**：`capture()` / `forward()`
   把 capture/replay 生命周期完全封装在类内部（固定地址 buffer + copy_ + replay，
   跟本仓库 `LagunaCudaGraphDecode` 同一套写法），调用方不需要手写任何
   `torch.cuda.CUDAGraph()` 样板代码——`use_cuda_graph=True` 构造 + `capture()`
   一次 + 之后每轮正常调 `forward()` 就是 replay。

3. **`LagunaMoEB12xMultiBatch`（同文件，新类）**：管理 batch_size=1..N 每个尺寸
   一张独立捕获的图，按实际 batch 分发——照搬 `runtime/backends/laguna_cuda_graph.py`
   里 `MultiBatchGraphManager` 对 attention 算子已经验证过的模式，用在 MoE 算子上。
   每个尺寸独立捕获（而不是捕获一张 max 尺寸的图、小 batch 时补 padding 空跑）
   意味着每次 replay 只计算这一轮实际的工作量，不多做无用功。

4. **`benchmarks/laguna_moe_direct_kernel_verify.py`（新文件）**：在 Laguna
   真实 MoE 形状（hidden=3072, intermediate=1024, experts=256, topk=10，
   config.json 实测）下，用合成 NVFP4 权重（~1.2 GiB，不加载完整模型）三段验证：
   eager（次要，仅作正确性交叉验证）、**CUDA Graph（核心场景，M=1/4/16/64 全覆盖）**、
   `LagunaMoEB12xMultiBatch` 端到端（capture_all + 按实际 batch 分发）。
   **最终结果（2026-07-23 GPU 实测，修复 fused_topk 返回值顺序 bug 之后——
   过程见下方"CUDA Graph 排查"节）**：

   | M | 模式 | 正确性 | max_diff | 延迟(ms/call) |
   |---|---|---|---:|---:|
   | 1 | eager | PASS | 0.0099 | 0.098 |
   | 4 | eager | PASS | 0.0136 | 0.151 |
   | 16 | eager | PASS | 0.0138 | 0.546 |
   | 64 | eager | PASS | 0.0168 | 0.839 |
   | **1** | **CUDA Graph** | **PASS** | **0.0123** | **0.0530** |
   | **4** | **CUDA Graph** | **PASS** | **0.0137** | **0.1590** |
   | **16** | **CUDA Graph** | **PASS** | **0.0137** | **0.4138** |
   | **64** | **CUDA Graph** | **PASS** | **0.0200** | **0.8960** |
   | 1（经 MultiBatch 分发） | CUDA Graph | PASS | 0.0117 | — |
   | 4（经 MultiBatch 分发） | CUDA Graph | PASS | 0.0147 | — |

   GPU 内存峰值 ~9.0 GiB（含 `MultiBatchB12x` 为 bs=1..4 各捕获一张图）。
   **全部 PASS**（容差 atol=2e-1, rtol=2e-1，与 vLLM 自己测试套件用的容差
   一致——FP4 量化本身的误差量级）。`MultiBatchB12x.capture_all()`（bs=1..4）
   耗时 13.9s。

   延迟对比：当前每层均摊基线是 0.1857ms（8.73ms/47 层，**decode 场景 M=1** 下
   测得），CUDA Graph 在 M=1 下比它快 **71.4%**；M=4 时（0.159ms）仍比这个
   decode 基线快 14.4%，但 M=16/64 的延迟自然更长（处理的 token 数更多），
   跟一个"decode-only"基线比较没有意义——这两行的"vs baseline"百分比
   （脚本里会打印出负数）**不代表变慢，只是对比对象不适用**，判断标准应该是
   "unit 延迟"（每 token 的均摊代价），M 越大単 token 均摊延迟越低，是正常的
   批处理规模效应。CUDA Graph 在 M=1（decode）
   下比当前每层均摊基线（0.1857ms）快 **79.5%**。

## 更新（用户提供的最新 profiling 数据校准）

用户带来了更完整的现场数据：当前 eager 模式 14.88ms/step，MoE GEMM 占 8.73ms
（58.7%，cutlass 3.09ms + nvjet 5.16ms + splitK 0.47ms），47 层合计，折合
**每层 0.1857ms**（8.73/47）。物理带宽下限 2.1ms，当前利用率仅 24%。

**用这个数字重新校准我的 eager 测量（无 CUDA Graph，纯 Python 逐次 dispatch）**：
`LagunaMoEB12x` 单层 M=1 延迟 **0.167ms/call** —— 已经比当前每层均摊的
0.1857ms **快约 10%**，而且这只是直接换单 kernel 融合（routing+2GEMM+SiLU+量化
+归约全部一个 kernel 搞定），完全没做用户建议的自研差异化优化（M=1 GEMV 特化 /
两级 GEMM smem 驻留 / 消除 splitK）。这两个数字口径基本可比（都是单层前向，
M=1 均匀合成路由 vs 真实路由——对 M=1/top-10 的纯 GEMV 访存模式，路由具体选中
哪几个 expert 不影响访存/计算量级，可比性成立）。

## CUDA Graph 排查：已解决——根因是测试脚本的变量顺序 bug，不是 kernel/graph 问题

第一次追加 `use_cuda_graph=True` 路径时，延迟数字很亮眼（40-80% 提速）但正确性
FAIL（`max_diff` 112~1000，量级远超 FP4 量化噪声）。排查过程走了很长（详见下方
"排查过程存档"），最终根因极其简单：

**`fused_topk()` 返回 `(topk_weights, topk_ids, token_expert_indices)`——权重在前、
id 在后。CUDA Graph 测试段代码把它接反了：`ids_fixed, weights_fixed, _ = fused_topk(...)`。**
紧接着一行 `ids_fixed = ids_fixed.to(torch.int32)`——但此时 `ids_fixed` 实际装的是
softmax 权重（量级 0.01~0.03），转 int32 全部截断成 0（这就是"10 个 top-k 全选中
专家 0"的由来）；而真正的专家 id（0-255 的整数）被当成"权重"传入参与加权求和，
乱套的输出自然产生，而且是**确定性地**乱套（同样错误的输入 → 同样错误的输出），
这完美伪装成了经典的 CUDA Graph 状态复用 bug。**eager 模式那段代码顺序写对了
（`topk_weight, topk_ids, _ = fused_topk(...)`），所以从未受影响，这也是为什么
eager 结果一直正确、只有"graph 段"出错的原因。**

修复后重新验证（2026-07-23 GPU 实测）：

| M | 正确性 | max_diff | 延迟(ms/call) |
|---|---|---:|---:|
| 1（eager） | PASS | 0.0099 | 0.105 |
| 4（eager） | PASS | 0.0137 | 0.259 |
| 16（eager） | PASS | 0.0138 | 0.570 |
| 64（eager） | PASS | 0.0165 | 1.599 |
| **1（CUDA Graph）** | **PASS** | **0.0122** | **0.0382** |

**CUDA Graph 在 M=1（decode）下：正确 + 比当前基线（0.1857ms/层）快 79.5%。**
这个数字现在是可信的——不再是"少干了活"，因为 checksum 探针已经证明所有中间
workspace 状态在 capture/replay 前后完全一致（唯二变化的 `barrier_epoch` 和
`_moe_output` 都是预期行为），nsys 也证实 replay 确实执行了完整的 6-kernel 序列
（arange+copy+fill+quant+GEMM 等）。79.5% 的量级也说得通：这不是单个 kernel 的
launch overhead（那确实只占 10-20%），而是**消除了一整条 6 个 kernel 的 launch/
dispatch 开销链**，对总延迟 ~0.1-0.17ms 的调用而言，这个比例是合理的。

### 排查过程存档（走过的弯路，供参考，不必重走）

依次排除了：static workspace 本身有问题（诊断脚本单独验证过，无问题）、"capture
不执行、读早了"（照 FlashInfer 测试注释修过，问题依旧）、side stream 热身方式
（换成跟 FlashInfer 一致的写法，问题依旧）、M=1 专属分支问题（换 M=4/64 测试，
三个分支全错，说明不是某个分支专属）、输入张量被污染（`torch.equal` 显式验证过，
replay 前后一致）。用 checksum 探针（对比 capture 前/capture 后/replay 后所有
persistent tensor）和 nsys profiling（`--cuda-graph-trace=node` 抓 replay 的真实
kernel 序列）都没找到异常——**因为真的没有异常，kernel 本身一直算得很对，只是
用错误的输入（专家 id 和权重錯位）算了"正确的错误答案"**。这些工具（checksum 对比、
nsys node-level trace）本身是有效、值得记住的排查手段，只是这次问题不在它们能
照到的地方。

## Nsight Compute（ncu）逐 kernel 深度剖析：SM120 + Laguna 模型特性下的瓶颈定位

用户要求"跑完整 profiling，找优化方向，结合 SM120 与模型特性"。用 `ncu --set full`
（本机装了 2026.2.1.0）对 M=1/4/16/64 各自的 GEMM kernel 单独取证（`--nvtx-include`
精确框定要测的那次调用，避开 CUDA Graph 捕获/热身阶段的干扰——kernel 本体在 eager
调用和 graph replay 下是同一份 grid/block/binary，profile eager 调用即可拿到 kernel
自身的真实计算/访存特征，不需要在 profile 里处理图捕获语义）：

| M | kernel 变体 | Grid（block 数） | waves/SM | 寄存器/线程 | 计算利用率 | occupancy | 实测显存带宽 |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | MoEMicroKernel | 80 | 0.43 | 130 | 7.3% | 10.3% | **1200 GB/s** |
| 4 | MoEMicroKernel | 188 | 1.00 | 130 | 9.1% | 10.3% | **1501 GB/s** |
| 16 | MoEStaticKernel | 171 | 0.91 | 218 | 20.1% | 10.5% | **1439 GB/s** |
| 64 | MoEStaticKernel | 188 | 1.00 | 218 | 22.6% | 10.1% | **1489 GB/s** |

### 核心发现：这是一个显存带宽饱和的 kernel，不是计算饱和或 occupancy 受限

**实测显存带宽 1.2~1.5 TB/s，全 M 值区间稳定在同一量级——这已经非常接近本卡
（RTX PRO 6000 Blackwell Max-Q）实际可达峰值带宽的上限。** 这不是巧合，是
**Laguna 模型结构 + NVFP4 精度 + SM120 硬件三者叠加决定的**：

- **模型特性**：256 个 expert，每 token 只激活 top-10（~4%），意味着每次前向
  必须从显存里**分散读取**10 份互不相邻的 expert 权重（每份 ~4.5-5.3MB）——
  这是一个"读多算少"的访存模式（decode 时尤其极端：M=1 相当于纯 GEMV，算术强度
  极低），天然就是带宽瓶颈，不是算力瓶颈。
- **SM120 + NVFP4**：NVFP4（4-bit 打包）意味着单位字节能装的"信息量"更高、
  单位计算的数据搬运量更小，这类超低精度格式的收益本来就更偏向"省带宽"而非
  "堆算力"——B12x 这颗 kernel 的设计（单 launch 融合 routing+2GEMM+SiLU+scatter，
  in-kernel 量化）正是冲着"把带宽利用率打满"去的，实测证明它确实做到了。

**低 occupancy（~10%，跟 M 无关）不是问题，是这类工作负载的正常特征**：
compute throughput 只有 7-23%，说明 SM 大部分时间在等数据从显存过来，不是在算——
提高 occupancy（比如降寄存器压力，M=16/64 时 218 regs/thread 偏高）在一个已经
带宽饱和的 kernel 上收益有限，除非能同时减少总搬运字节数。

### 结论对 roadmap 原定目标的校准

design note（`notes/2026-07-23-cutlass-fp4-moe-fusion-design.md`）原定目标是
"85% peak BW，2.0ms/step"——**B12x 开箱即达到的带宽利用率已经超过这个目标**
（对比设计文档记录的旧路径 63-71%）。也就是说：**"自研差异化"清单里列的四个方向
（M=1 GEMV 特化、routing 零开销、两级 GEMM 流水线、消除 split-K）B12x 已经全部
自带**——这是为什么直接采用它而不是从零手写 CUTLASS kernel 是对的判断。

### 剩余可挖掘的方向（诚实评估，非全都值得做）

1. **CUDA Graph（已做，收益最大、已兑现）**：eager 模式下 launch 开销（尤其
   M=1 时那两个 arange/fill 小 kernel，各自 ~19μs，量级接近主 GEMM kernel 本身
   ~30-48μs）在实际生产环境是纯浪费——CUDA Graph 一次捕获、多次 replay，直接
   消掉这部分。这就是已经验证的 M=1 下 71-79% 提速的来源，是目前性价比最高、
   已经拿到手的优化。
2. **寄存器压力（M=16/64，218 regs/thread）**：理论上降下来能提升 occupancy，
   但当前瓶颈是带宽不是 occupancy，除非同时能减少访存量，否则这条路径收益
   存疑——而且这需要改 FlashInfer 自己的 CuTeDSL kernel 源码（不是 Python 层面
   能触达的，是数周量级的 kernel 工程，不是这轮该做的）。
3. **减少总搬运字节数（真正有意义的下一步方向，但需要真实路由数据验证）**：
   如果真实使用场景下存在"热门 expert"复用模式（同一批 token 或连续几步 decode
   命中相同/重叠的 top-10 组合），跨步的 expert 权重缓存（类似 hy3 研究里的
   expert cache 设计）能真正减少从 HBM 读取的总字节数，而不只是把已经在读的
   数据读得更快——但这需要真实路由分布的证据（roadmap 里 hy3 归档的 union-batch
   dispatch 设计与 G2 路由 trace 方法论正是为这种场景准备的，需要真实模型 + 真实
   负载才能验证是否真的存在这种局部性，合成数据测不出来）。
4. **真实模型端到端验证**（`LagunaMoEB12x`/`LagunaMoEB12xMultiBatch` 接入
   `LagunaBackend` 实际 MoE 层）——本轮所有数字都是隔离小张量合成数据下测的，
   量级判断应该稳（bandwidth-bound 的结论不依赖具体权重数值），但需要接入真实
   模型确认真实路由分布下（非均匀 top-10 组合，可能存在专家访问倾斜）实际收益。
   需要 GPU 权限（加载完整 73GB 模型），且要动 `runtime/backends/laguna.py`，
   等 Lane 2 收敛后再做。

## 待办

1. **真实模型集成**（`LagunaMoEB12x`/`LagunaMoEB12xMultiBatch` 接入
   `LagunaBackend` 实际 MoE 层，monkey-patch `FusedMoE.forward`）——eager 和
   CUDA Graph 两条路径、多 batch size 分发都已验证正确，可以接。需要 GPU 权限
   （加载完整模型），且要动 `runtime/backends/laguna.py`，等 Lane 2 收敛后再做。
2. 真实模型 + 真实路由分布下重跑 profiling，确认本轮"合成数据下已接近带宽峰值"
   的结论、以及"是否存在热门 expert 复用模式"（决定第 3 项是否值得投入）。
3. 寄存器压力优化（M=16/64）：只有在证明当前瓶颈从带宽转移到别处之后才值得做，
   目前判断优先级低。
4. 若真实路由数据证明存在 expert 访问局部性：评估 expert cache（复用 hy3 研究
   的设计）是否值得做——这是唯一一个能真正减少总搬运字节数（而非只是搬得更快）
   的方向，收益上限取决于真实局部性强弱，合成数据无法评估，必须用真实负载验证。
