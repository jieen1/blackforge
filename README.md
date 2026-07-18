# Qwen SM120 Runtime

面向单张 RTX PRO 6000 Blackwell（SM120，95.59 GiB 可用显存）的
Qwen3.6-27B-NVFP4 专用推理 runtime。目标不是通用 serving，而是在最多四个
agent 并发的真实编程负载下，压低单请求延迟并提高 accepted tokens/s。

## 当前决策基线

- 当前主线已经从 HTTP bridge 转向直接模型执行器；最新事实、验证结果和下一步
  只以 [PROGRESS.md](PROGRESS.md) 为准。
- 真实 `nsys` 分解显示：NVFP4/FP8 GEMM 占 GPU kernel 时间约 **76%**，GDN
  约 **8%**，full attention 约 **1.5%**。因此近期优先级是 GEMM/权重布局、
  四并发+MTP 形状下的 launch-gap 复测、直接拥有 KV/GDN 状态，然后才是 GDN
  和 attention 的进一步优化。
- 项目完整阶段合同见 [项目实施规划.md](项目实施规划.md)。实现不得绕过其中的
  correctness gate、端到端 A/B 和真实 agent workload 验收。
- 本仓库已有并行开发中的未提交文件。学习外部源码时不得顺手清理、覆盖或回退
  其他人的工作区。

## Agent 开始工作前的阅读顺序

1. 阅读本 README，理解哪些外部项目是 oracle、算法参考或负面证据。
2. 阅读 [PROGRESS.md](PROGRESS.md) 的最上方最新记录和 `Next Work`，不要从旧阶段
   结论重新开始。
3. 阅读 [项目实施规划.md](项目实施规划.md) 中与任务对应的 phase 和 gate。
4. 再按下方 P0/P1 索引阅读外部源码。先记录调用链、tensor layout、支持架构和
   license，再提出移植方案。
5. 所有性能结论必须回到本机 SM120、batch/slot 1--4、MTP verify 形状和真实模型
   上验证；不能用 H100/B200/RTX 5090 的结果代替。

## 外部源码使用规则

- `/home/bot/project` 下的参考仓库默认只读。不要在参考树里实现本项目功能。
- 引用设计时记录仓库、commit、文件和适用 shape；复制代码前单独检查该文件及
  仓库 license。不要因为仓库整体开源就假定所有子目录可派生。
- CUTLASS 只使用 C++ template/CuTe C++ 路径。不要修改或派生另行授权的
  `nvidia-cutlass-dsl` 包。
- `kekzl-imp` 是 MIT 第三方只读参考；只用于交叉验证布局和数值方法。
- “支持 Blackwell”不等于支持 SM120；必须在 dispatch、CMake arch gate 和 PTX
  opcode 三处确认。SM100/SM103 的 `tcgen05` kernel 不能直接移植到 SM120。
- kernel microbenchmark 变快不代表 runtime 变快。每次落地都要同时给出正确性、
  kernel 数据、端到端 ITL/accepted tokens/s 和 launch 数量。

## P0：当前主线必须学习

### 1. vLLM：正确性 oracle 与现有模型语义

- 本地：[`/home/bot/vllm`](/home/bot/vllm)，快照 `e12b91b03`。
- 上游：[vllm-project/vllm](https://github.com/vllm-project/vllm)
- 重点入口：
  - [`qwen3_5.py`](/home/bot/vllm/vllm/model_executor/models/qwen3_5.py)：64 层混合
    模型、权重命名和 forward 语义。
  - [`qwen_gdn_linear_attn.py`](/home/bot/vllm/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py)：
    GDN 状态、metadata 和 kernel 调用链。
  - [`registry.py`](/home/bot/vllm/vllm/v1/attention/backends/registry.py) 与
    [`cuda.py`](/home/bot/vllm/vllm/platforms/cuda.py)：自定义 backend 的注册和优先级。
- 学习目标：提取模型层并直接驱动本项目的 slot-addressed KV/GDN buffer，而不是
  把 vLLM scheduler 搬进来。
- 边界：vLLM 是 oracle 和集成参考，不是最终 hot-path runtime。

### 2. SGLang：低延迟 runtime、GDN、MTP 与 SM120 低精度 dispatch

- 本地：[`/home/bot/project/sglang`](/home/bot/project/sglang)，已更新到
  `b296e1a503`（2026-07-16）。
- 上游：[sgl-project/sglang](https://github.com/sgl-project/sglang)
- 重点入口：
  - [`qwen3_next.py`](/home/bot/project/sglang/python/sglang/srt/models/qwen3_next.py)
    和 [`qwen3_next_mtp.py`](/home/bot/project/sglang/python/sglang/srt/models/qwen3_next_mtp.py)：
    Qwen 混合模型和 MTP 组织方式。
  - [`gdn_blackwell/`](/home/bot/project/sglang/python/sglang/kernels/ops/attention/linear/gdn_blackwell)：
    Blackwell GDN 专用实现。
  - [`compressed_tensors_w4a4_nvfp4.py`](/home/bot/project/sglang/python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py)
    和 [`modelopt_quant.py`](/home/bot/project/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py)：
    NVFP4 权重/scale 合同和 GEMM backend dispatch。
  - [`fp8_blockwise_scaled_mm_sm120.cuh`](/home/bot/project/sglang/python/sglang/jit_kernel/csrc/gemm/fp8_blockwise/fp8_blockwise_scaled_mm_sm120.cuh)：
    当前主线仍保留的 SM120 blockwise FP8 原生 kernel，可用于对照 launch、tile 和
    scale layout。
  - [`test_nvfp4_gemm_sm120.py`](/home/bot/project/sglang/test/registered/quant/test_nvfp4_gemm_sm120.py)：
    当前 SM120 NVFP4 backend 的端到端选择和质量测试入口。
  - [`decode_cuda_graph_runner.py`](/home/bot/project/sglang/python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py)
    和 [`cuda_piecewise_backend.py`](/home/bot/project/sglang/python/sglang/srt/compilation/cuda_piecewise_backend.py)：
    decode graph、piecewise graph 和动态边界处理。
- 学习目标：对比本项目真实 dominant GEMM shapes；提取 NVFP4 backend 选择、
  graph buffer ownership、固定批宽、MTP verify 和 GDN 融合的设计。注意旧快照中
  曾存在的 `nvfp4_scaled_mm_sm120.cuh`/`nvfp4_blockwise_moe.cuh` 已不在 7 月 16 日
  主线；需要历史比较时使用 `git show 97e3b8998d:<path>`，不要链接或修改已删除文件。
- 边界：不要整套 fork。SGLang 的通用调度和分布式路径超出四槽单卡目标。

### 3. sm120-flash-attention：本机 attention 成果与集成边界

- 本地：[`/home/bot/project/sm120-flash-attention`](/home/bot/project/sm120-flash-attention)。
- 先读 [`02-执行计划.md`](/home/bot/project/sm120-flash-attention/02-执行计划.md)、
  [`04-最新技术动向调研.md`](/home/bot/project/sm120-flash-attention/04-最新技术动向调研.md)
  和 [`notes/`](/home/bot/project/sm120-flash-attention/notes)。
- 重点：head_dim=256/GQA 24:4、paged decode v2、split-KV、SMEM 99KB 限制、
  P 两级量化和 vLLM backend 接入。
- 学习目标：复用已经验证的 attention 接口和数据，不重复旧 kernel 的死路。
- 边界：真实 profile 中 attention 只占约 1.5%；除非端到端数据改变，不得把它
  再提升为 GEMM 之前的主线。

### 4. CUTLASS 4.6.1：SM120 block-scaled GEMM 基线

- 本地：[`/home/bot/project/cutlass-4.6.1`](/home/bot/project/cutlass-4.6.1)，
  tag `v4.6.1`，commit `e05f953`。
- 上游：[NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass)
- 重点入口：
  - [`79a_blackwell_geforce_nvfp4_bf16_gemm.cu`](/home/bot/project/cutlass-4.6.1/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)
  - [`79d_blackwell_geforce_nvfp4_grouped_gemm.cu`](/home/bot/project/cutlass-4.6.1/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu)
  - [`87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise.cu`](/home/bot/project/cutlass-4.6.1/examples/87_blackwell_geforce_gemm_blockwise/87c_blackwell_geforce_fp8_bf16_grouped_gemm_groupwise.cu)
- 学习目标：围绕真实 QKV、gate-up、down、o_proj、MTP 和 lm_head shapes 建立
  可重复的 CUTLASS baseline，研究 tileN 8/16、scale layout、persistent/ping-pong
  和 grouped dispatch。
- 边界：先做 shape-by-shape A/B，不能因为版本更高就直接替换现有 backend。

### 5. FlashInfer：paged KV、SM120 GEMM 与量化 attention 参考

- 本地：[`/home/bot/project/flashinfer`](/home/bot/project/flashinfer)，快照
  `608657a7`。
- 上游：[flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
- 学习目标：paged KV 接口、SM120 FP8/NVFP4 GEMM dispatch、workspace 管理、
  CUDA Graph 安全性，以及 `nvfp4_attention_sm120` 的 P quantization。
- 边界：本项目不以 FlashInfer upstream 为验收目标；只采用能降低本机端到端
  延迟的部分。

## P1：按当前任务选择性学习

| 项目 | 本地快照 | 主要学习内容 | 关键边界 |
| --- | --- | --- | --- |
| [nano-vLLM](/home/bot/project/nano-vllm) | `bb823b3` | 最小 engine、paged cache、scheduler/runner 分层 | 教学骨架，不含本模型的 GDN/NVFP4/MTP 完整语义 |
| [flash-linear-attention](/home/bot/project/flash-linear-attention) | `b328e7c` | Gated DeltaNet/KDA 的分块、recurrent state 和融合策略 | 先对齐 Qwen3.6 shape/state，再谈移植 |
| [SageAttention](/home/bot/project/SageAttention) | `d1a57a5` | FP8/FP4 attention、outlier smoothing、P 重量化数值方法 | 官方 kernel 目标架构不等于 SM120；主要借鉴算法 |
| [DeepGEMM](/home/bot/project/DeepGEMM) | `559d79f` | JIT、masked/grouped GEMM、shape specialization | 当前源码无 SM120 路径；是设计参考，不是可直接 backend |
| [TensorRT-LLM](/home/bot/project/TensorRT-LLM) | `b602fa6` | C++ runtime、MTP、MoE、graph/plugin 生命周期 | 主要面向数据中心 Blackwell；不能假定 SM120 可编译 |
| [TransformerEngine](/home/bot/project/TransformerEngine) | `9d92fa0` | FP8/NVFP4 scaling、数值稳定性和 fused transformer 组织 | 更偏训练/数据中心实现 |
| [Model-Optimizer](/home/bot/project/Model-Optimizer) | `cba8a5c` | NVFP4 checkpoint、scale metadata、量化验证 | 用于格式和数值合同，不进入 hot-path runtime |
| [TileLang](/home/bot/project/tilelang) | `9ff4ef8` | 快速验证 tiling、融合和 persistent 调度假设 | 最终极限 kernel 仍需 CUDA/CUTLASS/PTX 证明 |
| [ExLlamaV2](/home/bot/project/exllamav2) | `7dc12af` | 消费卡单请求 runtime、静态 buffer、量化 cache | 模型和量化格式不同，只借鉴执行策略 |

## P2：专项历史和负面证据

- [`kekzl-imp/`](/home/bot/project/sm120-flash-attention/kekzl-imp)：SM120 fragment/
  scale layout、P 两级量化和“tensor-core MMA 不是主要耗时”的独立证据。只读。
- [`BlackFlash/`](/home/bot/project/sm120-flash-attention/BlackFlash)：SM120 attention
  实验参考，先核对 shape 与实现完整性。
- [`vllm-nvfp4-kv-sm120`](/home/bot/project/sm120-flash-attention/vllm-nvfp4-kv-sm120)：已有 SM120
  NVFP4 KV 尝试，适合核对接口和失败模式。
- [`vLLM-Moet`](/home/bot/project/sm120-flash-attention/vLLM-Moet)：MoE/调度专项参考；Qwen3.6 当前并非
  HY3 类大规模稀疏 MoE，不要把其复杂度引入主线。

## 当前建议的学习任务

Agent 接到“优化下一刀”任务时，先完成以下静态产物再写 kernel：

1. 从真实 profile 中列出前十个 GEMM call site：`M/N/K`、输入/权重/scale layout、
   batch/slot 1--4 和 MTP verify 下的调用次数与时间。
2. 对每个 dominant shape 对齐 vLLM、SGLang、FlashInfer、CUTLASS 4.6.1 的候选
   kernel 和 layout，形成一张可测试矩阵。
3. 单独画出直接 runner 的 KV/GDN state ownership 和 CUDA Graph address-stability
   合同，并与 vLLM/SGLang 对照。
4. 输出“可以复用 / 只借鉴 / 明确不适用”的结论；没有 SM120 编译和端到端证据的
   项目不得标记为可复用。
5. 选择一个最小切口实现，先过 oracle/correctness，再进行 kernel 与端到端 A/B。

## 环境

- CUDA 工具链固定为 `/usr/local/cuda-13.3`，构建前显式设置 `CUDA_HOME` 和
  `PATH`，不要依赖 shell 默认顺序。
- vLLM Python 环境为 `/home/bot/.venvs/vllm`；本项目也有独立 `.venv`。
- 单机只有一张 GPU。运行 `ncu`、`nsys`、`compute-sanitizer` 或模型服务前先确认
  没有其他项目占用 GPU；sanitizer 会严重干扰另一任务的延迟数据。
- 大型参考仓库目前是浅克隆且未初始化子模块。只有确定要构建某个项目时，才按需
  初始化其必要子模块，不要执行全量递归下载。
