# Laguna → E1 → server/ 双协议接线（Lane 1，CPU-only）

日期：2026-07-23
范围：`LagunaBackend → E1 抽象层 → server/ 双协议接线`（roadmap Track E，L2 的 server 生产形态部分）。本轮工作在独立 worktree（`worktree-laguna-e1-server-integration`）完成，全程未使用 GPU；与此并行的 Lane 2（CUDA Graph 多 batch 性能工作，`runtime/backends/laguna_cuda_graph.py`）在原 worktree 独立推进，互不干扰。

## 已完成

1. **LagunaBackend 补齐 E1 契约**（`runtime/backends/laguna.py`）：
   - `self.spec`：真实 `ModelSpec` 实例（此前是 `None` 占位符），`mtp_model_id=None` / `num_speculative_tokens=0` → `has_mtp=False`。
   - `self.block_table`：镜像 `DirectModelRunner.block_table` 的脏槽位标记语义（Laguna 无 block-table 间接层，恒为空列表）。
   - `reconcile_prefix_hit`：permanent-miss 桩（Laguna 无持久前缀缓存，TODO）。
   - `prefill_chunked_begin` / `prefill_chunked_step`：一次性（非增量）包装，匹配 `DirectModelRunner` 的分块 prefill 契约形状，`done=True` 立即返回。
   - `decode_batch_sampled`：签名改为与 `DirectModelRunner.decode_batch_sampled` 完全一致（`kv_lengths` 显式位置参数 + `return_logprobs`/`top_logprobs`），补齐 logprobs 支持（复用 `runtime/logprobs.compute_logprobs`）。此前的签名（3 个位置参数、无 logprobs）与 `ServerEngine` 调用约定不兼容，会直接 TypeError。

2. **ServerEngine 增加 backend 选择**（`server/engine.py`）：
   - 新增 `backend: str = "qwen36"` 构造参数（"qwen36" | "laguna"），非法值报错。
   - `MODEL` / `K` 按 backend 实例级覆盖（Qwen 路径的类属性默认值完全不变，等价性可证：覆盖分支只在 `backend != "qwen36"` 时执行）。
   - `eos_token_ids`（新，frozenset）泛化单值 `eos_token_id` 比较：Laguna 的 `generation_config.json` 声明两个停止符 `[2, 24]`（tokenizer 的 `.eos_token_id` 只覆盖 id 2），Qwen 路径的 `eos_token_ids` 恒等于 `{eos_token_id}`，行为可证明不变。三处比较点（`_activate_slot` / 采样解码 / MTP 解码）改为 `in eos_token_ids`。
   - `classify_decode_slots`（新，纯函数，同 `find_stale_slots` 风格提取，CPU 可测）：非 MTP backend（`spec.has_mtp=False`）时，全部活跃槽位（含"贪婪"请求）走 `decode_batch_sampled` 简单自回归路径，不再尝试 MTP verify/commit——贪婪本就是 temperature=0 的特例（B1），这不是行为分叉，只是路由分叉。Qwen 路径下此函数完全复现原有 inline 逻辑。
   - `_load_model` 按 `backend_name` 分派到 `_load_qwen36_model`（原逻辑原样重命名，未改动）或新增的 `_load_laguna_model`。

3. **server/app.py 接入**：
   - 新增 `QSR_SERVER_MODEL_BACKEND` 环境变量（默认 `qwen36`）。
   - Laguna 模式下若干环境变量默认值调整（显式 override 仍生效）：`enable_cudagraph=0`（Lane 2 未接入引擎）、`enable_prefix_cache=0`（无持久前缀缓存）、`capacity=1` / `num_slots=1`（见下方显存测算）、`blocks_per_slot=8192`（128K/槽，保守值，见下）、`kv_cache_dtype=auto`（fp8 未在 Laguna 路径验证过）。
   - 全部 21 处 `ServerEngine.MODEL`（类属性，硬编码 Qwen 字符串）改为 `engine.MODEL`（实例属性）——否则 Laguna 模式下 `/v1/models`、Prometheus 指标、`req.model` 默认值等会全部错误地报告 Qwen 的模型 id。Qwen 路径下 `engine.MODEL == ServerEngine.MODEL`，行为不变。

4. **测试**（`tests/test_laguna_server_integration.py`，12 项，用户批准后在 `/home/bot/.venvs/vllm`（import 级别，未建 CUDA context）跑通全绿；全仓库回归 `361 passed`，对比修改前基线 349，零回归）：
   - `classify_decode_slots` 4 个场景（含空输入、grammar 槽位、非 MTP 全路由）。
   - `ServerEngine(backend=...)` 真实构造（不调用 `.start()`，`__init__` 本身不碰 GPU）：非法 backend 报错、Qwen 默认值不变、Laguna 覆盖值正确、`eos_token_ids` 含 2 和 24。
   - `LagunaBackend` 新增方法的结构测试（`__new__` 绕过 `__init__`）：`reconcile_prefix_hit` 永久 miss、`prefill_chunked_begin` 的空输入/长度不匹配边界、`prefill_chunked_step` 遵循 `state.done`、`decode_batch_sampled` 与 `DirectModelRunner` 签名逐参数一致。

5. **意外发现并修复的真实 bug**：`ServerEngine.__init__` 的 `AutoTokenizer.from_pretrained(self.MODEL)` 原本不传 `trust_remote_code`——Laguna 有自定义 `configuration_laguna.py`，没有这个标志时 transformers 走通用校验路径，会在 yarn rope_parameters 上炸 `KeyError: 'original_max_position_embeddings'`（benchmark 脚本一直是传了 `trust_remote_code=True` 的，engine.py 这条路径此前从未被走过，所以没暴露）。修复：`trust_remote_code=(self.backend_name != "qwen36")`——Qwen 路径的实际取值仍是 `False`（未传时的默认值），行为不变。

6. **poolside_v1 tool-call 解析器 —— 含流式实时增量**（`server/formats/tools.py` + `server/formats/stream.py`）：读取本地已下载的 `chat_template.jinja` 拿到确切证据——Laguna 的 tool-call 内部格式与 Qwen 完全不同（`NAME<arg_key>K</arg_key><arg_value>V</arg_value>...`，无 `<function=>`/`<parameter=>` 包裹；两者共享外层 `<tool_call>...</tool_call>`）。
   - **最终解析**（`parse_tool_calls`）：按内部形状自动识别（先试 Qwen 的 `<function=` 形状，否则退到 Poolside 形状），对 Qwen 路径行为可证明不变（正常输出两种实现产出完全一致；不匹配任一形状的畸形块保留原有"不识别就留在可见文本里"的安全语义，靠 `_IDENTIFIER_RE` 过滤防止误把散文当零参数调用解析）。
   - **流式实时增量**（`StreamProcessor.drain_tool_deltas`）：同样按块自动识别两种形状，边生成边推送 `name`/`arguments_delta` 事件——Poolside 的函数名边界（无显式包裹标签）由首个 `<arg_key>` 或 `</tool_call>` 确定，在此之前不猜测、不提前发 name 事件（未知边界时按兵不动，等更多 token 到达），与 Qwen 分支"等 `<function=NAME>` 的收口 `>` 出现才发 name"的谨慎程度一致。Qwen 分支逐行核对过行为不变（现有 5 个流式测试原样全过）。
   - thinking 标签（`<think>...</think>`）两个模型一致，`strip_thinking` 无需改动。
   - 新增 13 个测试（8 个最终解析 + 5 个流式增量），全仓库 374 passed，零回归。
   - **仍未做**：这是基于 chat_template 的"应然"格式推出的，真实生成是否完全遵循模板仍需 GPU 冒烟核实。

## 重要发现（非显而易见，需记录）

### 1. Laguna 当前实现没有 SWA 环形 KV，实际显存开销比 L0 账本高 ~4 倍

`notes/2026-07-22-laguna-l0-memory-budget.md` 的 KV 增长模型（"24 KiB/token"）只计入 12 层全局 attention，假设 36 层滑窗（SWA，窗口 512）用有界环形 KV——但那是 roadmap L2 的**计划**特性，尚未实现。当前 `LagunaBackend.__init__` 对全部 48 层（含 36 层 SWA）一视同仁地按 `blocks_per_slot`（等于满上下文长度）分配分页 KV cache——`window_left` 参数只影响 FlashInfer kernel 的注意力计算范围，不影响存储分配。

实际每 token 每层 KV 字节（FP8，8 KV heads × 128 head_dim）：2048 B/层；× 48 层（非 L0 假设的 12 层）= **96 KiB/token**，约为 L0 账本的 4 倍。这意味着 L0 表格里"2×200K / 2×256K / 4×128K 均可行"的结论，在当前（无环形 KV）实现下**不成立**——按 96 KiB/token 计算，可用的 ~24.6 GiB KV+开销预算下，单槽安全上限约 128K token（12 GiB），2 槽 128K 已经吃满预算，2×200K 会 OOM。

`server/app.py` 的 Laguna 默认值（`capacity=1, num_slots=1, blocks_per_slot=8192`=128K）反映的是**当前的真实限制**，不是 roadmap 目标形态；等 L2 的 SWA 环形 KV 落地后应重新放宽。

### 2. Laguna 的 tool-call / thinking 解析器与 Qwen 不同 —— 已按 chat_template 证据修复，仍待真实样本核实

`generation_config.json` 声明 `tool_call_parser: "poolside_v1"`、`reasoning_parser: "poolside_v1"`——与 Qwen3.6 的格式是两回事。读 `chat_template.jinja` 拿到确切格式后，`server/formats/tools.py` + `server/formats/stream.py`（见"已完成 6"）已支持两种格式自动识别，**含流式实时增量**；`strip_thinking` 的 `<think>...</think>` 两个模型一致，不用改。

**仍未做**：这是基于模板"应然"格式的修复，真实生成是否 100% 遵循模板未验证，需要真实生成样本（即需要 GPU）核实。

### 3. Prefill 是逐槽单独调用，非批量

`prefill_chunked_begin` 循环调用 `self.prefill(slot, prompt)`（每槽一次独立前向），而非把同批次多个新请求的 prefill 合并成一次 forward——功能正确但吞吐不是最优。Qwen 路径的 `mtp_prefill_batch` 才是真正批量的。这是可接受的起点（correctness first），批量化是后续性能项，不在本轮范围内。

### 4. `enable_session_affinity` 未针对 Laguna 专门处理

`_finish_request` 的会话保留分支引用 `self.runner.slot_committed_tokens`（Laguna 已具备），但没有加任何针对 Laguna 的 gate——因为 `SERVER_ENABLE_SESSION_AFFINITY` 默认就是 `0`（两个 backend 共用同一个默认关闭），且 `enable_session_affinity` 依赖 `enable_prefix_cache`（Laguna 默认也是关闭），二者联动天然不会触发。如果有人显式为 Laguna 打开这两个开关，行为是"保留槽位但下一个请求必然冷启动"（`reconcile_prefix_hit` 恒返回 0）——浪费但不会产生错误结果。

## 待办（GPU 权限门禁，需要用户批准后才能执行）

1. ~~在 GPU 能力环境里跑 5 个结构测试~~ ✅ 已完成（用户批准 import 级别使用，12/12 全绿，全仓库回归 361 passed 零退化）。
2. **真实 GPU 冒烟**：`_load_laguna_model` 从未跑过——需要验证 `EngineArgs`/`create_engine_config` 组合、`blocks_per_slot=8192` 的显存是否真的够、以及 `LagunaBackend` 通过 `ServerEngine` 完整走一遍 prefill→decode→finish 的循环（含真实 HTTP 请求，双协议各打一轮）。
3. **确认 Laguna 真实输出格式**：拿到一段真实生成文本，核实 poolside_v1 tool-call 解析器（最终解析 + 流式实时增量均已按 chat_template 实现，见"已完成 6"）与真实模型输出是否完全吻合。
4. **确认 `kv_cache_dtype="auto"` 在 Laguna 路径下的实际行为**（是否真的退到 bf16，还是 vLLM 对 NVFP4 模型有别的默认值）。
