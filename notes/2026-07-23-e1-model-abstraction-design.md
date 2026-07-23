# E1 · 模型抽象层设计（2026-07-23）

## 目标

将 `DirectModelRunner` 中硬编码的 Qwen3.6-27B 假设显式化为 `ModelSpec` 接口，
使 runtime 基础设施（block pool、KV 管理、CUDA Graph、调度、前缀缓存）与模型特定逻辑解耦。

## 当前耦合点（4574 行 runner 中的模型假设）

### 1. 架构参数（硬编码或从 vllm_config 隐式推导）
- `attn_layer_names` / `gdn_layer_names`：通过遍历 `static_forward_context` 发现
- GDN state shape：`conv_state` + `ssm_state` per layer per slot
- Attention head config：从 vLLM model config 推导
- MTP K：从 `speculative_config.num_speculative_tokens` 获取

### 2. 模型特定逻辑
- `verify_batch_spec`：MTP verify forward（qo_len=K+1）
- `_mtp_forward` / `_mtp_forward_batch`：draft model forward
- `_mtp_sync_and_propose_batch`：draft→verify 同步
- GDN metadata 构建：`build_gdn_metadata_spec_batch`
- GDN state commit：accept/reject 后选择正确的 SSM row

### 3. 运行时基础设施（模型无关）
- Block pool / KV 分配
- CUDA Graph capture/replay
- 前缀缓存（content-addressed）
- 调度（固定槽位、chunked prefill）
- 采样（temperature/top-k/top-p）
- 指标/tracing

## 设计方案：ModelSpec 协议

```python
@dataclass
class ModelSpec:
    """模型特定配置，由模型注册时提供。"""
    # 架构标识
    model_id: str                    # "unsloth/Qwen3.6-27B-NVFP4"
    architecture: str                # "Qwen3_5ForConditionalGeneration"
    
    # 层结构
    num_layers: int                  # 48
    attn_layer_indices: list[int]    # [0, 3, 6, ...] (16 attention layers)
    gdn_layer_indices: list[int]     # [1, 2, 4, 5, ...] (32 GDN layers)
    
    # 注意力配置
    num_attention_heads: int         # 32
    num_key_value_heads: int         # 8
    head_dim: int                    # 128
    
    # GDN 配置
    gdn_conv_state_shape: tuple      # per-layer conv state shape
    gdn_ssm_state_shape: tuple       # per-layer SSM state shape
    
    # 投机解码
    mtp_model_id: str | None         # MTP draft model ID
    num_speculative_tokens: int      # K=3
    
    # KV 配置
    kv_dtype: str                    # "fp8_e4m3"
    block_size: int                  # 16
    
    # 词表
    vocab_size: int                  # 248044
    eos_token_id: int                # 248046


class ModelBackend(Protocol):
    """模型特定操作的接口。"""
    
    def prefill(self, slots, prompts, chunk_size) -> dict:
        """Prefill forward，返回 anchor + draft_tokens。"""
        ...
    
    def verify_and_commit(self, slots, anchors, drafts) -> dict:
        """MTP verify + accept/reject + GDN state commit。"""
        ...
    
    def decode_step(self, slots, tokens, kv_lengths) -> torch.Tensor:
        """单步 decode forward，返回 logits。"""
        ...
    
    def get_gdn_state_shape(self) -> dict:
        """返回 GDN state 分配所需的形状信息。"""
        ...
    
    def build_attention_metadata(self, **kwargs) -> Any:
        """构建注意力 metadata（模型特定的 kernel 参数）。"""
        ...
```

## 分步实施计划

### Phase 1: ModelSpec 提取（低风险）
- 从 runner.__init__ 中提取架构参数到 `ModelSpec` dataclass
- runner 通过 `self.spec` 访问，不再直接读 vllm_config
- 门禁：233 tests pass + speed baseline 无退化

### Phase 2: Qwen36Backend 提取（中风险）
- 将 MTP 相关方法（13 个，~2885 行）提取到 `runtime/backends/qwen36.py`
- runner 通过 `self.backend.verify_and_commit(...)` 调用
- 门禁：golden fixtures bit-exact + speed baseline 无退化

### Phase 3: 多模型注册（M4）
- `ModelRegistry` 支持按 architecture 字段选择 backend
- Laguna-S-2.1 backend 作为第二个实现验证接口完备性
- 门禁：Laguna L2 正确性 + 质量链

## 依赖与风险

- Phase 1 无外部依赖，可立即开始
- Phase 2 依赖 Phase 1 完成
- Phase 3 依赖 Laguna L2（M4）
- 风险：GDN state 管理深度耦合在 runner 中，Phase 2 需要仔细处理 state 所有权

## 与 B5 的关系

B5 模块化（已提取 4 域：block_pool / metadata_builders / cuda_graphs / mtp_accept）
是 E1 的前置工作。E1 Phase 2 的 Qwen36Backend 提取是 B5 的最终目标——
将 runner 从 4574 行降到 ~1500 行（纯调度 + KV 管理 + CUDA Graph 编排）。

---

*设计于 2026-07-23；基于 4574 行 runner 的实际耦合分析。*

## 缓存物种（前向兼容占位，审查非阻断⑥）

| 缓存物种 | 后端 | 状态 | 接口 |
|---|---|---|---|
| Full paged KV | LagunaBackend (12 full layers) | 已实现 | `kv_caches[name][block_table]` |
| SWA ring KV | LagunaBackend (36 SWA layers) | 已实现 | `kv_caches[name][ring_block_table]`, window=512, ring_blocks=34 |
| 前缀缓存（content-addressed） | — | TODO | 需「窗口快照」原语：SWA 层只保留窗口内 KV，前缀命中需额外快照机制（与 Qwen GDN checkpoint R1 同构） |
| Session affinity KV 保留 | — | TODO | 依赖前缀缓存 |
