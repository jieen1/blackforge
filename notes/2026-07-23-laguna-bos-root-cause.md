# Laguna E3 根因：缺 BOS token

日期：2026-07-23
状态：已修复

## 现象

LagunaBackend 直连 `model.forward()` 路径输出重复 token（" is is is..."），
而 stock vLLM LLM 基线输出正确（" Paris."）。

## 根因

`tokenizer.encode(prompt, add_special_tokens=False)` 跳过了 BOS token（id=2）。
Laguna 的 tokenizer 通过 `post_processor` 注入 BOS，`add_special_tokens=False`
会跳过这一步。所有下游组件（attention、MoE、GEMM）都在处理一个缺少 BOS 的
输入序列，导致 logits 系统性偏移。

## 修复

1. `tokenizer.encode(prompt)` — 不再传 `add_special_tokens=False`
2. EOS 停止条件从 Qwen 的 `(151643, 151645)` 改为 Laguna 的 `(2, 24)`
3. 护栏：oracle A/B 比对前先断言 prompt token ids 相等

## 方法论教训

**当「替换任何单组件都 bit 一致地错」时，立即停止换组件——病灶必在全部被换
组件的公共上游（输入 ids / 权重 / 配置），先做输入对齐断言再排查机器。**

本次从「疑点在 set_forward_context」到真正根因，绕了 ~15 轮实验。缺的正是
这条：先断言输入一致，再排查组件。

## 排除矩阵（全部 bit 一致，证明组件无问题）

| 实验 | 结果 |
|---|---|
| A2 patch ON/OFF | bit 一致 |
| vLLM 原生 set_forward_context | bit 一致 |
| model.forward() vs model() | bit 一致 |
| vLLM 原生 FlashInferMetadataBuilder | bit 一致 |
| 异构头数分组 (48/72) | 正确 |
| KV cache 写入 | 10229 nonzero ✅ |

## 验证

修复后：
- "The capital of France is" → " Paris.\n\nThe user is asking about..."
- 贪心确定性 ✅
- 吞吐 14.2 tok/s (single), 61.5 tok/s (batch=4)

## Server 层隐患（待 Laguna 接入时修复）

`server/app.py:132` 的 `_tokenize_encode` 使用 `add_special_tokens=False`。
Chat 路径（`_tokenize_chat`）通过 `apply_chat_template` 正确处理 BOS，不受影响。
但 raw text 路径（completions API）会在 Laguna 接入时复发 BOS 缺失。

修复方案：Laguna 接入 server 层时，`_tokenize_encode` 需要根据模型配置
决定是否传 `add_special_tokens=False`。建议：
- 在 engine/backend 层暴露 `tokenize_kwargs` 属性
- Laguna: `{}` (默认 add_special_tokens=True)
- Qwen: `{"add_special_tokens": False}` (保持现有行为)
