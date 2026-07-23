# CUDA Graph 确定性 + 正确性调查（2026-07-23）

## 最终根因

**`fast_decode_plan` 的调用方契约未被履行。**

FlashInfer 的 `fast_decode_plan`（cudagraph 模式）只做校验，不做 CPU→GPU 拷贝。
调用方必须自己把 indptr / last_page_len 写进 wrapper 的固定 GPU buffer。
vLLM 在 `_compute_flashinfer_kv_metadata` 里履行了这个契约；我们的 `replay()` 没有。

结果：kernel 每次 replay 读的是 capture 时 warmup 留下的元数据（kv=32, 2 页,
last_page_len=16），注意力在错误的 32-token 窗口上算 → cosine 0.93。

## 修复

`replay()` 中 `_run_plan` 前加两行：
```python
self._fi_indptr_gpu[:bs+1].copy_(self._fi_indptr_cpu[:bs+1], non_blocking=True)
self._fi_last_page_len_gpu[:bs].copy_(self._fi_last_page_len_cpu[:bs], non_blocking=True)
```

同时移除了 priming replay（同根因的 workaround，不再需要）。

## 验证结果

| 测试 | 修复前 | 修复后 |
|---|---|---|
| 单步 3× replay bit-exact | ✅ | ✅ |
| Graph vs eager cosine | 0.932 ❌ | 1.000000 ✅ (bit-exact) |
| 20 步 × 2 轮 token 一致 | ✅ | ✅ |
| 50 步 × 3 轮 token 一致 | run1≠run2 ❌ | ✅ |
| 50 步 graph vs eager token | 未测 | ✅ bit-exact |

## 排查过程中的关键裁决

1. **元数据构建正确**：手动 plan + 非 cudagraph wrapper → cos=1.0（洗清 metadata）
2. **fixed_split_size 无影响**：-1 vs 2048 在非 cudagraph 下 cos=1.0
3. **问题 100% 在 cudagraph wrapper 的 buffer 搬运路径**：
   非 cudagraph 分支（decode.py:3630-3632）直接重绑调用方张量，天然读到正确数据；
   cudagraph 分支（decode.py:3617-3628）只做校验，不拷贝。

## 性能现状

| 配置 | tok/s | ITL |
|---|---|---|
| 我们 Graph+Compile | 82.9 | 12.06ms |
| vLLM Graph+Compile | 94.5 | 10.58ms |
| 差距 | 12% | |

## 方法论教训

1. **fast path 函数省掉的那一步，永远要去读它的 docstring 里「谁来补这一步」。**
   vLLM 的 `fast_decode_plan` 用法是「先自己搬 buffer、再调 plan」，
   借用它的人必须连契约一起借。
2. **当「替换任何单组件都 bit 一致地错」时，立即停止换组件——病灶必在全部被换
   组件的公共上游。**
3. **对照组必须只变一个变量。** 0.93 那格同时变了 wrapper 模式 × plan 函数，
   导致误判为「kernel 有 bug」。
