# GPU 验证清单（模型下载完成后）

日期：2026-07-23
状态：等待模型下载

## 1. A2 GEMM Laguna shapes bit-exact 验证

```bash
/home/bot/.venvs/vllm/bin/python -m benchmarks.a2_laguna_shape_verify
```

验证 11 个 Laguna GEMM shape × 3 batch sizes (1/2/4) = 33 组合全部 bit-exact。
失败则需调整 tile config 或回退到 vLLM kernel。

## 2. 质量门 A/B 比对

```bash
/home/bot/.venvs/vllm/bin/python -m benchmarks.laguna_quality_gate
```

验证：
- Tokenizer 一致性（HF vs vLLM）
- 4 个 greedy prompt 的 token-level A/B 比对
- Prompt ids 断言护栏生效

## 3. LagunaBackend 性能基线

```bash
/home/bot/.venvs/vllm/bin/python -m benchmarks.laguna_backend_test
```

记录：
- Single decode tok/s（当前 14.2，目标 >21）
- Batch=4 decode tok/s（当前 61.5）
- ITL ms（当前 70.6，目标 <47）
- 贪心确定性

## 4. CUDA Graph capture 验证

```python
# 在 laguna_backend_test.py 中加入 CUDA Graph 路径
from runtime.backends.laguna_cuda_graph import LagunaDecodeGraph
graph = LagunaDecodeGraph(backend, batch_size=1)
graph.capture()
# 对比 eager vs graph 的 tok/s
```

## 5. 原生 vLLM 基线（如需更新）

```bash
/home/bot/.venvs/vllm/bin/python -m benchmarks.laguna_vllm_baseline
```

## 优先级

1. A2 GEMM bit-exact（门禁：不通过不能启用 custom GEMM）
2. 质量门 A/B（门禁：不通过不能合入）
3. 性能基线记录（数据为准）
4. CUDA Graph 验证（性能优化）
