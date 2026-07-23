# B12x MoE 集成 × CUDA Graph 不兼容 — 根因 + 修复

## 症状（集成后端端到端测试报告）

1. 性能不及预期
2. CUDA Graph 不兼容

两条症状同一根因。

## 根因

`runtime/backends/laguna.py::_patch_moe_b12x`（commit `2559465`，B12x MoE 集成骨架）
给每个 MoE 层构造了一个 `LagunaMoEB12x(..., use_cuda_graph=False)`，并且从不调用
它自己的 `.capture()` —— 也就是说 `forward()` 永远走 `_forward_impl()` 这条"eager"
分支，直接调用 `flashinfer.fused_moe.B12xMoEWrapper.run()`。

问题在于 `B12xMoEWrapper.run()` 自身的逻辑
（`flashinfer/fused_moe/cute_dsl/b12x_moe.py:526-533`）：

```python
else:
    if _is_cuda_graph_capturing():
        raise RuntimeError(
            "B12xMoEWrapper must be constructed with use_cuda_graph=True "
            "to run during CUDA graph capture."
        )
```

而 Laguna 已有的两条 CUDA Graph 捕获路径 —— `LagunaCudaGraphDecode.capture()`
(`runtime/backends/laguna_cuda_graph.py:367`) 和 DFlash 的
`DFlashVerifyCudaGraph.capture()` / `DFlashDraftCudaGraph.capture()`
(`runtime/backends/laguna_dflash_cudagraph.py:296` 等) —— 都是直接对
`backend.model.forward(...)` 做 `with torch.cuda.graph(graph): ...` 整图捕获，
包含全部 47 个 MoE 层。一旦 `QSR_MOE_B12X=1`，捕获会在第一个 MoE 层命中上面这个
`RuntimeError` —— 这就是"CUDA graph 不兼容"。

"性能不及预期"是同一个 `use_cuda_graph=False` 配置的另一面：即使不涉及外层图捕获，
`B12xMoEWrapper.run()` 在这个分支下每次调用都用 `torch.empty(...)` 现分配输出
buffer（`b12x_moe.py:534-538`），没有走 `use_cuda_graph=True` 时预分配好的固定
buffer 路径 —— 这正是此前 `notes/2026-07-23-laguna-moe-b12x-direct-kernel.md`
里验证过的"eager 比 graph-safe 慢 ~80%（M=1）"的那个差异，在集成后端上原样复现。

## 修复

`_patch_moe_b12x` 现在每层构造两个 `LagunaMoEB12x` 实例，按调用时的 `num_tokens`
分派（`runtime/backends/laguna.py:380-433`）：

- **graph-safe**（`use_cuda_graph=True, max_num_tokens=MOE_GRAPH_SAFE_MAX_TOKENS=16`）：
  服务 decode（`MultiBatchGraphManager`，batch_size ≤ 4）和 DFlash verify/draft
  （固定 `NUM_QUERY_PER_REQ=16`）—— 这些都会被外层 `torch.cuda.graph()` 捕获。
  **注意**：这个实例本身从不调用自己的 `.capture()`/`.replay()`（那是
  `laguna_moe_kernel.py` 文档里的"独立使用"模式）——它只是被构造成
  `use_cuda_graph=True`，让 `B12xMoEWrapper` 内部走预分配固定 buffer 的分支，
  然后仍然走 `forward()` 里的 eager `_forward_impl()` 分支被外层图直接录制。
  两层图（自己 capture 一次 vs 被外层 capture）语义不同，必须用后者，否则会是
  "图中启动另一张图的 replay"，同样不合法。
- **eager**（`use_cuda_graph=False`）：服务 prefill（分块 2048 tokens，从不被
  图捕获，远超 `MOE_GRAPH_SAFE_MAX_TOKENS`）。

`MOE_GRAPH_SAFE_MAX_TOKENS=16` 而不是直接设成 2048：graph-safe 模式在构造时就
`_allocate_buffers()` 预分配固定显存，47 层 × 2048 tokens 的常驻显存代价是
47 层 × 16 tokens 的 ~128 倍，纯浪费（decode/verify 实际用到的 num_tokens 上限
就是 16）。

## 验证

未做完整模型 GPU 测试（显存/并行工作纪律）。做了一个隔离 kernel 级复现+验证脚本
（复用与 `laguna_moe_direct_kernel_verify.py` 相同的小规模构造：
HIDDEN_SIZE=3072, NUM_EXPERTS=256, TOP_K=10, INTERMEDIATE_SIZE=1024, M=16）：

1. 旧配置（`use_cuda_graph=False`）在外层 `torch.cuda.graph()` capture 内调用
   → 复现出一模一样的 `RuntimeError: B12xMoEWrapper must be constructed with
   use_cuda_graph=True to run during CUDA graph capture.`
2. 新配置（`use_cuda_graph=True, max_num_tokens=16`）同样方式调用 → capture
   成功，`replay()` 结果与 eager 参考输出一致（数值差 0.0024，量级符合 NVFP4
   kernel 变体间的正常浮点误差，非正确性问题）。

结果：`RESULT: PASS`。

## 待办（留给完整集成测试）

- 这次修复只解决 MoE 层本身的 graph 兼容性；`LagunaCudaGraphDecode` 目前还没有
  接入 `server/engine.py` 的真实 decode 循环（只在 benchmark 里用到），DFlash
  的 CUDA Graph 才是当前唯一真正跑通生产路径的调用方 —— 完整端到端验证（真实
  model.forward + `QSR_MOE_B12X=1` + DFlash CUDA Graph 三者一起，多步 replay
  正确性）需要在有 GPU 余量、不影响并行工作时补一次真实模型级别的回归。
