# sparkinfer MoE 集成诊断报告 (2026-07-24)

## 结论

sparkinfer kernel 本身准确（vs 自带参考实现 cosine=0.998），但与 vLLM CUTLASS
的 E2E 输出不兼容（per-layer cosine 最高 0.94，47 层复合后 ≈ 0.06 → 乱码）。

## 根因

vLLM 的 block scale 格式与 sparkinfer 期望的格式不同：

| 配置 | vs vLLM cosine | vs sparkinfer ref cosine | 说明 |
|------|---------------|-------------------------|------|
| swizzle + orig_ws2 | 0.726 | 0.998 | kernel 正确但与 vLLM 不兼容 |
| 无 swizzle + orig_ws2 | 0.943 | 0.780 | 碰巧接近 vLLM 但 kernel 读错 |
| swizzle + ws2/inp | 0.519 | — | runtime alpha 路径对小值产生零 |

核心矛盾：
- sparkinfer kernel 需要 swizzled block scales（vs 参考实现 0.998）
- vLLM CUTLASS 使用 un-swizzled block scales
- 两种格式不兼容，无法同时满足

## 已修复的 5 个 bug

1. **Scale 折叠下溢**: block_scale/global_scale → fp8 下溢（值 < 0.002）
2. **权重来源**: 改为用 vLLM 已加载权重（scale 约定正确）
3. **Block scale 双重 swizzle**: vLLM 的 scale 不需要再 swizzle
4. **w13 布局**: vLLM 存 [gate,up]=w13，不是 [up,gate]=w31
5. **Routed scaling**: shared expert 输出需除以 routed_scaling_factor(2.5)

## 基线状态

基线（CUTLASS）完全正常：
- "The capital of France is" → " Paris" ✓
- 代码补全正确 ✓
- 399 CPU 测试通过 ✓

## 下一步方向

1. **从 checkpoint 直接加载权重**（绕过 vLLM 格式）：用 sparkinfer 的
   `swizzle_block_scale` 处理 checkpoint 原始 block scales，配合正确的
   global scale（1/checkpoint_gs）。需要解决 runtime alpha 路径对小值
   产生零的问题（可能是 sparkinfer 的 bug）。
2. **联系 sparkinfer 上游**：报告 runtime alpha 路径对 ~1e-4 值产生零的问题。
3. **暂缓 sparkinfer，转其他优化**：compile（35→27ms）、prefill（128K 26→18s）、
   步时延 L1 等。sparkinfer 作为后续自研 kernel 参考保留。

## 性能数据

| 配置 | ITL | tok/s | 正确性 |
|------|-----|-------|--------|
| CUTLASS 基线 (eager) | ~37ms | ~27 | ✓ |
| sparkinfer (eager) | 67-82ms | 12-15 | ✗ |
| sparkinfer standalone (CG) | 38μs/layer | — | ✓ (vs ref) |
