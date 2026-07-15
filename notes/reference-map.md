# Local Reference Map

The first implementation stage uses the following local sources as references,
without vendoring or modifying them:

| Reference | Location | Use |
| --- | --- | --- |
| nano-vLLM (`bb823b3`) | `~/project/nano-vllm` | Small scheduler, paged block ownership, CUDA Graph input buffers |
| vLLM (`e12b91b03`, locally modified) | `~/vllm` | Qwen3.5 graph, GDN/MTP semantics, NVFP4 loader, oracle outputs |
| FlashInfer | `~/project/flashinfer` | Paged attention and KV-cache behavior |
| SM120 attention study | `~/project/sm120-flash-attention` | NVFP4 KV design, harnesses, and profiler methodology |

Do not write into the vLLM checkout: it has pre-existing local modifications.
The selected Unsloth model snapshot is
`ccdaab7e68af2409599b8949a8f2685703c9bae5` under the Hugging Face cache.
Its `config.json` confirms the supported 64-layer topology (16
`full_attention`, 48 `linear_attention`/GDN), 5120 hidden size, four KV heads,
256 head dimension, and one MTP hidden layer. Treat the checkpoint config as
authoritative if this revision changes.

No vLLM Python environment or serving Docker image is currently installed, so
Phase 0 server measurements are blocked until that baseline environment is
provided or built. The GPU is otherwise available: RTX PRO 6000 Blackwell,
97,887 MiB, CUDA UMD 13.3.
