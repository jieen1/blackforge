"""DFlash configuration constants (no vLLM dependency).

Separated from laguna_dflash.py so CPU-only tests can validate
configuration without importing the full vLLM stack.
"""

# DFlash speculative decoding parameters (from model config.json)
NUM_SPECULATIVE_TOKENS = 15
NUM_QUERY_PER_REQ = 16  # 1 bonus + 15 mask

# Aux hidden state extraction layers — values passed to set_aux_hidden_state_layers().
# vLLM semantics: _maybe_add_hidden_state is called with (layer_idx + 1) after each
# layer completes, so value N means "capture output of layer N-1".
# Config eagle_aux_hidden_state_layer_ids = [2, 11, 20, 30, 39, 48]
# → captures after layers 1, 10, 19, 29, 38, 47 (matching dflash_config.target_layer_ids).
AUX_LAYER_IDS = (2, 11, 20, 30, 39, 48)

# Draft model architecture
MASK_TOKEN_ID = 12
DRAFT_NUM_LAYERS = 6
DRAFT_WINDOW = 512
DRAFT_NUM_QO_HEADS = 72
DRAFT_NUM_KV_HEADS = 8
DRAFT_HEAD_DIM = 128

# Default DFlash model path (HF cache)
DFLASH_MODEL_PATH = (
    "~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-DFlash-NVFP4/"
    "snapshots/723794750422b3efbf3a7b3af76dffb4ba035943/"
)
