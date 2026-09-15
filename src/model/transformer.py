import json
import mlx.core as mx
from .attention import init_attention, attention
from .mlp import init_swiglu, swiglu
from .norm import init_rms_norm, rms_norm
from .embeddings import init_embeddings, embed_tokens


def init_model(config_path: str = "configs/model_config.json") -> tuple[dict, dict, dict]:
    with open(config_path) as f:
        cfg = json.load(f)

    params = {
        "embeddings": init_embeddings(cfg["vocab_size"], cfg["hidden_size"]),
        "layers": [],
        "final_norm": init_rms_norm(cfg["hidden_size"]),
    }

    for _ in range(cfg["num_hidden_layers"]):
        layer = {
            "attn_norm": init_rms_norm(cfg["hidden_size"]),
            "attn": init_attention(cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]),
            "ffn_norm": init_rms_norm(cfg["hidden_size"]),
            "ffn": init_swiglu(cfg["hidden_size"], cfg["intermediate_size"]),
        }
        params["layers"].append(layer)

    if not cfg["tie_word_embeddings"]:
        params["lm_head"] = mx.random.normal((cfg["hidden_size"], cfg["vocab_size"]), scale=0.02)

    buffers = {}
    return params, buffers, cfg


def causal_mask(seq_len: int, dtype: mx.Dtype = mx.bfloat16) -> mx.array:
    mask = mx.full((seq_len, seq_len), float("-inf"), dtype=dtype)
    mask = mx.triu(mask, k=1)
    return mask.reshape(1, 1, seq_len, seq_len)


def forward(params: dict, buffers: dict, tokens: mx.array, cfg: dict,
            training: bool = False) -> mx.array:
    B, T = tokens.shape
    x = embed_tokens(tokens, params["embeddings"])
    mask = causal_mask(T, mx.bfloat16)
    dropout_rate = cfg.get("dropout", 0.0)
    rope_base = cfg.get("rope_theta", 10000.0)

    for layer in params["layers"]:
        h = rms_norm(x, layer["attn_norm"]["weight"])
        h = attention(h, layer["attn"],
                      cfg["num_attention_heads"], cfg["num_key_value_heads"], mask,
                      training=training, dropout_rate=dropout_rate,
                      rope_base=rope_base)
        x = x + h
        h = rms_norm(x, layer["ffn_norm"]["weight"])
        h = swiglu(h, layer["ffn"], training=training, dropout_rate=dropout_rate)
        x = x + h

    x = rms_norm(x, params["final_norm"]["weight"])

    if "lm_head" in params:
        logits = x @ params["lm_head"]
    else:
        logits = x @ params["embeddings"]["token_embedding"].T

    return logits


def count_parameters(params: dict) -> int:
    total = 0
    for k, v in params.items():
        if k.startswith("_"):
            continue
        if isinstance(v, mx.array):
            total += v.size
        elif isinstance(v, dict):
            total += count_parameters(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    total += count_parameters(item)
                elif isinstance(item, mx.array):
                    total += item.size
    return total
