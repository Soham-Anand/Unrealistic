import json
import os
import mlx.core as mx


def save_checkpoint(model, optimizer_state, step: int, save_dir: str = "checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"step_{step:06d}")
    os.makedirs(path, exist_ok=True)

    mx.save(os.path.join(path, "model.npz"), _flatten_params(model))

    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(model.config, f, indent=2)

    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump({"step": step}, f)

    print(f"Checkpoint saved: {path}")


def _flatten_params(model) -> dict:
    params = {}
    params["embeddings"] = model.embeddings.token_embedding
    for i, layer in enumerate(model.layers):
        params[f"layer_{i}.attn_norm"] = layer.attention_norm.weight
        params[f"layer_{i}.Wq"] = layer.attention.Wq
        params[f"layer_{i}.Wk"] = layer.attention.Wk
        params[f"layer_{i}.Wv"] = layer.attention.Wv
        params[f"layer_{i}.Wo"] = layer.attention.Wo
        params[f"layer_{i}.ffn_norm"] = layer.ffn_norm.weight
        params[f"layer_{i}.w_gate"] = layer.ffn.w_gate
        params[f"layer_{i}.w_up"] = layer.ffn.w_up
        params[f"layer_{i}.w_down"] = layer.ffn.w_down
    params["final_norm"] = model.final_norm.weight
    return params
