#!/usr/bin/env python3
"""Export MLX checkpoint -> HuggingFace Llama-layout safetensors.

Uses MLX's own checkpoint loader (correct namespace/dtype by construction).
MLX linears are x @ W (in,out); HF stores (out,in) -> transpose 2D projs.
Tied embeddings: no lm_head (tie_word_embeddings=true).

Usage:
  /tmp/export_venv/bin/python scripts/export_pt.py \
      --checkpoint checkpoints/step_355789 --out export_pt
"""

import os, sys, json, argparse, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import mlx.core as mx
from src.training.checkpointing import load_checkpoint


def interleaved_to_neox_rows(t, n_heads=8):
    """Permute q/k projection ROWS from GPT-J-interleaved to NeoX-half rotation.

    src rope.py uses mx.fast.rope(traditional=True) = interleaved pairs
    (2i,2i+1); HF Llama uses NeoX pairs (i,i+d/2) with identical frequencies,
    so a within-head row permutation exactly converts the weights.
    """
    import torch
    d = t.shape[0] // n_heads
    half = d // 2
    idx = torch.empty(t.shape[0], dtype=torch.long)
    for h in range(n_heads):
        base = h * d
        for i in range(half):
            idx[base + i] = base + 2 * i
            idx[base + i + half] = base + 2 * i + 1
    return t[idx].contiguous()

LAYER_MAP = [  # (mlx key, hf suffix, transpose?)
    ("attn_norm.weight", "input_layernorm.weight", False),
    ("ffn_norm.weight", "post_attention_layernorm.weight", False),
    ("attn.Wq", "self_attn.q_proj.weight", True),
    ("attn.Wk", "self_attn.k_proj.weight", True),
    ("attn.Wv", "self_attn.v_proj.weight", True),
    ("attn.Wo", "self_attn.o_proj.weight", True),
    ("ffn.w_gate", "mlp.gate_proj.weight", True),
    ("ffn.w_up", "mlp.up_proj.weight", True),
    ("ffn.w_down", "mlp.down_proj.weight", True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="export_pt")
    args = ap.parse_args()

    params, saved_cfg, _ = load_checkpoint(args.checkpoint, return_opt=True)
    nl = len(params["layers"])
    print(f"layers: {nl}", flush=True)

    import torch
    from safetensors.torch import save_file

    def conv(a):
        return torch.from_numpy(np.asarray(a.astype(mx.float32))).to(torch.bfloat16)

    tensors = {}
    for i, layer in enumerate(params["layers"]):
        for src, dst, transp in LAYER_MAP:
            node = layer
            for part in src.split("."):
                node = node[part]
            t = conv(node)
            if transp:
                t = t.t().contiguous()
            if dst in ("self_attn.q_proj.weight", "self_attn.k_proj.weight"):
                t = interleaved_to_neox_rows(t)
            tensors[f"model.layers.{i}.{dst}"] = t
    tensors["model.embed_tokens.weight"] = conv(params["embeddings"]["token_embedding"])
    tensors["model.norm.weight"] = conv(params["final_norm"]["weight"])
    print(f"HF tensors: {len(tensors)}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    save_file(tensors, os.path.join(args.out, "model.safetensors"))

    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 768, "intermediate_size": 4096,
        "num_hidden_layers": nl, "num_attention_heads": 8,
        "num_key_value_heads": 8, "vocab_size": 32000,
        "max_position_embeddings": 1024, "rope_theta": 10000.0,
        "rms_norm_eps": 1e-06, "hidden_act": "silu",
        "tie_word_embeddings": True, "torch_dtype": "bfloat16",
        "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 2,
    }
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=2)
    shutil.copy("data/tokenizer/phase1.model", os.path.join(args.out, "tokenizer.model"))
    json.dump({
        "tokenizer_class": "LlamaTokenizer", "bos_token": "<s>",
        "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "</s>",
        "add_bos_token": True, "add_eos_token": False,
    }, open(os.path.join(args.out, "tokenizer_config.json"), "w"), indent=2)
    print(f"WROTE {args.out}/ (model.safetensors + config + tokenizer)", flush=True)


if __name__ == "__main__":
    main()
