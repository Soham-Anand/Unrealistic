"""Alpha 450M LLaMA-style model (PyTorch). Mirrors configs/alpha_450m.json.

Decoder-only: RMSNorm pre-norm, GQA attention, RoPE, SwiGLU MLP,
tied embeddings. ~444M params.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def build_rope_cache(seq_len, head_dim, theta=100000.0, device="cpu"):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    # x: (B, H, T, D) NeoX half-rotation
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    c, s = cos[:, :x.shape[2], :], sin[:, :x.shape[2], :]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, kv, d = cfg["hidden_size"], cfg["num_key_value_heads"], cfg["hidden_size"] // cfg["num_attention_heads"]
        self.n_head, self.n_kv = cfg["num_attention_heads"], kv
        self.q = nn.Linear(h, cfg["num_attention_heads"] * d, bias=False)
        self.k = nn.Linear(h, kv * d, bias=False)
        self.v = nn.Linear(h, kv * d, bias=False)
        self.o = nn.Linear(cfg["num_attention_heads"] * d, h, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_head, -1).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv, -1).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv, -1).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.n_kv != self.n_head:
            rep = self.n_head // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h, f = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate = nn.Linear(h, f, bias=False)
        self.up = nn.Linear(h, f, bias=False)
        self.down = nn.Linear(f, h, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class AlphaLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg["num_hidden_layers"])])
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])

    def forward(self, ids):
        B, T = ids.shape
        cos, sin = build_rope_cache(T, self.cfg["hidden_size"] // self.cfg["num_attention_heads"],
                                    self.cfg.get("rope_theta", 100000.0), ids.device)
        cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
        x = self.tok(ids)
        for blk in self.layers:
            x = blk(x, cos, sin)
        return self.norm(x) @ self.tok.weight.T  # tied embeddings

    def count(self):
        return sum(p.numel() for p in self.parameters())


def tiny_config():
    """50M-class proxy with identical code paths (smoke tests)."""
    return {"hidden_size": 512, "intermediate_size": 1376, "num_hidden_layers": 8,
            "num_attention_heads": 8, "num_key_value_heads": 2,
            "max_position_embeddings": 512, "rope_theta": 100000.0,
            "rms_norm_eps": 1e-6, "vocab_size": 8000, "tie_word_embeddings": True}
