import math
import mlx.core as mx
from .rope import apply_rope


def init_attention(hidden_size: int, num_heads: int, num_kv_heads: int) -> dict:
    head_dim = hidden_size // num_heads
    scale = 0.02
    return {
        "Wq": mx.random.normal((hidden_size, hidden_size), scale=scale),
        "Wk": mx.random.normal((hidden_size, num_kv_heads * head_dim), scale=scale),
        "Wv": mx.random.normal((hidden_size, num_kv_heads * head_dim), scale=scale),
        "Wo": mx.random.normal((hidden_size, hidden_size), scale=scale),
    }


def attention(x: mx.array, params: dict,
              num_heads: int, num_kv_heads: int, mask: mx.array = None,
              training: bool = False, dropout_rate: float = 0.0,
              rope_base: float = 10000.0) -> mx.array:
    B, T, _ = x.shape
    head_dim = x.shape[-1] // num_heads
    num_queries_per_kv = num_heads // num_kv_heads

    q = x @ params["Wq"]
    k = x @ params["Wk"]
    v = x @ params["Wv"]

    q = q.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

    if num_queries_per_kv > 1:
        k = mx.repeat(k, num_queries_per_kv, axis=1)
        v = mx.repeat(v, num_queries_per_kv, axis=1)

    q = apply_rope(q, dims=head_dim, base=rope_base)
    k = apply_rope(k, dims=head_dim, base=rope_base)

    scale = 1.0 / math.sqrt(head_dim)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
    return out @ params["Wo"]
