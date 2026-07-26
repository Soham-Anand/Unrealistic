import mlx.core as mx
import math


def precompute_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> mx.array:
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    t = mx.arange(max_seq_len).astype(mx.float32)
    freqs = mx.outer(t, freqs)
    return mx.cos(freqs), mx.sin(freqs)


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    B, T, C = x.shape
    x1 = x[..., : C // 2]
    x2 = x[..., C // 2 :]

    cos = cos[:T].reshape(1, T, 1)
    sin = sin[:T].reshape(1, T, 1)

    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return mx.concatenate([out1, out2], axis=-1)
