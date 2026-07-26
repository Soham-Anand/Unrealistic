import mlx.core as mx
from .rope import precompute_freqs, apply_rope


class MultiHeadAttention:
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads

        scale = 0.02
        self.Wq = mx.random.normal((hidden_size, hidden_size), std=scale)
        self.Wk = mx.random.normal((hidden_size, num_kv_heads * self.head_dim), std=scale)
        self.Wv = mx.random.normal((hidden_size, num_kv_heads * self.head_dim), std=scale)
        self.Wo = mx.random.normal((hidden_size, hidden_size), std=scale)

        self._cos, self._sin = precompute_freqs(self.head_dim, 512)

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        B, T, _ = x.shape

        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv

        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.num_queries_per_kv > 1:
            k = mx.repeat(k, self.num_queries_per_kv, axis=1)
            v = mx.repeat(v, self.num_queries_per_kv, axis=1)

        q = apply_rope(q, self._cos, self._sin)
        k = apply_rope(k, self._cos, self._sin)

        scores = (q @ k.transpose(0, 1, 3, 2)) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores + mask

        attn = mx.softmax(scores, axis=-1)
        out = attn @ v

        out = out.transpose(0, 2, 1, 3).reshape(B, T, -1)
        return out @ self.Wo
