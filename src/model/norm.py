import mlx.core as mx
import math


class RMSNorm:
    def __init__(self, dims: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = mx.ones((dims,))

    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.rsqrt(mx.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return x * norm * self.weight
