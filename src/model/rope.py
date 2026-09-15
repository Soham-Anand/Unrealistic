import mlx.core as mx


def apply_rope(x: mx.array, dims: int, base: float = 10000.0) -> mx.array:
    return mx.fast.rope(x, dims=dims, traditional=True, base=base, scale=1.0, offset=0)
