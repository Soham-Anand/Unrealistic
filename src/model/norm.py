import mlx.core as mx


def rms_norm(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x, weight, eps)


def init_rms_norm(dims: int) -> dict:
    return {"weight": mx.ones((dims,))}
