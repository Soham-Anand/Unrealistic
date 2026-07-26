import mlx.core as mx
import math


def compute_perplexity(loss: float) -> float:
    return math.exp(loss)
