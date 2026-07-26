import mlx.core as mx


class SwiGLU:
    def __init__(self, hidden_size: int, intermediate_size: int):
        scale = 0.02
        self.w_gate = mx.random.normal((hidden_size, intermediate_size), std=scale)
        self.w_up = mx.random.normal((hidden_size, intermediate_size), std=scale)
        self.w_down = mx.random.normal((intermediate_size, hidden_size), std=scale)

    def __call__(self, x: mx.array) -> mx.array:
        gate = mx.silu(x @ self.w_gate)
        up = x @ self.w_up
        return (gate * up) @ self.w_down
