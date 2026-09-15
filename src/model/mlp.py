import mlx.core as mx


def init_swiglu(hidden_size: int, intermediate_size: int) -> dict:
    scale = 0.02
    return {
        "w_gate": mx.random.normal((hidden_size, intermediate_size), scale=scale),
        "w_up": mx.random.normal((hidden_size, intermediate_size), scale=scale),
        "w_down": mx.random.normal((intermediate_size, hidden_size), scale=scale),
    }


def swiglu(x: mx.array, params: dict, training: bool = False,
           dropout_rate: float = 0.0) -> mx.array:
    gate_input = x @ params["w_gate"]
    gate = gate_input * mx.sigmoid(gate_input)
    up = x @ params["w_up"]
    out = (gate * up) @ params["w_down"]

    if training and dropout_rate > 0:
        keep_prob = 1.0 - dropout_rate
        drop_mask = mx.random.uniform(shape=out.shape, dtype=out.dtype) < keep_prob
        out = out * drop_mask / keep_prob

    return out
