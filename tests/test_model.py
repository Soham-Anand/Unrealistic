import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
from src.model.transformer import init_model, forward, count_parameters, causal_mask


def test_model_forward_pass():
    params, buffers, cfg = init_model("configs/model_config.json")
    tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    logits = forward(params, buffers, tokens, cfg)
    assert logits.shape == (1, 5, cfg["vocab_size"])


def test_model_parameter_count():
    params, buffers, cfg = init_model("configs/model_config.json")
    count = count_parameters(params)
    assert count > 100_000_000, f"Expected >100M params, got {count}"
    assert count < 250_000_000, f"Expected <250M params, got {count}"


def test_causal_mask():
    mask = causal_mask(4)
    assert mask.shape == (1, 1, 4, 4)
    assert float(mask[0, 0, 0, 1]) == float("-inf")
    assert float(mask[0, 0, 0, 0]) == 0.0
