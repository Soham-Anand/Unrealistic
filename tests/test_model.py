import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
from src.model.transformer import UnrealisticModel


def test_model_forward_pass():
    model = UnrealisticModel("configs/model_config.json")
    tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    logits = model(tokens)
    assert logits.shape == (1, 5, model.config["vocab_size"])


def test_model_parameter_count():
    model = UnrealisticModel("configs/model_config.json")
    count = model.count_parameters()
    assert count > 100_000_000, f"Expected >100M params, got {count}"
    assert count < 250_000_000, f"Expected <250M params, got {count}"


def test_causal_mask():
    model = UnrealisticModel("configs/model_config.json")
    mask = model._causal_mask(4)
    assert mask.shape == (1, 1, 4, 4)
    assert float(mask[0, 0, 0, 1]) == float("-inf")
    assert float(mask[0, 0, 0, 0]) == 0.0
