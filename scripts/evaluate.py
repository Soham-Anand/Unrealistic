#!/usr/bin/env python3
"""Evaluate trained Unrealistic model."""

import os
import sys
import json
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
from src.model.transformer import init_model, forward
from src.data.dataset import StreamingDataset
from src.training.loss import cross_entropy_loss


def main():
    mx.set_default_device(mx.gpu)

    with open("configs/training_config.json") as f:
        config = json.load(f)

    params, buffers, model_cfg = init_model("configs/model_config.json")
    print(f"Parameters: {sum(v.size for v in params.values() if hasattr(v, 'size')):,}")

    tokens = np.fromfile("data/processed/tokens.bin", dtype=np.uint16).tolist()
    split = int(len(tokens) * 0.95)
    val_tokens = tokens[split:]

    dataset = StreamingDataset(val_tokens, seq_length=config["sequence_length"])

    total_loss = 0.0
    n_batches = 0

    for x, y in dataset.iterate_batches(8):
        if n_batches >= config.get("eval_steps", 50):
            break
        logits = forward(params, buffers, x, model_cfg)
        loss = cross_entropy_loss(logits, y)
        total_loss += float(loss)
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    ppl = math.exp(avg_loss)
    print(f"Eval loss: {avg_loss:.4f} | Perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
