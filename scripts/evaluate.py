#!/usr/bin/env python3
"""Evaluate trained Unrealistic model."""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import mlx.core as mx
from src.model.transformer import UnrealisticModel
from src.data.dataset import StreamingDataset
from src.training.loss import cross_entropy_loss


def main():
    mx.set_default_device(mx.gpu)

    with open("configs/training_config.json") as f:
        config = json.load(f)

    print("Loading model...")
    model = UnrealisticModel("configs/model_config.json")

    tokens = np.fromfile("data/processed/tokens.bin", dtype=np.uint16).tolist()
    split = int(len(tokens) * 0.95)
    val_tokens = tokens[split:]

    dataset = StreamingDataset(val_tokens, seq_length=config["sequence_length"])

    total_loss = 0.0
    n_batches = 0
    batch_size = 8

    for x, y in dataset.iterate_batches(batch_size):
        if n_batches >= config.get("eval_steps", 50):
            break
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        total_loss += float(loss)
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    import math
    ppl = math.exp(avg_loss)
    print(f"Eval loss: {avg_loss:.4f} | Perplexity: {ppl:.2f}")


if __name__ == "__main__":
    main()
