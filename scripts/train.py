#!/usr/bin/env python3
"""Train Unrealistic LLM from scratch."""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import mlx.core as mx

from src.model.transformer import UnrealisticModel
from src.data.dataset import StreamingDataset
from src.training.trainer import Trainer


def main():
    mx.set_default_device(mx.gpu)

    with open("configs/model_config.json") as f:
        model_config = json.load(f)
    with open("configs/training_config.json") as f:
        train_config = json.load(f)

    np.random.seed(train_config["seed"])

    print("Loading tokenized data...")
    tokens = np.fromfile("data/processed/tokens.bin", dtype=np.uint16).tolist()
    print(f"Loaded {len(tokens):,} tokens")

    split = int(len(tokens) * 0.95)
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    print("Initializing model...")
    model = UnrealisticModel("configs/model_config.json")
    print(f"Parameters: {model.count_parameters():,}")

    train_dataset = StreamingDataset(train_tokens, seq_length=train_config["sequence_length"])

    trainer = Trainer(model, train_config)
    trainer.train(train_dataset)


if __name__ == "__main__":
    main()
