#!/usr/bin/env python3
"""Evaluate a specific checkpoint against a given eval.bin dataset."""

import os
import sys
import json
import math
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
from mlx.utils import tree_map
from src.model.transformer import init_model, forward
from src.data.dataset import StreamingDataset
from src.training.loss import cross_entropy_loss
from src.training.checkpointing import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="checkpoint dir with model.npz + config.json")
    parser.add_argument("--eval-bin", required=True, help="eval .bin file")
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)

    params, model_cfg, _ = load_checkpoint(args.checkpoint, return_opt=True)
    params = tree_map(lambda v: v.astype(mx.bfloat16) if isinstance(v, mx.array) else v, params)
    buffers = {}
    mx.eval(params)
    print(f"Loaded {args.checkpoint}")

    dataset = StreamingDataset([args.eval_bin], seq_length=args.seq_length)

    compiled_forward = mx.compile(
        lambda p, b, x: forward(p, b, x, model_cfg, training=False)
    )

    total_loss = 0.0
    n_batches = 0
    for x, y in dataset.iterate_batches(args.batch_size):
        if n_batches >= args.eval_steps:
            break
        logits = compiled_forward(params, buffers, x)
        loss = cross_entropy_loss(logits, y)
        total_loss += float(loss)
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    ppl = math.exp(avg_loss)
    print(f"[eval] {os.path.basename(args.eval_bin)} | val_loss {avg_loss:.4f} | ppl {ppl:.2f} | batches {n_batches}")


if __name__ == "__main__":
    main()
