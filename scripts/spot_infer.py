#!/usr/bin/env python3
"""Tuned spot-inference against the latest (or given) checkpoint.

Runs the 5 knowledge/prose probes with the decoder knobs that make this
model readable: temperature=0.5, top_p=0.9, repetition_penalty=1.15.
"""

import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
from src.model.transformer import init_model
from src.training.checkpointing import load_checkpoint
from src.data.tokenizer import Tokenizer
from src.inference.generate import generate

DEFAULT_PROMPTS = [
    "Once upon a time,",
    "The capital of France is",
    "The opposite of hot is",
    "Albert Einstein was born in",
    "The water cycle",
]


def latest_checkpoint(root: str = "checkpoints") -> str:
    dirs = glob.glob(os.path.join(root, "step_*"))
    return max(dirs, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint dir (default: newest under checkpoints/)")
    parser.add_argument("--prompts", type=str, nargs="*", default=None,
                        help="Override probe prompts (default: 5 tuned probes)")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--self-consistency", type=int, default=0,
                        help="Sample N times per prompt (self-consistency). "
                             "0 = single sample (default).")
    parser.add_argument("--majority", action="store_true",
                        help="With --self-consistency, pick the most common "
                             "full response instead of printing all (rank by "
                             "exact-text frequency).")
    args = parser.parse_args()

    ckpt = args.checkpoint or latest_checkpoint()
    prompts = args.prompts or DEFAULT_PROMPTS

    mx.set_default_device(mx.gpu)
    params, _ = load_checkpoint(ckpt)
    _, buffers, model_cfg = init_model("configs/model_config.json")
    tokenizer = Tokenizer("data/tokenizer/phase1.model")

    print(f"Checkpoint: {ckpt}")
    print(f"temp={args.temperature} top_p={args.top_p} "
          f"rep_penalty={args.repetition_penalty} max_tokens={args.max_new_tokens}")
    print("=" * 60)

    for i, prompt in enumerate(prompts):
        tokens = tokenizer.encode(prompt)
        texts = []
        n = max(1, args.self_consistency)
        for s in range(n):
            mx.random.seed(args.seed + i * 100 + s)
            out = generate(params, buffers, model_cfg, tokens,
                           max_new_tokens=args.max_new_tokens,
                           temperature=args.temperature,
                           top_k=0, top_p=args.top_p,
                           repetition_penalty=args.repetition_penalty)
            texts.append(tokenizer.decode(out[len(tokens):]))

        print(f"--- {prompt!r} ---")
        if args.self_consistency > 1 and args.majority:
            from collections import Counter
            most_common, _ = Counter(texts).most_common(1)[0]
            print(most_common)
        else:
            for text in texts:
                print(text)
        print()

    print("\nTuning tips:")
    print("  best clarity so far: temp=0.4 top_p=0.9 rep_penalty=1.25")
    print("  stronger anti-collapse: temp=0.3 rep_penalty=1.3")
    print("  more variety: temp=0.7 top_p=0.95")


if __name__ == "__main__":
    main()
