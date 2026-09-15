#!/usr/bin/env python3
"""Interactive text generation with Unrealistic."""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
from src.model.transformer import init_model
from src.training.checkpointing import load_checkpoint
from src.data.tokenizer import Tokenizer
from src.inference.generate import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint dir (e.g. checkpoints/step_005000)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top-p", type=float, default=0.0, dest="top_p",
                        help="Nucleus sampling threshold (0 = disabled)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling (0 = disabled)")
    parser.add_argument("--max-new-tokens", type=int, default=200,
                        help="Maximum tokens to generate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible generation")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty (1.0 = disabled)")
    args = parser.parse_args()

    if args.seed is not None:
        mx.random.seed(args.seed)

    mx.set_default_device(mx.gpu)
    tokenizer = Tokenizer("data/tokenizer/phase1.model")

    if args.checkpoint:
        params, cfg = load_checkpoint(args.checkpoint)
        _, buffers, _ = init_model("configs/model_config.json")
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        params, buffers, cfg = init_model("configs/model_config.json")

    print("\nUnrealistic 187M — Interactive Generation")
    print(f"Temperature: {args.temperature}")
    print(f"Top-k: {args.top_k}")
    print(f"Top-p: {args.top_p}")
    print(f"Repetition penalty: {args.repetition_penalty}")
    print(f"Seed: {args.seed}")
    print("Type 'quit' to exit\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except EOFError:
            break
        if prompt.lower() in ("quit", "exit", "q"):
            break

        tokens = tokenizer.encode(prompt)
        out_tokens = generate(params, buffers, cfg, tokens,
                              max_new_tokens=args.max_new_tokens,
                              temperature=args.temperature,
                              top_k=args.top_k, top_p=args.top_p,
                              repetition_penalty=args.repetition_penalty)
        response = tokenizer.decode(out_tokens[len(tokens):])
        print(f"Unrealistic: {response}\n")


if __name__ == "__main__":
    main()
