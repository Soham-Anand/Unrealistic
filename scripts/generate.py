#!/usr/bin/env python3
"""Interactive text generation with Unrealistic."""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
from src.model.transformer import UnrealisticModel
from src.data.tokenizer import Tokenizer
from src.inference.generate import generate


def main():
    mx.set_default_device(mx.gpu)

    print("Loading model...")
    model = UnrealisticModel("configs/model_config.json")
    tokenizer = Tokenizer("data/tokenizer/unrealistic.model")

    print("\nUnrealistic 187M — Interactive Generation")
    print("Type 'quit' to exit\n")

    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            break

        tokens = tokenizer.encode(prompt)
        out_tokens = generate(model, tokens, max_new_tokens=200, temperature=0.8)
        response = tokenizer.decode(out_tokens[len(tokens):])
        print(f"Unrealistic: {response}\n")


if __name__ == "__main__":
    main()
