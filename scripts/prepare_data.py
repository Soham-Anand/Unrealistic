#!/usr/bin/env python3
"""Download and preprocess training data for Unrealistic."""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.tokenizer import Tokenizer
from src.data.preprocessing import preprocess_text


def download_tinystories(output_path: str = "data/raw/tinystories.txt"):
    """Download TinyStories dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        from datasets import load_dataset
        print("Downloading TinyStories...")
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for example in ds:
                text = preprocess_text(example["text"])
                if len(text) > 100:
                    f.write(text + "\n\n")
                    count += 1
                    if count % 10000 == 0:
                        print(f"  Processed {count:,} stories...")

        print(f"Saved {count:,} stories to {output_path}")
        return output_path

    except ImportError:
        print("datasets library not installed. Run: pip install datasets")
        sys.exit(1)


def train_tokenizer(text_file: str, vocab_size: int = 32000):
    """Train SentencePiece tokenizer on the corpus."""
    print(f"Training tokenizer with vocab_size={vocab_size}...")
    Tokenizer.train(text_file, vocab_size=vocab_size)
    print("Tokenizer trained.")


def tokenize_corpus(text_file: str, tokenizer_path: str = "data/tokenizer/unrealistic.model", output_path: str = "data/processed/tokens.bin"):
    """Tokenize the entire corpus and save as binary."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer = Tokenizer(tokenizer_path)

    all_tokens = []
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = tokenizer.encode(line)
                all_tokens.extend(tokens)
                all_tokens.append(tokenizer.eos_token)

    import numpy as np
    arr = np.array(all_tokens, dtype=np.uint16)
    arr.tofile(output_path)
    print(f"Tokenized {len(all_tokens):,} tokens -> {output_path}")
    return all_tokens


def main():
    print("=" * 50)
    print("  Unrealistic — Data Preparation")
    print("=" * 50)

    text_file = download_tinystories()
    train_tokenizer(text_file)
    tokenize_corpus(text_file)

    print("\nData preparation complete!")


if __name__ == "__main__":
    main()
