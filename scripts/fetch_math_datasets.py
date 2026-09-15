#!/usr/bin/env python3
"""Fetch GSM8K dataset and tokenize to uint16 bin for Phase 5 math training.

Output: data/phase5/gsm8k.bin
"""

import os, sys, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from datasets import load_dataset
from src.data.tokenizer import Tokenizer

DATA_DIR = "data/phase5"
TOKENIZER_PATH = "data/tokenizer/phase1.model"


def main():
    parser = argparse.ArgumentParser(description="Fetch GSM8K for math training")
    parser.add_argument("--max-tokens", type=int, default=15_000_000,
                        help="Max tokens to extract from GSM8K")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    tok = Tokenizer(TOKENIZER_PATH)
    EOS = tok.eos_token

    print("Loading GSM8K train split...")
    ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)

    out_path = f"{DATA_DIR}/gsm8k.bin"
    n = 0
    n_docs = 0
    with open(out_path, "wb") as f:
        for row in ds:
            q = row["question"].strip()
            a = row["answer"].strip()

            # Format as worked solution
            doc = f"Problem: {q}\n\nSolution:\n{a}"
            ids = tok.encode(doc)
            if not ids:
                continue
            if n + len(ids) > args.max_tokens:
                ids = ids[:args.max_tokens - n]
            if not ids:
                break
            f.write(np.array(ids, dtype=np.uint16).tobytes())
            f.write(np.uint16(EOS).tobytes())
            n += len(ids)
            n_docs += 1
            if n >= args.max_tokens:
                break

    print(f"  GSM8K: {n_docs} problems, {n:,} tokens -> {out_path}")


if __name__ == "__main__":
    main()
