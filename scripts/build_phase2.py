#!/usr/bin/env python3
"""Build Phase 2 dataset: FineWeb-Edu (820M train + 10M eval)."""

import os, sys, gc, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from datasets import load_dataset
from src.data.tokenizer import Tokenizer

DATA_DIR = "data/phase2"
TOKENIZER_PATH = "data/tokenizer/phase1.model"
REPO = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-10BT"
TEXT_KEY = "text"
BINARY_CHUNK = 1_000_000

OUTPUT_TRAIN = f"{DATA_DIR}/fwe_train.bin"
OUTPUT_EVAL = f"{DATA_DIR}/fwe_eval.bin"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tokens", type=int, default=820_000_000)
    parser.add_argument("--eval-tokens", type=int, default=10_000_000)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    tok = Tokenizer(TOKENIZER_PATH)
    total_needed = args.train_tokens + args.eval_tokens

    print(f"FineWeb-Edu build: {args.train_tokens:,} train + {args.eval_tokens:,} eval = {total_needed:,} total")
    print(f"Streaming from {REPO} ({CONFIG})...")

    ds = load_dataset(REPO, CONFIG, split="train", streaming=True)
    print(f"Tokenizer loaded (vocab={tok.vocab_size})")

    for path in [OUTPUT_TRAIN, OUTPUT_EVAL]:
        if os.path.exists(path):
            os.remove(path)

    eval_start = args.train_tokens
    buffer = []
    n_tokens = 0
    train_toks = eval_toks = train_docs = eval_docs = 0
    mode = "train"
    start_time = time.time()
    last_report = 0

    def flush_buffer(path):
        nonlocal buffer
        if not buffer:
            return
        arr = np.array(buffer, dtype=np.uint16)
        with open(path, "ab") as f:
            f.write(arr.tobytes())
        buffer = []

    for row in ds:
        text = row.get(TEXT_KEY, "")
        if not text or not text.strip():
            continue
        ids = tok.encode(text.strip())
        if not ids:
            continue

        doc_len = len(ids)

        if mode == "train" and n_tokens + doc_len > eval_start and n_tokens < eval_start:
            mid = eval_start - n_tokens
            ids = ids[:mid]
            doc_len = mid

        if mode == "train" and n_tokens + doc_len >= eval_start and eval_docs == 0:
            mode = "eval"

        buffer.extend(ids[:doc_len])
        buffer.append(tok.eos_token)

        n_tokens += doc_len
        if mode == "train":
            train_docs += 1
            train_toks += doc_len
            out_path = OUTPUT_TRAIN
        else:
            eval_docs += 1
            eval_toks += doc_len
            out_path = OUTPUT_EVAL
            if eval_toks >= args.eval_tokens:
                flush_buffer(out_path)
                break

        if len(buffer) >= BINARY_CHUNK:
            flush_buffer(out_path)

        if n_tokens - last_report >= 10_000_000:
            last_report = n_tokens
            elapsed = time.time() - start_time
            pct = min(100.0, 100.0 * n_tokens / total_needed)
            rate = n_tokens / elapsed if elapsed > 0 else 0
            eta = (total_needed - n_tokens) / rate if rate > 0 else 0
            print(f"  {train_docs:>8,} docs  {n_tokens:>12,} tokens  {pct:5.1f}%  {rate:>8,.0f} tok/s  ETA {eta:>7.0f}s")

    flush_buffer(OUTPUT_TRAIN if mode == "train" else out_path)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Train: {train_docs:,} docs, {train_toks:,} tokens -> {OUTPUT_TRAIN}")
    print(f"  Eval:  {eval_docs:,} docs, {eval_toks:,} tokens -> {OUTPUT_EVAL}")
    print(f"  Files: {os.path.getsize(OUTPUT_TRAIN)/1e6:.0f} MB + {os.path.getsize(OUTPUT_EVAL)/1e6:.0f} MB")


if __name__ == "__main__":
    main()
