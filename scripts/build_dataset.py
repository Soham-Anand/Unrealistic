#!/usr/bin/env python3
"""Build Phase 1 dataset: FineWeb-Edu (60%) + Wikipedia (20%) + SlimPajama (10%) + Cosmopedia (10%)."""

import os
import sys
import gc
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from datasets import load_dataset
from src.data.tokenizer import Tokenizer

DATA_DIR = "data/phase1"
TOKENIZER_DIR = "data/tokenizer"
TOKENIZER_MODEL_PREFIX = f"{TOKENIZER_DIR}/phase1"
TOKENIZER_MODEL_PATH = f"{TOKENIZER_MODEL_PREFIX}.model"
TOKENIZER_TRAIN_TEXT = f"{DATA_DIR}/tokenizer_train.txt"
OUTPUT_BIN = f"{DATA_DIR}/tokens.bin"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TOKENIZER_DIR, exist_ok=True)

TOKEN = os.environ.get("HF_TOKEN", "").strip()

# Dataset configs
DATASETS = [
    {
        "name": "FineWeb-Edu",
        "repo": "HuggingFaceFW/fineweb-edu",
        "config": "sample-10BT",
        "ratio": 0.60,
        "text_key": "text",
    },
    {
        "name": "Wikipedia",
        "repo": "wikimedia/wikipedia",
        "config": "20231101.en",
        "ratio": 0.20,
        "text_key": "text",
    },
    {
        "name": "SlimPajama",
        "repo": "gmongaras/SlimPajama-627B_Reupload",
        "config": None,
        "ratio": 0.10,
        "text_key": "text",
    },
    {
        "name": "Cosmopedia",
        "repo": "HuggingFaceTB/cosmopedia",
        "config": "stories",
        "ratio": 0.10,
        "text_key": "text",
    },
]

TOKENIZER_SAMPLE_DOCS = 5_000  # per dataset
BINARY_CHUNK = 1_000_000  # write tokens in 1M chunks


def compute_targets(target_tokens: int):
    for ds in DATASETS:
        ds["target"] = int(target_tokens * ds["ratio"])


def load_dataset_stream(ds_config):
    if ds_config["config"]:
        return load_dataset(ds_config["repo"], ds_config["config"],
                            split="train", streaming=True)
    return load_dataset(ds_config["repo"], split="train", streaming=True)


def print_progress(name: str, docs: int, tokens: int, target: int, elapsed: float):
    pct = min(100.0, 100.0 * tokens / target) if target > 0 else 0
    rate = tokens / elapsed if elapsed > 0 else 0
    eta = (target - tokens) / rate if rate > 0 else 0
    print(f"  {name:14s} {docs:>8,} docs  {tokens:>12,} tokens  "
          f"{pct:5.1f}%  {rate:>8,.0f} tok/s  ETA {eta:>7.0f}s")


def collect_tokenizer_sample():
    """Collect sample text from all sources for tokenizer training."""
    print("\n=== Collecting tokenizer training sample ===")
    all_samples = []

    for ds_config in DATASETS:
        print(f"  Sampling {ds_config['name']}...")
        n = 0
        try:
            ds = load_dataset_stream(ds_config)
            for row in ds:
                text = row.get(ds_config["text_key"], "").strip()
                if text:
                    all_samples.append(text)
                    n += 1
                    if n >= TOKENIZER_SAMPLE_DOCS:
                        break
        except Exception as e:
            print(f"  WARNING: Failed to sample {ds_config['name']}: {e}")
            continue
        print(f"    {n} docs sampled")
        del ds
        gc.collect()

    with open(TOKENIZER_TRAIN_TEXT, "w", encoding="utf-8") as f:
        f.write("\n".join(all_samples))

    size_mb = os.path.getsize(TOKENIZER_TRAIN_TEXT) / 1e6
    print(f"  Total: {size_mb:.1f} MB, {len(all_samples)} docs\n")


def train_tokenizer():
    """Train SentencePiece tokenizer on the sample."""
    print("=== Training tokenizer ===")
    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        input=TOKENIZER_TRAIN_TEXT,
        model_prefix=TOKENIZER_MODEL_PREFIX,
        vocab_size=32000,
        model_type="bpe",
        character_coverage=0.9995,
        byte_fallback=True,
        max_sentence_length=4096,
        split_digits=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
    )
    tok = Tokenizer(TOKENIZER_MODEL_PATH)
    print(f"  Vocab size: {tok.vocab_size}")
    print(f"  EOS id: {tok.eos_token}")
    print(f"  Model: {TOKENIZER_MODEL_PATH}\n")
    return tok


def tokenize_dataset(tok, ds_config, is_smoke: bool = False):
    """Tokenize one dataset and write to binary. Returns (docs, tokens)."""
    target = ds_config["target"] if not is_smoke else int(5e6 * ds_config["ratio"]) + 1
    name = ds_config["name"]

    ds = load_dataset_stream(ds_config)
    n_docs = 0
    n_tokens = 0
    buffer = []
    start_time = time.time()

    def flush_buffer():
        nonlocal buffer
        if not buffer:
            return
        arr = np.array(buffer, dtype=np.uint16)
        with open(OUTPUT_BIN, "ab") as f:
            f.write(arr.tobytes())
        buffer = []

    for row in ds:
        text = row.get(ds_config["text_key"], "")
        if not text or not text.strip():
            continue

        ids = tok.encode(text.strip())
        if not ids:
            continue

        buffer.extend(ids)
        buffer.append(tok.eos_token)

        n_docs += 1
        n_tokens += len(ids)

        if len(buffer) >= BINARY_CHUNK:
            flush_buffer()

        if n_docs % 50_000 == 0:
            elapsed = time.time() - start_time
            print_progress(name, n_docs, n_tokens, target, elapsed)

        if n_tokens >= target:
            break

    flush_buffer()
    elapsed = time.time() - start_time
    print_progress(name, n_docs, n_tokens, target, elapsed)
    print()

    del ds
    gc.collect()
    return n_docs, n_tokens


def verify_binary(tok):
    """Verify the binary by decoding random sequences."""
    print("=== Verifying output ===")
    tokens = np.fromfile(OUTPUT_BIN, dtype=np.uint16)
    print(f"  Total tokens: {len(tokens):,}")
    print(f"  File size: {os.path.getsize(OUTPUT_BIN)/1e9:.2f} GB")

    n_docs = int(np.sum(tokens == tok.eos_token))
    print(f"  Documents: {n_docs:,}")

    rng = np.random.default_rng(42)
    for _ in range(3):
        start = rng.integers(0, max(1, len(tokens) - 200))
        snippet = tokens[start:start + 200]
        text = tok.decode(snippet.tolist())
        print(f"\n  Random sample (offset {start}):")
        print(f"    {text[:200]}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Build Phase 1 dataset")
    parser.add_argument("--target-tokens", type=int, default=5_000_000_000,
                        help="Target total tokens (default: 5B)")
    parser.add_argument("--smoke", type=int, default=0,
                        help="Smoke test: process N total tokens and verify")
    args = parser.parse_args()

    target_tokens = args.smoke if args.smoke > 0 else args.target_tokens
    is_smoke = args.smoke > 0
    compute_targets(target_tokens)

    if is_smoke:
        print(f"\n{'='*60}")
        print(f"  SMOKE TEST MODE — {target_tokens:,} tokens")
        print(f"{'='*60}\n")

    # Remove old binary
    if os.path.exists(OUTPUT_BIN):
        os.remove(OUTPUT_BIN)

    # Step 1: Collect tokenizer training sample
    if not os.path.exists(TOKENIZER_MODEL_PATH):
        collect_tokenizer_sample()
        tok = train_tokenizer()
    else:
        print(f"Tokenizer exists at {TOKENIZER_MODEL_PATH}, reusing")
        tok = Tokenizer(TOKENIZER_MODEL_PATH)

    # Step 2: Tokenize each dataset
    print("=== Tokenizing datasets ===")
    total_docs = 0
    total_tokens = 0

    for ds_config in DATASETS:
        print(f"--- {ds_config['name']} ---")
        docs, toks = tokenize_dataset(tok, ds_config, is_smoke=is_smoke)
        total_docs += docs
        total_tokens += toks

    print(f"\n  TOTAL: {total_docs:,} docs, {total_tokens:,} tokens")

    # Step 3: Verify
    verify_binary(tok)

    print("Phase 1 dataset build complete!")
    if is_smoke:
        print(f"Smoke test passed: {total_tokens:,} tokens written to {OUTPUT_BIN}")


if __name__ == "__main__":
    main()
