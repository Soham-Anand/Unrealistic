#!/usr/bin/env python3
"""Build Phase 3 dataset: Wikipedia (350M) + Cosmopedia (150M), merged doc-level 70/30, + 10M wiki eval."""

import os, sys, time, argparse, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from datasets import load_dataset
from src.data.tokenizer import Tokenizer

DATA_DIR = "data/phase3"
TOKENIZER_PATH = "data/tokenizer/phase1.model"
OUT_TRAIN = f"{DATA_DIR}/train.bin"
OUT_EVAL = f"{DATA_DIR}/eval.bin"
WIKI_RAW = f"{DATA_DIR}/wiki_raw.bin"
COSMO_RAW = f"{DATA_DIR}/cosmo_raw.bin"
EOS = None


def tokenize_stream(tok, ds, text_key, raw_path, eval_path, total_target, eval_at, sample_fn=None):
    """Write docs (ids+eos) to raw_path, switching to eval_path after eval_at tokens."""
    n = 0
    start = time.time()
    last_report = 0
    f = open(raw_path, "wb")
    switched = False

    def report(final=False):
        el = time.time() - start
        rate = n / el if el else 0
        eta = (total_target - n) / rate if rate else 0
        tag = "done" if final else ""
        print(f"    {n:>12,}/{total_target:,} tok  {100*n/total_target:5.1f}%  {rate:,.0f} tok/s  ETA {eta:.0f}s {tag}")

    for row in ds:
        if sample_fn is not None and not sample_fn(row):
            continue
        text = row.get(text_key, "")
        if not text or not text.strip():
            continue
        ids = tok.encode(text.strip())
        if not ids:
            continue
        if n < eval_at and n + len(ids) > eval_at:
            ids = ids[: eval_at - n]
        if n + len(ids) > total_target:
            ids = ids[: total_target - n]
        if not ids:
            break
        if eval_path is not None and n >= eval_at and not switched:
            f.close()
            f = open(eval_path, "wb")
            switched = True
        f.write(np.array(ids, dtype=np.uint16).tobytes())
        f.write(np.uint16(EOS).tobytes())
        n += len(ids)
        if n - last_report >= 20_000_000:
            last_report = n
            report()
        if n >= total_target:
            break

    f.close()
    report(final=True)
    return n


def iter_docs(path):
    """Yield docs (list of ids) from a uint16 bin, split on EOS (handles chunk boundaries)."""
    with open(path, "rb") as f:
        carry = np.zeros(0, dtype=np.uint16)
        while True:
            chunk = f.read(1 << 30)
            if not chunk:
                break
            arr = np.concatenate([carry, np.frombuffer(chunk, dtype=np.uint16)])
            carry = np.zeros(0, dtype=np.uint16)
            start = 0
            for i, v in enumerate(arr):
                if v == EOS:
                    yield arr[start:i].tolist()
                    start = i + 1
            if start < len(arr):
                carry = arr[start:]


def merge_sources(sources, out_path, seed=42):
    """sources: list of (path, token_target). Doc-level weighted interleave."""
    import random
    rng = random.Random(seed)
    active = list(sources)
    iters = {p: iter(iter_docs(p)) for p, _ in sources}
    remaining = {p: t for p, t in sources}
    total_used = 0
    total_target = sum(t for _, t in sources)
    start = time.time()
    last_report = 0
    with open(out_path, "wb") as out:
        while active:
            weights = [remaining[p] for p, _ in active]
            chosen = rng.choices([p for p, _ in active], weights=weights, k=1)[0]
            try:
                doc = next(iters[chosen])
            except StopIteration:
                active = [s for s in active if s[0] != chosen]
                iters.pop(chosen, None)
                remaining.pop(chosen, None)
                continue
            out.write(np.array(doc, dtype=np.uint16).tobytes())
            out.write(np.uint16(EOS).tobytes())
            remaining[chosen] -= len(doc)
            total_used += len(doc)
            if remaining[chosen] <= 0:
                active = [s for s in active if s[0] != chosen]
                iters.pop(chosen, None)
                remaining.pop(chosen, None)
            if total_used - last_report >= 20_000_000:
                last_report = total_used
                el = time.time() - start
                rate = total_used / el if el else 0
                eta = (total_target - total_used) / rate if rate else 0
                print(f"    merged {total_used:>12,}/{total_target:,}  {100*total_used/total_target:5.1f}%  {rate:,.0f} tok/s  ETA {eta:.0f}s")
    print(f"    merge done: {total_used:,} tok -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki-tokens", type=int, default=350_000_000)
    parser.add_argument("--cosmo-tokens", type=int, default=150_000_000)
    parser.add_argument("--eval-tokens", type=int, default=10_000_000)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    tok = Tokenizer(TOKENIZER_PATH)
    global EOS
    EOS = tok.eos_token
    print(f"Phase 3 build: wiki {args.wiki_tokens/1e6:.0f}M + cosmo {args.cosmo_tokens/1e6:.0f}M + eval {args.eval_tokens/1e6:.0f}M")

    wiki_keep = lambda row: (int(hashlib.md5(row.get("id", "").encode()).hexdigest()[:8], 16) % 9) == 0

    print("Streaming Wikipedia (20231101.en, ~11% reservoir)...")
    wiki_ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    tokenize_stream(tok, wiki_ds, "text", WIKI_RAW, OUT_EVAL,
                    args.wiki_tokens + args.eval_tokens, args.wiki_tokens, sample_fn=wiki_keep)

    print("Streaming Cosmopedia (khanacademy)...")
    cosmo_ds = load_dataset("HuggingFaceTB/cosmopedia", "khanacademy", split="train", streaming=True)
    tokenize_stream(tok, cosmo_ds, "text", COSMO_RAW, None, args.cosmo_tokens, args.cosmo_tokens)

    print("Merging (70/30 doc interleave)...")
    merge_sources([(WIKI_RAW, args.wiki_tokens), (COSMO_RAW, args.cosmo_tokens)], OUT_TRAIN)

    print("\nDone:")
    for p in [OUT_TRAIN, OUT_EVAL]:
        print(f"  {p}: {os.path.getsize(p)/1e6:.0f} MB")


if __name__ == "__main__":
    main()
