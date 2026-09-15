#!/usr/bin/env python3
"""Build Phase 5 dataset: mixed refresh (extended, code-preserving + arithmetic).

Mix (220M tokens) to rebalance skills before SFT:
  FineWeb-Edu      27%  (60M)  <- data/phase2/fwe_train.bin
  Wiki/Cosmopedia  23%  (50M)  <- data/phase3/train.bin
  Phase-4 code     23%  (50M)  <- data/phase4/train.bin  (build Phase 4 first)
  Arithmetic        9%  (20M)  <- synthetic + GSM8K (NEW)
  OpenWebText       9%  (20M)  <- data/phase1/owt_train.bin
  SlimPajama        9%  (20M)  <- streamed (only online step)

Arithmetic added to fix the "2+2=4" deficit from Phase 4.
Code reduced to 23% to make room. FWE+Wiki at 50% for facts.

Local sources are sliced at random doc boundaries (EOS-delimited bins),
then merged with weighted doc-level interleave (seed 42). Eval = 10M tokens
from the head of the merged train.
"""

import os, sys, time, argparse, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from datasets import load_dataset
from src.data.tokenizer import Tokenizer

DATA_DIR = "data/phase5"
TOKENIZER_PATH = "data/tokenizer/phase1.model"
OUT_TRAIN = f"{DATA_DIR}/train.bin"
OUT_EVAL = f"{DATA_DIR}/eval.bin"
EOS = None

TOTAL_TARGET = 220_000_000

# (label, local bin, target tokens)
SOURCES = [
    ("fwe",  "data/phase2/fwe_train.bin", 60_000_000),
    ("wiki", "data/phase3/train.bin",     50_000_000),
    ("code", "data/phase4/train.bin",     50_000_000),
    ("owt",  "data/phase1/owt_train.bin", 20_000_000),
]

ARITH_TARGET = 20_000_000
ARITH_RAW = f"{DATA_DIR}/arith_raw.bin"

SLIM_LABEL = "slim"
SLIM_TARGET = 20_000_000
SLIM_RAW = f"{DATA_DIR}/slim_raw.bin"
SLIM_REPOS = ["gmongaras/SlimPajama-627B_Reupload", "cerebras/SlimPajama-627B"]

EVAL_TOKENS = 10_000_000


def eos_indices(path):
    """Global indices of EOS tokens (doc ends), chunked scan to bound memory."""
    indices = []
    offset = 0
    with open(path, "rb") as f:
        while True:
            chunk = np.frombuffer(f.read(1 << 28), dtype=np.uint16)
            if chunk.size == 0:
                break
            idx = np.nonzero(chunk == EOS)[0]
            indices.extend((offset + idx).tolist())
            offset += chunk.size
    if not indices:
        return np.zeros(0, dtype=np.int64)
    return np.array(indices, dtype=np.int64)


def slice_bin_docs(path, target, seed, out_path):
    """Random doc-boundary slice of `target` tokens from a uint16 EOS bin."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing source: {path}")
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    seps = eos_indices(path)
    n_docs = len(seps)
    if n_docs == 0:
        raise ValueError(f"No documents (no EOS) in {path}")
    starts = np.concatenate(([-1], seps[:-1] + 1))
    rng = random.Random(seed)
    start_doc = rng.randrange(n_docs)
    n = 0
    t0 = time.time()
    with open(out_path, "wb") as f:
        for i in range(n_docs):
            di = (start_doc + i) % n_docs
            lo, hi = int(starts[di]), int(seps[di])
            f.write(arr[lo:hi].tobytes())
            f.write(np.uint16(EOS).tobytes())
            n += hi - lo
            if n >= target:
                break
    del arr
    print(f"  {os.path.basename(path):22s} -> {n:>10,}/{target:,} tok  {time.time()-t0:.0f}s")
    return n


def tokenize_slim(tok, raw_path, target):
    """Stream SlimPajama to raw_path until `target` tokens."""
    ds = None
    for repo in SLIM_REPOS:
        try:
            print(f"  streaming {repo} ...")
            ds = load_dataset(repo, split="train", streaming=True)
            break
        except Exception as e:
            print(f"    failed ({e}); trying next repo")
    if ds is None:
        raise RuntimeError("All SlimPajama repos unavailable")
    n = 0
    t0 = time.time()
    with open(raw_path, "wb") as f:
        for row in ds:
            text = row.get("text", "")
            if not text or not text.strip():
                continue
            ids = tok.encode(text.strip())
            if not ids:
                continue
            if n + len(ids) > target:
                ids = ids[: target - n]
            if not ids:
                break
            f.write(np.array(ids, dtype=np.uint16).tobytes())
            f.write(np.uint16(EOS).tobytes())
            n += len(ids)
            if n >= target:
                break
    print(f"  slim -> {n:>10,}/{target:,} tok  {time.time()-t0:.0f}s")
    return n


def iter_docs(path):
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
    """sources: list of (path, token_target). Weighted doc-level interleave."""
    rng = random.Random(seed)
    active = list(sources)
    iters = {p: iter(iter_docs(p)) for p, _ in sources}
    remaining = {p: t for p, t in sources}
    total_used = 0
    total_target = sum(t for _, t in sources)
    t0 = time.time()
    with open(out_path, "wb") as out:
        while active:
            weights = [max(1, remaining[p]) for p, _ in active]
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
    print(f"  merged {total_used:,} tok -> {out_path}  {time.time()-t0:.0f}s")


def slice_head(path, target, out_path):
    """First `target` tokens of a bin at doc boundaries (for eval)."""
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    seps = eos_indices(path)
    n = 0
    with open(out_path, "wb") as f:
        start = 0
        for se in seps:
            lo, hi = int(start), int(se)
            if hi - lo == 0:
                start = se + 1
                continue
            f.write(arr[lo:hi].tobytes())
            f.write(np.uint16(EOS).tobytes())
            n += hi - lo
            start = se + 1
            if n >= target:
                break
    del arr
    print(f"  eval -> {n:,} tok -> {out_path}")


def generate_arithmetic(tok, out_path, target, seed=42):
    """Generate synthetic arithmetic problems and write tokenized to out_path."""
    rng = random.Random(seed)
    n = 0
    t0 = time.time()

    def write_doc(text):
        nonlocal n
        ids = tok.encode(text.strip())
        if not ids:
            return
        if n + len(ids) > target:
            ids = ids[:target - n]
        if not ids:
            return
        f.write(np.array(ids, dtype=np.uint16).tobytes())
        f.write(np.uint16(EOS).tobytes())
        n += len(ids)

    with open(out_path, "wb") as f:
        # === Addition (0-100) ===
        for _ in range(target // 6):
            a, b = rng.randint(0, 100), rng.randint(0, 100)
            ans = a + b
            fmt = rng.choice([
                f"{a} + {b} = {ans}",
                f"What is {a} + {b}? The answer is {ans}.",
                f"Calculate {a} + {b}. Solution: {ans}",
                f"{a} plus {b} equals {ans}",
                f"If you have {a} and add {b}, you get {ans}.",
                f"{a} + {b} = ? Answer: {ans}",
            ])
            write_doc(fmt)
            if n >= target:
                break

        # === Subtraction (0-100) ===
        for _ in range(target // 6):
            a = rng.randint(10, 100)
            b = rng.randint(0, a)
            ans = a - b
            fmt = rng.choice([
                f"{a} - {b} = {ans}",
                f"What is {a} minus {b}? The answer is {ans}.",
                f"Calculate {a} - {b}. Solution: {ans}",
                f"{a} subtract {b} equals {ans}",
                f"{a} - {b} = ? Answer: {ans}",
            ])
            write_doc(fmt)
            if n >= target:
                break

        # === Multiplication (0-12) ===
        for _ in range(target // 6):
            a, b = rng.randint(0, 12), rng.randint(0, 12)
            ans = a * b
            fmt = rng.choice([
                f"{a} * {b} = {ans}",
                f"What is {a} times {b}? The answer is {ans}.",
                f"Calculate {a} x {b}. Solution: {ans}",
                f"{a} multiplied by {b} equals {ans}",
                f"{a} * {b} = ? Answer: {ans}",
            ])
            write_doc(fmt)
            if n >= target:
                break

        # === Division (clean, 0-144) ===
        for _ in range(target // 6):
            b = rng.randint(1, 12)
            q = rng.randint(0, 12)
            a = b * q
            fmt = rng.choice([
                f"{a} / {b} = {q}",
                f"What is {a} divided by {b}? The answer is {q}.",
                f"Calculate {a} / {b}. Solution: {q}",
                f"{a} divided by {b} equals {q}",
                f"{a} / {b} = ? Answer: {q}",
            ])
            write_doc(fmt)
            if n >= target:
                break

        # === Square roots (perfect squares 0-144) ===
        for _ in range(target // 6):
            root = rng.randint(0, 12)
            sq = root * root
            fmt = rng.choice([
                f"The square root of {sq} is {root}.",
                f"What is sqrt({sq})? The answer is {root}.",
                f"sqrt({sq}) = {root}",
                f"The square root of {sq} is {root}.",
            ])
            write_doc(fmt)
            if n >= target:
                break

    print(f"  arithmetic -> {n:>10,}/{target:,} tok  {time.time()-t0:.0f}s")
    return n


def fetch_gsm8k(tok, out_path, target, seed=42):
    """Fetch GSM8K train split and tokenize to out_path."""
    print("  streaming openai/gsm8k ...")
    ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    rng = random.Random(seed)
    n = 0
    t0 = time.time()
    with open(out_path, "wb") as f:
        for row in ds:
            q = row["question"].strip()
            a = row["answer"].strip()
            # Format as Q&A pair
            doc = f"Question: {q}\nAnswer: {a}"
            ids = tok.encode(doc)
            if not ids:
                continue
            if n + len(ids) > target:
                ids = ids[:target - n]
            if not ids:
                break
            f.write(np.array(ids, dtype=np.uint16).tobytes())
            f.write(np.uint16(EOS).tobytes())
            n += len(ids)
            if n >= target:
                break
    print(f"  gsm8k -> {n:>10,}/{target:,} tok  {time.time()-t0:.0f}s")
    return n


def main():
    parser = argparse.ArgumentParser(description="Build Phase 5 mixed refresh dataset")
    parser.add_argument("--smoke", type=int, default=0,
                        help="Smoke test: scale targets to N total tokens")
    parser.add_argument("--skip-slim", action="store_true",
                        help="Skip the SlimPajama stream (offline smoke)")
    parser.add_argument("--skip-arithmetic", action="store_true",
                        help="Skip arithmetic generation (use cached arith_raw.bin)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    tok = Tokenizer(TOKENIZER_PATH)
    global EOS
    EOS = tok.eos_token

    scale = (args.smoke / TOTAL_TARGET) if args.smoke > 0 else 1.0
    tag = "SMOKE" if scale < 1.0 else "FULL"
    print(f"[{tag}] refresh build: scale={scale:.4f}")

    sources = [(label, path, max(1, int(target * scale))) for label, path, target in SOURCES]
    slim_target = 0 if args.skip_slim else max(1, int(SLIM_TARGET * scale))
    arith_target = max(1, int(ARITH_TARGET * scale))

    # Local slices
    raw_parts = []
    for label, path, target in sources:
        raw = f"{DATA_DIR}/{label}_raw.bin"
        try:
            slice_bin_docs(path, target, args.seed, raw)
        except FileNotFoundError as e:
            if scale < 1.0:
                print(f"  WARNING: {e} -- skipping (smoke mode)")
                continue
            raise
        raw_parts.append((raw, target))

    # Arithmetic data (synthetic + GSM8K)
    if not args.skip_arithmetic:
        print("Generating arithmetic data...")
        arith_synth = f"{DATA_DIR}/arith_synth.bin"
        arith_gsm = f"{DATA_DIR}/arith_gsm8k.bin"
        generate_arithmetic(tok, arith_synth, arith_target // 2, args.seed)
        fetch_gsm8k(tok, arith_gsm, arith_target // 2, args.seed)
        # Merge arithmetic sources into one file
        merge_sources([
            (arith_synth, arith_target // 2),
            (arith_gsm, arith_target // 2),
        ], ARITH_RAW, seed=args.seed)
    raw_parts.append((ARITH_RAW, arith_target))

    # SlimPajama stream (only online step)
    if slim_target > 0:
        tokenize_slim(tok, SLIM_RAW, slim_target)
        raw_parts.append((SLIM_RAW, slim_target))

    # Weighted merge
    print("Merging (weighted doc interleave)...")
    merge_sources(raw_parts, OUT_TRAIN, seed=args.seed)

    # Eval from head of merged train
    print("Slicing eval head...")
    slice_head(OUT_TRAIN, max(1, int(EVAL_TOKENS * scale)), OUT_EVAL)

    print("\nDone:")
    for p in [OUT_TRAIN, OUT_EVAL]:
        sz = os.path.getsize(p) / 1e6
        n_tok = len(np.fromfile(p, dtype=np.uint16, count=-1)) if os.path.getsize(p) < 2e9 else -1
        print(f"  {p}: {sz:.0f} MB, {n_tok:,} tokens" if n_tok >= 0 else f"  {p}: {sz:.0f} MB")


if __name__ == "__main__":
    main()
