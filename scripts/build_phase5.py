#!/usr/bin/env python3
"""Build Phase 5 dataset: Math Specialization (100M tokens).

Sources:
  Synthetic arithmetic    25M  (gen_math_data.py -> arith_syn.bin)
  Synthetic algebra       20M  (gen_math_data.py -> algebra_syn.bin)
  Synthetic problem-solve 15M  (gen_math_data.py -> problem_syn.bin)
  Error identification    10M  (gen_math_data.py -> error_syn.bin)
  Code-as-solver          10M  (gen_math_data.py -> code_syn.bin)
  Clean-completion Q&A    10M  (gen_math_data.py -> clean_syn.bin)
  GSM8K (fetched)         10M  (fetch_math_datasets.py -> gsm8k.bin)
  FWE + Wiki + Code        5M  (existing bins, prevent forgetting)
  Simplicity + other       5M  (gen_math_data.py -> simplify_syn.bin)

Run gen_math_data.py and fetch_math_datasets.py first.
"""

import os, sys, time, random, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.data.tokenizer import Tokenizer

DATA_DIR = "data/phase5"
TOKENIZER_PATH = "data/tokenizer/phase1.model"
OUT_TRAIN = f"{DATA_DIR}/train.bin"
OUT_EVAL = f"{DATA_DIR}/eval.bin"
EOS = None

TOTAL_TARGET = 100_000_000

# (label, local bin, target tokens)
SYNTH_SOURCES = [
    ("arith",     f"{DATA_DIR}/arith_syn.bin",    25_000_000),
    ("algebra",   f"{DATA_DIR}/algebra_syn.bin",   20_000_000),
    ("problem",   f"{DATA_DIR}/problem_syn.bin",   15_000_000),
    ("error",     f"{DATA_DIR}/error_syn.bin",     10_000_000),
    ("code",      f"{DATA_DIR}/code_syn.bin",      10_000_000),
    ("gsm8k",     f"{DATA_DIR}/gsm8k.bin",         10_000_000),
    ("simplify",  f"{DATA_DIR}/simplify_syn.bin",   5_000_000),
    ("clean",     f"{DATA_DIR}/clean_syn.bin",     10_000_000),
]

# Small anti-forgetting mix from existing data
FORGETTING_SOURCES = [
    ("fwe",  "data/phase2/fwe_train.bin",  2_000_000),
    ("wiki", "data/phase3/train.bin",      2_000_000),
    ("code", "data/phase4/train.bin",      1_000_000),
]

EVAL_TOKENS = 10_000_000


def eos_indices(path):
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
    rng = random.Random(seed)
    active = list(sources)
    iters = {p: iter(iter_docs(p)) for p, _ in sources}
    remaining = {p: t for p, t in sources}
    total_used = 0
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


def main():
    parser = argparse.ArgumentParser(description="Build Phase 5 math dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-synth", action="store_true",
                        help="Skip synth generation (use cached bins)")
    parser.add_argument("--skip-gsm8k", action="store_true",
                        help="Skip GSM8K fetch (use cached bin)")
    parser.add_argument("--smoke", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    tok = Tokenizer(TOKENIZER_PATH)
    global EOS
    EOS = tok.eos_token

    scale = (args.smoke / TOTAL_TARGET) if args.smoke > 0 else 1.0
    tag = "SMOKE" if scale < 1.0 else "FULL"
    print(f"[{tag}] Phase 5 math build: scale={scale:.4f}")

    # Step 1: Generate synthetic math data
    if not args.skip_synth:
        print("\nGenerating synthetic math data...")
        os.system(f"python3 scripts/gen_math_data.py --tokens {int(70_000_000 * scale)}")

    # Step 2: Fetch GSM8K
    if not args.skip_gsm8k:
        print("\nFetching GSM8K...")
        os.system(f"python3 scripts/fetch_math_datasets.py --max-tokens {int(10_000_000 * scale)}")

    # Step 3: Slice synthetic sources at doc boundaries
    print("\nSlicing synthetic sources...")
    raw_parts = []
    for label, path, target in SYNTH_SOURCES:
        raw = f"{DATA_DIR}/{label}_raw.bin"
        try:
            slice_bin_docs(path, max(1, int(target * scale)), args.seed, raw)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} -- skipping")
            continue
        raw_parts.append((raw, max(1, int(target * scale))))

    # Step 4: Slice anti-forgetting sources
    print("\nSlicing anti-forgetting sources...")
    for label, path, target in FORGETTING_SOURCES:
        raw = f"{DATA_DIR}/{label}_raw.bin"
        try:
            slice_bin_docs(path, max(1, int(target * scale)), args.seed, raw)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} -- skipping")
            continue
        raw_parts.append((raw, max(1, int(target * scale))))

    # Step 5: Weighted merge
    print("\nMerging (weighted doc interleave)...")
    merge_sources(raw_parts, OUT_TRAIN, seed=args.seed)

    # Step 6: Eval from head
    print("\nSlicing eval head...")
    slice_head(OUT_TRAIN, max(1, int(EVAL_TOKENS * scale)), OUT_EVAL)

    print("\nDone:")
    for p in [OUT_TRAIN, OUT_EVAL]:
        sz = os.path.getsize(p) / 1e6
        n_tok = len(np.fromfile(p, dtype=np.uint16, count=-1)) if os.path.getsize(p) < 2e9 else -1
        print(f"  {p}: {sz:.0f} MB, {n_tok:,} tokens" if n_tok >= 0 else f"  {p}: {sz:.0f} MB")


if __name__ == "__main__":
    main()
