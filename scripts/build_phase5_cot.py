#!/usr/bin/env python3
"""Build the CoT-heavy mathematics (+code) dataset for Phase 5.

Slots between Stage C and Phase 6 Refresh. Teaches the 190M model to
COMPUTE step-by-step (scratchpad / chain-of-thought) rather than jump to
a memorized answer. Sources are the reasoning-rich generator bins from
gen_math_data.py plus GSM8K and a small code slice for reinforcement.

  1. cot_scratchpad    long multiplication/division, column add, powers
  2. arithmetic_worked worked multi-step arithmetic (solve, fractions, %, decimals)
  3. algebra           equations/simplify (multi-step kinds emit Step N:)
  4. problem           word problems with worked solutions
  5. verification      classify candidate solutions (valid/invalid)
  6. error             find the mistake in a faulty solution
  7. code              Python code solving math (stepwise-logic reinforcement)
  8. gsm8k             grade-school math word problems

Sources are sliced at document boundaries and weighted-interleaved into
one train.bin. Eval = head slice.

Usage:
  python3 scripts/build_phase5_cot.py
  python3 scripts/build_phase5_cot.py --smoke 200000
"""

import os, sys, time, random, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.data.tokenizer import Tokenizer

DATA_DIR = "data/phase5"
TOKENIZER_PATH = "data/tokenizer/phase1.model"
OUT_TRAIN = f"{DATA_DIR}/cot_train.bin"
OUT_EVAL = f"{DATA_DIR}/cot_eval.bin"
EOS = None

# (source_bin_relative_to_DATA_DIR, target_tokens)
SOURCES = [
    ("cot_scratchpad_syn.bin",    10_000_000),
    ("arithmetic_worked_syn.bin", 8_000_000),
    ("algebra_syn.bin",           6_000_000),
    ("problem_syn.bin",           6_000_000),
    ("verification_syn.bin",      6_000_000),
    ("error_syn.bin",             4_000_000),
    ("code_syn.bin",              4_000_000),
    ("gsm8k.bin",                 8_000_000),
]

EVAL_TOKENS = 5_000_000


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


def slice_bin_docs(path, target, seed, out_path, scale):
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
    target = max(1, int(target * scale))
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
    print(f"  {os.path.basename(path):28s} {n:>12,}/{target:,} tok  {time.time()-t0:.0f}s")
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
    print(f"  merged {total_used:,} tok -> {os.path.basename(out_path)}  {time.time()-t0:.0f}s")


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
    print(f"  eval -> {n:,} tok -> {os.path.basename(out_path)}")


def main():
    ap = argparse.ArgumentParser(description="Build CoT-heavy math+code dataset (Phase 5)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", type=int, default=0,
                    help="Scale all budgets to N total tokens")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    tok = Tokenizer(TOKENIZER_PATH)
    global EOS
    EOS = tok.eos_token

    total = sum(t for _, t in SOURCES)
    scale = (args.smoke / total) if args.smoke > 0 else 1.0
    tag = "SMOKE" if scale < 1.0 else "FULL"
    print(f"[{tag}] CoT-heavy build  scale={scale:.4f}")

    raw_parts = []
    for src, tgt in SOURCES:
        path = f"{DATA_DIR}/{src}"
        raw = f"{DATA_DIR}/cot_src_{src.replace('/','_').replace('.bin','')}.bin"
        try:
            slice_bin_docs(path, tgt, args.seed, raw, scale)
        except (FileNotFoundError, ValueError) as e:
            print(f"    WARNING: {e} -- skipping")
            continue
        raw_parts.append((raw, max(1, int(tgt * scale))))

    print("  merging CoT-heavy train...")
    merge_sources(raw_parts, OUT_TRAIN, seed=args.seed)
    for raw, _ in raw_parts:
        if os.path.exists(raw):
            os.remove(raw)

    print("  slicing eval head...")
    slice_head(OUT_TRAIN, max(1, int(EVAL_TOKENS * scale)), OUT_EVAL)

    print(f"\nCoT-heavy total: {sum(t for _, t in SOURCES):,} tokens budget")

    if os.path.exists(OUT_TRAIN):
        print(f"Train: {OUT_TRAIN} ({os.path.getsize(OUT_TRAIN)/1e6:.1f} MB)")
    print(f"Done. -> {OUT_TRAIN}")


if __name__ == "__main__":
    main()
