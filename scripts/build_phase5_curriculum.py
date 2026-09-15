#!/usr/bin/env python3
"""Build Prabhakar (Phase 5) 3-stage curriculum datasets.

Prabhakar is a curriculum of three sequential stages, each trained as its
own `train.py` run from the previous stage's final checkpoint:

  Stage A — Foundations (~30M) : arithmetic, algebra, simplify, clean
  Stage B — Advanced   (~30M) : advanced algebra, number theory, geometry,
                                trig, combinatorics
  Stage C — Reasoning  (~40M) : GSM8K, verification, error, code,
                                arithmetic reinforcement + anti-forgetting

Stage data is produced by slicing the standalone generator bins from
gen_math_data.py (data/phase5/<gen>_syn.bin) at document boundaries and
weighted-interleaving them into one train.bin per stage. Anti-forgetting
uses existing Phase 2/3/4 bins.

Run gen_math_data.py (--all) and fetch_math_datasets.py (GSM8K) first, or
use the integrated flags below.

Usage:
  python3 scripts/build_phase5_curriculum.py               # full build
  python3 scripts/build_phase5_curriculum.py --skip-synth --skip-gsm8k
  python3 scripts/build_phase5_curriculum.py --smoke 200000

Output (data/phase5/):
  stage_a_train.bin, stage_a_eval.bin
  stage_b_train.bin, stage_b_eval.bin
  stage_c_train.bin, stage_c_eval.bin
"""

import os, sys, time, random, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.data.tokenizer import Tokenizer

DATA_DIR = "data/phase5"
TOKENIZER_PATH = "data/tokenizer/phase1.model"
EOS = None

# (stage_label, [(source_bin, target_tokens), ...])
# source_bin is relative to DATA_DIR unless it starts with "data/".
STAGES = {
    "a": [
        ("arithmetic",       16_000_000),
        ("algebra",           9_000_000),
        ("simplify",          2_000_000),
        ("clean",             3_000_000),
    ],
    "b": [
        ("advanced_algebra",  7_000_000),
        ("number_theory",     6_000_000),
        ("geometry",          6_000_000),
        ("trig",              5_000_000),
        ("combinatorics",     6_000_000),
    ],
    "c": [
        ("data/phase5/gsm8k.bin", 12_000_000),
        ("verification",      7_000_000),
        ("error",             5_000_000),
        ("code",              5_000_000),
        ("arithmetic",        4_000_000),
        ("advanced_algebra",  3_000_000),
        # anti-forgetting (absolute paths)
        ("data/phase2/fwe_train.bin", 2_000_000),
        ("data/phase3/train.bin",     1_000_000),
        ("data/phase4/train.bin",     1_000_000),
    ],
}

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
    print(f"  {os.path.basename(path):24s} {n:>12,}/{target:,} tok  {time.time()-t0:.0f}s")
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
    ap = argparse.ArgumentParser(description="Build Prabhakar 3-stage curriculum (Phase 5)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-synth", action="store_true",
                    help="Skip synthetic generator run (use cached bins)")
    ap.add_argument("--skip-gsm8k", action="store_true",
                    help="Skip GSM8K fetch (use cached bin)")
    ap.add_argument("--smoke", type=int, default=0,
                    help="Scale all budgets to N total tokens")
    ap.add_argument("--synth-tokens", type=int, default=100_000_000,
                    help="Token budget for gen_math_data.py --all")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    tok = Tokenizer(TOKENIZER_PATH)
    global EOS
    EOS = tok.eos_token

    # Scale factor for smoke tests
    stage_total = sum(t for _, lst in STAGES.items() for _, t in lst)
    scale = (args.smoke / stage_total) if args.smoke > 0 else 1.0
    tag = "SMOKE" if scale < 1.0 else "FULL"
    print(f"[{tag}] Prabhakar curriculum build  scale={scale:.4f}")

    # Step 0: generate synthetic data
    if not args.skip_synth:
        print("\nGenerating synthetic math data (--all)...")
        os.system(f"python3 scripts/gen_math_data.py --all --tokens {args.synth_tokens}")

    # Step 1: fetch GSM8K (for Stage C)
    if not args.skip_gsm8k:
        print("\nFetching GSM8K...")
        os.system(f"python3 scripts/fetch_math_datasets.py --max-tokens {int(13_000_000 * scale)}")

    # Step 2: build each stage
    for label, specs in STAGES.items():
        print(f"\n=== Building Stage {label.upper()} ===")
        raw_parts = []
        for src, tgt in specs:
            if src.startswith("data/"):
                path = src
            else:
                path = f"{DATA_DIR}/{src}_syn.bin"
            raw = f"{DATA_DIR}/stage_{label}_src_{os.path.basename(src)}.bin"
            try:
                slice_bin_docs(path, tgt, args.seed, raw, scale)
            except (FileNotFoundError, ValueError) as e:
                print(f"    WARNING: {e} -- skipping")
                continue
            raw_parts.append((raw, max(1, int(tgt * scale))))

        train_path = f"{DATA_DIR}/stage_{label}_train.bin"
        print(f"  merging Stage {label.upper()}...")
        merge_sources(raw_parts, train_path, seed=args.seed)

        # clean up intermediate raw parts
        for raw, _ in raw_parts:
            if os.path.exists(raw):
                os.remove(raw)

        eval_path = f"{DATA_DIR}/stage_{label}_eval.bin"
        print(f"  slicing eval head for Stage {label.upper()}...")
        slice_head(train_path, max(1, int(EVAL_TOKENS * scale)), eval_path)

    print("\nStage budget summary (tokens per stage):")
    for label, specs in STAGES.items():
        print(f"  Stage {label.upper()}: {sum(t for _, t in specs):,} tokens")
    print("\nDone. Stage data written to data/phase5/stage_{a,b,c}_{train,eval}.bin")


if __name__ == "__main__":
    main()
