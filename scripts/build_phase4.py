#!/usr/bin/env python3
"""Build Phase 4 dataset: code + StackExchange + prose + math, merged weighted. 500M train + 10M eval."""

import multiprocessing
multiprocessing.set_start_method("fork", force=True)

import os, sys, time, argparse, json, hashlib
import warnings
warnings.filterwarnings("ignore", message="resource_tracker")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from datasets import load_dataset
from src.data.tokenizer import Tokenizer

DATA_DIR = "data/phase4"
TOKENIZER_PATH = "data/tokenizer/phase1.model"
OUT_TRAIN = f"{DATA_DIR}/train.bin"
OUT_EVAL = f"{DATA_DIR}/eval.bin"
MANIFEST_PATH = f"{DATA_DIR}/manifest.json"
EOS = None

# (label, target_tokens)
SOURCES = [
    ("python",         90_000_000),
    ("c",              40_000_000),
    ("cpp",            50_000_000),
    ("javascript",     30_000_000),
    ("typescript",     20_000_000),
    ("html",           10_000_000),
    ("css",            10_000_000),
    ("stackexchange", 120_000_000),
    ("prose",          70_000_000),
    ("math",           60_000_000),
]

STARCODER_LANGS = ["python", "c", "cpp", "javascript", "typescript", "html", "css"]
STARCODER_DATA_FILES = {lang: f"{lang}/*.parquet" for lang in STARCODER_LANGS}

EVAL_TOKENS = 10_000_000
PROSE_TRAIN = 90_000_000  # Python train portion (rest goes to eval)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 24)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def tokenize_stream(tok, ds, text_key, raw_path, total_target, fmt=None):
    n = 0
    start = time.time()
    last_report = 0
    f = open(raw_path, "wb")
    for row in ds:
        text = row.get(text_key, "")
        if not text or not text.strip():
            continue
        if fmt is not None:
            text = fmt(text, row)
        ids = tok.encode(text.strip())
        if not ids:
            continue
        if n + len(ids) > total_target:
            ids = ids[: total_target - n]
        if not ids:
            break
        f.write(np.array(ids, dtype=np.uint16).tobytes())
        f.write(np.uint16(EOS).tobytes())
        n += len(ids)
        if n - last_report >= 20_000_000:
            last_report = n
            el = time.time() - start
            rate = n / el if el else 0
            eta = (total_target - n) / rate if rate else 0
            print(f"      {n:>12,}/{total_target:,}  {100*n/total_target:5.1f}%  {rate:,.0f} tok/s  ETA {eta:.0f}s")
        if n >= total_target:
            break
    f.close()
    print(f"      done {n:,} tok in {time.time()-start:.0f}s")
    return n


def slice_prose(raw_in, raw_out, target):
    """Reuse FineWeb-Edu docs from an existing tokenized bin (no re-download)."""
    n = 0
    start = time.time()
    with open(raw_out, "wb") as f:
        for doc in iter_docs(raw_in):
            f.write(np.array(doc, dtype=np.uint16).tobytes())
            f.write(np.uint16(EOS).tobytes())
            n += len(doc)
            if n >= target:
                break
    print(f"      done {n:,} tok in {time.time()-start:.0f}s")


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


def count_tokens(path):
    """Count actual token count in an EOS-delimited uint16 bin."""
    n = 0
    for doc in iter_docs(path):
        n += len(doc)
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-tokens", type=int, default=EVAL_TOKENS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    tok = Tokenizer(TOKENIZER_PATH)
    global EOS
    EOS = tok.eos_token
    total = sum(t for _, t in SOURCES)
    print(f"Phase 4 build: {total/1e6:.0f}M train + {args.eval_tokens/1e6:.0f}M eval")
    print(f"Sources: {len(SOURCES)}")

    manifest = {
        "version": "1.0",
        "tokenizer": TOKENIZER_PATH,
        "total_tokens": total,
        "eval_tokens": args.eval_tokens,
        "seed": args.seed,
        "sources": {},
    }

    # Python: reserve eval tail (90M train + eval tokens, split at 90M boundary)
    print("Streaming python (train + eval)...")
    py_ds = load_dataset("bigcode/starcoderdata", data_files="python/*.parquet", split="train", streaming=True)
    py_raw = f"{DATA_DIR}/python_raw.bin"
    py_actual = tokenize_stream(tok, py_ds, "content", py_raw, PROSE_TRAIN + args.eval_tokens)
    with open(py_raw, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint16)
    with open(py_raw, "wb") as f:
        f.write(data[:PROSE_TRAIN].tobytes())
    with open(OUT_EVAL, "wb") as f:
        f.write(data[PROSE_TRAIN:].tobytes())
    del data
    manifest["sources"]["python"] = {
        "requested": PROSE_TRAIN,
        "actual": min(PROSE_TRAIN, py_actual),
        "eval_tokens": py_actual - PROSE_TRAIN if py_actual > PROSE_TRAIN else 0,
    }

    # StarCoder languages (c, cpp, javascript, typescript, html, css)
    for label, target in SOURCES[1:7]:
        print(f"Streaming {label}...")
        ds = load_dataset("bigcode/starcoderdata", data_files=STARCODER_DATA_FILES[label], split="train", streaming=True)
        actual = tokenize_stream(tok, ds, "content", f"{DATA_DIR}/{label}_raw.bin", target)
        manifest["sources"][label] = {"requested": target, "actual": actual}

    # StackExchange
    print("Streaming stackexchange...")
    so_ds = load_dataset("bigcode/stackoverflow-clean", split="train", streaming=True)
    so_actual = tokenize_stream(tok, so_ds, "content", f"{DATA_DIR}/stackexchange_raw.bin", 120_000_000)
    manifest["sources"]["stackexchange"] = {"requested": 120_000_000, "actual": so_actual}

    # Prose from FineWeb-Edu (existing bin, no re-download)
    print("Slicing prose from FineWeb-Edu...")
    slice_prose("data/phase2/fwe_train.bin", f"{DATA_DIR}/prose_raw.bin", 70_000_000)
    prose_actual = count_tokens(f"{DATA_DIR}/prose_raw.bin")
    manifest["sources"]["prose"] = {
        "requested": 70_000_000,
        "actual": prose_actual,
        "source_bin": "data/phase2/fwe_train.bin",
    }

    # OpenWebMath (math)
    print("Streaming OpenWebMath...")
    owm_ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    math_actual = tokenize_stream(tok, owm_ds, "text", f"{DATA_DIR}/math_raw.bin", 60_000_000)
    manifest["sources"]["math"] = {"requested": 60_000_000, "actual": math_actual}

    # Merge all sources
    print("Merging (weighted doc interleave)...")
    raw_bins = [(f"{DATA_DIR}/{l}_raw.bin", t) for l, t in SOURCES]
    merge_sources(raw_bins, OUT_TRAIN, seed=args.seed)

    # Final stats
    train_tokens = count_tokens(OUT_TRAIN)
    eval_tokens = count_tokens(OUT_EVAL)
    print(f"\nDone:")
    print(f"  {OUT_TRAIN}: {os.path.getsize(OUT_TRAIN)/1e6:.0f} MB ({train_tokens:,} tokens)")
    print(f"  {OUT_EVAL}: {os.path.getsize(OUT_EVAL)/1e6:.0f} MB ({eval_tokens:,} tokens)")

    # Build manifest
    manifest["train_tokens_actual"] = train_tokens
    manifest["eval_tokens_actual"] = eval_tokens
    manifest["checksums"] = {
        "train.bin": sha256_file(OUT_TRAIN),
        "eval.bin": sha256_file(OUT_EVAL),
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
