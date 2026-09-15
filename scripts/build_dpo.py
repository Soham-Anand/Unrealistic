#!/usr/bin/env python3
"""Phase 7d DPO data: UltraFeedback binarized (MIT) preference pairs.

Fetches HuggingFaceH4/ultrafeedback_binarized, formats chat-style, filters to
fit seq 512 (prompt<=256, response<=256), samples 4K train + 200 eval, writes
padded uint16 npz pairs + prompt lens.

Usage:
  PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/build_dpo.py [--n-train 4000]
  -> data/phase7d/dpo_train.npz + dpo_eval.npz
"""

import os, sys, argparse, shutil, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.data.tokenizer import Tokenizer

OUT_DIR = "data/phase7d"
SEQ = 512
PROMPT_MAX = 256
RESP_MAX = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    from datasets import load_dataset
    cache = "/tmp/dpo_cache"
    print("fetch ultrafeedback_binarized...", flush=True)
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs",
                      cache_dir=cache)
    print(f"rows: {len(ds)}", flush=True)
    idx = rng.sample(range(len(ds)), len(ds))

    tok = Tokenizer("data/tokenizer/phase1.model")
    EOS = tok.eos_token

    pairs = []
    for i in idx:
        r = ds[int(i)]
        def assistant_text(msgs):
            if isinstance(msgs, list):
                turns = [m.get("content", "") for m in msgs
                         if isinstance(m, dict) and m.get("role") == "assistant"]
                return turns[-1].strip() if turns else ""
            return str(msgs or "").strip()

        prompt = str(r.get("prompt", "")).strip()
        ch = assistant_text(r.get("chosen"))
        rj = assistant_text(r.get("rejected"))
        if not prompt or not ch or not rj or ch == rj:
            continue
        pre = f"User: {prompt}\nAssistant: "
        pre_ids = tok.encode(pre)
        if len(pre_ids) > PROMPT_MAX or len(pre_ids) < 4:
            continue
        c_ids = (pre_ids + tok.encode(ch)[:RESP_MAX])[:SEQ]
        r_ids = (pre_ids + tok.encode(rj)[:RESP_MAX])[:SEQ]
        if len(c_ids) < len(pre_ids) + 4 or len(r_ids) < len(pre_ids) + 4:
            continue
        pairs.append((pre_ids, c_ids, r_ids))
        if len(pairs) >= args.n_train + args.n_eval:
            break
    print(f"usable pairs: {len(pairs)}", flush=True)
    shutil.rmtree(cache, ignore_errors=True)

    rng.shuffle(pairs)
    eval_pairs, train_pairs = pairs[:args.n_eval], pairs[args.n_eval:args.n_eval + args.n_train]
    os.makedirs(OUT_DIR, exist_ok=True)

    def write(ps, path):
        n = len(ps)
        C = np.full((n, SEQ), EOS, dtype=np.uint16)
        R = np.full((n, SEQ), EOS, dtype=np.uint16)
        P = np.zeros((n,), dtype=np.uint16)
        for i, (pre, c, r) in enumerate(ps):
            C[i, :len(c)] = np.array(c, dtype=np.uint16)
            R[i, :len(r)] = np.array(r, dtype=np.uint16)
            P[i] = len(pre)
        np.savez_compressed(path, chosen=C, rejected=R, prompt_len=P,
                            n=np.array([n]))
        return n

    nt = write(train_pairs, f"{OUT_DIR}/dpo_train.npz")
    ne = write(eval_pairs, f"{OUT_DIR}/dpo_eval.npz")
    Cl = float(np.mean([len(c) for _, c, _ in train_pairs])) if train_pairs else 0
    print(f"WROTE train {nt} pairs (avg len {Cl:.0f}) + eval {ne} pairs -> {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
