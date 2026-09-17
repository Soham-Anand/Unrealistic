#!/usr/bin/env python3
"""Phase 7g combined recovery: custom Openers + HEAVY Anchors + Phase 7 slice.

All three bins share the tokenizer (uint16 + EOS) -> byte-merge is exact.
Anchors repeated heavy (bhayankar): knowledge pins must dominate.
Phase7 slice: proven SFT grounding (anti-Wiki-damage insurance).

Usage:
  PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/build_7g.py
  -> data/phase7g/combined_train.bin + combined_eval.bin
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

OUT_DIR = "data/phase7g"
TRAIN_BIN = f"{OUT_DIR}/combined_train.bin"
EVAL_BIN = f"{OUT_DIR}/combined_eval.bin"
EOS = 2
ANCHOR_REPEAT = 40
PHASE7_TOKENS = 5_000_000


def read(p):
    return np.fromfile(p, dtype=np.uint16)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    openers = read(f"{OUT_DIR}/greet_train.bin")
    anchors = read("data/phase8/anchor_train.bin")
    print(f"openers: {len(openers):,} tok | anchors: {len(anchors):,} tok", flush=True)

    # heavy anchors: repeat the verified pins
    heavy = np.tile(anchors, ANCHOR_REPEAT)
    print(f"anchors x{ANCHOR_REPEAT}: {len(heavy):,} tok", flush=True)

    # phase7 slice at EOS boundary
    s7 = read("data/phase7/train.bin")
    cut = PHASE7_TOKENS
    while cut < len(s7) and int(s7[cut]) != EOS:
        cut += 1
    s7 = s7[:cut + 1]
    print(f"phase7 slice: {len(s7):,} tok", flush=True)

    train = np.concatenate([openers, heavy, s7])
    # eval: small held-out sliver of openers + anchors (NOT phase7)
    ev_o = openers[:len(openers) // 10]
    ev_a = anchors[:len(anchors) // 10]
    ev = np.concatenate([ev_o, ev_a])
    train.tofile(TRAIN_BIN)
    ev.tofile(EVAL_BIN)
    print(f"WROTE train {len(train):,} tokens -> {TRAIN_BIN}", flush=True)
    print(f"WROTE eval {len(ev):,} tokens -> {EVAL_BIN}", flush=True)
    print(f"epochs over 1500 steps: {1500 * 8192 / len(train):.2f}", flush=True)


if __name__ == "__main__":
    main()
