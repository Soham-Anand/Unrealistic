#!/usr/bin/env python3
"""Phase 7f conversational SFT + knowledge recovery (~20M tokens, 5000 steps).

Chat volume (UltraChat + smol-smoltalk + greeting synthetics) installs
conversation; knowledge grounding (anchors x40 + SFT-mix slice) prevents a
repeat of the Wiki damage. All Tier-B.

Usage:
  PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/build_sft3.py
  -> data/phase7f/sft3_train.bin + sft3_eval.bin
"""

import os, sys, random, argparse, shutil, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.data.tokenizer import Tokenizer

OUT_DIR = "data/phase7f"
TRAIN_BIN = f"{OUT_DIR}/sft3_train.bin"
EVAL_BIN = f"{OUT_DIR}/sft3_eval.bin"
EOS = None
rng = random.Random(31337)


def chat(q, a):
    return f"User: {q}\nAssistant: {a}"


def msgs_to_doc(msgs):
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None
    users = [m.get("content", "").strip() for m in msgs
             if isinstance(m, dict) and m.get("role") == "user"]
    assts = [m.get("content", "").strip() for m in msgs
             if isinstance(m, dict) and m.get("role") == "assistant"]
    if not users or not assts:
        return None
    q, a = users[0], assts[-1]
    if not q or not a or len(q) > 1500 or len(a) > 1500:
        return None
    return chat(q, a)


def fetch_hf(repo, split, pred, cap_total, cap_src):
    from datasets import load_dataset
    cache = "/tmp/sft3_cache"
    docs, seen = [], {}
    try:
        ds = load_dataset(repo, split=split, streaming=True, cache_dir=cache)
        for r in ds:
            d, src = pred(r)
            if d is None:
                continue
            seen[src] = seen.get(src, 0) + 1
            if seen[src] > cap_src:
                continue
            docs.append(d)
            if len(docs) >= cap_total:
                break
    except Exception as e:
        print(f"  {repo} FAILED: {str(e)[:120]}", flush=True)
    shutil.rmtree(cache, ignore_errors=True)
    print(f"  {repo}: +{len(docs)} docs from {len(seen)} sources", flush=True)
    return docs


def smol_pred(r):
    src = str(r.get("source", ""))
    if not src.startswith("smol-"):
        return None, src
    return msgs_to_doc(r.get("messages")), src


def ultra_pred(r):
    return msgs_to_doc(r.get("messages")), "ultrachat"


GREETINGS = [
    ("Hey!", "Hello! How can I help?"),
    ("Hey Dude!", "Hey! How can I help you?"),
    ("Hello!", "Hi there! What can I do for you?"),
    ("Hi!", "Hello! How can I help you today?"),
    ("Hi there!", "Hello! What can I do for you?"),
    ("How are you?", "I am good. How can I help you?"),
    ("How is it going?", "All good here. What can I do for you?"),
    ("Good morning!", "Good morning! How can I help you?"),
    ("Good afternoon!", "Good afternoon! How can I help?"),
    ("Good evening!", "Good evening! How can I help you?"),
    ("Good night!", "Good night! Sleep well!"),
    ("Bye!", "Goodbye! Have a great day!"),
    ("See you later!", "See you later! Take care!"),
    ("Thank you!", "You are welcome!"),
    ("Thanks a lot!", "You are most welcome!"),
    ("What is your name?", "I am Unrealistic, an AI assistant."),
    ("Who are you?", "I am Unrealistic, a small AI model."),
    ("Tell me about yourself.", "I am Unrealistic, a 190M AI model trained from scratch. I can answer questions and help with math, science, and general knowledge."),
    ("What can you do?", "I can answer questions and help with math, science, and general knowledge."),
    ("Are you human?", "No, I am an AI assistant."),
    ("Are you a robot?", "I am an AI program, not a robot."),
    ("How old are you?", "I was trained in 2026."),
    ("Where are you from?", "I was built on a MacBook Air."),
    ("Nice to meet you!", "Nice to meet you too! How can I help?"),
    ("How was your day?", "Every day is a good day for answering questions. What is yours?"),
]


def greeting_synthetics():
    out = []
    variants = ["", "!", "!!"]
    for q, a in GREETINGS:
        out.append(chat(q, a))
        base = q.rstrip("!.")
        for v in variants[1:]:
            out.append(chat(base + v, a))
    # small-talk follow-ups (two-turn shape)
    out.append("User: Hey!\nAssistant: Hello! How can I help?\nUser: What is 2 + 2?\nAssistant: 4.")
    out.append("User: Hi!\nAssistant: Hello! How can I help you today?\nUser: Who built the Taj Mahal?\nAssistant: Shah Jahan built the Taj Mahal.")
    return out


def anchor_docs():
    """Fresh dump of ALL build_anchor.py lists (re-dumped every build)."""
    spec = importlib.util.spec_from_file_location(
        "ba", os.path.join(os.path.dirname(__file__), "build_anchor.py"))
    ba = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ba)
    docs = []
    for q, expr, forms in ba.MATH:
        for f in forms:
            docs.append(f"User: {q}\nAssistant: {f}")
    for lst in ("GK", "LP", "MUGHAL", "CODE"):
        for q, a in getattr(ba, lst, []):
            docs.append(f"User: {q}\nAssistant: {a}")
    # also pick up any extra lists added later
    for name in dir(ba):
        if name.isupper() and name not in ("MATH", "GK", "LP", "MUGHAL", "CODE"):
            v = getattr(ba, name)
            if isinstance(v, list) and v and isinstance(v[0], tuple):
                for q, a in v:
                    docs.append(f"User: {q}\nAssistant: {a}")
    print(f"  anchors dumped: {len(docs)} unique docs", flush=True)
    return docs


def phase7_slice(max_tokens=3_000_000):
    p = "data/phase7/train.bin"
    try:
        a = np.fromfile(p, dtype=np.uint16)
    except Exception as e:
        print(f"  phase7 slice FAILED: {e}", flush=True)
        return b""
    if len(a) <= max_tokens:
        return a.tobytes()
    cut = max_tokens
    while cut < len(a) and int(a[cut]) != 2:
        cut += 1
    print(f"  phase7 slice: {cut:,} tokens", flush=True)
    return a[:cut + 1].tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-smol", type=int, default=20000)
    ap.add_argument("--n-ultra", type=int, default=30000)
    ap.add_argument("--anchor-repeat", type=int, default=40)
    args = ap.parse_args()

    docs = []
    docs.extend(fetch_hf("HuggingFaceTB/smol-smoltalk", "train", smol_pred,
                         args.n_smol, 6000))
    docs.extend(fetch_hf("HuggingFaceH4/ultrachat_200k", "train_sft", ultra_pred,
                         args.n_ultra, args.n_ultra))
    n0 = len(docs)
    docs.extend(greeting_synthetics())
    print(f"greetings: +{len(docs) - n0} docs", flush=True)
    anchors = anchor_docs()
    for _ in range(args.anchor_repeat):
        docs.extend(anchors)
    print(f"anchors x{args.anchor_repeat} sprinkled", flush=True)

    seen, uniq = set(), []
    for d in docs:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    # NOTE: anchor repeats intentionally collapse in dedup; re-expand after
    uniq_anchors = []
    for _ in range(args.anchor_repeat):
        uniq_anchors.extend(anchors)
    full = uniq + uniq_anchors
    rng.shuffle(full)
    n_eval = max(1000, len(full) // 10)
    eval_docs, train_docs = full[:n_eval], full[n_eval:]
    print(f"unique: {len(uniq)} + anchor reps => {len(full)} docs", flush=True)

    global EOS
    tok = Tokenizer("data/tokenizer/phase1.model")
    EOS = tok.eos_token
    os.makedirs(OUT_DIR, exist_ok=True)

    def write(ds, path):
        n = 0
        with open(path, "wb") as f:
            for d in ds:
                ids = tok.encode(d)
                if not ids:
                    continue
                f.write(np.array(ids, dtype=np.uint16).tobytes())
                f.write(np.uint16(EOS).tobytes())
                n += len(ids)
        return n

    s7_raw = phase7_slice()
    nt = write(train_docs, TRAIN_BIN)
    if s7_raw:
        with open(TRAIN_BIN, "ab") as f:
            f.write(s7_raw)
        nt += len(s7_raw) // 2
    ne = write(eval_docs, EVAL_BIN)
    print(f"WROTE train {nt:,} tokens -> {TRAIN_BIN}", flush=True)
    print(f"WROTE eval {ne:,} tokens -> {EVAL_BIN}", flush=True)


if __name__ == "__main__":
    main()
