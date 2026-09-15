#!/usr/bin/env python3
"""Phase 7e SUPER-DIVERSE SFT: maximum task-type variety for instruction-following.

Sources (all Tier-B):
  smol-smoltalk (Apache-2.0 native subsets only: source startswith 'smol-')
  UltraChat-200K (MIT)                      \
  OA chat slice (Apache-2.0, local merge)    }  chat diversity
  Constraint synthetics (CC0, deterministic) }  task-type diversity
  Anchors x3 (grounding sprinkle; real pinning is 8b)

Synthetic task types: say/repeat/yes-no/keyword/starts-with/letter-count/
upper/lower/list/opposite/compare/sort/arithmetic/true-false/translate/
summarize/explain/capitalize/rhyme/days/months/alphabet (~20 types x templates).

Usage:
  PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/build_sft2.py
  -> data/phase7e/sft2_train.bin + sft2_eval.bin
"""

import os, sys, random, argparse, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.data.tokenizer import Tokenizer

OUT_DIR = "data/phase7e"
TRAIN_BIN = f"{OUT_DIR}/sft2_train.bin"
EVAL_BIN = f"{OUT_DIR}/sft2_eval.bin"
EOS = None
rng = random.Random(99)


def chat(q, a):
    return f"User: {q}\nAssistant: {a}"


def msgs_to_doc(msgs):
    """First user turn + last assistant turn -> chat doc. None if unusable."""
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None
    users = [m.get("content", "").strip() for m in msgs
             if isinstance(m, dict) and m.get("role") == "user"]
    assts = [m.get("content", "").strip() for m in msgs
             if isinstance(m, dict) and m.get("role") == "assistant"]
    if not users or not assts:
        return None
    q, a = users[0], assts[-1]
    if not q or not a or len(q) > 1200 or len(a) > 1200:
        return None
    return chat(q, a)


def fetch_hf(repo, split, pred, cap_total, cap_note):
    from datasets import load_dataset
    cache = "/tmp/sft2_cache"
    docs, seen_src = [], {}
    try:
        ds = load_dataset(repo, split=split, streaming=True, cache_dir=cache)
        for r in ds:
            d, src = pred(r)
            if d is None:
                continue
            seen_src[src] = seen_src.get(src, 0) + 1
            if seen_src[src] > cap_note:
                continue
            docs.append(d)
            if len(docs) >= cap_total:
                break
    except Exception as e:
        print(f"  {repo} FAILED: {str(e)[:120]}", flush=True)
    shutil.rmtree(cache, ignore_errors=True)
    print(f"  {repo}: +{len(docs)} docs, sources={dict(sorted(seen_src.items()))}", flush=True)
    return docs


def smol_pred(r):
    src = str(r.get("source", ""))
    if not src.startswith("smol-"):
        return None, src
    return msgs_to_doc(r.get("messages")), src


def ultra_pred(r):
    return msgs_to_doc(r.get("messages")), "ultrachat"


WORDS = ("red blue green yellow orange purple pink brown black white cat dog bird fish "
         "lion tiger bear apple mango book table chair river mountain sun moon star tree").split()
ANIMALS = "cat dog lion tiger bear elephant monkey bird fish horse".split()
FRUITS = "apple mango banana orange grapes".split()
COLORS = "red blue green yellow orange purple".split()
PLANETS = "Mercury Venus Earth Mars Jupiter Saturn".split()
DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
OPPOSITES = [("hot", "cold"), ("day", "night"), ("up", "down"), ("big", "small"),
             ("fast", "slow"), ("light", "dark"), ("in", "out"), ("on", "off"),
             ("tall", "short"), ("happy", "sad"), ("open", "closed"), ("man", "woman")]
RHYMES = [("cat", "hat"), ("sun", "fun"), ("day", "play"), ("night", "light"),
          ("green", "bean"), ("tree", "free"), ("car", "star"), ("dog", "frog")]
SPANISH = {"hello": "hola", "water": "agua", "cat": "gato", "dog": "perro",
           "sun": "sol", "moon": "luna", "book": "libro", "house": "casa",
           "food": "comida", "friend": "amigo"}
MONTHS = {"January": 31, "February": 28, "March": 31, "April": 30, "May": 31,
          "June": 30, "July": 31, "August": 31, "September": 30, "October": 31,
          "November": 30, "December": 31}
YESNO = [("Paris is the capital of France.", "Yes"), ("Jupiter is the largest planet.", "Yes"),
         ("The sun rises in the east.", "Yes"), ("Water is made of hydrogen and oxygen.", "Yes"),
         ("2 + 2 equals 5.", "No"), ("The moon orbits the sun.", "No"),
         ("Delhi is the capital of India.", "Yes"), ("Diwali is the festival of lights.", "Yes"),
         ("Fish can fly.", "No"), ("Fire is cold.", "No"), ("Tokyo is the capital of Japan.", "Yes"),
         ("Oxygen helps fire burn.", "Yes"), ("Ice is hot.", "No"), ("Birds are mammals.", "No")]
EXPLAIN = [("gravity", "Gravity pulls objects toward the Earth."),
           ("photosynthesis", "Plants use sunlight to make food."),
           ("evaporation", "Water turns into vapor when heated."),
           ("democracy", "People choose their leaders by voting.")]


def synthetics():
    r = random.Random(2024)
    out = []

    def add(q, a):
        out.append(chat(q, a))

    for w in r.choices(WORDS, k=120):
        add(f"Say the word {w}.", f"{w}.")
        add(f"Repeat after me: {w}.", f"{w}.")
        add(f"Write '{w}' in uppercase.", f"{w.upper()}.")
        add(f"How many letters are in '{w}'?", f"{len(w)}.")
    for w in r.choices(WORDS, k=60):
        add(f"Write a sentence containing the word '{w}'.", f"The {w} is here.")
        add(f"Write a sentence that starts with '{w}'.", f"{w.capitalize()} is nice.")
    for a, b in OPPOSITES:
        add(f"What is the opposite of {a}?", f"{b}.")
        add(f"True or false: the opposite of {a} is {b}.", "True.")
    for a, b in RHYMES:
        add(f"What rhymes with '{a}'?", f"{b}.")
    for q, yn in YESNO:
        add(f"Answer yes or no: {q}", f"{yn}.")
        add(f"True or false: {q}", ("True." if yn == "Yes" else "False."))
    for cat, items in [("colors", COLORS), ("animals", ANIMALS), ("fruits", FRUITS),
                       ("planets", PLANETS)]:
        picks = r.sample(items, 3)
        add(f"List three {cat}.", f"1. {picks[0]}\n2. {picks[1]}\n3. {picks[2]}")
    for _ in range(400):
        a, b = r.randint(0, 50), r.randint(0, 50)
        op = r.choice(["+", "-"])
        ans = a + b if op == "+" else a - b
        t = r.choice([f"What is {a} {op} {b}?", f"Compute {a} {op} {b}.",
                      f"Solve: {a} {op} {b}."])
        add(t, f"{ans}.")
    for _ in range(200):
        a, b = r.randint(0, 100), r.randint(0, 100)
        big = max(a, b)
        add(f"Which is bigger, {a} or {b}?", f"{big}.")
    for _ in range(200):
        ns = [r.randint(0, 99) for _ in range(3)]
        add(f"Sort these numbers: {ns[0]}, {ns[1]}, {ns[2]}.",
            f"{sorted(ns)[0]}, {sorted(ns)[1]}, {sorted(ns)[2]}.")
    for w, s in SPANISH.items():
        add(f"How do you say '{w}' in Spanish?", f"{s}.")
    for d in DAYS:
        i = DAYS.index(d)
        add(f"What day comes after {d}?", f"{DAYS[(i + 1) % 7]}.")
    for m, n in MONTHS.items():
        add(f"How many days are in {m}?", f"{n}.")
    for n in range(1, 27):
        add(f"What is the {n}th letter of the alphabet?", f"{chr(64 + n)}.")
    for c, e in EXPLAIN:
        add(f"Explain {c} simply.", e)
        add(f"What is {c}?", e)
    facts = ["Paris is the capital of France.", "Water boils at 100 degrees Celsius.",
             "The sun rises in the east.", "Jupiter is the largest planet."]
    for f in facts:
        add(f"Summarize in one sentence: {f}", f)
    return out


def oa_slice(max_tokens=2_000_000):
    """Byte-slice local SFT-mix bin at EOS boundary (same tokenizer).
    phase7/train.bin = synthetic + OA (all Tier-B), repeats proven format."""
    p = "data/phase7/train.bin"
    try:
        a = np.fromfile(p, dtype=np.uint16)
    except Exception as e:
        print(f"  oa slice FAILED: {e}", flush=True)
        return b""
    if len(a) <= max_tokens:
        return a.tobytes()
    cut = max_tokens
    while cut < len(a) and int(a[cut]) != 2:
        cut += 1
    print(f"  oa slice: {cut:,} tokens", flush=True)
    return a[:cut + 1].tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-smol", type=int, default=15000)
    ap.add_argument("--n-ultra", type=int, default=15000)
    args = ap.parse_args()

    docs = []
    if True:
        docs.extend(fetch_hf("HuggingFaceTB/smol-smoltalk", "train", smol_pred,
                             args.n_smol, 4000))
        docs.extend(fetch_hf("HuggingFaceH4/ultrachat_200k", "train_sft", ultra_pred,
                             args.n_ultra, args.n_ultra))
    n0 = len(docs)
    docs.extend(synthetics())
    print(f"synthetics: +{len(docs) - n0} docs", flush=True)
    for _ in range(3):
        for line in open("scripts/_anchor_docs.txt"):
            docs.append(line.rstrip("\n"))
    print(f"anchors x3 sprinkled", flush=True)

    seen, uniq = set(), []
    for d in docs:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    rng.shuffle(uniq)
    n_eval = max(1000, len(uniq) // 10)
    eval_docs, train_docs = uniq[:n_eval], uniq[n_eval:]
    print(f"unique docs: {len(uniq)} (train {len(train_docs)}, eval {len(eval_docs)})", flush=True)

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

    oa_raw = oa_slice()
    nt = write(train_docs, TRAIN_BIN)
    if oa_raw:
        with open(TRAIN_BIN, "ab") as f:
            f.write(oa_raw)
        nt += len(oa_raw) // 2
    ne = write(eval_docs, EVAL_BIN)
    print(f"WROTE train {nt:,} tokens -> {TRAIN_BIN}", flush=True)
    print(f"WROTE eval {ne:,} tokens -> {EVAL_BIN}", flush=True)


if __name__ == "__main__":
    main()
