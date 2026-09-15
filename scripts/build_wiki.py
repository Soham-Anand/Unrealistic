#!/usr/bin/env python3
"""Phase 7b (Wiki): CC0 entity-knowledge injection between SFT and RFT.

Source: English Wikipedia lead sections (CC BY-SA, same posture as existing
wiki data) via category walk + extracts API: bands/singers/actors/films/books
+ embedded essentials (Linkin Park etc.). Math: GSM8K (MIT, local) +
Orca-Math (MIT/Apache-2.0, HF) + SVAMP (MIT, HF). NO KELM, NO WDQS (outage).

Mix: entity statements + QA-form pairs + Wikidata math people/constants +
arithmetic keep-in (protects SFT math routing).

Usage:
  PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/build_wiki.py [--fetch-only|--no-fetch]
  -> data/phase7b/wiki_train.bin + wiki_eval.bin
"""

import os, sys, json, random, argparse, time, urllib.request, urllib.parse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.data.tokenizer import Tokenizer

OUT_DIR = "data/phase7b"
TRAIN_BIN = f"{OUT_DIR}/wiki_train.bin"
EVAL_BIN = f"{OUT_DIR}/wiki_eval.bin"
UA = {"User-Agent": "UnrealisticBot/1.0 (research; contact local)",
      "Accept": "application/sparql-results+json"}
EOS = None
rng = random.Random(7)


API = "https://en.wikipedia.org/w/api.php"

UA_WP = {"User-Agent": "UnrealisticBot/1.0 (research; contact local)"}


def wp(params: dict, timeout: int = 60, retries: int = 4):
    """MediaWiki API GET with retries. Returns parsed JSON or None."""
    params = dict(params, format="json", formatversion="2")
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA_WP)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:
            print(f"  wp attempt {attempt + 1} failed: {str(e)[:100]}", flush=True)
            time.sleep(4 * (attempt + 1))
    return None


def cat_members(cat: str, cap: int, depth: int = 1):
    """Collect page titles from a category (+1 subcat level). Returns title list."""
    titles, seen_cats = [], set()

    def walk(c, d):
        cont = None
        while len(titles) < cap:
            p = {"action": "query", "list": "categorymembers",
                 "cmtitle": f"Category:{c}", "cmtype": "page|subcat",
                 "cmlimit": "500"}
            if cont:
                p["cmcontinue"] = cont
            d_ = wp(p)
            if not d_:
                break
            members = d_.get("query", {}).get("categorymembers", [])
            for m in members:
                t = m.get("title", "")
                if m.get("ns") != 0 or not t or ":" in t.split(" (")[0]:
                    if m.get("ns") == 14 and d > 0 and t not in seen_cats:
                        seen_cats.add(t)
                        walk(t.replace("Category:", ""), d - 1)
                    continue
                if len(t) < 100:
                    titles.append(t)
                if len(titles) >= cap:
                    break
            cont = d_.get("continue", {}).get("cmcontinue")
            if not cont:
                break
            time.sleep(2.0)
    walk(cat, depth)
    return titles


CATEGORIES = [
    ("American rock music groups", 1200), ("English rock music groups", 1200),
    ("American pop music groups", 600), ("South Korean boy bands", 200),
    ("South Korean girl groups", 200), ("American singers", 1500),
    ("English singers", 1000), ("American rappers", 600),
    ("American films", 2000), ("English-language films", 2000),
    ("American novels", 1000), ("American actors", 1500),
    ("American mathematicians", 300),
]


def fetch_leads(titles, batch=50):
    """Lead-section plaintext per title. Returns [(title, text)]."""
    out = []
    for i in range(0, len(titles), batch):
        chunk = titles[i:i + batch]
        d_ = wp({"action": "query", "prop": "extracts", "exintro": "1",
                 "explaintext": "1", "redirects": "1", "titles": "|".join(chunk)})
        if not d_:
            continue
        for pg in d_.get("query", {}).get("pages", []):
            t, x = pg.get("title", ""), pg.get("extract", "")
            if t and x and len(x) > 200:
                out.append((t, x))
        time.sleep(2.0)
        if (i // batch) % 20 == 0:
            print(f"  leads {i + len(chunk)}/{len(titles)}", flush=True)
    return out


import re as _re
_SENT = _re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")

def split_sentences(text):
    sents = []
    for s in _SENT.split(text.replace("\n", " ")):
        s = s.strip()
        n = len(s.split())
        if 8 <= n <= 60 and "Coordinates" not in s and s.count("(") == s.count(")"):
            sents.append(s)
    return sents


STATIC = [
    ("Linkin Park", ["Linkin Park is a rock band from the United States, formed in 1996.",
     "User: Who was the lead singer of Linkin Park?\nAssistant: Chester Bennington was the lead singer of Linkin Park.",
     "User: When was Linkin Park formed?\nAssistant: Linkin Park was formed in 1996.",
     "Hybrid Theory is an album by Linkin Park, released in 2000.",
     "In the End is a song by Linkin Park, released in 2001."]),
    ("The Beatles", ["The Beatles is a rock band from England, formed in 1960.",
     "User: Which country is the band The Beatles from?\nAssistant: The Beatles is from England."]),
    ("Queen", ["Queen is a rock band from England, formed in 1970.",
     "Freddie Mercury is a singer from Tanzania, born in 1946.",
     "Freddie Mercury died in 1991."]),
    ("Taylor Swift", ["Taylor Swift is a singer from the United States, born in 1989.",
     "User: When was Taylor Swift born?\nAssistant: Taylor Swift was born in 1989."]),
    ("BTS", ["BTS is a band from South Korea, formed in 2013."]),
    ("Michael Jackson", ["Michael Jackson is a singer from the United States, born in 1958.",
     "Michael Jackson died in 2009."]),
    ("Elvis Presley", ["Elvis Presley is a singer from the United States, born in 1935.",
     "Elvis Presley died in 1977."]),
    ("Chester Bennington", ["Chester Bennington is a singer from the United States, born in 1976.",
     "Chester Bennington died in 2017."]),
    ("Titanic", ["Titanic is a film directed by James Cameron, released in 1997."]),
    ("Avatar", ["Avatar is a film directed by James Cameron, released in 2009."]),
    ("Harry Potter", ["Harry Potter is a book by J. K. Rowling, published in 1997."]),
    ("Albert Einstein", ["Albert Einstein is a physicist from Germany, born in 1879.",
     "Albert Einstein died in 1955."]),
    ("Isaac Newton", ["Isaac Newton is a mathematician from England, born in 1643.",
     "Isaac Newton died in 1727."]),
    ("Srinivasa Ramanujan", ["Srinivasa Ramanujan is a mathematician from India, born in 1887.",
     "Srinivasa Ramanujan died in 1920."]),
    ("Aryabhata", ["Aryabhata is a mathematician from India, born in 476."]),
    ("Shakuntala Devi", ["Shakuntala Devi is a mathematician from India, born in 1929.",
     "Shakuntala Devi died in 2013."]),
    ("Taj Mahal", ["The Taj Mahal is a monument in Agra, India, built in 1653."]),
    ("Mount Everest", ["Mount Everest is a mountain in Nepal, the highest in the world."]),
    ("Nile", ["The Nile is a river in Africa, the longest in the world."]),
    ("Oxygen", ["Oxygen is a chemical element with the symbol O.",
     "Oxygen has the atomic number 8."]),
    ("Gold", ["Gold is a chemical element with the symbol Au.",
     "Gold has the atomic number 79."]),
]

MATH_CONSTANTS = [
    "Pi is a mathematical constant, about 3.14159.",
    "User: What is the value of pi?\nAssistant: Pi is about 3.14159.",
    "The number e is a mathematical constant, about 2.71828.",
    "Zero is the number that means nothing.",
    "User: What is 10 divided by 2?\nAssistant: 10 divided by 2 is 5.",
    "A triangle has 3 sides.",
    "A square has 4 sides.",
    "A circle has 360 degrees.",
    "User: How many sides does a triangle have?\nAssistant: A triangle has 3 sides.",
    "Infinity means without end.",
]


def fetch_math_datasets(n_orca=40000):
    """Real math datasets (all permissive licenses), chat-formatted docs.

    - GSM8K (MIT): merged at write time from local data/phase5/gsm8k.bin
    - Orca-Math-200K (MIT/Apache-2.0): HF fetch, seeded sample
    - SVAMP (MIT): HF fetch, full 1k
    Each returns list of 'User: ...\\nAssistant: ...' strings. Failures skip gracefully.
    """
    import shutil
    docs = []
    cache = "/tmp/wiki_math_cache"
    os.makedirs(cache, exist_ok=True)
    try:
        from datasets import load_dataset
        print("fetch orca-math (sample 40k)...", flush=True)
        ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train",
                          cache_dir=cache)
        idx = rng.sample(range(len(ds)), min(n_orca, len(ds)))
        for i in idx:
            q = ds[i]["question"].strip()
            a = ds[i]["answer"].strip()
            if q and a and len(q) < 1500 and len(a) < 1500:
                docs.append(f"User: {q}\nAssistant: {a}")
        print(f"  orca-math: +{len(docs)} docs", flush=True)
    except Exception as e:
        print(f"  orca-math FAILED: {e}", flush=True)
    try:
        from datasets import load_dataset
        print("fetch svamp...", flush=True)
        ds = load_dataset("ChilleD/SVAMP", split="train", cache_dir=cache)
        n0 = len(docs)
        for r in ds:
            q = str(r.get("question", "")).strip()
            a = str(r.get("answer", "")).strip()
            if q and a:
                docs.append(f"User: {q}\nAssistant: {a}")
        print(f"  svamp: +{len(docs) - n0} docs", flush=True)
    except Exception as e:
        print(f"  svamp FAILED (skipped): {e}", flush=True)
    shutil.rmtree(cache, ignore_errors=True)
    return docs


def gsm8k_bytes():
    """Raw bytes of local GSM8K bin (same tokenizer, EOS-delimited). Merged at write."""
    p = "data/phase5/gsm8k.bin"
    if os.path.exists(p):
        with open(p, "rb") as f:
            return f.read()
    print("  WARNING: data/phase5/gsm8k.bin missing, GSM8K skipped", flush=True)
    return b""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true", help="skip wiki fetch, static+math only")
    ap.add_argument("--fetch-only", action="store_true", help="skip math datasets")
    ap.add_argument("--max-titles", type=int, default=25000)
    args = ap.parse_args()

    docs = []
    if not args.no_fetch:
        titles, seen = [], set()
        for cat, cap in CATEGORIES:
            print(f"category {cat} (cap {cap})...", flush=True)
            try:
                got = cat_members(cat, cap)
            except Exception as e:
                print(f"  {cat}: FAILED {str(e)[:100]}", flush=True)
                continue
            n0 = len(titles)
            for t in got:
                if t not in seen:
                    seen.add(t)
                    titles.append(t)
            print(f"  {cat}: +{len(titles) - n0} titles (total {len(titles)})", flush=True)
            if len(titles) >= args.max_titles:
                break
        titles = titles[:args.max_titles]
        print(f"fetching leads for {len(titles)} titles...", flush=True)
        n0 = len(docs)
        for title, lead in fetch_leads(titles):
            for s in split_sentences(lead)[:6]:
                docs.append(s)
        print(f"leads -> +{len(docs) - n0} sentence docs", flush=True)

    # essentials always present (STATIC + math constants)
    for _, ss in STATIC:
        docs.extend(ss)

    docs.extend(MATH_CONSTANTS)
    gsm_raw = b""
    if not args.fetch_only:
        docs.extend(fetch_math_datasets())
        gsm_raw = gsm8k_bytes()
        print(f"gsm8k local: {len(gsm_raw) // 2:,} tokens", flush=True)

    # dedup + shuffle + split
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

    def write(docs, path):
        n = 0
        with open(path, "wb") as f:
            for d in docs:
                ids = tok.encode(d)
                if not ids:
                    continue
                f.write(np.array(ids, dtype=np.uint16).tobytes())
                f.write(np.uint16(EOS).tobytes())
                n += len(ids)
        return n

    nt = write(train_docs, TRAIN_BIN)
    if gsm_raw:
        with open(TRAIN_BIN, "ab") as f:
            f.write(gsm_raw)
        nt += len(gsm_raw) // 2
    ne = write(eval_docs, EVAL_BIN)
    print(f"WROTE train {nt:,} tokens -> {TRAIN_BIN}", flush=True)
    print(f"WROTE eval {ne:,} tokens -> {EVAL_BIN}", flush=True)


if __name__ == "__main__":
    main()
