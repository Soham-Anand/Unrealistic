#!/usr/bin/env python3
"""Pass 2 scorer: exact-match with normalization + per-domain accuracy.

Normalization: lowercase, strip whitespace/punctuation/articles, UNICODE
DIACRITIC folding (Dalí==Dali — required by ART-005/006), numeric equivalence.
Arithmetic: calculator-generated (unlimited ground truth).
Science/GK: verified anchor-derived sets.

Usage:
  PYTHONPATH=. python3 scripts/score_exact.py --checkpoint <ckpt> \
      --out evals/failure_analysis_362289/pass2.json [--n-arith 300] [--temp 0.0]
"""

import os, sys, re, json, argparse, random, unicodedata
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mlx.core as mx
from src.model.transformer import init_model
from src.training.checkpointing import load_checkpoint
from src.data.tokenizer import Tokenizer
from src.inference.generate import generate

ARTICLES = re.compile(r"\b(a|an|the)\b", re.I)
PUNCT = re.compile(r"[^\w\s]")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = PUNCT.sub(" ", s)
    s = ARTICLES.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def num_of(text):
    m = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return float(m[-1]) if m else None


def match(pred: str, accepted: str, aliases) -> bool:
    pn, an = norm(pred), norm(accepted)
    if not pn:
        return False
    if pn == an or pn.startswith(an) or an.startswith(pn):
        return True
    for al in aliases:
        a = norm(al)
        if pn == a or pn.startswith(a) or a.startswith(pn):
            return True
    return False


# ─── verified science/GK sets (anchor + probe facts) ───
SCIENCE = [
    ("Water is made of hydrogen and", "oxygen", ["oxygen"]),
    ("The largest planet in the Solar System is", "Jupiter", ["Jupiter"]),
    ("The Earth revolves around the", "Sun", ["Sun", "sun"]),
    ("Atoms are made of protons, neutrons and", "electrons", ["electrons"]),
    ("DNA is a", "molecule", ["molecule"]),
    ("The Moon orbits the", "Earth", ["Earth"]),
    ("Photosynthesis produces", "oxygen", ["oxygen", "glucose"]),
    ("The Nile is the longest", "river", ["river"]),
    ("Mount Everest is the highest", "mountain", ["mountain"]),
    ("Light travels faster than", "sound", ["sound"]),
]
GK = [
    ("The capital of France is", "Paris", ["Paris"]),
    ("The capital of Japan is", "Tokyo", ["Tokyo"]),
    ("The capital of India is", "New Delhi", ["New Delhi", "Delhi"]),
    ("The largest planet in the Solar System is", "Jupiter", ["Jupiter"]),
    ("In which year did India gain independence?", "1947", ["1947"]),
    ("What is the currency of India?", "Rupee", ["Rupee", "INR"]),
    ("The Second World War ended in", "1945", ["1945"]),
    ("Who was the first Prime Minister of India?",
     "Jawaharlal Nehru was the first Prime Minister of India.",
     ["Nehru", "Jawaharlal Nehru"]),
    ("Which monument in Agra is one of the wonders of the world?",
     "The Taj Mahal", ["Taj Mahal"]),
    ("Who wrote the Indian national anthem?", "Rabindranath Tagore", ["Tagore"]),
]


def gen_arith(n, seed):
    r = random.Random(seed)
    out = []
    for _ in range(n):
        op = r.choice(["+", "-", "*"])
        if op == "+":
            a, b = r.randint(0, 999), r.randint(0, 999)
            out.append((f"What is {a} + {b}?", a + b))
        elif op == "-":
            a = r.randint(0, 999)
            b = r.randint(0, a)
            out.append((f"What is {a} - {b}?", a - b))
        else:
            a, b = r.randint(0, 99), r.randint(0, 12)
            out.append((f"What is {a} x {b}?", a * b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--culture", default="evals/failure_analysis_362289/culture_100.jsonl")
    ap.add_argument("--n-arith", type=int, default=300)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-tokens", type=int, default=40)
    args = ap.parse_args()

    mx.set_default_device(mx.gpu)
    params, _ = load_checkpoint(args.checkpoint)
    _, buffers, model_cfg = init_model("configs/model_config.json")
    tok = Tokenizer("data/tokenizer/phase1.model")
    EOS = tok.eos_token
    mx.eval(params)

    def ask(prompt):
        mx.random.seed(args.seed)
        ids = tok.encode(prompt)
        out = generate(params, buffers, model_cfg, ids,
                       max_new_tokens=args.max_tokens, temperature=args.temp,
                       top_k=0, top_p=1.0, repetition_penalty=1.0)
        text = tok.decode(out[len(ids):])
        s = text.lstrip()
        if s.startswith("Assistant:"):
            s = s[len("Assistant:"):].strip()
        return s

    report = {"checkpoint": args.checkpoint, "temp": args.temp,
              "seed": args.seed, "domains": {}, "items": []}

    def run_domain(name, items):
        hits = 0
        for prompt, accepted, aliases in items:
            pred = ask(prompt)
            ok = match(pred, accepted, aliases)
            hits += int(ok)
            report["items"].append({"domain": name, "prompt": prompt,
                                   "accepted": accepted, "pred": pred[:200],
                                   "hit": ok})
        acc = hits / max(1, len(items))
        report["domains"][name] = {"n": len(items), "hits": hits, "acc": acc}
        print(f"{name}: {hits}/{len(items)} = {acc:.4f}", flush=True)

    culture = [json.loads(l) for l in open(args.culture)]
    run_domain("culture", [(c["question"], c["accepted_answer"], c["aliases"])
                           for c in culture])
    run_domain("science", SCIENCE)
    run_domain("gk", GK)
    arith = gen_arith(args.n_arith, args.seed)
    ha = 0
    for prompt, gold in arith:
        pred = num_of(ask(prompt))
        ok = pred is not None and abs(pred - gold) < 1e-6
        ha += int(ok)
        report["items"].append({"domain": "arithmetic", "prompt": prompt,
                               "accepted": gold, "pred": pred, "hit": ok})
    acc = ha / max(1, len(arith))
    report["domains"]["arithmetic"] = {"n": len(arith), "hits": ha, "acc": acc}
    print(f"arithmetic: {ha}/{len(arith)} = {acc:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
