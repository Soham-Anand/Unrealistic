#!/usr/bin/env python3
"""Pass 3: stability (5-order permutations, S_q) + association intrusions.

S_q = fraction of runs producing the modal answer.
Classes: STABLE_CORRECT / STABLE_WRONG / UNSTABLE_CORRECT / UNSTABLE_WRONG /
ASSOCIATION_INTRUSION.
Intrusions: Delhi in non-capital India answers; Beatles/Michael in LP
answers; stray years (1947/1996/2014/1879/1526) where unasked.

Usage:
  PYTHONPATH=. python3 scripts/contamination.py --checkpoint <ckpt> \
      --out evals/failure_analysis_362289/pass3.json
"""

import os, sys, re, json, argparse, random
from collections import Counter
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
from src.model.transformer import init_model
from src.training.checkpointing import load_checkpoint
from src.data.tokenizer import Tokenizer
from src.inference.generate import generate
from score_exact import norm, match, gen_arith, num_of

YEARS = ["1947", "1996", "2014", "1879", "1526", "1945", "1789"]


def stability_set():
    """Small discriminative set: anchors + known-fragile prompts."""
    from score_exact import SCIENCE, GK
    items = [("science", q, a, al) for q, a, al in SCIENCE]
    items += [("gk", q, a, al) for q, a, al in GK]
    items += [("culture", "The lead singer of Linkin Park was",
               "Chester Bennington", ["Chester Bennington", "Chester"])]
    items += [("culture", "Who built the Taj Mahal?", "Shah Jahan",
               ["Shah Jahan", "Shah Jahan built the Taj Mahal"])]
    items += [("arithmetic", "What is 2 + 2?", 4, [])]
    items += [("arithmetic", "What is 8 + 7?", 15, [])]
    return items


def intrusion(pred: str, prompt: str):
    tags = []
    pl = norm(pred)
    if "delhi" in pl and "capital" not in norm(prompt).split("capital")[0]:
        if "capital" not in norm(prompt):
            tags.append("DELHI_INTRUSION")
    if any(k in pl for k in ["beatles", "michael hey", "michael,"]):
        if "linkin" in norm(prompt) or "chester" in norm(prompt):
            tags.append("LP_BEATLES_INTRUSION")
    for y in YEARS:
        if y in pl and y not in norm(prompt):
            tags.append(f"YEAR_INTRUSION_{y}")
            break
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--orders", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-tokens", type=int, default=40)
    args = ap.parse_args()

    mx.set_default_device(mx.gpu)
    params, _ = load_checkpoint(args.checkpoint)
    _, buffers, model_cfg = init_model("configs/model_config.json")
    tok = Tokenizer("data/tokenizer/phase1.model")
    mx.eval(params)

    def ask(prompt, seed):
        mx.random.seed(seed)
        ids = tok.encode(prompt)
        out = generate(params, buffers, model_cfg, ids,
                       max_new_tokens=args.max_tokens, temperature=args.temp,
                       top_k=0, top_p=1.0, repetition_penalty=1.0)
        s = tok.decode(out[len(ids):]).lstrip()
        if s.startswith("Assistant:"):
            s = s[len("Assistant:"):].strip()
        return s

    items = stability_set()
    order = list(range(len(items)))
    report = {"checkpoint": args.checkpoint, "temp": args.temp, "orders": args.orders,
              "questions": [], "intrusion_counts": {}, "class_counts": {}}

    for o in range(args.orders):
        rng = random.Random(args.seed + o)
        rng.shuffle(order)
        for idx in order:
            dom, prompt, accepted, aliases = items[idx]
            pred = ask(prompt, args.seed * 1000 + o)
            if isinstance(accepted, (int, float)):
                v = num_of(pred)
                ok = v is not None and abs(v - accepted) < 1e-6
            else:
                ok = match(pred, accepted, aliases)
            tags = intrusion(pred, prompt)
            q = next((x for x in report["questions"]
                      if x["prompt"] == prompt), None)
            if q is None:
                q = {"domain": dom, "prompt": prompt, "accepted": accepted,
                     "runs": []}
                report["questions"].append(q)
            q["runs"].append({"pred": pred[:200], "hit": ok, "intrusions": tags})
            for t in tags:
                report["intrusion_counts"][t] = report["intrusion_counts"].get(t, 0) + 1

    for q in report["questions"]:
        preds = [r["pred"] for r in q["runs"]]
        modal, cnt = Counter(preds).most_common(1)[0]
        sq = cnt / len(preds)
        hits = sum(r["hit"] for r in q["runs"])
        intr = any(r["intrusions"] for r in q["runs"])
        if intr and not all(r["hit"] for r in q["runs"]):
            cls = "ASSOCIATION_INTRUSION"
        elif sq == 1.0 and hits == len(preds):
            cls = "STABLE_CORRECT"
        elif sq == 1.0:
            cls = "STABLE_WRONG"
        elif hits > 0:
            cls = "UNSTABLE_CORRECT"
        else:
            cls = "UNSTABLE_WRONG"
        q["S_q"] = sq
        q["modal"] = modal[:200]
        q["class"] = cls
        report["class_counts"][cls] = report["class_counts"].get(cls, 0) + 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    print("classes:", report["class_counts"], flush=True)
    print("intrusions:", report["intrusion_counts"], flush=True)
    print(f"WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
