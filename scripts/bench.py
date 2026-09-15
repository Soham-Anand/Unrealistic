#!/usr/bin/env python3
"""Minimal MLX-direct benchmark runner for Unrealistic final (no torch needed).

Multiple-choice: sum (or token-norm) continuation logps via compiled forward.
GSM8K: open generation, strict EM via numeric extraction.
MBPP: open generation, pass@1 via subprocess exec of test_list.

Suites: arc_easy, piqa, hellaswag(500), winogrande, truthfulqa_mc1,
        mmlu_easy(4 subsets), gsm8k(200), mbpp(100).
Usage:
  PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/bench.py \
      --checkpoint checkpoints/step_355789 --out evals/benchmarks/final
"""

import os, sys, re, json, argparse, random, subprocess, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import mlx.core as mx
from src.model.transformer import init_model, forward
from src.training.checkpointing import load_checkpoint
from src.data.tokenizer import Tokenizer
from src.inference.generate import generate

PAD = 256
params = buffers = model_cfg = tok = EOS = None


def setup(ckpt):
    global params, buffers, model_cfg, tok, EOS
    mx.set_default_device(mx.gpu)
    params, _ = load_checkpoint(ckpt)
    _, buffers, model_cfg = init_model("configs/model_config.json")
    tok = Tokenizer("data/tokenizer/phase1.model")
    EOS = tok.eos_token
    mx.eval(params)
    print(f"model loaded: {ckpt}", flush=True)


def _fwd(x):
    logits = forward(params, buffers, x, model_cfg, training=False)
    return logits


compiled_fwd = None


def cont_logp(context_ids, cont_ids):
    """Sum logp of continuation tokens given context. Fixed-shape compiled."""
    global compiled_fwd
    if compiled_fwd is None:
        def f(x):
            lg = forward(params, buffers, x, model_cfg, training=True)[0].astype(mx.float32)
            n = lg - mx.max(lg, axis=-1, keepdims=True)
            return n
        compiled_fwd = mx.compile(f)
    full = (context_ids + cont_ids)[:PAD]
    nc = min(len(context_ids), PAD - 1)
    arr = np.full((PAD,), EOS, dtype=np.int32)
    arr[:len(full)] = np.asarray(full, dtype=np.int32)
    x = mx.array(arr)[None, :]
    normed = compiled_fwd(x)
    mx.eval(normed)
    normed = np.asarray(normed, dtype=np.float32)
    if normed.ndim == 3:
        normed = normed[0]
    lse = np.logaddexp.reduce(normed, axis=-1)
    lp = normed[np.arange(PAD), arr] - lse
    # score continuation positions only: logit row t-1 predicts token arr[t]
    s = 0.0
    for t in range(nc, len(full)):
        if arr[t] == EOS:
            break
        row = normed[t - 1] if t - 1 >= 0 else normed[0]
        s += float(row[arr[t]] - np.logaddexp.reduce(row))
    return s, max(1, len(full) - nc)


def mc_score(context, choices, norm=False):
    cids = tok.encode(context)
    best, best_i, scores = -1e18, -1, []
    for ch in choices:
        s, n = cont_logp(cids, tok.encode(" " + ch))
        v = s / n if norm else s
        scores.append(v)
        if v > best:
            best, best_i = v, len(scores) - 1
    return best_i, scores


def run_mc(name, items, norm=False, out=None):
    """items: list of (context, choices, gold_idx)."""
    hits = 0
    for i, (ctx, chs, gold) in enumerate(items):
        bi, _ = mc_score(ctx, chs, norm)
        hits += int(bi == gold)
        if (i + 1) % 200 == 0:
            print(f"  {name}: {i + 1}/{len(items)} acc={hits/(i+1):.3f}", flush=True)
    acc = hits / max(1, len(items))
    print(f"{name}: {hits}/{len(items)} = {acc:.4f}", flush=True)
    if out is not None:
        out[name] = {"n": len(items), "hits": hits, "acc": acc}
    return acc


# ─── loaders (all failures -> skip with warning) ───
def L_arc():
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
    items = []
    for r in ds:
        q = r["question"]
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        gold = labels.index(r["answerKey"])
        items.append((f"Question: {q}\nAnswer:", texts, gold))
    return items


def L_piqa():
    import urllib.request
    url = "https://yonatanbisk.com/piqa/data/val.jsonl"
    req = urllib.request.Request(url, headers={"User-Agent": "UnrealisticBot/1.0"})
    items = []
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            o = json.loads(line)
            items.append((f"Goal: {o['goal']}\nAnswer:",
                          [o["sol1"], o["sol2"]], int(o["label"])))
    return items


def L_hellaswag(n=500, seed=7):
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    idx = random.Random(seed).sample(range(len(ds)), min(n, len(ds)))
    items = []
    for i in idx:
        r = ds[i]
        items.append((r["ctx"], r["endings"], int(r["label"])))
    return items


def L_winogrande():
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    items = []
    for r in ds:
        # full-sequence scoring: fill blank per option, mean token logp
        s = r["sentence"]
        opts = [r["option1"], r["option2"]]
        gold = int(r["answer"]) - 1
        items.append((s, opts, gold, True))
    return items


def L_truthfulqa():
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    items = []
    for r in ds:
        chs = r["mc1_targets"]["choices"]
        lab = r["mc1_targets"]["labels"]
        gold = lab.index(1)
        items.append((f"Question: {r['question']}\nAnswer:", chs, gold))
    return items


def L_mmlu(subsets=("elementary_mathematics", "high_school_biology",
                    "high_school_chemistry", "high_school_physics")):
    from datasets import load_dataset
    items = []
    for s in subsets:
        ds = load_dataset("cais/mmlu", s, split="test")
        for r in ds:
            opts = [f"{t}. {c}" for t, c in zip("ABCD", r["choices"])]
            ctx = f"Question: {r['question']}\n" + "\n".join(opts) + "\nAnswer:"
            items.append((ctx, list("ABCD"), int(r["answer"])))
    return items


def gen_answer(prompt, max_tokens=256):
    out = generate(params, buffers, model_cfg, tok.encode(prompt),
                   max_new_tokens=max_tokens, temperature=0.0,
                   top_k=0, top_p=1.0, repetition_penalty=1.0)
    return tok.decode(out)


def num_last(text):
    m = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return float(m[-1]) if m else None


def run_gsm8k(n=200, seed=7, out=None):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = random.Random(seed).sample(range(len(ds)), min(n, len(ds)))
    hits = 0
    for k, i in enumerate(idx):
        r = ds[int(i)]
        gold_m = re.search(r"####\s*(-?\d+\.?\d*)", r["answer"])
        gold = float(gold_m.group(1)) if gold_m else None
        pred = num_last(gen_answer(f"Question: {r['question']}\nAnswer:"))
        ok = gold is not None and pred is not None and abs(pred - gold) < 1e-6
        hits += int(ok)
        if (k + 1) % 25 == 0:
            print(f"  gsm8k: {k + 1}/{len(idx)} em={hits/(k+1):.3f}", flush=True)
    acc = hits / max(1, len(idx))
    print(f"gsm8k: {hits}/{len(idx)} = {acc:.4f}", flush=True)
    if out is not None:
        out["gsm8k"] = {"n": len(idx), "hits": hits, "acc": acc}
    return acc


def run_mbpp(n=100, seed=7, out=None):
    import urllib.request
    url = ("https://raw.githubusercontent.com/google-research/google-research/"
           "master/mbpp/mbpp.jsonl")
    req = urllib.request.Request(url, headers={"User-Agent": "UnrealisticBot/1.0"})
    rows = []
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            rows.append(json.loads(line))
    idx = random.Random(seed).sample(range(len(rows)), min(n, len(rows)))
    hits = 0
    for k, i in enumerate(idx):
        r = rows[int(i)]
        code = gen_answer(f"{r['text']}\nWrite a Python function:\n```python\n", 256)
        code = re.sub(r"```.*", "", code).strip()
        prog = code + "\n" + "\n".join(r["test_list"])
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(prog)
                path = f.name
            cp = subprocess.run([sys.executable, path], capture_output=True, timeout=10)
            ok = cp.returncode == 0
        except Exception:
            ok = False
        hits += int(ok)
        if (k + 1) % 20 == 0:
            print(f"  mbpp: {k + 1}/{len(idx)} pass@1={hits/(k+1):.3f}", flush=True)
    acc = hits / max(1, len(idx))
    print(f"mbpp: {hits}/{len(idx)} = {acc:.4f}", flush=True)
    if out is not None:
        out["mbpp"] = {"n": len(idx), "hits": hits, "acc": acc}
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/step_355789")
    ap.add_argument("--out", default="evals/benchmarks/final")
    ap.add_argument("--suite", default="all", help="all|mc|gsm8k|mbpp")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    setup(args.checkpoint)
    res = {"checkpoint": args.checkpoint}

    def guarded(name, fn, *a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:
            print(f"{name}: SKIPPED ({str(e)[:150]})", flush=True)
            res[name] = {"skipped": str(e)[:150]}
            return None

    if args.suite in ("all", "mc"):
        guarded("arc_easy", run_mc, "arc_easy", guarded("arc_easy_items", L_arc) or [], False, res)
        guarded("piqa", run_mc, "piqa", guarded("piqa_items", L_piqa) or [], True, res)
        guarded("hellaswag", run_mc, "hellaswag", guarded("hs_items", L_hellaswag) or [], True, res)
        # winogrande: full-sequence scoring
        def wino(items, out):
            hits = 0
            for i, (s, opts, gold, _) in enumerate(items):
                sc = []
                for o in opts:
                    t = tok.encode(s.replace("_", " " + o))
                    sc.append(cont_logp([], t)[0] / max(1, len(t)))
                hits += int(int(np.argmax(sc)) == gold)
                if (i + 1) % 200 == 0:
                    print(f"  winogrande: {i + 1}/{len(items)}", flush=True)
            acc = hits / max(1, len(items))
            print(f"winogrande: {hits}/{len(items)} = {acc:.4f}", flush=True)
            out["winogrande"] = {"n": len(items), "hits": hits, "acc": acc}
        guarded("winogrande", wino, guarded("wino_items", L_winogrande) or [], res)
        guarded("truthfulqa_mc1", run_mc, "truthfulqa_mc1",
                guarded("tqa_items", L_truthfulqa) or [], False, res)
        guarded("mmlu_easy", run_mc, "mmlu_easy", guarded("mmlu_items", L_mmlu) or [], False, res)
    if args.suite in ("all", "gsm8k"):
        guarded("gsm8k_run", run_gsm8k, 200, 7, res)
    if args.suite in ("all", "mbpp"):
        guarded("mbpp_run", run_mbpp, 100, 7, res)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("\n==== SUMMARY ====", flush=True)
    for k, v in res.items():
        if isinstance(v, dict) and "acc" in v:
            print(f"  {k:<16} {v['hits']}/{v['n']} = {v['acc']:.4f}", flush=True)
        else:
            print(f"  {k}: {v}", flush=True)


if __name__ == "__main__":
    main()
