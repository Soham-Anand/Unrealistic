#!/usr/bin/env python3
"""First-token probability probes for the Phase-3 gate.

For each probe (prompt + target tokens from evals/probes.json), reports:
  - rank and P(target) in the next-token distribution
  - top-k alternatives
  - greedy (argmax) continuation
  - for multi-token targets (years): per-token ranks + sequence log-probability

Appends a JSONL record to evals/first_token_history.json for trend tracking.
"""

import os
import sys
import json
import glob
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import mlx.core as mx
from mlx.utils import tree_map
from src.model.transformer import forward
from src.training.checkpointing import load_checkpoint
from src.data.tokenizer import Tokenizer

PROBE_VERSION = "1.0"


def latest_checkpoint(root: str = "checkpoints") -> str:
    dirs = glob.glob(os.path.join(root, "step_*"))
    return max(dirs, key=os.path.getmtime)


def step_from_path(path: str) -> int:
    m = os.path.basename(path).split("_")
    return int(m[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--probes", type=str, default="evals/probes.json")
    parser.add_argument("--out", type=str, default="evals/first_token_history.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    mx.set_default_device(mx.gpu)
    ckpt = args.checkpoint or latest_checkpoint()
    step = step_from_path(ckpt)

    params, model_cfg, _ = load_checkpoint(ckpt, return_opt=True)
    params = tree_map(lambda v: v.astype(mx.bfloat16) if isinstance(v, mx.array) else v, params)
    buffers = {}
    mx.eval(params)
    print(f"Checkpoint: {ckpt} (step {step})")

    tokenizer = Tokenizer("data/tokenizer/phase1.model")
    with open(args.probes) as f:
        probes = json.load(f)

    record = {
        "checkpoint": ckpt,
        "step": step,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "probe_version": PROBE_VERSION,
        "probes": [],
    }

    for probe in probes:
        prompt_ids = probe["prompt_ids"]
        x = mx.array([prompt_ids], dtype=mx.int32)
        logits = forward(params, buffers, x, model_cfg, training=False)
        probs = np.asarray(mx.softmax(logits[0, -1, :]).astype(mx.float32))
        order = np.argsort(probs)[::-1]
        top1_id = int(order[0])
        greedy_text = tokenizer.decode([top1_id])

        topk = [
            {"token": tokenizer.decode([int(i)]), "id": int(i),
             "prob": float(probs[int(i)])}
            for i in order[: args.top_k]
        ]

        p_entry = {
            "domain": probe["domain"],
            "prompt": probe["prompt"],
            "prompt_ids": prompt_ids,
            "top1": {"token": greedy_text, "id": top1_id, "prob": float(probs[top1_id])},
            "topk": topk,
            "targets": [],
        }

        for target in probe["targets"]:
            ids = target["ids"]
            entry = {"text": target["text"], "kind": target["kind"], "ids": ids}
            if target["kind"] == "tok":
                tid = ids[0]
                pos = int(np.where(order == tid)[0][0])
                entry["rank"] = pos + 1
                entry["prob"] = float(probs[tid])
            else:
                prefix = list(prompt_ids)
                step_ranks = []
                logp = 0.0
                for tid in ids:
                    xx = mx.array([prefix], dtype=mx.int32)
                    ll = forward(params, buffers, xx, model_cfg, training=False)
                    pp = np.asarray(mx.softmax(ll[0, -1, :]).astype(mx.float32))
                    oo = np.argsort(pp)[::-1]
                    pos = int(np.where(oo == tid)[0][0])
                    step_ranks.append({"id": tid, "rank": pos + 1,
                                       "prob": float(pp[tid])})
                    logp += np.log(float(pp[tid]))
                    prefix.append(tid)
                entry["step_ranks"] = step_ranks
                entry["seq_logp"] = float(logp)
            p_entry["targets"].append(entry)

        record["probes"].append(p_entry)

        print(f'\n--- [{p_entry["domain"]}] {probe["prompt"]!r} ---')
        print(f'  greedy: {greedy_text!r} (P={probs[top1_id]:.4f})')
        for t in p_entry["targets"]:
            if t["kind"] == "tok":
                print(f'  {t["text"]!r:<14} rank {t["rank"]:<4} P={t["prob"]:.5f}')
            else:
                print(f'  {t["text"]!r:<14} seq_logp={t["seq_logp"]:.3f} '
                      f'steps={[(s["id"], s["rank"]) for s in t["step_ranks"]]}')
        print(f'  top{args.top_k}: ' + ", ".join(
            f'{k["token"]!r}:{k["prob"]:.3f}' for k in topk))

    domains = {}
    for p in record["probes"]:
        d = p["domain"]
        hits3 = any(t.get("rank", 99) <= 3 for t in p["targets"]
                    if t["kind"] == "tok")
        hits10 = any(t.get("rank", 99) <= 10 for t in p["targets"]
                     if t["kind"] == "tok")
        if d not in domains:
            domains[d] = [0, 0, 0]
        domains[d][0] += 1
        domains[d][1] += int(hits3)
        domains[d][2] += int(hits10)

    print("\n" + "=" * 60)
    print("Breadth coverage (single-token targets, any-correct):")
    for d, (n, h3, h10) in sorted(domains.items()):
        print(f"  {d:<10} top-3 {h3}/{n} ({100*h3/n:.0f}%)   top-10 {h10}/{n} ({100*h10/n:.0f}%)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"\nappended to {args.out}")


if __name__ == "__main__":
    main()
