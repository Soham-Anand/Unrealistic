#!/usr/bin/env python3
"""Publish MLX checkpoints (params-only repacks) + all baselines to Hub.

Strips optimizer state (opt.*) via MLX's own loader — keeps params.* only
(~380MB vs 1.1G each). Streams: repack one -> upload -> delete.
Usage: HF_USER=SohamProgrammer python3 scripts/publish_mlx.py
"""
import os, sys, shutil, glob

REPO = f"{os.environ['HF_USER']}/Unrealistic-v1"

import mlx.core as mx
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.training.checkpointing import load_checkpoint
from huggingface_hub import HfApi


def _flatten(d, prefix, out):
    if isinstance(d, dict):
        for k, v in d.items():
            _flatten(v, prefix + "." + k, out)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            _flatten(v, f"{prefix}.{i}", out)
    else:
        out[prefix] = d

api = HfApi()
stage = "/tmp/mlx_stage"
os.makedirs(stage, exist_ok=True)

jobs = [("final", "checkpoints/step_355789")]
for d in sorted(glob.glob("baselines/*")):
    jobs.append((os.path.basename(d), d))

shutil.copy("data/tokenizer/phase1.model", f"{stage}/tokenizer.model")
api.upload_file(path_or_fileobj=f"{stage}/tokenizer.model",
                   path_in_repo=f"mlx/tokenizer.model", repo_id=REPO)
print("tokenizer up", flush=True)

import glob as _glob

for name, src in jobs:
    out = f"{stage}/{name}"
    os.makedirs(out, exist_ok=True)
    # old-style baselines nest under step_*/ subdir
    if not os.path.exists(os.path.join(src, "model.npz")):
        sub = sorted(_glob.glob(os.path.join(src, "step_*")))
        if sub:
            print(f"{name}: descending into {sub[0]}", flush=True)
            src = sub[0]
    print(f"repack {name} from {src}...", flush=True)
    params, _, _ = load_checkpoint(src, return_opt=True)
    flat = {}
    _flatten(params, "params", flat)
    mx.savez(os.path.join(out, "model.npz"), **flat)
    for f in ("config.json", "meta.json"):
        p = os.path.join(src, f)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(out, f))
    api.upload_folder(folder_path=out, path_in_repo=f"mlx/{name}", repo_id=REPO)
    print(f"uploaded mlx/{name}", flush=True)
    shutil.rmtree(out, ignore_errors=True)

print(f"MLX DONE: https://huggingface.co/{REPO}/tree/main/mlx", flush=True)
