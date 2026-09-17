#!/usr/bin/env python3
"""Publish Unrealistic-v1 to HuggingFace Hub.

Usage:
  HF_TOKEN=hf_xxx HF_USER=Soham-Anand python3 scripts/publish_hf.py
Uploads: config, safetensors, tokenizer, model card, 3 GGUFs, Modelfile.
"""
import os, sys, shutil

REPO = f"{os.environ['HF_USER']}/Unrealistic-v1"

from huggingface_hub import HfApi, create_repo

api = HfApi()
create_repo(REPO, exist_ok=True)
print(f"repo ready: {REPO}", flush=True)

# stage upload dir
stage = "/tmp/hf_stage"
os.makedirs(stage, exist_ok=True)
for f in ["config.json", "model.safetensors", "tokenizer.model",
          "tokenizer_config.json"]:
    shutil.copy(f"export_pt/{f}", f"{stage}/{f}")
shutil.copy("MODEL_CARD.md", f"{stage}/README.md")
for f in ["unrealistic-v1-f16.gguf", "unrealistic-v1-Q8_0.gguf",
          "unrealistic-v1-Q4_K_M.gguf"]:
    shutil.copy(f"gguf/{f}", f"{stage}/{f}")
shutil.copy("Modelfile", f"{stage}/Modelfile")

api.upload_folder(folder_path=stage, repo_id=REPO, repo_type="model")
print(f"PUBLISHED: https://huggingface.co/{REPO}", flush=True)
