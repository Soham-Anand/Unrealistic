#!/usr/bin/env bash
# One-command correct setup for Unrealistic v1 on Ollama.
# Fetches the tested Modelfile from HuggingFace and builds the record:
#   curl -sL https://raw.githubusercontent.com/Soham-Anand/Unrealistic/main/Modelfile.hf -o /tmp/uv1.Modelfile && ollama create unrealistic-v1 -f /tmp/uv1.Modelfile && ollama run unrealistic-v1
# (This file documents the flow; the one-liner above is the actual installer.)
set -u
REPO="${HF_REPO:-SohamProgrammer/Unrealistic-v1}"
NAME="${MODEL_NAME:-unrealistic-v1}"
TMPMF="$(mktemp /tmp/uv1.Modelfile.XXXXXX)"
trap 'rm -f "$TMPMF"' EXIT
echo "Fetching Modelfile from https://huggingface.co/${REPO} ..."
curl -sL -m 120 "https://huggingface.co/${REPO}/resolve/main/Modelfile" -o "$TMPMF" || {
    echo "Download failed (network?)." >&2; exit 1
}
grep -q "^FROM " "$TMPMF" || { echo "Bad Modelfile content." >&2; exit 1; }
ollama create "$NAME" -f "$TMPMF" || exit 1
echo "Done. Run: ollama run $NAME"
