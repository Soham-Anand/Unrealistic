#!/usr/bin/env bash
# One-command correct setup for Unrealistic v1 on Ollama.
# NOTE: plain `FROM hf.co/...` fails: HF serves GGUFs via Xet CDN and
# Ollama blocks the cross-host redirect. So we download first, then create
# with the tested Modelfile (FROM rewritten to the local file).
# Usage:  curl -sL <this-file-raw-URL> | bash
#         ollama run unrealistic-v1
set -u
REPO="${HF_REPO:-SohamProgrammer/Unrealistic-v1}"
NAME="${MODEL_NAME:-unrealistic-v1}"
GGUF="${GGUF_FILE:-unrealistic-v1-f16.gguf}"
BASE="https://huggingface.co/${REPO}/resolve/main"

echo "Downloading model (~380MB) ..."
curl -sL -m 1200 --retry 3 -o "$GGUF" "${BASE}/${GGUF}" || {
    echo "Model download failed (network?)." >&2; exit 1; }
[ -s "$GGUF" ] || { echo "Empty download." >&2; exit 1; }

echo "Fetching tested Modelfile ..."
MF="./.uv1.Modelfile.tmp"
trap 'rm -f "$MF" "$MF.bak"' EXIT
curl -sL -m 60 "${BASE}/Modelfile" -o "$MF" || {
    echo "Modelfile download failed." >&2; exit 1; }
grep -q "^TEMPLATE " "$MF" || { echo "Bad Modelfile content." >&2; exit 1; }
sed -i.bak "s|^FROM .*|FROM ./$GGUF|" "$MF" && rm -f "$MF.bak"
ollama create "$NAME" -f "$MF" || exit 1
echo "Done (kept ./$GGUF for re-use; delete it if short on disk). Run: ollama run $NAME"
