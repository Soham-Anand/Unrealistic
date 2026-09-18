---
language: en
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
- 190m
- from-scratch
- mlx
- tiny
---

# Unrealistic v1 (190M)

Decoder-only transformer LM (189,748,992 params) trained from scratch on a
MacBook Air M4 with MLX. LLaMA-family: 14 layers, hidden 768, FFN 4096,
8 heads, RoPE θ=10k, RMSNorm, SiLU/SwiGLU, tied embeddings, bf16,
32K SentencePiece vocab, 1024 context.

## Training

~356K steps / ~1.1B tokens. Chain: Stage C → CoT (math+code reasoning) →
Refresh (240M topper) → Realization (judgment tasks) → SFT (31.7M incl.
OpenAssistant) → Wiki entities+math (13.35M) → SFT-refresh → DPO
(UltraFeedback, β=0.5) → diverse-SFT (smol-smoltalk + UltraChat + task
synthetics) → pristine anchors. AdamW throughout. Full log:
`docs/TRAINING_REPORT.md`.

## Evaluation (final `step_355789`)

- Math free-gen 10/15 · GK 16/16 · Code 3/5 · Culture 4/7
- First-token probes 20/40 top-3
- Standard benchmarks: `evals/benchmarks/final/results.json`
  (ARC-Easy, PIQA, HellaSwag-sample, Winogrande, TruthfulQA-MC1,
  MMLU-easy, GSM8K-sample, MBPP-sample)
- Full detail: `docs/EVAL_REPORT.md`

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("REPO_ID")
model = AutoModelForCausalLM.from_pretrained("REPO_ID")
```

On Apple Silicon the native MLX checkpoint + `scripts/spot_infer.py` is fastest.
Suggested decode: temperature 0.4, top_p 0.9, repetition_penalty 1.25.

Raw GGUF (`unrealistic-v1-f16.gguf`, exact weights, no quantization) + a
local-Ollama `Modelfile` ship alongside. (Ollama library listing excluded:
publisher under 18; local use unaffected.)

## Ollama (recommended setup — required for correct behavior)

Direct `ollama run hf.co/...` uses wrong sampling defaults (temp 0.8) and may
ignore the embedded template. One-command correct setup (fetches the tested
`Modelfile`, builds the record — no manual config):

```bash
curl -sL https://huggingface.co/SohamProgrammer/Unrealistic-v1/resolve/main/setup-ollama.sh | bash
ollama run unrealistic-v1
```
(Manual alternative: download `Modelfile` from Files, `ollama create unrealistic-v1 -f Modelfile`.)
Note: `FROM hf.co/...` Modelfiles fail on this repo because HF serves large
files via Xet CDN, whose cross-host redirects Ollama blocks. The installer
above downloads first and creates locally — the supported path.

This pins temperature 0.4, top_p 0.9, repeat_penalty 1.25, ctx 1024, the
`User:/Assistant:` chat template, and `User:` stop — the exact configuration
the model was validated with. No system prompt (untrained distribution).
Phone users: re-download the Q4 file (post-fix bytes with BOS disabled).

## Limitations

Weak: large-number word problems, deep code, biology, multi-step reasoning,
instruction-following beyond basics, non-English. May hallucinate entities
and dates. 190M memorization model, not a reasoner.

## Data & license

Weights: Apache-2.0. Training-data attribution: `docs/DATA_REPORT.md`
(CC BY-SA content attributed; 0.6% GPL-family trace in code data measured
and documented; no verbatim reproduction possible at this scale).
