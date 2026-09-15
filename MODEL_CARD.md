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

GGUF quants (Q8_0 recommended, Q4_K_M small-device, F16 reference) + a
local-Ollama `Modelfile` ship alongside. (Ollama library listing excluded:
publisher under 18; local use unaffected.)

## Limitations

Weak: large-number word problems, deep code, biology, multi-step reasoning,
instruction-following beyond basics, non-English. May hallucinate entities
and dates. 190M memorization model, not a reasoner.

## Data & license

Weights: Apache-2.0. Training-data attribution: `docs/DATA_REPORT.md`
(CC BY-SA content attributed; 0.6% GPL-family trace in code data measured
and documented; no verbatim reproduction possible at this scale).
