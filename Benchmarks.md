# Unrealistic v1 — Benchmark Results (final `step_355789`)

Runner: `scripts/bench.py` (MLX-direct, no torch). MC = continuation-logp
argmax (sum, token-norm where noted). GSM8K = open generation, strict EM.
MBPP = open generation, subprocess-exec pass@1. Raw: `evals/benchmarks/final/results.json`.

## Standard benchmarks

| Benchmark | n | v1 190M | Chance | SmolLM2-135M | SmolLM2-360M |
|---|---|---|---|---|---|
| ARC-Easy (val) | 570 | **34.0%** (194) | 25% | 37.3 | 43.7 |
| HellaSwag (sample) | 500 | 25.2% (126) | 25% | 40.9 | 52.1 |
| Winogrande (xl val) | 1267 | 50.8% (644) | 50% | — | — |
| TruthfulQA MC1 | 817 | 25.0% (204) | ~25% | — | — |
| MMLU-easy (elem-math, HS-bio/chem/phys) | 1042 | 23.0% (240) | 25% | 29.3 | 32.8 (cloze) |
| GSM8K (strict EM, temp-0) | 200 | **2.5%** (5) | ~0% | 1.4 | 7.43 |
| MBPP (exec pass@1) | 100 | 0.0% (0) | ~0% | — | — |
| PIQA | — | omitted (upstream 404, all mirrors dead) | 50% | 66.3 | 70.8 |

Peer figures: HuggingFaceTB model cards. Methods differ (theirs: 5-shot
GSM8K, cloze MMLU, full-val; ours as above) — directional, not exact.

## Custom batteries (final_eval.sh, temp 0.4/0.9/1.25)

| Set | Score |
|---|---|
| Math free-gen (15) | 10/15 |
| General knowledge (16) | 16/16 |
| Code (5) | 3/5 |
| Culture + greetings (7) | 4/7 |
| **Total** | **33/43 (77%)** |

Raw: `evals/battery/final_step_355789/`.

## First-token probes (40)

| Domain | top-3 | top-10 |
|---|---|---|
| geography | 5/8 | 6/8 |
| math | 8/12 | 9/12 |
| science | 6/8 | 6/8 |
| history | 1/6 | 1/6 |
| culture | 0/3 | 2/3 |
| biology | 0/3 | 0/3 |
| **Total** | **20/40 (50%)** | 24/40 |

Raw: `evals/battery/final_step_355789/probe.jsonl`. History: per-checkpoint
mixes + probes in `evals/battery/`, 277k→355789.
