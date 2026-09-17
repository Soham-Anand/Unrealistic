# Unrealistic v1 — Eval Report (final `step_355789`)

Decode everywhere: temp 0.4, top_p 0.9, rep_penalty 1.25 (temp-0 collapses).
History: `evals/battery/` per-checkpoint mixes + probes, 277k→355789.

## Free generation (final_eval.sh sets)

| Set | Score | Notes |
|---|---|---|
| Math (15) | **10/15** | 8+7, 12×4, √49, 2+2, 64/8, 15−7, 7×8, 9×9, 144/12, Janet→7. Miss: 100/10, big-number word problems |
| GK (16) | **16/16** | Paris, oxygen, Jupiter, 1879, Delhi, Nehru, Taj, tiger, Rupee, 1947, Diwali, cold, seasons, east, DNA |
| Code (5) | **3/5** | add, factorial, print→4. Miss: is_prime, fibonacci |
| Culture (7) | **4/7** | LP band, Chester Bennington, born-1976, greetings work. Miss: Titanic, LP year (1986 vs 1996), Beatles detail |
| **Total** | **33/43 (77%)** | |

## First-token probes (40, any-correct)

| Domain | top-3 | top-10 |
|---|---|---|
| geography | 5/8 (62%) | 6/8 |
| math | 8/12 (67%) | 9/12 |
| science | 6/8 (75%) | 6/8 |
| history | 1/6 (17%) | 1/6 |
| culture | 0/3 | 2/3 |
| biology | 0/3 | 0/3 |
| **Total** | **20/40 (50%)** | 24/40 |

Trajectory: 20/40 (338000) → 17/40 (wiki damage) → 22/40 (7c repair) →
20/40 (final; science best-ever, geo/math −1 under 7e dilution).

## Standard benchmarks (`evals/benchmarks/final/results.json`)

MLX-direct runner (`scripts/bench.py`, no torch): MC sum/token-norm logps;
GSM8K-200 strict EM; MBPP-100 exec pass@1. Method notes in script header.

| Benchmark | n | v1 190M | Chance | SmolLM2-135M | SmolLM2-360M |
|---|---|---|---|---|---|
| ARC-Easy (val) | 570 | **34.0%** | 25% | 37.3 | 43.7 |
| PIQA | — | omitted (upstream dead) | 50% | 66.3 | 70.8 |
| HellaSwag | 500 | 25.2% | 25% | 40.9 | 52.1 |
| Winogrande | 1267 | 50.8% | 50% | — | — |
| TruthfulQA MC1 | 817 | 25.0% | ~25% | — | — |
| MMLU-easy (4 subsets) | 1042 | 23.0% | 25% | 29.3 | 32.8 (cloze) |
| GSM8K (strict EM) | 200 | **2.5%** | ~0% | 1.4 | 7.43 |
| MBPP (exec pass@1) | 100 | 0.0% | ~0% | — | — |

Peer figures: HuggingFaceTB model cards (Sep 2026). Methodologies differ
(theirs: 5-shot GSM8K, cloze MMLU, full-val; ours: temp-0 strict EM-200,
letter-scoring easy-subsets, val-570) — treat gaps as directional, not exact.
Reads: **ARC-Easy within striking distance of 135M (34.0 vs 37.3)**;
**GSM8K beats 135M (2.5 vs 1.4)** — the Orca-heavy diet shows. HellaSwag
and MMLU gaps reflect their 2–4T pretraining tokens vs our ~1.1B.

Raw: `evals/benchmarks/final/results.json`. Only ARC-Easy clears chance —
consistent with a 190M memorization model: first-token probes (20/40) and
constrained free-gen (33/43) show knowledge the argmax harness can't surface.
GSM8K/MBPP floors match predictions; generation quality lives in the battery.

## Limitations (ship with these)

Big-number word problems · deep code · biology 0/3 · multi-step reasoning ·
instruction-following beyond basics · non-English · entity/date hallucination
possible · Euro-history/culture soup outside drilled facts.
