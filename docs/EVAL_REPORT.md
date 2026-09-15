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

| Benchmark | n | Score | Chance | Peer (SmolLM2) |
|---|---|---|---|---|
| ARC-Easy | 2376 | *running* | 25% | 43.7 (360M) |
| PIQA | 1838 | *running* | 50% | 70.8 |
| HellaSwag | 500 | *running* | 25% | 52.1 |
| Winogrande | 1267 | *running* | 50% | — |
| TruthfulQA MC1 | 817 | *running* | ~25% | — |
| MMLU-easy (4 subsets) | ~1000 | *running* | 25% | 32.8 (cloze) |
| GSM8K | 200 | *running* | ~0% | 7.43 (360M) |
| MBPP | 100 | *running* | ~0% | — |

*(Table fills when the background suite completes; results.json is authoritative.)*

## Limitations (ship with these)

Big-number word problems · deep code · biology 0/3 · multi-step reasoning ·
instruction-following beyond basics · non-English · entity/date hallucination
possible · Euro-history/culture soup outside drilled facts.
