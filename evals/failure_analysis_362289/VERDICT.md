# Pre-Release Failure Analysis — Verdict (362289 primary, 355789 control)

Frozen protocol: temp 0.0, fixed seed/template/tokens, single-turn, no history.
Executed 2026-09-16/17. No training occurred at any point (verified: no
trainer processes, no new checkpoints, forward-only code).

## Pass 1 — Deterministic transitions (37 shared prompts scored)

| Class | Count | Members |
|---|---|---|
| correct→correct | 30 | core math, Paris/Jupiter/Taj/Nehru/1947/Diwali, code basics |
| wrong→correct (repair) | 2 | Einstein→1879, is_prime |
| correct→wrong (regression) | 3 | 7×8, LP lead singer, Chester bio |
| wrong→wrong (persistent) | 2 | In the End, Titanic |

Probes top-3: 355789 = 20/40 → 362289 = 19/40 (science 6→5 the only move;
geography 5, history 1, math 8, bio/culture 0 both).
Net: knowledge preserved; LP-entity routing regressed under greedy.

## Pass 2 — Exact match (temp 0.0)

| Domain | 355789 | 362289 |
|---|---|---|
| culture/100 | 2% | 3% |
| science/10 | 80% | 70% |
| gk/10 | 100% | 80% |
| arithmetic/300 | 6.7% | **34.3%** |

Headline: arithmetic 7%→34% is the genuine 7f–8b repair signal. Culture floor
reflects greedy harshness on multi-word answers (temp-0.4 chat recovers
Chester/Paris — probes agree the knowledge exists). GK slip 100→80% is
sampling-thin (n=10) plus currency-attribution noise (below).

## Pass 3 — Stability (temp 0.0, 5 orders, 23 questions)

Classes: STABLE_CORRECT 17, STABLE_WRONG 4, ASSOCIATION_INTRUSION 2.
Intrusions: YEAR_1945 ×10 (LP question answered with WWII bleed — REAL),
YEAR_1947 ×5 (flagged on the 1947 question itself — detector artifact).

### Post-hoc matcher audit (methodological honesty)
Three "STABLE_WRONG" verdicts are scorer false negatives, not model failures:
currency→"Rupee (INR)" ✓, water→contains "oxygen (H2O)" ✓,
photosynthesis→"energy and oxygen" ✓ — all correct answers embedded
mid-sentence that prefix-only matching rejected. True stable-wrong: 1
(light→Earth). True intrusions: 1 family (1945→LP).
Caveat also stands: S_q≡1.0 at temp-0 by construction (deterministic argmax);
stochastic stability needs a temp-0.4 rerun (proposed follow-up, NOT run —
freeze respected).

## Failure map (for Alpha)

1. **LP-entity routing**: Chester/Beatles/Michael attractors compete; greedy
   collapses to years (1945). Needs mass, not prompts.
2. **Year attractors** (1947/1996/2014/1879): quantified intruders; anchor
   over-repetition is implicated — future work: frequency balancing.
3. **Word-problem arithmetic**: method present, arithmetic fragile (car/boxes).
4. **Biology/history depth**: absent (0/3 bio probes) — needs pretraining tokens.
5. **Greedy vs sampled gap**: knowledge is sampling-accessible but not
   argmax-accessible — routing, not storage.
6. **Scorer lesson**: substring/contains matching required; diacritic folding
   required; intrusion detectors must exempt asked-about entities.

## Verdict
Ship-blockers for factual Q&A: LP stability, year intrusions, word problems.
Shippable as-is: core math facts, GK anchors, code basics, greetings.
Recommended Alpha interventions in order: massed entity coverage →
frequency-balanced anchors → reasoning-heavy math → biology pretraining.
