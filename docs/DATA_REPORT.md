# Unrealistic v1 — Data Report

Every bin: uint16 token stream (32K SPM), EOS-delimited docs.
Build scripts in `scripts/build_*.py`. Bins are git-ignored; recipes are the record.

## Phase data

| Phase | Train bin | Tokens | Contents |
|---|---|---|---|
| Stage C | stage_c_train | ~25M | foundations mix |
| CoT 5 | cot_train | 39.2M | 8-src merge: arithmetic/cot-scratchpad/worked/problems/algebra/combinatorics/verification/clean |
| Refresh 6 | train | 240M | fwe60/wiki50/code50/owt30/slim20 + ARITH30 + COT_KEEP20 |
| Realization | realization_train | 79.7M | 9 judgment tasks (negation/contradiction/correction × math/GK/code) + 24M real recovery |
| SFT 7 | train | 28.6M (55M file) | 25M synthetic (math-CoT, India 10-Q, reasoning, creative, code) + **6M OpenAssistant** (oasst1+oasst2, en, rank-0, quality≥0.5, toxicity<0.2: 1,810 chat + 5,217 remembrance) |
| SFT eval | eval | 3.1M | held-out |
| Wiki 7b | wiki_train | 13.35M | 11.5K Wikipedia leads (bands/singers/actors/films/novels/math people) + Orca-Math 36,779 + GSM8K 1.45M merged + constants |
| DPO 7d | dpo_train.npz | 4,000 pairs | UltraFeedback binarized (last-assistant-turn pairs, ≤512 tok) |
| div-SFT 7e | sft2_train | ~10.4M | smol-smoltalk 14,282 (Apache-native subsets) + UltraChat 15,000 + ~20 synthetic task types + OA-format slice 2M + anchors×3 |
| Anchors 8b | anchor_train | 32,261 tok (2,312 docs) | 214 hand-verified pins: 21 math (calculator-checked) + GK + Linkin Park + Mughals + Modi + games/NFS + slang/celebs + religion + cooking + CS/science + greetings/persona |

## StarCoder GPL audit (Sep 12)

Full decode audit of code slices: 135,658 docs, **~807 true copyleft (0.6%)**.
No verbatim reproduction possible at 190M (verified by inspection). Risk: nil.
Tooling: `scripts/audit_starcoder_slices.py`, `scripts/classify_licenses.py`.
Recommendation carried to Alpha: license-filter code (Stack v2 MIT/Apache/BSD).

## Notes

- Wikipedia/StackExchange-derived content (CC BY-SA): attributed per
  `DATASETS_LICENSES.md`; industry-norm posture (attribution + report).
- Synthetic sentences (anchors, math variety, verbalizations): own templates,
  CC0-equivalent; facts are not copyrightable.
- Decontamination: HumanEval/MBPP solutions filtered from code data
  (StarCoder2-style rules in pipeline).
