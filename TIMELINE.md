# Unrealistic v1 — Project Timeline

> Reconstructed 2026-09-14. A Sep-13 bulk filesystem event wiped most file
> mtimes (everything touched 08:53), so entries are marked:
> **solid** = git/log/session evidence · *~approximate* = inferred spans.

| Date | Event |
|---|---|
| **Jul 26** | Repo scaffolded (only git commit: `335e6c0`) |
| **Jul 29–30** | Phase 1 data builds (500M/5B), OWT train/eval bins |
| *~Jul 31–Aug 2* | *Phase 1 training (English foundations)* |
| **Aug 3** | Phase 2 (FineWeb-Edu 1.5G) + Phase 3 data |
| *~Aug 3–20* | *Stage A → B → C pretraining (incl. Stage-B SIGTERM resume at step 264k)* |
| **Aug 21** | Phase 4 data (code shell: python/c/js/math/prose/stackexchange) |
| *~Aug 22–28* | *Stage C → CoT math+code reasoning* |
| **Aug 29–31** | GSM8K + Stage A/B/C bins; phase5_a/b/c training runs |
| **Sep 9** | CoT training; Refresh (phase 6, 240M topper) data build |
| **Sep 12** | StarCoder GPL audit (0.6% copyleft, cleared); Realization rebuild (79.7M); SFT build + OpenAssistant fetch (31.7M total) |
| **Sep 13 AM** | SFT training (paused 339,040 overnight → resumed 07:10, mid-phase reset-optimizer noted) |
| **Sep 13 PM** | Probe/infer marathon (342500→344762, 14:05–17:14); Linkin Park 0/6; India/Pakistan boundary test; RFT handoff watcher armed 09:27 |
| **Sep 13 ~20:49** | SFT died at step 346,790 (1 short of max) → discovered trainer finals land at max−1; watcher hardened |
| **Sep 13 eve** | Phase 7b Wiki inserted (WDQS outage → Wikipedia category-walk 11.5K leads + Orca-Math 36K + GSM8K merge, 13.35M tok); chain watcher SFT→wiki→RFT |
| **Sep 14 AM** | Wiki DONE at 349,790 (~12:30); sampled-RFT attempts fail twice (31% then 10% noisy — model can't emit clean targets) |
| **Sep 14 PM** | Wiki damage measured (probes 22/40→17/40, free-gen collapse); recovery plan B adopted: 7c SFT-refresh launched, pristine anchors built (math+GK+LP+Mughals+Modi+games+NFS+slang+religion+cooking) |
| **Sep 14 eve** | DPO built from scratch (MLX, UltraFeedback MIT 4K pairs, β=0.5, smoke-tested); DPO ran 351790→352789 (win-rate 51/100); **7e super-diverse SFT launched 20:57** (smol-smoltalk + UltraChat + 20 synthetic task types + OA slice, 10.4M tok) → 354790 |
| **Next** | 8b anchors →355790 → `final_eval.sh` + probes → PT export → GGUF Q8/Q4 → HuggingFace → release reports |

## Step-count spine (absolute)

C 296,583 → CoT 306,583 → Refresh 330,790 → Realization 336,790 →
SFT 346,790 → Wiki 349,790 → SFT-refresh 351,789 → DPO 352,789 →
diverse-SFT 354,790 (planned) → anchors 355,790 (planned) = DONE.
