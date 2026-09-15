# Unrealistic v1 — Training Report (MacBook Air M4, 16GB, MLX)

Trained direct via `scripts/train.py` (the `train_manager.py` chain script
was retired mid-project; `training/status.json` is stale). Monitored via
stdout logs. ~18h/day uptime. Full history: `TIMELINE.md`.

## Phase table (absolute steps)

| Phase | Range | Data | LR / seq | Notes |
|---|---|---|---|---|
| Stage C | →296,583 | stage_c 51M | —/1024 | foundations end |
| CoT | →306,583 | cot 39.2M (8-src merge) | —/512 | free-gen math 0/15 trough; probes held 67–75% |
| Refresh | →330,790 | 240M topper (fwe/wiki/code/owt/slim + ARITH30 + COT_KEEP20) | —/1024 | prose shell rebuild |
| Realization | →336,790 | 79.7M (9 judgment tasks incl. GK/code negation) | —/1024 | truth-judgment |
| SFT | →346,790 | 31.7M (25M synth + 6M OpenAssistant EN/rank-0) | 5e-05/1024 | format lands: Paris/Jupiter/oxygen |
| Wiki 7b | →349,790 | 13.35M (11.5K WP leads + Orca-36K + GSM8K) | 5e-05/1024 | **damaged model**: probes 22→17/40, Orca-ramble + `<<>>` bleed |
| SFT-refresh 7c | →351,789 | phase7 31.7M again | 5e-05/1024 | repair: probes back to 22/40, geo 7/8 best |
| DPO 7d | →352,789 | UltraFeedback MIT 4K pairs, β=0.5, lr 1e-6 | 1e-6/512 | from-scratch MLX DPO; win-rate 45→51/100 |
| diverse-SFT 7e | →354,790 | 10.4M (smol-smoltalk 14K + UltraChat 15K + 20 synthetic task types + OA slice) | 5e-05/1024 | chattiness installed; late dilution |
| Anchors 8b | →355,790 | 32K pristine verified pins (math/GK/LP/Mughals/Modi/games/culture/CS/greetings) | 5e-05/512 | final lock |

Totals: ~356K steps, ~1.1B tokens, ~6 days wall (interrupted).

## Key events

1. **Stage-B SIGTERM resume** (step 264k) — resumed clean, no loss spike.
2. **Manager → direct-train.py switch** — status.json abandoned; logs authoritative.
3. **CoT trough** — free-gen math collapsed to fraction/Step-1 bleed while
   probes held: knowledge/routing dissociation (recurs at every phase start).
4. **SFT died at 346,790** (1 short of max) — discovered finals land at
   max_steps−1; all gates hardened to `≥ FINAL−1` + actual-latest-dir.
5. **Stale-detached log** (phase7, truncated under live process) — 5K steps
   of loss lines voided; monitoring moved to checkpoint dirs. Logs now
   append-only, rotated only between runs.
6. **WDQS outage** — RFT sampling data source pivoted to Wikipedia
   category-walk + HF math datasets.
7. **Sampled-RFT failed twice** (31% then 10% noisy accepts: `####`/`<<>>`
   bleed) — replaced with deterministic verified anchors.
8. **Wiki damage + 7c repair** — measured −5 probe points, restored +5.
9. **SFT-final weights lost** (baseline overwrite during watcher restart +
   keep_last pruning) — behavior record survives in batteries; continuity
   unaffected. Documented as process failure.
10. **Reset-optimizer discipline** — reset at phase starts only; mid-phase
    resumes keep Adam (one accidental mid-SFT reset recovered without harm).

## Loss landmarks

SFT start ~2.3 → 0.13–0.21 stable · Wiki start 2.36 (Orca shock) · 7c start
0.70→0.18 (instant recognition) · DPO loss 0–8.7 (sigmoid saturation noise) ·
7e 3.6→0.12 (diversity → chat settling) · 8b 1.1→0.003 (memorization).
