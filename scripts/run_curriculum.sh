#!/bin/bash
# Curriculum runner: Phase 4 → Phase 5 (Prabhakar 3-stage) → Phase 6 (Refresh) → Phase 7 (SFT) → Phase 8 (RFT)
set -u

log() { echo "[curriculum] $(date '+%H:%M:%S') $1" | tee -a curriculum.log; }
die() { log "FATAL: $1"; exit 1; }
# Keep each baselines/<folder> to 1 latest checkpoint (SSD safety)
prune_baseline() {
  local dir="$1"
  local cnt=$(ls -1 "$dir" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$cnt" -gt 1 ]; then
    ls -1 "$dir" 2>/dev/null | sort -t_ -k2 -n | sed '$d' | while read sub; do
      [ -z "$sub" ] && continue
      rm -rf "${dir}/${sub}"
      log "Pruned old baseline ${dir}/${sub} (keep 1)"
    done
  fi
}

# ─── Gate evaluation helper ──────────────────────────────────────────
run_gate() {
    local phase=$1 step=$2
    log "Running gate evaluation for ${phase} at step ${step}..."

    PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/probe_first_token.py \
        --checkpoint "checkpoints/step_${step}" \
        --out "evals/${phase}_gate_${step}.jsonl" \
        >> "evals/${phase}_gate.log" 2>&1
    [ $? -ne 0 ] && die "Gate probe eval failed for ${phase}"

    PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/spot_infer.py \
        --checkpoint "checkpoints/step_${step}" \
        > "evals/${phase}_spot_${step}.txt" 2>&1
    [ $? -ne 0 ] && die "Gate spot inference failed for ${phase}"

    log "Gate evaluation saved to evals/${phase}_gate_${step}.jsonl"
}

# ─── Find latest checkpoint step number ──────────────────────────────
latest_step() {
    ls -td checkpoints/step_* 2>/dev/null | head -1 | sed 's/.*step_//'
}

# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: Code + math + technical (continue to 246,584)
# ═══════════════════════════════════════════════════════════════════════
PH4_START=$(latest_step)
log "════════════════════════════════════════════════════════════"
log "PHASE 4 RESUMING (step ${PH4_START} → 246,584, code+math+technical)"
log "════════════════════════════════════════════════════════════"

[ -f data/phase4/train.bin ] || die "Phase 4 data not found."
log "Phase 4 data: $(du -h data/phase4/train.bin | cut -f1)"

PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/train_manager.py \
    --config configs/phase4_config.json \
    --model-config configs/model_config.json \
    --phase "phase4" \
    >> training_phase4.log 2>&1
[ $? -ne 0 ] && die "Phase 4 training failed"

PH4_STEP=$(latest_step)
log "Phase 4 complete at step ${PH4_STEP}"

run_gate "phase4" "${PH4_STEP}"

mkdir -p baselines/post_phase4/step_${PH4_STEP}
cp checkpoints/step_${PH4_STEP}/*.npz baselines/post_phase4/step_${PH4_STEP}/
cp checkpoints/step_${PH4_STEP}/*.json baselines/post_phase4/step_${PH4_STEP}/ 2>/dev/null
log "Phase 4 checkpoint archived to baselines/post_phase4/step_${PH4_STEP}/"
prune_baseline baselines/post_phase4

# ═══════════════════════════════════════════════════════════════════════
# PHASE 5: Prabhakar — 3-stage Math Curriculum (50,000 steps: 246,584 → 296,584)
#   Stage A Foundations (15K) : arithmetic, algebra, simplify, clean
#   Stage B Advanced    (15K) : number theory, geometry/trig, combinatorics, adv algebra
#   Stage C Reasoning   (20K) : GSM8K, verification, error, code + anti-forgetting
# ═══════════════════════════════════════════════════════════════════════
log "════════════════════════════════════════════════════════════"
log "PHASE 5 STARTING (Prabhakar 3-stage curriculum → 296,584)"
log "════════════════════════════════════════════════════════════"

# Build Phase 5 curriculum data (3 stages, ~100M total: synthetic math + GSM8K + anti-forgetting)
if [ ! -f data/phase5/stage_a_train.bin ]; then
    log "Building Prabhakar curriculum data (synthetic math + GSM8K + anti-forgetting)..."
    PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/build_phase5_curriculum.py >> build_phase5.log 2>&1
    [ $? -ne 0 ] && die "Phase 5 curriculum data build failed"
    log "Phase 5 curriculum data built"
else
    log "Phase 5 curriculum data already present"
fi
for st in a b c; do
    log "Stage ${st} data: $(du -h data/phase5/stage_${st}_train.bin | cut -f1)"
done

run_prabhakar_stage() {
    local st=$1 cfg=$2 tag=$3
    log "── Prabhakar Stage ${st} (${tag}) starting at step $(latest_step) ──"
    PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/train_manager.py \
        --config "${cfg}" \
        --model-config configs/model_config.json \
        --reset-optimizer \
        --phase "prabhakar_stage_${st}" \
        >> "training_phase5_${st}.log" 2>&1
    [ $? -ne 0 ] && die "Prabhakar Stage ${st} training failed"
    local stp
    stp=$(latest_step)
    log "Prabhakar Stage ${st} complete at step ${stp}"
    run_gate "phase5_${st}" "${stp}"
    mkdir -p baselines/post_phase5_${st}/step_${stp}
    cp checkpoints/step_${stp}/*.npz baselines/post_phase5_${st}/step_${stp}/
    cp checkpoints/step_${stp}/*.json baselines/post_phase5_${st}/step_${stp}/ 2>/dev/null
    log "Stage ${st} checkpoint archived to baselines/post_phase5_${st}/step_${stp}/"
    prune_baseline baselines/post_phase5_${st}
}

run_prabhakar_stage a configs/phase5a_config.json "Foundations"
run_prabhakar_stage b configs/phase5b_config.json "Advanced"
run_prabhakar_stage c configs/phase5c_config.json "Reasoning"

PH5_STEP=$(latest_step)
log "Prabhakar (Phase 5) complete at step ${PH5_STEP}"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 5.x-CoT: CoT-heavy math + code (10,000 steps: 296,584 → 306,584)
#   Teaches the 190M model to COMPUTE step-by-step (chain-of-thought /
#   scratchpad): long mul/div, column add, powers, worked arithmetic,
#   word problems, verification, error-ID, + code. Builds the procedural
#   "actual thinking" layer on top of Stage C, before general refresh.
# ═══════════════════════════════════════════════════════════════════════
log "════════════════════════════════════════════════════════════"
log "PHASE 5.x-CoT STARTING (step ${PH5_STEP} → 306,584, CoT-heavy math+code)"
log "════════════════════════════════════════════════════════════"

# Build CoT-heavy dataset if needed (~40M tokens: scratchpad/worked/problem/verification/error/code/gsm8k)
if [ ! -f data/phase5/cot_train.bin ]; then
    log "Building CoT-heavy math+code dataset..."
    PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/build_phase5_cot.py >> build_phase5_cot.log 2>&1
    [ $? -ne 0 ] && die "CoT-heavy data build failed"
    log "CoT-heavy dataset built"
else
    log "CoT-heavy dataset already present"
fi
log "CoT-heavy data: $(du -h data/phase5/cot_train.bin | cut -f1)"

# Pre-CoT baseline (before CoT, = post-Stage C)
mkdir -p baselines/pre_phase5_cot/step_${PH5_STEP}
cp checkpoints/step_${PH5_STEP}/*.npz baselines/pre_phase5_cot/step_${PH5_STEP}/ 2>/dev/null || log "WARNING: pre-CoT checkpoint step_${PH5_STEP} not found (pruned)"
cp checkpoints/step_${PH5_STEP}/*.json baselines/pre_phase5_cot/step_${PH5_STEP}/ 2>/dev/null || true
log "Pre-CoT baseline archived to baselines/pre_phase5_cot/step_${PH5_STEP}/ (post-Stage C)"
prune_baseline baselines/pre_phase5_cot

PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/train_manager.py \
    --config configs/phase5_cot_config.json \
    --model-config configs/model_config.json \
    --reset-optimizer \
    --phase "prabhakar_cot_math" \
    >> training_phase5_cot.log 2>&1
[ $? -ne 0 ] && die "CoT-heavy training failed"

PHC_STEP=$(latest_step)
log "CoT-heavy complete at step ${PHC_STEP}"

run_gate "phase5_cot" "${PHC_STEP}"

mkdir -p baselines/post_phase5_cot/step_${PHC_STEP}
cp checkpoints/step_${PHC_STEP}/*.npz baselines/post_phase5_cot/step_${PHC_STEP}/
cp checkpoints/step_${PHC_STEP}/*.json baselines/post_phase5_cot/step_${PHC_STEP}/ 2>/dev/null
log "CoT-heavy checkpoint archived to baselines/post_phase5_cot/step_${PHC_STEP}/"
prune_baseline baselines/post_phase5_cot

# ═══════════════════════════════════════════════════════════════════════
# PHASE 6: Refresh (24,207 steps: 306,584 → 330,791)
# ═══════════════════════════════════════════════════════════════════════
log "════════════════════════════════════════════════════════════"
log "PHASE 6 STARTING (step ${PHC_STEP} → 330,791, mixed refresh)"
log "════════════════════════════════════════════════════════════"

# Build Phase 6 data if needed (200M tokens: FWE/Wiki/Code/OWT/Slim)
if [ ! -f data/phase6/train.bin ]; then
    log "Building Phase 6 refresh data (200M tokens)..."
    PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/build_phase6.py >> build_phase6.log 2>&1
    [ $? -ne 0 ] && die "Phase 6 data build failed"
    log "Phase 6 data built"
else
    log "Phase 6 data already present"
fi
log "Phase 6 data: $(du -h data/phase6/train.bin | cut -f1)"

# Pre-Refresh baseline (before Refresh, = post-CoT)
mkdir -p baselines/pre_phase6/step_${PHC_STEP}
cp checkpoints/step_${PHC_STEP}/*.npz baselines/pre_phase6/step_${PHC_STEP}/ 2>/dev/null || log "WARNING: pre-Refresh checkpoint step_${PHC_STEP} not found"
cp checkpoints/step_${PHC_STEP}/*.json baselines/pre_phase6/step_${PHC_STEP}/ 2>/dev/null || true
log "Pre-Refresh baseline archived to baselines/pre_phase6/step_${PHC_STEP}/ (post-CoT)"
prune_baseline baselines/pre_phase6

PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/train_manager.py \
    --config configs/phase6_refresh_config.json \
    --model-config configs/model_config.json \
    --reset-optimizer \
    --phase "phase6_refresh" \
    >> training_phase6.log 2>&1
[ $? -ne 0 ] && die "Phase 6 training failed"

PH6_STEP=$(latest_step)
log "Phase 6 complete at step ${PH6_STEP}"

run_gate "phase6" "${PH6_STEP}"

mkdir -p baselines/post_phase6/step_${PH6_STEP}
cp checkpoints/step_${PH6_STEP}/*.npz baselines/post_phase6/step_${PH6_STEP}/
cp checkpoints/step_${PH6_STEP}/*.json baselines/post_phase6/step_${PH6_STEP}/ 2>/dev/null
log "Phase 6 checkpoint archived to baselines/post_phase6/step_${PH6_STEP}/"
prune_baseline baselines/post_phase6

# ═══════════════════════════════════════════════════════════════════════
# PHASE 5.x-Realization: right after Refresh (6,000 steps: 330,791 → 336,791)
#   Teaches the model to JUDGE truth: negation, contradiction, correction
#   across math + science, plus real Code/English/GK recovery. Sits after
#   the general refresh so truth-judging lands on a refreshed model,
#   immediately before SFT/RFT reasoning alignment.
# ═══════════════════════════════════════════════════════════════════════
log "════════════════════════════════════════════════════════════"
log "PHASE 5.x-REALIZATION STARTING (step ${PH6_STEP} → 336,791)"
log "════════════════════════════════════════════════════════════"

if [ ! -f data/phase6/realization_train.bin ]; then
    log "Building Realization dataset (~30M tokens: judgement + code/english/GK)..."
    PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/generate_realization_data.py >> build_realization.log 2>&1
    [ $? -ne 0 ] && die "Realization data build failed"
    log "Realization dataset built"
else
    log "Realization dataset already present"
fi
log "Realization data: $(du -h data/phase6/realization_train.bin | cut -f1)"

# Pre-Realization baseline (before Realization, = post-Refresh)
mkdir -p baselines/pre_realization/step_${PH6_STEP}
cp checkpoints/step_${PH6_STEP}/*.npz baselines/pre_realization/step_${PH6_STEP}/ 2>/dev/null || log "WARNING: pre-Realization checkpoint step_${PH6_STEP} not found"
cp checkpoints/step_${PH6_STEP}/*.json baselines/pre_realization/step_${PH6_STEP}/ 2>/dev/null || true
log "Pre-Realization baseline archived to baselines/pre_realization/step_${PH6_STEP}/ (post-Refresh)"
prune_baseline baselines/pre_realization

PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/train_manager.py \
    --config configs/phase5_realization_config.json \
    --model-config configs/model_config.json \
    --reset-optimizer \
    --phase "prabhakar_realization" \
    >> training_realization.log 2>&1
[ $? -ne 0 ] && die "Realization training failed"

PHZ_STEP=$(latest_step)
log "Realization complete at step ${PHZ_STEP}"

run_gate "realization" "${PHZ_STEP}"

mkdir -p baselines/post_realization/step_${PHZ_STEP}
cp checkpoints/step_${PHZ_STEP}/*.npz baselines/post_realization/step_${PHZ_STEP}/
cp checkpoints/step_${PHZ_STEP}/*.json baselines/post_realization/step_${PHZ_STEP}/ 2>/dev/null
log "Realization checkpoint archived to baselines/post_realization/step_${PHZ_STEP}/"
prune_baseline baselines/post_realization

# ═══════════════════════════════════════════════════════════════════════
# PHASE 7: SFT (10,000 steps: 336,791 → 346,791)
# ═══════════════════════════════════════════════════════════════════════
log "════════════════════════════════════════════════════════════"
log "PHASE 7 STARTING (step ${PHZ_STEP} → 346,791, SFT)"
log "════════════════════════════════════════════════════════════"

# Build Phase 7 SFT data if needed (~8M tokens)
if [ ! -f data/phase7/train.bin ]; then
    log "Building Phase 7 SFT data..."
    PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/build_sft.py >> build_sft.log 2>&1
    [ $? -ne 0 ] && die "Phase 7 data build failed"
    log "Phase 7 SFT data built"
else
    log "Phase 7 SFT data already present"
fi
log "Phase 7 data: $(du -h data/phase7/train.bin | cut -f1)"

# Pre-SFT baseline (before SFT, = post-Realization)
mkdir -p baselines/pre_phase7/step_${PHZ_STEP}
cp checkpoints/step_${PHZ_STEP}/*.npz baselines/pre_phase7/step_${PHZ_STEP}/ 2>/dev/null || log "WARNING: pre-SFT checkpoint step_${PHZ_STEP} not found"
cp checkpoints/step_${PHZ_STEP}/*.json baselines/pre_phase7/step_${PHZ_STEP}/ 2>/dev/null || true
log "Pre-SFT baseline archived to baselines/pre_phase7/step_${PHZ_STEP}/ (post-Realization)"
prune_baseline baselines/pre_phase7

PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/train_manager.py \
    --config configs/phase7_sft_config.json \
    --model-config configs/model_config.json \
    --reset-optimizer \
    --phase "phase7_sft" \
    >> training_phase7.log 2>&1
[ $? -ne 0 ] && die "Phase 7 training failed"

PH7_STEP=$(latest_step)
log "Phase 7 complete at step ${PH7_STEP}"

run_gate "phase7" "${PH7_STEP}"

mkdir -p baselines/post_phase7/step_${PH7_STEP}
cp checkpoints/step_${PH7_STEP}/*.npz baselines/post_phase7/step_${PH7_STEP}/
cp checkpoints/step_${PH7_STEP}/*.json baselines/post_phase7/step_${PH7_STEP}/ 2>/dev/null
log "Phase 7 checkpoint archived to baselines/post_phase7/step_${PH7_STEP}/"
prune_baseline baselines/post_phase7

# ═══════════════════════════════════════════════════════════════════════
# PHASE 7b: WIKI — CC0 Wikidata entity injection (346,791 → 349,791)
# Live Wikidata (CC0) template-verbalized + GSM8K/Orca-Math/SVAMP math.
# Fixes tail-entity gaps (bands/musicians/films/books/places), math kept fresh.
# ═══════════════════════════════════════════════════════════════════════
log "════════════════════════════════════════════════════════════"
log "PHASE 7b STARTING (Wiki: CC0 entities + math, step ${PH7_STEP} → 349,791)"
log "════════════════════════════════════════════════════════════"

if [ ! -f data/phase7b/wiki_train.bin ]; then
    log "Building Phase 7b data (WDQS CC0 + math datasets)..."
    PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/build_wiki.py \
        >> build_phase7b.log 2>&1
    [ $? -ne 0 ] && die "Phase 7b data build failed"
fi
log "Phase 7b data: $(du -h data/phase7b/wiki_train.bin | cut -f1)"

mkdir -p baselines/pre_phase7b/step_${PH7_STEP}
cp checkpoints/step_${PH7_STEP}/*.npz baselines/pre_phase7b/step_${PH7_STEP}/ 2>/dev/null || log "WARNING: pre-Wiki checkpoint step_${PH7_STEP} not found"
cp checkpoints/step_${PH7_STEP}/*.json baselines/pre_phase7b/step_${PH7_STEP}/ 2>/dev/null || true
log "Pre-Wiki baseline archived to baselines/pre_phase7b/step_${PH7_STEP}/ (post-SFT)"
prune_baseline baselines/pre_phase7b

PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/train_manager.py \
    --config configs/phase7b_wiki_config.json \
    --model-config configs/model_config.json \
    --reset-optimizer \
    --phase "phase7b_wiki" \
    >> training_phase7b.log 2>&1
[ $? -ne 0 ] && die "Phase 7b training failed"

PH7B_STEP=$(latest_step)
log "Phase 7b complete at step ${PH7B_STEP}"

run_gate "phase7b" "${PH7B_STEP}"

mkdir -p baselines/post_phase7b/step_${PH7B_STEP}
cp checkpoints/step_${PH7B_STEP}/*.npz baselines/post_phase7b/step_${PH7B_STEP}/
cp checkpoints/step_${PH7B_STEP}/*.json baselines/post_phase7b/step_${PH7B_STEP}/ 2>/dev/null
log "Phase 7b checkpoint archived to baselines/post_phase7b/step_${PH7B_STEP}/"
prune_baseline baselines/post_phase7b

# ═══════════════════════════════════════════════════════════════════════
# PHASE 7c: SFT-REFRESH — repair Wiki-phase damage (349,790 → 351,790)
# Wiki 7b regressed probes 22/40 -> 17/40 + wrecked chat format (Orca-ramble,
# <<>> artifacts). Refresh on proven SFT data restores format + facts.
# ═══════════════════════════════════════════════════════════════════════
log "════════════════════════════════════════════════════════════"
log "PHASE 7c STARTING (SFT-refresh: repair, step ${PH7B_STEP} → 351,790)"
log "════════════════════════════════════════════════════════════"

mkdir -p baselines/pre_phase7c/step_${PH7B_STEP}
cp checkpoints/step_${PH7B_STEP}/*.npz baselines/pre_phase7c/step_${PH7B_STEP}/ 2>/dev/null || log "WARNING: pre-7c checkpoint step_${PH7B_STEP} not found"
cp checkpoints/step_${PH7B_STEP}/*.json baselines/pre_phase7c/step_${PH7B_STEP}/ 2>/dev/null || true
log "Pre-7c baseline archived to baselines/pre_phase7c/step_${PH7B_STEP}/ (post-Wiki)"
prune_baseline baselines/pre_phase7c

PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/train_manager.py \
    --config configs/phase7c_sftrefresh_config.json \
    --model-config configs/model_config.json \
    --reset-optimizer \
    --phase "phase7c_sftrefresh" \
    >> training_phase7c.log 2>&1
[ $? -ne 0 ] && die "Phase 7c training failed"

PH7C_STEP=$(latest_step)
log "Phase 7c complete at step ${PH7C_STEP}"

run_gate "phase7c" "${PH7C_STEP}"

mkdir -p baselines/post_phase7c/step_${PH7C_STEP}
cp checkpoints/step_${PH7C_STEP}/*.npz baselines/post_phase7c/step_${PH7C_STEP}/
cp checkpoints/step_${PH7C_STEP}/*.json baselines/post_phase7c/step_${PH7C_STEP}/ 2>/dev/null
log "Phase 7c checkpoint archived to baselines/post_phase7c/step_${PH7C_STEP}/"
prune_baseline baselines/post_phase7c

# ═══════════════════════════════════════════════════════════════════════
# PHASE 8b: ANCHOR — pristine synthesized targets (replaces sampled RFT)
# Sampled RFT failed twice (31% noisy, 10% noisy: model can't emit clean
# targets post-Wiki). Anchors are deterministic + calculator/fact-verified:
# 20 math prompts + GK + Linkin Park + code, ~72 docs x12.
# ═══════════════════════════════════════════════════════════════════════
log "════════════════════════════════════════════════════════════"
log "════════════════════════════════════════════════════════════"
log "PHASE 7d STARTING (DPO: UltraFeedback prefs, step ${PH7C_STEP} → 352,790)"
log "════════════════════════════════════════════════════════════"

if [ ! -f data/phase7d/dpo_train.npz ]; then
    log "Building Phase 7d DPO data (UltraFeedback MIT)..."
    PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/build_dpo.py \
        >> build_dpo.log 2>&1
    [ $? -ne 0 ] && die "Phase 7d DPO data build failed"
fi
log "Phase 7d data: train pairs $(python3 -c "import numpy as np; print(int(np.load('data/phase7d/dpo_train.npz')['n'][0]))")"

mkdir -p baselines/pre_phase7d/step_${PH7C_STEP}
cp checkpoints/step_${PH7C_STEP}/*.npz baselines/pre_phase7d/step_${PH7C_STEP}/ 2>/dev/null || log "WARNING: pre-7d checkpoint step_${PH7C_STEP} not found"
cp checkpoints/step_${PH7C_STEP}/*.json baselines/pre_phase7d/step_${PH7C_STEP}/ 2>/dev/null || true
log "Pre-7d baseline archived to baselines/pre_phase7d/step_${PH7C_STEP}/ (post-7c; DPO reference)"
prune_baseline baselines/pre_phase7d

PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/dpo.py \
    --train-config configs/phase7d_dpo_config.json \
    --model-config configs/model_config.json \
    --ref-checkpoint baselines/pre_phase7d \
    >> training_phase7d.log 2>&1
[ $? -ne 0 ] && die "Phase 7d training failed"

PH7D_STEP=$(latest_step)
log "Phase 7d complete at step ${PH7D_STEP}"

run_gate "phase7d" "${PH7D_STEP}"

mkdir -p baselines/post_phase7d/step_${PH7D_STEP}
cp checkpoints/step_${PH7D_STEP}/*.npz baselines/post_phase7d/step_${PH7D_STEP}/
cp checkpoints/step_${PH7D_STEP}/*.json baselines/post_phase7d/step_${PH7D_STEP}/ 2>/dev/null
log "Phase 7d checkpoint archived to baselines/post_phase7d/step_${PH7D_STEP}/"
prune_baseline baselines/post_phase7d

# ═══════════════════════════════════════════════════════════════════════
# PHASE 8b: ANCHOR — pristine synthesized targets (replaces sampled RFT)
# ═══════════════════════════════════════════════════════════════════════
log "════════════════════════════════════════════════════════════"
log "════════════════════════════════════════════════════════════"
log "PHASE 7e STARTING (Super-diverse SFT, step ${PH7D_STEP} → 354,790)"
log "════════════════════════════════════════════════════════════"

if [ ! -f data/phase7e/sft2_train.bin ]; then
    log "Building Phase 7e data (smol-smoltalk + ultrachat + synthetics)..."
    PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/build_sft2.py \
        >> build_sft2.log 2>&1
    [ $? -ne 0 ] && die "Phase 7e data build failed"
fi
log "Phase 7e data: $(du -h data/phase7e/sft2_train.bin | cut -f1)"

mkdir -p baselines/pre_phase7e/step_${PH7D_STEP}
cp checkpoints/step_${PH7D_STEP}/*.npz baselines/pre_phase7e/step_${PH7D_STEP}/ 2>/dev/null || log "WARNING: pre-7e checkpoint step_${PH7D_STEP} not found"
cp checkpoints/step_${PH7D_STEP}/*.json baselines/pre_phase7e/step_${PH7D_STEP}/ 2>/dev/null || true
log "Pre-7e baseline archived to baselines/pre_phase7e/step_${PH7D_STEP}/ (post-DPO)"
prune_baseline baselines/pre_phase7e

PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/train_manager.py \
    --config configs/phase7e_sft2_config.json \
    --model-config configs/model_config.json \
    --reset-optimizer \
    --phase "phase7e_sft2" \
    >> training_phase7e.log 2>&1
[ $? -ne 0 ] && die "Phase 7e training failed"

PH7E_STEP=$(latest_step)
log "Phase 7e complete at step ${PH7E_STEP}"

run_gate "phase7e" "${PH7E_STEP}"

mkdir -p baselines/post_phase7e/step_${PH7E_STEP}
cp checkpoints/step_${PH7E_STEP}/*.npz baselines/post_phase7e/step_${PH7E_STEP}/
cp checkpoints/step_${PH7E_STEP}/*.json baselines/post_phase7e/step_${PH7E_STEP}/ 2>/dev/null
log "Phase 7e checkpoint archived to baselines/post_phase7e/step_${PH7E_STEP}/"
prune_baseline baselines/post_phase7e

# ═══════════════════════════════════════════════════════════════════════
# PHASE 8b: ANCHOR — pristine synthesized targets (replaces sampled RFT)
# ═══════════════════════════════════════════════════════════════════════
log "════════════════════════════════════════════════════════════"
log "PHASE 8b STARTING (Anchor: pristine targets, step ${PH7E_STEP} → 355,790)"
log "════════════════════════════════════════════════════════════"

# Build Phase 8 RFT data: sample from Phase 7b model, verify, save accepted
if [ ! -f data/phase8/rft_train.bin ]; then
    log "Building Phase 8 RFT data (sampling + calculator verification)..."
    PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/rft.py \
        --checkpoint "checkpoints/step_${PH7B_STEP}" \
        --out data/phase8/rft_train.bin \
        --num-samples 32 \
        --temperature 0.4 \
        >> build_phase8.log 2>&1
    [ $? -ne 0 ] && log "WARNING: RFT data build returned non-zero. Supervised SFT already applied; continuing."
fi
log "Phase 8 data: $(du -h data/phase8/rft_train.bin | cut -f1)"

# Anchor data: synthesized pristine targets (no sampling; sampled RFT failed twice)
if [ ! -f data/phase8/anchor_train.bin ]; then
    log "Building Phase 8b anchor data (deterministic + verified)..."
    PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/build_anchor.py \
        >> build_phase8b.log 2>&1
    [ $? -ne 0 ] && die "Phase 8b anchor build failed"
fi
log "Phase 8b data: $(du -h data/phase8/anchor_train.bin | cut -f1)"

# Pre-anchor baseline (before 8b, = post-7e)
mkdir -p baselines/pre_phase8b/step_${PH7E_STEP}
cp checkpoints/step_${PH7E_STEP}/*.npz baselines/pre_phase8b/step_${PH7E_STEP}/ 2>/dev/null || log "WARNING: pre-8b checkpoint step_${PH7E_STEP} not found"
cp checkpoints/step_${PH7E_STEP}/*.json baselines/pre_phase8b/step_${PH7E_STEP}/ 2>/dev/null || true
log "Pre-8b baseline archived to baselines/pre_phase8b/step_${PH7E_STEP}/ (post-7e)"
prune_baseline baselines/pre_phase8b

if [ -f data/phase8/anchor_train.bin ] && [ "$(du -k data/phase8/anchor_train.bin | cut -f1)" -gt 0 ]; then
    PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/train_manager.py \
        --config configs/phase8b_anchor_config.json \
        --model-config configs/model_config.json \
        --reset-optimizer \
        --phase "phase8b_anchor" \
        >> training_phase8b.log 2>&1
    [ $? -ne 0 ] && die "Phase 8b training failed"

    PH8_STEP=$(latest_step)
    log "Phase 8b complete at step ${PH8_STEP}"

    run_gate "phase8b" "${PH8_STEP}"

    mkdir -p baselines/post_phase8b/step_${PH8_STEP}
    cp checkpoints/step_${PH8_STEP}/*.npz baselines/post_phase8b/step_${PH8_STEP}/
    cp checkpoints/step_${PH8_STEP}/*.json baselines/post_phase8b/step_${PH8_STEP}/ 2>/dev/null
    log "Phase 8b checkpoint archived to baselines/post_phase8b/step_${PH8_STEP}/"
    prune_baseline baselines/post_phase8b
else
    log "Skipping Phase 8b training (no anchor data). Model stays at Phase 7c."
    PH8_STEP="${PH7C_STEP}"
fi

# ═══════════════════════════════════════════════════════════════════════
# CURRICULUM COMPLETE — GGUF → HuggingFace → Ollama
# ═══════════════════════════════════════════════════════════════════════
log "════════════════════════════════════════════════════════════"
log "CURRICULUM COMPLETE"
log "  Phase 4 end: baselines/post_phase4/"
log "  Phase 5 end (Prabhakar): baselines/post_phase5_a/ (A: Foundations)"
log "  Phase 5 end (Prabhakar): baselines/post_phase5_b/ (B: Advanced)"
log "  Phase 5 end (Prabhakar): baselines/post_phase5_c/ (C: Reasoning)"
log "  Phase 5.x end (CoT math+code): baselines/post_phase5_cot/step_${PHC_STEP}/"
log "  Phase 6 end: baselines/post_phase6/step_${PH6_STEP}/"
log "  Phase 5.x end (Realization): baselines/post_realization/step_${PHZ_STEP}/"
log "  Phase 7 end: baselines/post_phase7/step_${PH7_STEP}/"
log "  Phase 7b end (Wiki): baselines/post_phase7b/step_${PH7B_STEP}/"
log "  Phase 7c end (SFT-refresh): baselines/post_phase7c/step_${PH7C_STEP}/"
log "  Phase 7d end (DPO): baselines/post_phase7d/step_${PH7D_STEP}/"
log "  Phase 8b end (Anchor): baselines/post_phase8b/step_${PH8_STEP}/"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 7f: CONVERSATIONAL SFT + knowledge recovery (355,790 → 360,790)
# 5,000 steps: UltraChat-30K + smol-smoltalk-20K + greetings + anchors-x40
# + phase7 slice (grounding). Installs conversation, re-pins knowledge.
# ═══════════════════════════════════════════════════════════════════════
log "════════════════════════════════════════════════════════════"
log "PHASE 7f STARTING (Convo-SFT: step ${PH8_STEP} → 360,790)"
log "════════════════════════════════════════════════════════════"

if [ ! -f data/phase7f/sft3_train.bin ]; then
    log "Building Phase 7f data..."
    PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/build_sft3.py \
        >> build_sft3.log 2>&1
    [ $? -ne 0 ] && die "Phase 7f data build failed"
fi
log "Phase 7f data: $(du -h data/phase7f/sft3_train.bin | cut -f1)"

mkdir -p baselines/pre_phase7f/step_${PH8_STEP}
cp checkpoints/step_${PH8_STEP}/*.npz baselines/pre_phase7f/step_${PH8_STEP}/ 2>/dev/null || log "WARNING: pre-7f checkpoint step_${PH8_STEP} not found"
cp checkpoints/step_${PH8_STEP}/*.json baselines/pre_phase7f/step_${PH8_STEP}/ 2>/dev/null || true
log "Pre-7f baseline archived to baselines/pre_phase7f/step_${PH8_STEP}/ (post-anchors)"
prune_baseline baselines/pre_phase7f

PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/train_manager.py \
    --config configs/phase7f_convosft_config.json \
    --model-config configs/model_config.json \
    --reset-optimizer \
    --phase "phase7f_convosft" \
    >> training_phase7f.log 2>&1
[ $? -ne 0 ] && die "Phase 7f training failed"

PH7F_STEP=$(latest_step)
log "Phase 7f complete at step ${PH7F_STEP}"

run_gate "phase7f" "${PH7F_STEP}"

mkdir -p baselines/post_phase7f/step_${PH7F_STEP}
cp checkpoints/step_${PH7F_STEP}/*.npz baselines/post_phase7f/step_${PH7F_STEP}/
cp checkpoints/step_${PH7F_STEP}/*.json baselines/post_phase7f/step_${PH7F_STEP}/ 2>/dev/null
log "Phase 7f checkpoint archived to baselines/post_phase7f/step_${PH7F_STEP}/"
prune_baseline baselines/post_phase7f
log "  Phase 7f end (Convo-SFT): baselines/post_phase7f/step_${PH7F_STEP}/"

# ═══════════════════════════════════════════════════════════════════════
# PHASE 7g: COMBINED RECOVERY (B-modified: openers + HEAVY anchors + phase7)
# 7f damaged free-gen routing (math 2/15, GK 9/16) while probes held.
# One phase repairs all three: custom openers + anchors-x40 + phase7 slice.
# Data: data/phase7g/combined_train.bin (6.41M). 360790 → 362290.
# ═══════════════════════════════════════════════════════════════════════
log "  Next: GGUF export → HuggingFace → Ollama"
log "════════════════════════════════════════════════════════════"
