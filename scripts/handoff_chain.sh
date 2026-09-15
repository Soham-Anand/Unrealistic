#!/usr/bin/env bash
# Full chain handoff: SFT(346791) -> Wiki(349791) -> RFT data -> RFT(355791).
# NOTE: trainer's final save lands at max_steps-1 (loop `range(step, max)` then
# saves last iterated step), so gates accept latest >= FINAL-1 and use the
# ACTUAL latest dir for baselines + rft.py. Safe to start any time.
#   nohup bash scripts/handoff_chain.sh > handoff_chain.log 2>&1 &

set -u

SFT_FINAL=346791
WIKI_FINAL=349791
WIKI_BIN="data/phase7b/wiki_train.bin"
RFT_BIN="data/phase8/rft_train.bin"
LATEST_DIR=""

log() { echo "[$(date '+%H:%M:%S')] $*"; }

latest_dir() { ls -td checkpoints/step_* 2>/dev/null | head -1; }
latest_step() { latest_dir | sed -E 's/.*step_([0-9]+)/\1/'; }

# Gate: fires when trainer has exited AND latest ckpt >= FINAL-1.
# If trainer died early (latest < FINAL-1), warns until user resumes.
wait_gate() { # $1=FINAL $2=log-name
    while true; do
        if pgrep -f "scripts/train.py" >/dev/null; then
            sleep 60
            continue
        fi
        cur=$(latest_step)
        if [ -n "$cur" ] && [ "$cur" -ge $(( $1 - 1 )) ]; then
            LATEST_DIR=$(latest_dir)
            log "Gate $1 passed: trainer exited, latest=${LATEST_DIR}"
            return 0
        fi
        log "WARNING: no train.py but latest step ${cur:-none} < $(( $1 - 1 )) — crashed early? check $2"
        sleep 60
    done
}

wait_exit() {
    for _ in $(seq 1 60); do
        pgrep -f "scripts/train.py" >/dev/null || break
        sleep 10
    done
}

log "Chain watcher started. Gate 1: SFT -> ${SFT_FINAL} (accepts >= $((SFT_FINAL - 1)))"
if [ "$(latest_step)" -gt $((SFT_FINAL + 500)) ]; then
    log "Latest $(latest_dir) already far past SFT — skipping Gate 1 block (restart guard)."
else
    wait_gate "${SFT_FINAL}" "training_phase7.log"
    log "SFT final: ${LATEST_DIR}"

bn=$(basename "${LATEST_DIR}")
rm -rf "baselines/post_phase7" "baselines/pre_phase7b"
cp -r "${LATEST_DIR}" "baselines/post_phase7"
cp -r "${LATEST_DIR}" "baselines/pre_phase7b"
log "Archived post_phase7 + pre_phase7b from ${bn}."

[ -s "${WIKI_BIN}" ] || { log "ERROR: ${WIKI_BIN} missing/empty — build first. ABORT."; exit 1; }
log "Wiki data present: $(du -h ${WIKI_BIN} | cut -f1). Launching Phase 7b..."
PYTHONUNBUFFERED=1 PYTHONPATH=. nohup /opt/homebrew/bin/python3 -u \
    scripts/train.py --train-config configs/phase7b_wiki_config.json \
    --model-config configs/model_config.json --reset-optimizer \
    >> training_phase7b.log 2>&1 &
sleep 30
pgrep -f "scripts/train.py" >/dev/null \
    && log "Phase 7b training live -> training_phase7b.log" \
    || { log "ERROR: Phase 7b exited immediately. Check training_phase7b.log"; exit 1; }
fi

log "Gate 2: Wiki -> ${WIKI_FINAL} (accepts >= $((WIKI_FINAL - 1)))"
wait_gate "${WIKI_FINAL}" "training_phase7b.log"
log "Wiki final: ${LATEST_DIR}"

bn=$(basename "${LATEST_DIR}")
rm -rf "baselines/post_phase7b" "baselines/pre_phase8"
cp -r "${LATEST_DIR}" "baselines/post_phase7b"
cp -r "${LATEST_DIR}" "baselines/pre_phase8"
log "Archived post_phase7b + pre_phase8 from ${bn}."

log "Rejection sampling from ${LATEST_DIR} (42 x 32, temp 0.4)..."
PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/rft.py \
    --checkpoint "${LATEST_DIR}" --out "${RFT_BIN}" \
    --num-samples 32 --temperature 0.4 2>&1 | tee build_phase8.log
[ ${PIPESTATUS[0]} -ne 0 ] && { log "ERROR: rft.py failed. ABORT."; exit 1; }
[ -s "${RFT_BIN}" ] || { log "ERROR: ${RFT_BIN} empty. ABORT."; exit 1; }
ACCEPTED=$(grep -E "accepted:" build_phase8.log | sed -E 's/.*accepted:[[:space:]]*([0-9]+).*/\1/')
[ "${ACCEPTED:-0}" -eq 0 ] && { log "ERROR: 0 accepted. ABORT."; exit 1; }
log "Accepted: ${ACCEPTED}. Launching RFT..."

PYTHONUNBUFFERED=1 PYTHONPATH=. nohup /opt/homebrew/bin/python3 -u \
    scripts/train.py --train-config configs/phase8_rft_config.json \
    --model-config configs/model_config.json --reset-optimizer \
    >> training_phase8.log 2>&1 &
sleep 30
pgrep -f "scripts/train.py" >/dev/null \
    && log "RFT live (target 355791) -> training_phase8.log. CHAIN COMPLETE." \
    || { log "ERROR: RFT exited immediately. Check training_phase8.log"; exit 1; }
