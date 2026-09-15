#!/usr/bin/env bash
# Auto-handoff: wait for SFT to hit max_steps (346791) and finish,
# then archive baselines, build RFT rejection-sampled data, and launch RFT.
# Safe to start now — it just waits until SFT is done.
#   nohup bash scripts/handoff_phase8.sh > handoff_phase8.log 2>&1 &

set -u

FINAL=346791
FINAL_DIR="checkpoints/step_${FINAL}"
OUT="data/phase8/rft_train.bin"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Handoff watcher started. Waiting for SFT to complete at ${FINAL_DIR}..."
while [ ! -d "${FINAL_DIR}" ]; do
    if ! pgrep -f "scripts/train.py" >/dev/null; then
        log "WARNING: no train.py running but ${FINAL_DIR} missing. SFT may have stalled — check training_phase7.log"
    fi
    sleep 60
done
log "Final SFT checkpoint found: ${FINAL_DIR}"

# Wait for the SFT trainer to actually exit (final save lands a moment before exit)
for _ in $(seq 1 60); do
    if ! pgrep -f "scripts/train.py" >/dev/null; then
        break
    fi
    sleep 10
done
log "SFT trainer exited. Current step: $(grep '^step ' training_phase7.log | tail -1)"

rm -rf baselines/post_phase7 baselines/pre_phase8
cp -r "${FINAL_DIR}" baselines/post_phase7
cp -r "${FINAL_DIR}" baselines/pre_phase8
log "Baselines archived: baselines/post_phase7, baselines/pre_phase8"

log "Running rejection sampling (42 prompts x 32 samples, temp 0.4)..."
PYTHONUNBUFFERED=1 PYTHONPATH=. python3 -u scripts/rft.py \
    --checkpoint "${FINAL_DIR}" \
    --out "${OUT}" \
    --num-samples 32 \
    --temperature 0.4 2>&1 | tee build_phase8.log
STATUS=${PIPESTATUS[0]}
if [ ${STATUS} -ne 0 ]; then
    log "ERROR: rft.py failed. Not launching RFT."
    exit 1
fi
[ -s "${OUT}" ] || { log "ERROR: ${OUT} missing/empty. Not launching RFT."; exit 1; }

ACCEPTED=$(grep -E "accepted:" build_phase8.log | sed -E 's/.*accepted:[[:space:]]*([0-9]+).*/\1/')
if [ "${ACCEPTED:-0}" -eq 0 ]; then
    log "ERROR: 0 accepted answers from rft.py. Not launching RFT."
    exit 1
fi
log "Accepted answers: ${ACCEPTED}. Launching RFT..."

PYTHONUNBUFFERED=1 PYTHONPATH=. nohup /opt/homebrew/bin/python3 -u \
    scripts/train.py --train-config configs/phase8_rft_config.json \
    --model-config configs/model_config.json --reset-optimizer \
    >> training_phase8.log 2>&1 &
RFT_PID=$!
sleep 20
if kill -0 ${RFT_PID} 2>/dev/null; then
    log "RFT launched (PID ${RFT_PID}). Training to max_steps 352791 -> training_phase8.log"
else
    log "ERROR: RFT process ${RFT_PID} exited immediately. Check training_phase8.log"
fi