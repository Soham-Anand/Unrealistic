#!/usr/bin/env bash
# Gate: wait for Phase 7c -> 351790 (accepts >= 351789), archive, launch 8b anchor.
#   nohup bash scripts/watch_7c.sh > watch_7c.log 2>&1 &
set -u
FINAL=351790
log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "Watching for step >= $((FINAL - 1))..."
while true; do
    if ! pgrep -f "scripts/train.py" >/dev/null; then
        cur=$(ls -td checkpoints/step_* 2>/dev/null | head -1 | sed -E 's/.*step_([0-9]+)/\1/')
        if [ -n "$cur" ] && [ "$cur" -ge $((FINAL - 1)) ]; then
            LATEST=$(ls -td checkpoints/step_* | head -1)
            log "Gate passed: ${LATEST}"
            break
        fi
        log "WARNING: no train.py but latest ${cur:-none} < $((FINAL - 1)) — check training_phase7c.log"
    fi
    sleep 60
done
bn=$(basename "$LATEST")
rm -rf baselines/post_phase7c baselines/pre_phase8b
cp -r "$LATEST" baselines/post_phase7c
cp -r "$LATEST" baselines/pre_phase8b
log "Archived post_phase7c + pre_phase8b from ${bn}."
[ -s data/phase8/anchor_train.bin ] || { log "ERROR: anchor bin missing. ABORT."; exit 1; }
PYTHONUNBUFFERED=1 PYTHONPATH=. nohup /opt/homebrew/bin/python3 -u \
    scripts/train.py --train-config configs/phase8b_anchor_config.json \
    --model-config configs/model_config.json --reset-optimizer \
    >> training_phase8b.log 2>&1 &
sleep 30
pgrep -f "scripts/train.py" >/dev/null \
    && log "Phase 8b live (target 352790). WATCH COMPLETE." \
    || { log "ERROR: 8b exited immediately."; exit 1; }
