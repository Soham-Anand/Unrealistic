#!/usr/bin/env bash
# Chain: 7e -> 354790 -> archive post_phase7e + pre_phase8b -> anchor 8b -> 355790.
#   nohup bash scripts/watch_7e8b.sh > watch_7e8b.log 2>&1 &
set -u
log() { echo "[$(date '+%H:%M:%S')] $*"; }
latest() { ls -td checkpoints/step_* 2>/dev/null | head -1; }
latest_n() { latest | sed -E 's/.*step_([0-9]+)/\1/'; }
wait_gate() {
    while true; do
        if ! pgrep -f "scripts/train.py" >/dev/null; then
            cur=$(latest_n)
            if [ -n "$cur" ] && [ "$cur" -ge $(( $1 - 1 )) ]; then
                log "Gate $1 passed: latest=$(latest)"
                return 0
            fi
            log "WARNING: nothing training but latest ${cur:-none} < $(( $1 - 1 )) — check $2"
        fi
        sleep 60
    done
}
log "Gate 7e -> 354790"
wait_gate 354790 "training_phase7e.log"
L=$(latest); B=$(basename "$L")
rm -rf baselines/post_phase7e baselines/pre_phase8b
cp -r "$L" baselines/post_phase7e
cp -r "$L" baselines/pre_phase8b
log "Archived post_phase7e + pre_phase8b from ${B}."
[ -s data/phase8/anchor_train.bin ] || { log "ERROR: anchor bin missing. ABORT."; exit 1; }
log "Launching anchor 8b..."
PYTHONUNBUFFERED=1 PYTHONPATH=. nohup /opt/homebrew/bin/python3 -u \
    scripts/train.py --train-config configs/phase8b_anchor_config.json \
    --model-config configs/model_config.json --reset-optimizer \
    >> training_phase8b.log 2>&1 &
sleep 30
pgrep -f "scripts/train.py" >/dev/null \
    && log "8b live (target 355790)." \
    || { log "ERROR: 8b exited immediately."; exit 1; }
log "Gate 8b -> 355790 (final)"
wait_gate 355790 "training_phase8b.log"
LF=$(latest)
rm -rf baselines/post_final
cp -r "$LF" baselines/post_final
log "FINAL MODEL: ${LF} archived to baselines/post_final. ALL TRAINING COMPLETE."
