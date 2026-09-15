#!/usr/bin/env bash
# Chain: 7c -> 351790 -> archive pre_phase7d -> DPO 7d -> 352790 ->
#        archive post_phase7d + pre_phase8b -> anchor 8b -> 353790. Done.
#   nohup bash scripts/watch_chain2.sh > watch_chain2.log 2>&1 &
set -u
log() { echo "[$(date '+%H:%M:%S')] $*"; }
latest() { ls -td checkpoints/step_* 2>/dev/null | head -1; }
latest_n() { latest | sed -E 's/.*step_([0-9]+)/\1/'; }

wait_gate() { # $1=FINAL $2=log-name
    while true; do
        if ! pgrep -f "scripts/train.py|scripts/dpo.py" >/dev/null; then
            cur=$(latest_n)
            # dpo.py saves via same save_checkpoint -> same step_ dirs
            if [ -n "$cur" ] && [ "$cur" -ge $(( $1 - 1 )) ]; then
                log "Gate $1 passed: latest=$(latest)"
                return 0
            fi
            log "WARNING: nothing training but latest ${cur:-none} < $(( $1 - 1 )) — check $2"
        fi
        sleep 60
    done
}

log "Gate A: 7c -> 351790"
wait_gate 351790 "training_phase7c.log"
L7C=$(latest); B7C=$(basename "$L7C")
rm -rf baselines/post_phase7c
cp -r "$L7C" baselines/post_phase7c
cp -r "$L7C" baselines/pre_phase7d
log "Archived post_phase7c + pre_phase7d (DPO ref) from ${B7C}."
[ -s data/phase7d/dpo_train.npz ] || { log "ERROR: DPO data missing. ABORT."; exit 1; }

log "Launching DPO 7d (ref=baselines/pre_phase7d)..."
PYTHONUNBUFFERED=1 PYTHONPATH=. nohup python3 -u scripts/dpo.py \
    --train-config configs/phase7d_dpo_config.json \
    --model-config configs/model_config.json \
    --ref-checkpoint baselines/pre_phase7d \
    >> training_phase7d.log 2>&1 &
sleep 45
pgrep -f "scripts/dpo.py" >/dev/null \
    && log "DPO live -> training_phase7d.log" \
    || { log "ERROR: DPO exited immediately. Check training_phase7d.log"; exit 1; }

log "Gate B: 7d -> 352790"
wait_gate 352790 "training_phase7d.log"
L7D=$(latest); B7D=$(basename "$L7D")
rm -rf baselines/post_phase7d baselines/pre_phase8b
cp -r "$L7D" baselines/post_phase7d
cp -r "$L7D" baselines/pre_phase8b
log "Archived post_phase7d + pre_phase8b from ${B7D}."
[ -s data/phase8/anchor_train.bin ] || { log "ERROR: anchor bin missing. ABORT."; exit 1; }

log "Launching anchor 8b..."
PYTHONUNBUFFERED=1 PYTHONPATH=. nohup /opt/homebrew/bin/python3 -u \
    scripts/train.py --train-config configs/phase8b_anchor_config.json \
    --model-config configs/model_config.json --reset-optimizer \
    >> training_phase8b.log 2>&1 &
sleep 30
pgrep -f "scripts/train.py" >/dev/null \
    && log "8b live (target 353790). CHAIN ARMED THROUGH FINAL." \
    || { log "ERROR: 8b exited immediately."; exit 1; }

log "Gate C: 8b -> 353790 (final)"
wait_gate 353790 "training_phase8b.log"
LF=$(latest)
rm -rf baselines/post_final
cp -r "$LF" baselines/post_final
log "FINAL MODEL: ${LF} archived to baselines/post_final. ALL TRAINING COMPLETE."
