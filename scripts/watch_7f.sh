#!/usr/bin/env bash
# Gate: 7f -> 360790 (accepts >= 360789), archive post_phase7f, report.
#   nohup bash scripts/watch_7f.sh > watch_7f.log 2>&1 &
set -u
log() { echo "[$(date '+%H:%M:%S')] $*"; }
latest() { ls -td checkpoints/step_* 2>/dev/null | head -1; }
latest_n() { latest | sed -E 's/.*step_([0-9]+)/\1/'; }
log "Gate 7f -> 360790"
while true; do
    if ! pgrep -f "scripts/train.py" >/dev/null; then
        cur=$(latest_n)
        if [ -n "$cur" ] && [ "$cur" -ge 360789 ]; then
            L=$(latest)
            log "Gate passed: ${L}"
            break
        fi
        log "WARNING: nothing training but latest ${cur:-none} < 360789 — check training_phase7f.log"
    fi
    sleep 60
done
bn=$(basename "$L")
rm -rf baselines/post_phase7f
cp -r "$L" baselines/post_phase7f
log "Archived post_phase7f from ${bn}. CONVO-SFT COMPLETE — run final_eval + probes."
