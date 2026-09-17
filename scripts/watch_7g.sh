#!/usr/bin/env bash
# Gate: 7g -> 362290 (accepts >= 362289), archive post_phase7g, report.
#   nohup bash scripts/watch_7g.sh > watch_7g.log 2>&1 &
set -u
log() { echo "[$(date '+%H:%M:%S')] $*"; }
latest() { ls -td checkpoints/step_* 2>/dev/null | head -1; }
latest_n() { latest | sed -E 's/.*step_([0-9]+)/\1/'; }
log "Gate 7g -> 362290 (openers + heavy anchors + phase7)"
while true; do
    if ! pgrep -f "scripts/train.py" >/dev/null; then
        cur=$(latest_n)
        if [ -n "$cur" ] && [ "$cur" -ge 362289 ]; then
            L=$(latest)
            log "Gate passed: ${L}"
            break
        fi
        log "WARNING: nothing training but latest ${cur:-none} < 362289 — check training_phase7g.log"
    fi
    sleep 60
done
bn=$(basename "$L")
rm -rf baselines/post_phase7g
cp -r "$L" baselines/post_phase7g
log "Archived post_phase7g from ${bn}. COMBINED RECOVERY COMPLETE — run final_eval + probes + greeting battery."
