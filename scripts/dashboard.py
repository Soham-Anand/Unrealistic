#!/usr/bin/env python3
"""Training Dashboard: web server for monitoring and controlling training.

Endpoints:
  GET  /              -> Dashboard UI
  GET  /api/status    -> Current training status
  POST /api/control   -> Send control command (pause/resume/stop/checkpoint/timer)
  GET  /api/log       -> Last N lines of training log

Usage:
    python3 scripts/dashboard.py [--port 8080]
"""

import os, sys, json, time, argparse
from pathlib import Path

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    print("Installing dependencies...")
    os.system(f"{sys.executable} -m pip install fastapi uvicorn -q")
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn

STATUS_FILE = "training/status.json"
CONTROL_FILE = "training/control.json"
LOG_DIR = "training"

app = FastAPI(title="Unrealistic Training Dashboard")


def read_status():
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "state": "no_data",
            "phase": "unknown",
            "step": 0,
            "max_steps": 0,
            "loss": 0,
            "lr": 0,
            "tok_per_sec": 0,
            "mem_mb": 0,
            "elapsed": "0:00:00",
            "eta": "unknown",
            "checkpoint": "none",
            "pid": 0,
            "timer_remaining": "none",
            "last_update": 0,
        }


def write_control(cmd):
    os.makedirs(os.path.dirname(CONTROL_FILE), exist_ok=True)
    with open(CONTROL_FILE, "w") as f:
        json.dump(cmd, f)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


@app.get("/api/status")
async def api_status():
    status = read_status()
    # Check if stale (no update in 30s)
    if status.get("last_update", 0) > 0:
        age = time.time() - status["last_update"]
        if age > 30:
            status["state"] = "stale"
    return JSONResponse(status)


@app.post("/api/control")
async def api_control(request: Request):
    body = await request.json()
    action = body.get("action", "")

    if action == "pause":
        write_control({"action": "pause"})
        return JSONResponse({"ok": True, "message": "Pause requested"})
    elif action == "resume":
        write_control({"action": "resume"})
        return JSONResponse({"ok": True, "message": "Resume requested"})
    elif action == "stop":
        write_control({"action": "stop"})
        return JSONResponse({"ok": True, "message": "Stop requested"})
    elif action == "checkpoint":
        write_control({"action": "checkpoint"})
        return JSONResponse({"ok": True, "message": "Checkpoint requested"})
    elif action == "set-timer":
        seconds = body.get("seconds", 0)
        write_control({"action": "set-timer", "seconds": seconds})
        return JSONResponse({"ok": True, "message": f"Timer set: {seconds}s"})
    elif action == "cancel-timer":
        write_control({"action": "cancel-timer"})
        return JSONResponse({"ok": True, "message": "Timer cancelled"})
    else:
        return JSONResponse({"ok": False, "message": f"Unknown action: {action}"}, status_code=400)


@app.get("/api/log")
async def api_log(lines: int = 50):
    log_files = sorted(Path(LOG_DIR).glob("*.log"), reverse=True)
    if not log_files:
        return JSONResponse({"lines": [], "file": "none"})

    latest = log_files[0]
    with open(latest) as f:
        all_lines = f.readlines()

    return JSONResponse({
        "lines": all_lines[-lines:],
        "file": str(latest),
        "total_lines": len(all_lines),
    })


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Unrealistic Training Dashboard</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'SF Mono', 'Fira Code', monospace; background: #0d1117; color: #c9d1d9; min-height: 100vh; }
  .header { background: #161b22; border-bottom: 1px solid #30363d; padding: 16px 24px; display: flex; justify-content: space-between; align-items: center; }
  .header h1 { font-size: 18px; color: #58a6ff; }
  .header .status-badge { padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; }
  .badge-running { background: #238636; color: #fff; }
  .badge-paused { background: #d29922; color: #fff; }
  .badge-stopped { background: #da3633; color: #fff; }
  .badge-idle { background: #30363d; color: #8b949e; }
  .badge-completed { background: #1f6feb; color: #fff; }
  .badge-stale { background: #6e40c9; color: #fff; }

  .container { max-width: 1200px; margin: 0 auto; padding: 24px; }

  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .card .label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
  .card .value { font-size: 28px; font-weight: bold; color: #58a6ff; }
  .card .value.loss { color: #f0883e; }
  .card .value.mem { color: #a371f7; }

  .progress-container { margin-bottom: 24px; }
  .progress-bar { width: 100%; height: 24px; background: #21262d; border-radius: 12px; overflow: hidden; position: relative; }
  .progress-fill { height: 100%; background: linear-gradient(90deg, #238636, #3fb950); transition: width 0.5s ease; border-radius: 12px; }
  .progress-text { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 12px; font-weight: bold; color: #fff; }

  .controls { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 24px; }
  .btn { padding: 10px 20px; border: 1px solid #30363d; border-radius: 6px; background: #21262d; color: #c9d1d9; font-family: inherit; font-size: 13px; cursor: pointer; transition: all 0.2s; }
  .btn:hover { background: #30363d; border-color: #8b949e; }
  .btn-primary { background: #238636; border-color: #238636; color: #fff; }
  .btn-primary:hover { background: #2ea043; }
  .btn-warning { background: #d29922; border-color: #d29922; color: #fff; }
  .btn-warning:hover { background: #e3b341; }
  .btn-danger { background: #da3633; border-color: #da3633; color: #fff; }
  .btn-danger:hover { background: #f85149; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }

  .timer-section { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 24px; }
  .timer-section h3 { font-size: 14px; color: #8b949e; margin-bottom: 12px; }
  .timer-input { display: flex; gap: 8px; align-items: center; }
  .timer-input input { width: 80px; padding: 8px; background: #0d1117; border: 1px solid #30363d; border-radius: 4px; color: #c9d1d9; font-family: inherit; font-size: 13px; text-align: center; }
  .timer-input label { font-size: 12px; color: #8b949e; }

  .log-container { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 16px; max-height: 400px; overflow-y: auto; }
  .log-container h3 { font-size: 14px; color: #8b949e; margin-bottom: 12px; }
  .log-line { font-size: 11px; color: #8b949e; line-height: 1.6; white-space: pre-wrap; word-break: break-all; }
  .log-line.highlight { color: #3fb950; }

  .checkpoint { font-size: 14px; color: #58a6ff; }
</style>
</head>
<body>

<div class="header">
  <h1>Unrealistic Training Dashboard</h1>
  <span id="state-badge" class="status-badge badge-idle">IDLE</span>
</div>

<div class="container">
  <!-- Status Cards -->
  <div class="grid">
    <div class="card">
      <div class="label">Phase</div>
      <div class="value" id="phase">-</div>
    </div>
    <div class="card">
      <div class="label">Step</div>
      <div class="value" id="step">-</div>
    </div>
    <div class="card">
      <div class="label">Loss</div>
      <div class="value loss" id="loss">-</div>
    </div>
    <div class="card">
      <div class="label">Learning Rate</div>
      <div class="value" id="lr">-</div>
    </div>
    <div class="card">
      <div class="label">Speed</div>
      <div class="value" id="speed">-</div>
    </div>
    <div class="card">
      <div class="label">Memory</div>
      <div class="value mem" id="mem">-</div>
    </div>
    <div class="card">
      <div class="label">ETA</div>
      <div class="value" id="eta">-</div>
    </div>
    <div class="card">
      <div class="label">Checkpoint</div>
      <div class="checkpoint" id="checkpoint">-</div>
    </div>
  </div>

  <!-- Progress Bar -->
  <div class="progress-container">
    <div class="progress-bar">
      <div class="progress-fill" id="progress-fill" style="width: 0%"></div>
      <div class="progress-text" id="progress-text">0%</div>
    </div>
  </div>

  <!-- Controls -->
  <div class="controls">
    <button class="btn btn-primary" id="btn-resume" onclick="control('resume')">Resume</button>
    <button class="btn btn-warning" id="btn-pause" onclick="control('pause')">Pause</button>
    <button class="btn" id="btn-checkpoint" onclick="control('checkpoint')">Save Checkpoint</button>
    <button class="btn btn-danger" id="btn-stop" onclick="control('stop')">Stop Training</button>
  </div>

  <!-- Timer -->
  <div class="timer-section">
    <h3>Auto-Stop Timer</h3>
    <div class="timer-input">
      <input type="number" id="timer-hours" placeholder="0" min="0" max="48">
      <label>hours</label>
      <input type="number" id="timer-mins" placeholder="0" min="0" max="59">
      <label>mins</label>
      <button class="btn" onclick="setTimer()">Set Timer</button>
      <button class="btn" onclick="control('cancel-timer')">Cancel</button>
      <span id="timer-display" style="margin-left: 16px; color: #d29922; font-size: 13px;"></span>
    </div>
  </div>

  <!-- Log -->
  <div class="log-container">
    <h3>Training Log (last 30 lines)</h3>
    <div id="log-output"></div>
  </div>
</div>

<script>
const API = '';

async function fetchStatus() {
  try {
    const res = await fetch(API + '/api/status');
    const s = await res.json();

    // Update badge
    const badge = document.getElementById('state-badge');
    const state = s.state || 'idle';
    badge.textContent = state.toUpperCase();
    badge.className = 'status-badge badge-' + (state === 'no_data' ? 'idle' : state);

    // Update cards
    document.getElementById('phase').textContent = s.phase || '-';
    document.getElementById('step').textContent = s.step ? s.step.toLocaleString() : '-';
    document.getElementById('loss').textContent = s.loss ? s.loss.toFixed(4) : '-';
    document.getElementById('lr').textContent = s.lr ? s.lr.toExponential(2) : '-';
    document.getElementById('speed').textContent = s.tok_per_sec ? s.tok_per_sec.toLocaleString() + ' t/s' : '-';
    document.getElementById('mem').textContent = s.mem_mb ? s.mem_mb + ' MB' : '-';
    document.getElementById('eta').textContent = s.eta || '-';
    document.getElementById('checkpoint').textContent = s.checkpoint || '-';

    // Progress bar
    if (s.max_steps > 0) {
      const pct = Math.min(100, (s.step / s.max_steps) * 100);
      document.getElementById('progress-fill').style.width = pct + '%';
      document.getElementById('progress-text').textContent = pct.toFixed(1) + '% (' + s.step.toLocaleString() + '/' + s.max_steps.toLocaleString() + ')';
    }

    // Timer
    document.getElementById('timer-display').textContent = s.timer_remaining !== 'none' ? '⏰ ' + s.timer_remaining : '';

    // Button states
    document.getElementById('btn-pause').disabled = (state !== 'running');
    document.getElementById('btn-resume').disabled = (state !== 'paused');
    document.getElementById('btn-checkpoint').disabled = (state !== 'running' && state !== 'paused');
    document.getElementById('btn-stop').disabled = (state === 'idle' || state === 'stopped' || state === 'completed' || state === 'no_data');
  } catch (e) {
    document.getElementById('state-badge').textContent = 'OFFLINE';
    document.getElementById('state-badge').className = 'status-badge badge-idle';
  }
}

async function fetchLog() {
  try {
    const res = await fetch(API + '/api/log?lines=30');
    const data = await res.json();
    const logEl = document.getElementById('log-output');
    logEl.innerHTML = data.lines.map(l => {
      const cls = l.includes('step ') ? 'highlight' : '';
      return '<div class="log-line ' + cls + '">' + escapeHtml(l.trimEnd()) + '</div>';
    }).join('');
    logEl.scrollTop = logEl.scrollHeight;
  } catch (e) {}
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function control(action) {
  if (action === 'stop' && !confirm('Stop training? This cannot be undone.')) return;
  try {
    await fetch(API + '/api/control', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action})
    });
    setTimeout(fetchStatus, 500);
  } catch (e) { alert('Control failed: ' + e); }
}

async function setTimer() {
  const h = parseInt(document.getElementById('timer-hours').value) || 0;
  const m = parseInt(document.getElementById('timer-mins').value) || 0;
  const seconds = h * 3600 + m * 60;
  if (seconds <= 0) { alert('Enter a time > 0'); return; }
  if (!confirm('Auto-stop in ' + h + 'h ' + m + 'm?')) return;
  try {
    await fetch(API + '/api/control', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action: 'set-timer', seconds})
    });
    setTimeout(fetchStatus, 500);
  } catch (e) { alert('Timer failed: ' + e); }
}

// Poll every 3 seconds
setInterval(fetchStatus, 3000);
setInterval(fetchLog, 5000);
fetchStatus();
fetchLog();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Training Dashboard")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    print(f"Dashboard: http://localhost:{args.port}")
    print(f"API: http://localhost:{args.port}/api/status")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
