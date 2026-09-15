#!/usr/bin/env python3
"""Training Manager v2: wraps train.py with real-time status + control.

Fixes:
- Threaded control checker (reads control.json every 0.5s, not just on log lines)
- Non-blocking subprocess reads
- Proper macOS signal handling

Usage:
    python3 scripts/train_manager.py --config configs/phase5_math_config.json [--model-config configs/model_config.json]
"""

import os, sys, json, time, signal, subprocess, argparse, threading, re
from pathlib import Path

STATUS_FILE = "training/status.json"
CONTROL_FILE = "training/control.json"
LOG_DIR = "training"


class TrainingManager:
    def __init__(self):
        self.process = None
        self.paused = False
        self.stopped = False
        self.timer_seconds = 0
        self.timer_start = 0
        self._lock = threading.Lock()
        self.status = {
            "state": "idle",
            "phase": "idle",
            "step": 0,
            "max_steps": 0,
            "loss": 0.0,
            "lr": 0.0,
            "tok_per_sec": 0,
            "mem_mb": 0,
            "elapsed": "0:00:00",
            "eta": "unknown",
            "checkpoint": "none",
            "pid": 0,
            "timer_remaining": "none",
            "last_update": 0,
        }
        os.makedirs(LOG_DIR, exist_ok=True)
        self._write_status()

    def _write_status(self):
        self.status["last_update"] = time.time()
        os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
        tmp = STATUS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.status, f, indent=2)
        os.replace(tmp, STATUS_FILE)

    def _read_control(self):
        try:
            if os.path.exists(CONTROL_FILE):
                with open(CONTROL_FILE) as f:
                    cmd = json.load(f)
                os.remove(CONTROL_FILE)
                return cmd
        except (json.JSONDecodeError, FileNotFoundError):
            pass
        return None

    def _parse_log_line(self, line):
        """Parse training log line for status updates."""
        m = re.search(
            r"step ([\d,]+)\s*\|\s*loss ([\d.]+)\s*\|\s*lr ([\d.e+-]+)\s*\|\s*(\d+)\s*tok/s.*?mem\s*(\d+)/(\d+)MB",
            line,
        )
        if m:
            with self._lock:
                self.status["step"] = int(m.group(1).replace(",", ""))
                self.status["loss"] = float(m.group(2))
                self.status["lr"] = float(m.group(3))
                self.status["tok_per_sec"] = int(m.group(4))
                self.status["mem_mb"] = int(m.group(5))
                self._calc_eta()
                self._write_status()
            return

        m = re.search(r"\[eval\]\s*step\s*([\d,]+)", line)
        if m:
            with self._lock:
                self.status["checkpoint"] = f"step_{m.group(1).replace(',', '')}"
                self._write_status()

        m = re.search(r"Checkpoint saved:\s*checkpoints/(step_\d+)", line)
        if m:
            with self._lock:
                self.status["checkpoint"] = m.group(1)
                self._write_status()

    def _calc_eta(self):
        if self.status["step"] == 0 or self.status["tok_per_sec"] == 0:
            self.status["eta"] = "calculating..."
            return
        remaining = self.status["max_steps"] - self.status["step"]
        if remaining <= 0:
            self.status["eta"] = "complete"
            return
        tok_remaining = remaining * 8192
        sec_remaining = tok_remaining / max(1, self.status["tok_per_sec"])
        hours = int(sec_remaining // 3600)
        mins = int((sec_remaining % 3600) // 60)
        self.status["eta"] = f"{hours}h {mins}m"

    def _control_loop(self):
        """Background thread: check for control commands every 0.5s, keep status alive."""
        while self._running:
            cmd = self._read_control()
            if cmd:
                self._process_control(cmd)
            # Keep status alive while process runs
            if self.process and self.process.poll() is None:
                with self._lock:
                    self.status["last_update"] = time.time()
                    self._write_status()
            # Timer check
            if self.timer_seconds > 0 and self.timer_start > 0:
                remaining = self.timer_seconds - (time.time() - self.timer_start)
                if remaining <= 0:
                    print("[manager] Timer expired. Stopping...")
                    self.timer_seconds = 0
                    self._do_stop()
                    break
                h = int(remaining // 3600)
                m = int((remaining % 3600) // 60)
                s = int(remaining % 60)
                with self._lock:
                    self.status["timer_remaining"] = f"{h}h {m}m {s}s"
                    self._write_status()
            time.sleep(1)

    def _read_output(self):
        """Background thread: read subprocess stdout line-by-line."""
        try:
            for raw in iter(self.process.stdout.readline, b""):
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
                self._log_file.write(line)
                self._log_file.flush()
                self._parse_log_line(line)
        except (ValueError, OSError):
            pass

    def _process_control(self, cmd):
        action = cmd.get("action", "")
        if action == "pause":
            self._do_pause()
        elif action == "resume":
            self._do_resume()
        elif action == "stop":
            self._do_stop()
        elif action == "checkpoint":
            self._do_checkpoint()
        elif action == "set-timer":
            self.timer_seconds = cmd.get("seconds", 0)
            self.timer_start = time.time()
            print(f"[manager] Timer set: {self.timer_seconds}s")
        elif action == "cancel-timer":
            self.timer_seconds = 0
            self.timer_start = 0
            with self._lock:
                self.status["timer_remaining"] = "none"
                self._write_status()

    def _do_pause(self):
        if self.process and self.process.poll() is None:
            try:
                self.process.send_signal(signal.SIGSTOP)
            except ProcessLookupError:
                pass
            self.paused = True
            with self._lock:
                self.status["state"] = "paused"
                self._write_status()
            print("[manager] PAUSED")

    def _do_resume(self):
        if self.process and self.process.poll() is None:
            try:
                self.process.send_signal(signal.SIGCONT)
            except ProcessLookupError:
                pass
            self.paused = False
            with self._lock:
                self.status["state"] = "running"
                self._write_status()
            print("[manager] RESUMED")

    def _do_stop(self):
        if self.process and self.process.poll() is None:
            try:
                self.process.send_signal(signal.SIGCONT)  # unstop first
                time.sleep(0.1)
                self.process.terminate()
                self.process.wait(timeout=10)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    self.process.kill()
                except ProcessLookupError:
                    pass
        with self._lock:
            self.status["state"] = "stopped"
            self._write_status()
        self.stopped = True
        self._running = False
        print("[manager] STOPPED")

    def _do_checkpoint(self):
        """Write a trigger file that train.py can pick up, or just log the request."""
        trigger = os.path.join(LOG_DIR, ".checkpoint_trigger")
        Path(trigger).touch()
        print("[manager] Checkpoint requested (trigger file written)")

    def run(self, train_cmd, phase_name="training"):
        """Run training with monitoring."""
        self.phase = phase_name
        with self._lock:
            self.status["phase"] = phase_name
            self.status["state"] = "running"

        # Parse max_steps
        for i, arg in enumerate(train_cmd):
            if arg == "--train-config" and i + 1 < len(train_cmd):
                try:
                    with open(train_cmd[i + 1]) as f:
                        cfg = json.load(f)
                    self.status["max_steps"] = cfg.get("max_steps", 0)
                except:
                    pass

        print(f"[manager] Starting {phase_name}")
        print(f"[manager] Status: {STATUS_FILE}")
        print(f"[manager] Control: {CONTROL_FILE}")

        self.process = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        with self._lock:
            self.status["pid"] = self.process.pid
            self._write_status()

        # Start control checker thread
        self._running = True
        ctrl_thread = threading.Thread(target=self._control_loop, daemon=True)
        ctrl_thread.start()

        # Read output via background thread (macOS select() doesn't work on pipes)
        log_path = os.path.join(LOG_DIR, f"{phase_name}.log")
        self._log_file = open(log_path, "a")
        reader = threading.Thread(target=self._read_output, daemon=True)
        reader.start()

        # Wait for process to finish
        self.process.wait()
        self._running = False
        reader.join(timeout=5)

        self.process.wait()
        exit_code = self.process.returncode

        with self._lock:
            if self.stopped:
                self.status["state"] = "stopped"
            elif exit_code == 0:
                self.status["state"] = "completed"
            elif exit_code in (-15, -9):
                self.status["state"] = "stopped"
            else:
                self.status["state"] = f"failed (exit {exit_code})"
            self._write_status()

        self._running = False
        return exit_code


def main():
    parser = argparse.ArgumentParser(description="Training Manager v2")
    parser.add_argument("--config", required=True, help="Training config JSON")
    parser.add_argument("--model-config", default="configs/model_config.json")
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--phase", default="training")
    args = parser.parse_args()

    if os.path.exists(CONTROL_FILE):
        os.remove(CONTROL_FILE)

    manager = TrainingManager()

    train_cmd = [
        sys.executable, "-u", "scripts/train.py",
        "--train-config", args.config,
        "--model-config", args.model_config,
    ]
    if args.reset_optimizer:
        train_cmd.append("--reset-optimizer")

    exit_code = manager.run(train_cmd, phase_name=args.phase)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
