"""Run the chat app with a 395% CPU ceiling, preferring the Linux cgroup limit."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

CPU_LIMIT_CORES = 3.95
SAMPLE_SECONDS = 0.1
TICKS_PER_SECOND = os.sysconf("SC_CLK_TCK") if sys.platform == "linux" else 100


def _apply_cgroup_limit() -> bool:
    """Apply a hard 395% cap when this container exposes a writable cgroup v2."""
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    if not cpu_max.is_file():
        print("[cpu-limit] cgroup v2 cpu.max is unavailable; using process monitor.", flush=True)
        return False
    try:
        cpu_max.write_text("395000 100000", encoding="ascii")
    except OSError as exc:
        print(f"[cpu-limit] cgroup limit unavailable ({exc}); using process monitor.", flush=True)
        return False
    print("[cpu-limit] cgroup CPU ceiling set to 395%.", flush=True)
    return True


def _cpu_seconds(pid: int) -> float:
    """Return total user+system CPU time across a process's threads."""
    total_ticks = 0
    task_root = Path("/proc") / str(pid) / "task"
    for task in task_root.iterdir():
        try:
            stat = (task / "stat").read_text(encoding="ascii")
        except FileNotFoundError:
            continue
        fields = stat[stat.rfind(")") + 2 :].split()
        if len(fields) >= 13:
            total_ticks += int(fields[11]) + int(fields[12])
    return total_ticks / TICKS_PER_SECOND


def main() -> int:
    if sys.platform != "linux":
        os.execv(sys.executable, [sys.executable, "app.py"])

    cgroup_limited = _apply_cgroup_limit()
    limit_cores = CPU_LIMIT_CORES if cgroup_limited else 3.8
    child = subprocess.Popen([sys.executable, "app.py"])

    def forward_signal(signum: int, _frame: object) -> None:
        if child.poll() is None:
            try:
                os.kill(child.pid, signal.SIGCONT)
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, forward_signal)
    signal.signal(signal.SIGINT, forward_signal)

    last_cpu = _cpu_seconds(child.pid)
    period_started = time.monotonic()
    while child.poll() is None:
        if cgroup_limited:
            time.sleep(1.0)
            continue
        time.sleep(SAMPLE_SECONDS)
        if child.poll() is not None:
            break

        now = time.monotonic()
        cpu_now = _cpu_seconds(child.pid)
        allowed_cpu = limit_cores * (now - period_started)
        excess_cpu = cpu_now - last_cpu - allowed_cpu
        if excess_cpu > 0:
            pause = excess_cpu / limit_cores
            os.kill(child.pid, signal.SIGSTOP)
            time.sleep(pause)
            if child.poll() is None:
                os.kill(child.pid, signal.SIGCONT)
            last_cpu = cpu_now
            period_started = time.monotonic()

    return child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
