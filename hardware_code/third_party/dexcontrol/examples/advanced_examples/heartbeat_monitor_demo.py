# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Heartbeat monitor demo with GIL stress testing.

Spawns CPU-bound Python threads to create GIL contention while monitoring
the heartbeat. The Rust-based monitor is unaffected; a Python-threaded
subscriber would false-timeout under this load.

Usage:
    python heartbeat_monitor_demo.py                     # with GIL stress (default)
    python heartbeat_monitor_demo.py --no-gil-stress     # without GIL stress
    python heartbeat_monitor_demo.py --num-workers 8     # more workers
"""

import os
import random
import sys
import threading
import time

import typer
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.table import Table

from dexcontrol.robot import Robot
from dexcontrol.utils.constants import DISABLE_HEARTBEAT_ENV_VAR


def _gil_worker(stop_event: threading.Event):
    """CPU-bound worker: pure Python matrix multiply in a tight loop."""
    size = 50
    a = [[random.random() for _ in range(size)] for _ in range(size)]
    b = [[random.random() for _ in range(size)] for _ in range(size)]

    while not stop_event.is_set():
        result = [[0.0] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                s = 0.0
                for k in range(size):
                    s += a[i][k] * b[k][j]
                result[i][j] = s


def _build_status_table(heartbeat, gil_stress: bool, num_workers: int) -> Table:
    """Build the heartbeat status table (same format as heartbeat.show())."""
    status = heartbeat.get_status()

    table = Table(title="Heartbeat Monitor Status")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value")

    # Mode
    mode_parts = []
    if not status["enabled"]:
        mode_parts.append("[yellow]Exit Disabled[/]")
    if status["paused"]:
        mode_parts.append("[yellow]Paused[/]")
    if not mode_parts:
        mode_parts.append("[green]Active[/]")
    table.add_row("Mode", " | ".join(mode_parts))

    # Signal
    active_style = "green" if status["is_active"] else "red"
    table.add_row(
        "Signal",
        f"[{active_style}]{'Receiving' if status['is_active'] else 'No Signal'}[/]",
    )

    # Server uptime
    if status["last_value"] is not None:
        uptime_str = heartbeat.format_uptime(status["last_value"])
        table.add_row("Server Uptime", f"[blue]{uptime_str}[/]")

    # Time since last heartbeat
    if status["time_since_last"] is not None:
        time_since = float(status["time_since_last"])
        timeout = (
            float(status["timeout_seconds"])
            if status["timeout_seconds"] is not None
            else 1.0
        )
        time_style = (
            "red"
            if time_since > timeout
            else "yellow"
            if time_since > timeout * 0.5
            else "green"
        )
        table.add_row("Last Heartbeat", f"[{time_style}]{time_since:.1f}s ago[/]")

    # Timeout
    table.add_row("Timeout", f"[blue]{status['timeout_seconds']}s[/]")

    # GIL stress indicator
    if gil_stress:
        table.add_row("GIL Stress", f"[bold magenta]{num_workers} workers[/]")

    return table


def main(
    monitor_duration: float = 30.0,
    disable_heartbeat: bool = False,
    gil_stress: bool = True,
    num_workers: int = 4,
):
    """Demonstrate heartbeat monitoring with optional GIL stress testing.

    Args:
        monitor_duration: How long to monitor the heartbeat in seconds.
        disable_heartbeat: If True, disable heartbeat monitoring.
        gil_stress: If True, spawn CPU-bound threads to create GIL contention.
        num_workers: Number of CPU-bound worker threads for GIL stress.
    """
    if disable_heartbeat:
        os.environ[DISABLE_HEARTBEAT_ENV_VAR] = "1"

    # Set up GIL stress
    old_interval = sys.getswitchinterval()
    stop_event = threading.Event()
    workers: list[threading.Thread] = []

    if gil_stress:
        sys.setswitchinterval(0.1)  # 100ms (default 5ms)
        for i in range(num_workers):
            t = threading.Thread(
                target=_gil_worker,
                args=(stop_event,),
                name=f"gil-stress-{i}",
                daemon=True,
            )
            t.start()
            workers.append(t)
        logger.info(f"GIL stress: {num_workers} workers, switch_interval=100ms")

    bot = Robot()

    try:
        logger.info(f"Monitoring heartbeat for {monitor_duration}s...")

        console = Console()
        start_time = time.time()
        with Live(console=console, refresh_per_second=4) as live:
            while time.time() - start_time < monitor_duration:
                live.update(_build_status_table(bot.heartbeat, gil_stress, num_workers))
                time.sleep(0.25)

        logger.info("Demo completed successfully")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        stop_event.set()
        for w in workers:
            w.join(timeout=2.0)
        if gil_stress:
            sys.setswitchinterval(old_interval)
        bot.shutdown()


if __name__ == "__main__":
    typer.run(main)
