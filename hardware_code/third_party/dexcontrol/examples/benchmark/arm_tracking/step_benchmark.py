# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Step response latency benchmark for all 7 joints of an arm.

This script tests ALL 7 joints sequentially by applying a step input to each
joint individually while keeping other joints at their zero position. For each
joint it generates a summary plot of command vs actual and saves raw data.

Replaces the separate step_test.py + plot_step.py workflow with a single
self-contained benchmark.
"""

import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import tyro
from common import NUM_JOINTS, ZERO_POS, verify_zero_position
from dexcomm import RateLimiter
from loguru import logger
from matplotlib.axes import Axes

from dexcontrol.robot import Robot
from dexcontrol.utils.trajectory_utils import generate_linear_trajectory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent


def plot_joint(
    ax: Axes,
    times: np.ndarray,
    cmd: np.ndarray,
    actual: np.ndarray,
    joint_idx: int,
    step_amplitude: float,
    step_time: float,
) -> None:
    """Plot command vs actual for one joint on the given axes.

    Args:
        ax: Matplotlib axes to draw on.
        times: (N,) time array in seconds.
        cmd: (N, 7) command array.
        actual: (N, 7) state array.
        joint_idx: Index of the tested joint.
        step_amplitude: Step amplitude in radians (for the title).
        step_time: Time at which the step occurs in seconds.
    """
    ax.plot(times, cmd[:, joint_idx], color="tab:blue", label="Command")
    ax.plot(times, actual[:, joint_idx], color="tab:orange", label="Actual")
    ax.axvline(
        step_time, color="tab:green", linestyle="--", alpha=0.7, label="Step time"
    )
    ax.set_title(f"Joint {joint_idx} - Step ({np.rad2deg(step_amplitude):.1f} deg)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (rad)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@dataclass
class Args:
    """Step response latency benchmark for all 7 joints of an arm."""

    side: Literal["left", "right"] = "right"
    """Which arm to test (left or right)."""

    duration: float = 5.0
    """Duration of the step test per joint in seconds."""

    control_hz: int = 200
    """Control loop frequency in Hz."""

    step_amplitude: float = np.deg2rad(30.0)
    """Amplitude of the step input in radians."""

    step_time: float = 0.3
    """Time at which the step transition starts in seconds."""

    transition_time: float = 0.3
    """Duration of the step transition in seconds."""

    max_vel: float | None = None
    """Maximum velocity in rad/s. If None, auto-calculated from transition_time."""

    output_dir: str = "results"
    """Base output directory for results."""

    no_confirm: bool = False
    """Skip the interactive safety confirmation prompt."""


def main(args: Args) -> None:
    """Run the step response latency benchmark on all 7 joints."""

    # ------------------------------------------------------------------
    # Safety warning
    # ------------------------------------------------------------------
    warnings.warn(
        "This benchmark moves the robot arm through step trajectories. "
        "No collision checking is performed. Ensure the workspace is clear.",
        stacklevel=1,
    )
    logger.warning(
        "The robot arm will move. Make sure the workspace is clear "
        "and the e-stop is accessible."
    )

    if not args.no_confirm:
        answer = input("Continue? [y/N]: ").strip().lower()
        if answer != "y":
            logger.info("Aborted by user.")
            return

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    bot = Robot()
    arm = bot.left_arm if args.side == "left" else bot.right_arm
    arm_name = f"{args.side} arm"

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = SCRIPT_DIR / args.output_dir
    result_dir = output_base / f"{timestamp_str}_{args.side}_step"
    result_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {result_dir}")

    # ------------------------------------------------------------------
    # Pre-compute step trajectory (shared across all joints)
    # ------------------------------------------------------------------
    max_vel = args.max_vel
    if max_vel is None:
        max_vel = abs(args.step_amplitude) / args.transition_time

    transition_trajectory, _ = generate_linear_trajectory(
        np.array([0.0]), np.array([args.step_amplitude]), max_vel, args.control_hz
    )

    num_samples_total = int(args.duration * args.control_hz)
    num_samples_before = int(args.step_time * args.control_hz)

    pos_traj = np.concatenate(
        [
            np.full(num_samples_before, 0.0),
            transition_trajectory.flatten(),
            np.full(
                max(
                    0,
                    num_samples_total - num_samples_before - len(transition_trajectory),
                ),
                args.step_amplitude,
            ),
        ]
    )

    # Truncate to exact duration
    if len(pos_traj) > num_samples_total:
        pos_traj = pos_traj[:num_samples_total]

    # ------------------------------------------------------------------
    # Move to zero position and verify
    # ------------------------------------------------------------------
    logger.info(f"Moving {arm_name} to zero position")
    arm.set_joint_pos(ZERO_POS, wait_time=3.0)
    time.sleep(0.5)

    verify_zero_position(arm)

    # ------------------------------------------------------------------
    # Test each joint
    # ------------------------------------------------------------------
    all_commands: list[np.ndarray] = []
    all_states: list[np.ndarray] = []

    for joint_idx in range(NUM_JOINTS):
        logger.info(f"--- Testing joint {joint_idx} ---")

        # Return to zero between joints (skip before joint 0)
        if joint_idx > 0:
            arm.set_joint_pos(ZERO_POS, wait_time=2.0)

        commands: list[np.ndarray] = []
        states: list[np.ndarray] = []
        rate_limiter = RateLimiter(args.control_hz)
        start_time = time.time()
        traj_idx = 0

        while time.time() - start_time < args.duration and traj_idx < len(pos_traj):
            step_value = pos_traj[traj_idx]

            joint_pos = ZERO_POS.copy()
            joint_pos[joint_idx] += step_value

            arm.set_joint_pos(joint_pos)
            commands.append(joint_pos.copy())
            states.append(arm.get_joint_pos().copy())
            rate_limiter.sleep()
            traj_idx += 1

        cmd_array = np.array(commands)
        state_array = np.array(states)
        all_commands.append(cmd_array)
        all_states.append(state_array)

    # ------------------------------------------------------------------
    # Summary plot (4x2 grid, last cell off)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes_flat = axes.flatten()

    for joint_idx in range(NUM_JOINTS):
        ax = axes_flat[joint_idx]
        cmd_array = all_commands[joint_idx]
        state_array = all_states[joint_idx]
        num_samples = len(cmd_array)
        times = np.arange(num_samples) / args.control_hz
        plot_joint(
            ax,
            times,
            cmd_array,
            state_array,
            joint_idx,
            args.step_amplitude,
            args.step_time,
        )

    # Turn off the unused 8th subplot
    axes_flat[NUM_JOINTS].set_visible(False)

    fig.suptitle(
        f"Step Benchmark ({args.side} arm) | amp={np.rad2deg(args.step_amplitude):.1f} deg, "
        f"step_t={args.step_time} s, "
        f"dur={args.duration} s, "
        f"ctrl={args.control_hz} Hz",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(result_dir / "summary.png", dpi=150)
    plt.close(fig)
    logger.info(f"Summary plot saved to {result_dir / 'summary.png'}")

    # ------------------------------------------------------------------
    # Save raw data
    # ------------------------------------------------------------------
    save_dict: dict[str, Any] = {
        "side": args.side,
        "duration": args.duration,
        "control_hz": args.control_hz,
        "step_amplitude": args.step_amplitude,
        "step_time": args.step_time,
        "transition_time": args.transition_time,
        "zero_pos": ZERO_POS,
    }
    for joint_idx in range(NUM_JOINTS):
        save_dict[f"commands_{joint_idx}"] = all_commands[joint_idx]
        save_dict[f"states_{joint_idx}"] = all_states[joint_idx]

    data_path = result_dir / "data.npz"
    np.savez(data_path, **save_dict)
    logger.info(f"Raw data saved to {data_path}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    logger.info("Returning to zero position")
    arm.set_joint_pos(ZERO_POS, wait_time=3.0)
    bot.shutdown()
    logger.info("Benchmark complete")


if __name__ == "__main__":
    main(tyro.cli(Args))
