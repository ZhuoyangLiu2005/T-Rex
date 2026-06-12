#!/usr/bin/env python3
"""Simple trajectory replay script for robot control.

Replays pre-recorded NPZ trajectory files with optional smoothing and velocity compensation.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dexcomm import RateLimiter
from loguru import logger
from scipy.ndimage import gaussian_filter1d

from dexcontrol.robot import Robot


def load_trajectory(filepath: Path) -> tuple[dict[str, np.ndarray], float]:
    """Load trajectory from NPZ file."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    data = np.load(filepath)
    trajectory = {}
    control_hz = 500.0  # default

    for key in data.files:
        if key in ["control_hz", "control_frequency"]:
            control_hz = float(data[key].item())
            logger.info(f"Found control frequency: {control_hz}Hz")
        else:
            trajectory[key] = data[key]
            logger.info(f"Loaded {key}: shape {data[key].shape}")

    return trajectory, control_hz


def resample_trajectory(
    trajectory: dict[str, np.ndarray], speed_factor: float
) -> dict[str, np.ndarray]:
    """Resample trajectory by speed factor (>1 = faster, <1 = slower)."""
    if speed_factor == 1.0:
        return trajectory

    resampled = {}
    for part, positions in trajectory.items():
        old_len = len(positions)
        new_len = int(np.ceil(old_len / speed_factor))
        old_indices = np.linspace(0, old_len - 1, old_len)
        new_indices = np.linspace(0, old_len - 1, new_len)

        # Interpolate each joint
        new_positions = np.zeros((new_len, positions.shape[1]))
        for j in range(positions.shape[1]):
            new_positions[:, j] = np.interp(new_indices, old_indices, positions[:, j])

        resampled[part] = new_positions

    return resampled


def smooth_trajectory(
    trajectory: dict[str, np.ndarray], sigma_time: float, hz: float
) -> dict[str, np.ndarray]:
    """Apply Gaussian smoothing to trajectory.

    Args:
        trajectory: Trajectory data
        sigma_time: Smoothing window in seconds (e.g., 0.1 for 100ms window)
        hz: Control frequency

    Returns:
        Smoothed trajectory
    """
    if sigma_time <= 0:
        return trajectory

    # Convert time-based sigma to sample-based sigma
    # sigma_samples = sigma_time * hz
    # For Gaussian filter, sigma is the standard deviation, so roughly 99.7% of
    # the weight is within ±3*sigma samples
    sigma_samples = sigma_time * hz

    logger.info(
        f"Smoothing with {sigma_time:.3f}s window = {sigma_samples:.1f} samples at {hz}Hz"
    )

    return {
        part: gaussian_filter1d(positions, sigma=sigma_samples, axis=0, mode="nearest")
        for part, positions in trajectory.items()
    }


def compute_velocities(
    trajectory: dict[str, np.ndarray], hz: float, smooth_time: float = 0.01
) -> dict[str, np.ndarray]:
    """Compute velocities using finite differences.

    Args:
        trajectory: Position trajectory
        hz: Control frequency
        smooth_time: Smoothing window in seconds for velocity estimation

    Returns:
        Velocity trajectory
    """
    dt = 1.0 / hz
    velocities = {}

    for part, positions in trajectory.items():
        if len(positions) < 2:
            velocities[part] = np.zeros_like(positions)
            continue

        # Smooth before differentiation (convert time to samples)
        if smooth_time > 0:
            sigma_samples = smooth_time * hz
            positions = gaussian_filter1d(
                positions, sigma=sigma_samples, axis=0, mode="nearest"
            )

        # Compute velocities
        vel = np.zeros_like(positions)
        vel[0] = (positions[1] - positions[0]) / dt
        vel[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
        vel[-1] = (positions[-1] - positions[-2]) / dt

        velocities[part] = vel

    return velocities


def visualize_trajectory(
    trajectory: dict[str, np.ndarray],
    velocities: dict[str, np.ndarray] | None,
    hz: float,
) -> None:
    """Visualize trajectory with separate figures for each part."""
    # Set up time array
    num_frames = len(next(iter(trajectory.values())))
    time_array = np.arange(num_frames) / hz

    # Define colors for different joints
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, 10))

    # Create figures for each part
    for part, positions in trajectory.items():
        # Create figure with two subplots always; hide velocity axis if not used
        if velocities and part in velocities:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            fig.suptitle(
                f"{part.upper()} - Trajectory Commands", fontsize=18, fontweight="bold"
            )
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            fig.suptitle(
                f"{part.upper()} - Position Commands", fontsize=18, fontweight="bold"
            )
            ax2.axis("off")

        # Plot positions
        num_joints = positions.shape[1]
        for j in range(num_joints):
            ax1.plot(
                time_array,
                positions[:, j],
                color=colors[j % len(colors)],
                linewidth=2,
                label=f"Joint {j + 1}",
                alpha=0.8,
            )

        ax1.set_title("Position Commands", fontsize=14)
        ax1.set_xlabel("Time (s)", fontsize=12)
        ax1.set_ylabel("Position (rad)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best", frameon=True, fancybox=True, shadow=True)

        # Add min/max annotations
        for j in range(min(num_joints, 3)):  # Limit annotations to avoid clutter
            joint_data = positions[:, j]
            min_idx = np.argmin(joint_data)
            max_idx = np.argmax(joint_data)

            # Annotate min
            ax1.annotate(
                f"J{j + 1}: {joint_data[min_idx]:.3f}",
                xy=(time_array[min_idx], joint_data[min_idx]),
                xytext=(5, -15),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
                arrowprops=dict(arrowstyle="->", alpha=0.5),
            )

            # Annotate max
            ax1.annotate(
                f"J{j + 1}: {joint_data[max_idx]:.3f}",
                xy=(time_array[max_idx], joint_data[max_idx]),
                xytext=(5, 15),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
                arrowprops=dict(arrowstyle="->", alpha=0.5),
            )

        # Plot velocities if available
        if velocities and part in velocities:
            vels = velocities[part]
            for j in range(num_joints):
                ax2.plot(
                    time_array,
                    vels[:, j],
                    color=colors[j % len(colors)],
                    linewidth=2,
                    label=f"Joint {j + 1}",
                    alpha=0.8,
                )

            ax2.set_title("Velocity Commands", fontsize=14)
            ax2.set_xlabel("Time (s)", fontsize=12)
            ax2.set_ylabel("Velocity (rad/s)", fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="best", frameon=True, fancybox=True, shadow=True)

            # Add zero line
            ax2.axhline(y=0, color="black", linestyle="--", alpha=0.3)

            # Add max velocity annotations
            for j in range(min(num_joints, 3)):  # Limit annotations
                vel_data = vels[:, j]
                max_vel_idx = np.argmax(np.abs(vel_data))
                ax2.annotate(
                    f"J{j + 1}: {vel_data[max_vel_idx]:.3f}",
                    xy=(time_array[max_vel_idx], vel_data[max_vel_idx]),
                    xytext=(5, 10),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.7,
                    arrowprops=dict(arrowstyle="->", alpha=0.5),
                )

        plt.tight_layout()

    # Show all figures at once
    plt.show()


def check_goal_difference(diff: dict[str, np.ndarray], max_goal_diff: float) -> bool:
    """Check maximum absolute joint difference and compare with threshold.

    Args:
        diff: Mapping from part name to per-joint delta array (frame - current).
        max_goal_diff: Maximum allowed absolute joint delta in radians.

    Returns:
        True if all absolute joint deltas are within the allowed threshold.
        False if any joint delta exceeds the threshold (also logs details).
    """
    max_diff_part: str | None = None
    max_joint_idx: int = -1
    max_diff: float = 0.0

    for part, delta in diff.items():
        abs_delta = np.abs(delta)
        part_max_idx = int(np.argmax(abs_delta))
        part_max_val = float(abs_delta[part_max_idx])
        if part_max_val > max_diff:
            max_diff = part_max_val
            max_diff_part = part
            max_joint_idx = part_max_idx
    if max_diff > max_goal_diff:
        logger.warning(
            f"Max |diff| between current and target position: {max_diff:.3f} rad in {max_diff_part} joint {max_joint_idx}"
        )
        logger.warning(
            f"This is greater than the allowed goal difference of {max_goal_diff:.3f} rad"
        )
        logger.warning("Exiting...")
        return False

    return True


def run_replay_loop(
    robot: Robot,
    trajectory: dict[str, np.ndarray],
    velocities: dict[str, np.ndarray] | None,
    hz: float,
    num_frames: int,
    rate_limiter: RateLimiter,
    max_goal_diff: float | None = None,
) -> bool:
    """Run the main trajectory replay loop.

    Args:
        robot: Robot instance used to send commands.
        trajectory: Mapping from part name to per-frame joint positions.
        velocities: Optional mapping from part name to per-frame joint velocities.
        hz: Control frequency in Hz.
        num_frames: Number of frames to replay.
        rate_limiter: Rate limiter to maintain control frequency.
        max_goal_diff: Maximum allowed absolute joint delta in radians.

    Returns:
        True if the trajectory was executed successfully, False otherwise.
    """
    # Move to start position
    start_pos = {part: pos[0] for part, pos in trajectory.items()}
    # Filter start_pos to only include components available on this robot
    available_start_pos = {
        part: pos for part, pos in start_pos.items() if robot.has_component(part)
    }
    robot.set_joint_pos(available_start_pos, wait_time=3.0, exit_on_reach=True)
    is_success = True
    try:
        start_time = time.time()
        for i in range(num_frames):
            # Prepare frame data
            frame_pos = {part: pos[i] for part, pos in trajectory.items()}
            if max_goal_diff is not None:
                # Only get positions for components that exist on the robot
                current_pos = {
                    part: getattr(robot, part).get_joint_pos()
                    for part in trajectory.keys()
                    if robot.has_component(part)
                }
                # get the difference between the current position and the frame position
                diff = {
                    part: frame_pos[part] - current_pos[part]
                    for part in current_pos.keys()
                }
                # enforce goal difference threshold
                if not check_goal_difference(diff, max_goal_diff):
                    is_success = False
                    break

            if velocities:
                # Try position+velocity control for each part
                for part, pos in frame_pos.items():
                    if not robot.has_component(part):
                        continue  # Skip components that don't exist on this robot
                    component = getattr(robot, part)
                    if velocities and part in velocities:
                        component.set_joint_pos_vel(pos, velocities[part][i])
                    else:
                        component.set_joint_pos(pos)
            else:
                # Position-only control - filter to only components that exist
                valid_frame_pos = {
                    part: pos
                    for part, pos in frame_pos.items()
                    if robot.has_component(part)
                }
                robot.set_joint_pos(valid_frame_pos, wait_time=0.0)

            # Progress update every 3 seconds
            if i % int(hz * 3) == 0:
                elapsed = time.time() - start_time
                progress = (i / num_frames) * 100
                logger.info(f"Progress: {progress:.0f}% - Time: {elapsed:.1f}s")

            rate_limiter.sleep()

    except KeyboardInterrupt:
        logger.info("Interrupted")
    return is_success


def replay_trajectory(
    filepath: Path,
    control_hz: float | None = None,
    gaussian_sigma: float = 0.0,
    use_velocity: bool = False,
    velocity_sigma: float = 1.0,
    speed_factor: float = 1.0,
    visualize: bool = False,
    max_goal_diff: float = 0.7,
):
    """Main replay function."""
    # Load trajectory
    trajectory, file_hz = load_trajectory(filepath)
    if not trajectory:
        logger.error("Empty trajectory")
        return

    # Set control frequency
    hz = control_hz or file_hz

    # Get original number of frames and duration
    original_frames = len(next(iter(trajectory.values())))
    original_duration = original_frames / hz

    # Apply speed factor by resampling
    if speed_factor != 1.0:
        logger.info(f"Applying speed factor {speed_factor}x")
        trajectory = resample_trajectory(trajectory, speed_factor)

    # Get actual number of frames after resampling
    num_frames = len(next(iter(trajectory.values())))
    duration = num_frames / hz

    logger.info(f"Original: {original_frames} frames, {original_duration:.1f}s")
    logger.info(f"Playback: {num_frames} frames @ {hz}Hz ({duration:.1f}s)")
    logger.info(f"Parts: {', '.join(trajectory.keys())}")

    # Apply smoothing
    if gaussian_sigma > 0:
        logger.info(f"Applying Gaussian smoothing (sigma={gaussian_sigma})")
        trajectory = smooth_trajectory(trajectory, gaussian_sigma, hz)

    # Compute velocities
    velocities = None
    if use_velocity:
        logger.info(f"Computing velocities (sigma={velocity_sigma})")
        velocities = compute_velocities(trajectory, hz, velocity_sigma)

    # Visualize if requested
    if visualize:
        logger.info("Visualizing trajectory...")
        visualize_trajectory(trajectory, velocities, hz)
        if input("Continue with execution? [y/N]: ").lower() != "y":
            logger.info("Execution cancelled")
            return

    # Initialize robot
    robot = Robot()
    rate_limiter = RateLimiter(rate_hz=hz)
    logger.warning("Setting joint positions...")
    logger.warning("Press e-stop if needed!")
    logger.warning(
        "Please ensure the arms and the torso have sufficient space to move."
    )
    logger.warning(
        "If you have an end effector attached, some pre-existing trajectories may cause collisions with the robot."
    )
    logger.warning(
        "Will move the left arm to folded, the right arm to folded, and the head to home pose, then the torso to crouch20_medium."
    )
    if input("Continue? [y/N]: ").lower() != "y":
        logger.info("Execution cancelled")
        return

    # Build initial pose dict based on available components
    init_pose = {
        "left_arm": robot.left_arm.get_predefined_pose("folded"),
        "right_arm": robot.right_arm.get_predefined_pose("folded"),
    }
    if robot.has_component("head"):
        init_pose["head"] = robot.head.get_predefined_pose("home")

    robot.set_joint_pos(init_pose, wait_time=5.0, exit_on_reach=True)

    if robot.has_component("torso"):
        robot.torso.go_to_pose("crouch20_medium", wait_time=5.0, exit_on_reach=True)

    if robot.have_hand("left"):
        robot.left_hand.close_hand()
    if robot.have_hand("right"):
        robot.right_hand.close_hand()

    input("Press Enter to start the replay...")
    # Replay
    run_replay_loop(
        robot=robot,
        trajectory=trajectory,
        velocities=velocities if use_velocity else None,
        hz=hz,
        num_frames=num_frames,
        rate_limiter=rate_limiter,
        max_goal_diff=max_goal_diff,
    )

    logger.info("Done")
    robot.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Replay robot trajectory from NPZ file"
    )
    parser.add_argument("file", type=Path, help="NPZ trajectory file")
    parser.add_argument(
        "--hz", type=float, help="Control frequency (default: from file or 500Hz)"
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.1,
        help="Position smoothing time window in seconds (e.g., 0.1 for 100ms)",
    )
    parser.add_argument(
        "--velocity", action="store_true", help="Use velocity compensation"
    )
    parser.add_argument(
        "--vel-smooth",
        type=float,
        default=0.5,
        help="Velocity smoothing time window in seconds (default: 0.5s = 500ms)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Speed factor (2=2x faster, 0.5=2x slower)",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize trajectory before execution"
    )
    parser.add_argument(
        "--max-goal-diff",
        type=float,
        default=1.0,
        help="Maximum goal difference in radians (default: 1.0)",
    )

    args = parser.parse_args()
    logger.warning(
        "Warning: Be ready to press e-stop if needed! "
        "This example does not check for self-collisions."
    )
    logger.warning(
        "Please ensure the arms and the torso have sufficient space to move."
    )
    if input("Continue? [y/N]: ").lower() != "y":
        return

    if args.file.name == "vega-1_dance.npz":
        speed = np.clip(args.speed, 0.2, 3.0)
        if speed != args.speed:
            logger.warning(f"Speed factor clamped to {speed} (valid range: 0.2-3.0)")
            args.speed = speed
    else:
        speed = args.speed

    if speed > 3.0:
        logger.warning("Speed factor is greater than 3.0!!! This can be dangerous!!!")
        if input("Continue? [y/N]: ").lower() != "y":
            return

    replay_trajectory(
        args.file,
        control_hz=args.hz,
        gaussian_sigma=args.smooth,
        use_velocity=args.velocity,
        velocity_sigma=args.vel_smooth,
        speed_factor=speed,
        visualize=args.visualize,
        max_goal_diff=args.max_goal_diff,
    )


if __name__ == "__main__":
    main()
