# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""
Keyboard Joint Control for Robot Components

This script allows controlling individual joints of various robot components (arms, torso, head)
using keyboard inputs. User can specify which component to control and can switch between joints
during operation using number keys.

Keys:
    '0'-'9': Select joint by index
    'w': Increase selected joint angle
    's': Decrease selected joint angle
    'q': Quit
    Ctrl+C: Quit (emergency exit)

Usage:
    python keyboard_joint_control.py --component left_arm --step_size 5.0
"""

import signal
import sys
import termios
import time
import tty
from typing import Literal

import numpy as np
import tyro
from loguru import logger

from dexcontrol.robot import Robot


def get_key() -> str:
    """Get a single keypress from the user without requiring Enter.

    Returns:
        The character corresponding to the key pressed by the user
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def restore_terminal():
    """Restore terminal settings."""
    print("\033[?25h", end="", flush=True)  # Show cursor


def control_joints(
    robot: Robot,
    component: Literal["left_arm", "right_arm", "torso", "head"],
    step_size: float,
) -> None:
    """
    Interactive keyboard control of joints with ability to switch between joints.

    Args:
        robot: Robot instance
        component: Component to control ("left_arm", "right_arm", "torso", "head")
        step_size: Angle change in degrees per keypress
    """
    wait_time = 0.5
    # Get the component object
    if component == "left_arm":
        comp_obj = robot.left_arm
        max_joint_idx = 6  # 7 joints (0-6)
    elif component == "right_arm":
        comp_obj = robot.right_arm
        max_joint_idx = 6  # 7 joints (0-6)
    elif component == "torso":
        comp_obj = robot.torso
        max_joint_idx = 2  # 3 joints (0-2)
    elif component == "head":
        comp_obj = robot.head
        comp_obj.set_mode("enable")
        max_joint_idx = 2  # 3 joints (0-2)
    else:
        raise ValueError(f"Invalid component: {component}")

    # Set default joint to the last one in the component
    joint_idx = max_joint_idx

    # Safety limits for angle changes (in degrees)
    max_angle_changes = {
        "left_arm": 10.0,
        "right_arm": 10.0,
        "torso": 5.0,
        "head": 10.0,
    }
    max_angle_change = max_angle_changes[component]

    # Adjust step size if it exceeds maximum angle change
    if step_size > max_angle_change:
        logger.warning(f"Step size limited to {max_angle_change} degrees for safety")
        step_size = max_angle_change

    # Setup Ctrl+C handler to ensure proper cleanup
    def sigint_handler(sig, frame):
        print("\nEmergency stop triggered by Ctrl+C")
        restore_terminal()
        raise KeyboardInterrupt

    original_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, sigint_handler)

    # Initialize control
    print(f"\nControlling {component}, default joint index: {joint_idx}")
    print(f"Step size: {step_size} degrees")
    print("Commands:")
    print("  w: Increase joint angle")
    print("  s: Decrease joint angle")
    print(f"  0-{max_joint_idx}: Select joint by index")
    print("  q: Quit")
    print("  Ctrl+C: Quit")
    print("\nPress any key to start... (then press 'q' to quit)")
    get_key()  # Wait for initial keypress

    print("\033[?25l", end="")  # Hide cursor
    try:
        # Get initial joint positions
        current_pos = np.array(comp_obj.get_joint_pos())
        current_pos_deg = np.rad2deg(current_pos)

        # Display initial position
        print(
            f"\rControlling joint {joint_idx}: {current_pos_deg[joint_idx]:.2f}° | Press 0-{max_joint_idx} to select joint, w/s to move, q to quit",
            end="",
            flush=True,
        )

        # Main control loop
        last_command_time = (
            time.time() - 1.0
        )  # Set to a past time to allow first command to execute
        rate_limit = 0.1  # Minimum time between commands in seconds

        while True:
            # Non-blocking key check
            if sys.stdin.isatty():
                key = get_key()

                # Check for Ctrl+C (ASCII value 3)
                if ord(key) == 3:
                    print("\nQuitting...")
                    break

                current_time = time.time()

                if key == "q":
                    print("\nQuitting...")
                    break
                # Handle joint selection with number keys
                elif key.isdigit() and int(key) <= max_joint_idx:
                    new_joint_idx = int(key)
                    if 0 <= new_joint_idx <= max_joint_idx:
                        joint_idx = new_joint_idx
                        # Update current positions before displaying
                        current_pos = np.array(comp_obj.get_joint_pos())
                        current_pos_deg = np.rad2deg(current_pos)
                        print(
                            f"\rControlling joint {joint_idx}: {current_pos_deg[joint_idx]:.2f}° | Press 0-{max_joint_idx} to select joint, w/s to move, q to quit",
                            end="",
                            flush=True,
                        )
                        last_command_time = current_time

                # Movement commands
                elif key == "w" or key == "s":
                    # Only apply rate limiting if too soon after last command
                    if current_time - last_command_time < rate_limit:
                        continue

                    # Process command
                    if key == "w":
                        # Increase joint angle using relative control
                        rel_joint_pos = np.zeros(len(current_pos))
                        rel_joint_pos[joint_idx] = np.deg2rad(step_size)
                    else:  # key == 's'
                        # Decrease joint angle using relative control
                        rel_joint_pos = np.zeros(len(current_pos))
                        rel_joint_pos[joint_idx] = np.deg2rad(-step_size)

                    # Use relative control API
                    comp_obj.set_joint_pos(
                        rel_joint_pos, relative=True, wait_time=wait_time
                    )
                    last_command_time = current_time

                    # Update display
                    current_pos = np.array(comp_obj.get_joint_pos())
                    current_pos_deg = np.rad2deg(current_pos)
                    print(
                        f"\rControlling joint {joint_idx}: {current_pos_deg[joint_idx]:.2f}° | Press 0-{max_joint_idx} to select joint, w/s to move, q to quit",
                        end="",
                        flush=True,
                    )

    except KeyboardInterrupt:
        print("\nControl interrupted by user")
    finally:
        # Restore signal handler
        signal.signal(signal.SIGINT, original_sigint_handler)
        # Restore terminal
        restore_terminal()


def main(
    component: Literal["left_arm", "right_arm", "torso", "head"],
    step_size: float = 5.0,
) -> None:
    """Keyboard joint control for robot components.

    Args:
        component: Component to control ("left_arm", "right_arm", "torso", "head")
        step_size: Step size in degrees for each keypress (default: 5.0)
    """
    # Initialize robot
    logger.warning("Warning: Be ready to press e-stop if needed!")
    logger.warning("Please ensure the robot has enough space to move safely.")

    # Create robot instance
    bot = Robot()

    try:
        # Start joint control
        control_joints(bot, component, step_size)
    except Exception as e:
        logger.error(f"Error during joint control: {e}")
    finally:
        # Ensure robot disconnects even if there's an error
        bot.shutdown()
        logger.info("Robot disconnected")


if __name__ == "__main__":
    tyro.cli(main)
