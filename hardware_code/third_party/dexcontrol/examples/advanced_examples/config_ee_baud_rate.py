# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Interactive end-effector baud rate configuration.

This script provides a menu-driven interface to query and set the RS485 baud rate
for the end-effector communication on the robot arms.

Usage:
    python config_ee_baud_rate.py
"""

from typing import Literal

from loguru import logger

from dexcontrol.robot import Robot

BAUD_RATE_OPTIONS = [115200, 460800, 921600, 1000000, 3000000]


def select_arm() -> Literal["left", "right", "both"] | None:
    """Prompt user to select which arm to configure.

    Returns:
        Selected arm ('left', 'right', 'both') or None if cancelled.
    """
    logger.info("Select arm:")
    logger.info("  [1] Left")
    logger.info("  [2] Right")
    logger.info("  [3] Both")
    logger.info("  [0] Cancel")

    choice = input("\nEnter choice: ").strip()

    if choice == "1":
        return "left"
    elif choice == "2":
        return "right"
    elif choice == "3":
        return "both"
    else:
        return None


def select_baud_rate() -> int | None:
    """Prompt user to select a baud rate.

    Returns:
        Selected baud rate or None if cancelled.
    """
    logger.info("Select baud rate:")
    for i, rate in enumerate(BAUD_RATE_OPTIONS, 1):
        logger.info(f"  [{i}] {rate}")
    logger.info("  [0] Cancel (keep current)")

    choice = input("\nEnter choice: ").strip()

    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(BAUD_RATE_OPTIONS):
            return BAUD_RATE_OPTIONS[idx - 1]

    return None


def display_current_baud_rate(
    robot: Robot, arm: Literal["left", "right", "both"]
) -> None:
    """Display current baud rate for selected arm(s).

    Args:
        robot: Robot instance.
        arm: Which arm to query.
    """
    logger.info("Current baud rate:")

    if arm in ("left", "both"):
        result = robot.left_arm.get_ee_baud_rate()
        if result.get("success"):
            logger.info(f"  Left arm:  {result.get('baud_rate')}")
        else:
            logger.error(f"  Left arm:  Error - {result.get('message')}")

    if arm in ("right", "both"):
        result = robot.right_arm.get_ee_baud_rate()
        if result.get("success"):
            logger.info(f"  Right arm: {result.get('baud_rate')}")
        else:
            logger.error(f"  Right arm: Error - {result.get('message')}")


def set_baud_rate(
    robot: Robot, arm: Literal["left", "right", "both"], baud_rate: int
) -> None:
    """Set baud rate for selected arm(s).

    Args:
        robot: Robot instance.
        arm: Which arm to configure.
        baud_rate: Baud rate to set.
    """
    logger.info(f"Setting baud rate to {baud_rate}...")

    if arm in ("left", "both"):
        result = robot.left_arm.set_ee_baud_rate(baud_rate)
        if not result.get("success"):
            logger.error(f"  Left arm:  Failed - {result.get('message')}")

    if arm in ("right", "both"):
        result = robot.right_arm.set_ee_baud_rate(baud_rate)
        if not result.get("success"):
            logger.error(f"  Right arm: Failed - {result.get('message')}")


def main() -> None:
    """Interactive end-effector baud rate configuration."""
    arm = select_arm()
    if arm is None:
        logger.info("Cancelled.")
        return

    # Connect to robot
    logger.info("Connecting to robot...")
    with Robot() as robot:
        # Display current baud rate
        display_current_baud_rate(robot, arm)

        # Select new baud rate
        new_baud_rate = select_baud_rate()
        if new_baud_rate is None:
            logger.info("Cancelled. Keeping current baud rate.")
            return

        # Set new baud rate
        set_baud_rate(robot, arm, new_baud_rate)

        # Display updated baud rate
        logger.info("Verifying new baud rate...")
        display_current_baud_rate(robot, arm)


if __name__ == "__main__":
    main()
