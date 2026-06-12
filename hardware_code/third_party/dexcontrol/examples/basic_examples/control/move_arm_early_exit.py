# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script demonstrating early exit functionality in robot arm movements.

This script demonstrates the difference between standard pose movement and early exit
pose movement. It shows how the `exit_on_reach` parameter can be used to exit
movement commands as soon as the target position is reached, rather than waiting
for the full specified wait time.
"""

import time

import tyro
from loguru import logger

from dexcontrol.robot import Robot


def main() -> None:
    """Demonstrates early exit functionality in robot arm pose movements.

    This function performs two pose movements:
    1. Standard movement to "L_shape" pose with full wait time
    2. Early exit movement to "folded" pose that exits when position is reached

    The function measures and displays the actual time taken for each movement
    to demonstrate the efficiency improvement of early exit functionality.
    """
    # Safety warnings and user confirmation
    logger.warning("Warning: Be ready to press e-stop if needed!")
    logger.warning("Please make sure the arms have enough space to move.")

    user_input = input("Continue? [y/N]: ").lower()
    if user_input != "y":
        logger.info("Operation cancelled by user.")
        return

    # Initialize robot and movement parameters
    robot = Robot()
    wait_time = 6.0
    tolerance = 0.05

    # Movement 1: Standard pose movement (waits full time)
    logger.info("Starting standard pose movement to 'L_shape'...")
    start_time = time.time()
    robot.left_arm.go_to_pose("L_shape", wait_time=wait_time)
    actual_time = time.time() - start_time

    logger.info("Standard movement completed:")
    logger.info(f"  Max wait time: {wait_time:.2f} seconds")
    logger.info(f"  Actual time taken: {actual_time:.2f} seconds")

    # Movement 2: Early exit pose movement
    logger.info("Starting early exit pose movement to 'folded'...")
    start_time = time.time()
    robot.left_arm.go_to_pose(
        "folded",
        wait_time=wait_time,
        exit_on_reach=True,
        exit_on_reach_kwargs={"tolerance": tolerance},
    )
    actual_time = time.time() - start_time

    logger.info("Early exit movement completed:")
    logger.info(f"  Max wait time: {wait_time:.2f} seconds")
    logger.info(f"  Actual time taken: {actual_time:.2f} seconds")
    logger.info(f"  Position tolerance: {tolerance:.3f} radians")


if __name__ == "__main__":
    tyro.cli(main)
