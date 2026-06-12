#!/usr/bin/env python3
# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script for moving the robot's torso to a predefined pose.

This script demonstrates how to move the robot's torso to a predefined pose
from the pose library. It includes safety prompts and proper shutdown procedures.
"""

import tyro
from loguru import logger

from dexcontrol.robot import Robot
from dexcontrol.utils.compat import supported_models


@supported_models("vega_1", "vega_1p")
def main(pose: str = "crouch45_high") -> None:
    """Moves the robot's torso to a predefined pose.

    Args:
        pose: Name of the predefined torso pose to move to.
    """
    # Safety confirmation
    logger.warning(
        "Warning: Be ready to press e-stop if needed! "
        "This example does not check for self-collisions."
    )
    logger.warning("Please ensure the torso has sufficient space to move.")
    if input("Continue? [y/N]: ").lower() != "y":
        return

    # Initialize robot
    bot = Robot()

    try:
        logger.info(f"Moving torso to {pose} position")
        bot.torso.go_to_pose(pose, wait_time=6.0)
    finally:
        # Ensure robot is properly shut down
        logger.info("Shutting down robot")
        bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
