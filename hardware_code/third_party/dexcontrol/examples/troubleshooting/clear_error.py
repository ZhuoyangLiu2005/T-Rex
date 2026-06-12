# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to clear robot component errors.

This script demonstrates how to clear error states for all robot components
including left arm, right arm, head, and chassis.
"""

import tyro
from loguru import logger

from dexcontrol.robot import Robot


def main() -> None:
    """Clear errors for all robot components.

    Initializes a robot instance and attempts to clear error states
    for all major components (left_arm, right_arm, head, chassis).
    """
    # Initialize robot with default configuration
    robot = Robot()

    # List of components that support error clearing
    components = ["left_arm", "right_arm", "head", "chassis", "left_hand", "right_hand"]

    logger.info("Starting error clearing process for all components...")

    for component in components:
        try:
            logger.info(f"Clearing error state for {component}...")
            robot.clear_error(component)
        except Exception as e:
            logger.error(f"Failed to clear error for {component}: {e}")

    logger.info("Error clearing process completed")

    # Clean shutdown
    robot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
