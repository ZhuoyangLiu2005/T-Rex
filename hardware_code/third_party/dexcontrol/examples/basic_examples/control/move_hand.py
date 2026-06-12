# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to control robot hand movements.

This script demonstrates basic hand control by opening and closing the hand.
"""

import tyro
from loguru import logger

from dexcontrol.robot import Robot


def main() -> None:
    """Move robot hand through open and close sequence."""
    logger.warning("Warning: Be ready to press e-stop if needed!")
    logger.warning("Please ensure adequate clearance around robot before proceeding.")
    if input("Continue? [y/N]: ").lower() != "y":
        return

    with Robot() as bot:
        logger.info("Moving left hand")

        # Get and log initial position
        current_position = bot.left_hand.get_joint_pos()
        logger.info(f"Initial position: {current_position}")

        # Close hand
        logger.info("Closing hand")
        bot.left_hand.close_hand(wait_time=2.0)

        # Log closed position
        closed_position = bot.left_hand.get_joint_pos()
        logger.info(f"Closed position: {closed_position}")

        # Open hand
        logger.info("Opening hand")
        bot.left_hand.open_hand(wait_time=2.0)

        # Log final position
        final_position = bot.left_hand.get_joint_pos()
        logger.info(f"Final position: {final_position}")


if __name__ == "__main__":
    tyro.cli(main)
