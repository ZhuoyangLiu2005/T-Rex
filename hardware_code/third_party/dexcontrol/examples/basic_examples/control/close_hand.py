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

This script demonstrates basic hand control by opening both hands.
"""

import time

import tyro
from loguru import logger

from dexcontrol.robot import Robot


def main() -> None:
    """Close both robot hands and then shut down.

    This function initializes the robot, closes both hands simultaneously,
    waits briefly, and then shuts down the robot properly.
    """
    logger.info("Initializing robot")
    with Robot() as bot:
        logger.info("Closing both hands")
        bot.left_hand.close_hand()
        bot.right_hand.close_hand()

        logger.info("Waiting for hand movement to complete")
        time.sleep(2)


if __name__ == "__main__":
    tyro.cli(main)
