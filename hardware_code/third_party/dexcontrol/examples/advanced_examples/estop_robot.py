# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to trigger the estop of the robot.

This script demonstrates how to trigger the estop of the robot.
By default, it will activate the estop. Use --deactivate to deactivate it.
"""

import time

import tyro
from loguru import logger

from dexcontrol.robot import Robot


def main(deactivate: bool = False) -> None:
    """Trigger the estop of the robot.

    Args:
        deactivate: If True, deactivate the estop. If False (default), activate the estop.
    """
    # Initialize robot with default configuration
    bot = Robot()

    enable = not deactivate

    try:
        if enable:
            bot.estop.activate()
        else:
            bot.estop.deactivate()

        time.sleep(3)
        bot.estop.show()

    except Exception as e:
        logger.error(f"Failed to set estop: {e}")
    finally:
        # Display robot system information
        bot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
