# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to reboot robot components.

This script demonstrates how to use the standalone RobotQueryInterface
to reboot specific robot components without initializing the full Robot class.
This is more lightweight and efficient when you only need to perform
simple query operations.
"""

from typing import Literal

import tyro
from loguru import logger

from dexcontrol.core.robot_query_interface import RobotQueryInterface


def main(part: Literal["arm", "chassis", "torso"]) -> None:
    """Reboot part of the robot using standalone query interface.

    Args:
        part: Part of the robot to reboot.
    """
    query_interface = RobotQueryInterface.create()

    try:
        logger.info(f"Rebooting {part}...")
        query_interface.reboot_component(part)
        logger.info(f"Reboot command sent for {part}")
    finally:
        # Clean up the zenoh session
        query_interface.close()


if __name__ == "__main__":
    tyro.cli(main)
