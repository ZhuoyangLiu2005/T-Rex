# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to disable robot head control.

This script demonstrates how to disable the robot head by setting it to disable mode,
which allows manual manipulation of the head joints.
"""

import time

import tyro
from loguru import logger

from dexcontrol.robot import Robot


def main() -> None:
    """Disable robot head control to allow manual manipulation.

    Sets the head to disable mode, which turns off active control and allows
    the head joints to be moved manually by hand.
    """
    logger.info("Initializing robot and disabling head control...")

    with Robot() as bot:
        bot.head.set_mode("disable")
        time.sleep(2.0)
        logger.info(
            "Head control disabled successfully. You can now move the head manually."
        )


if __name__ == "__main__":
    tyro.cli(main)
