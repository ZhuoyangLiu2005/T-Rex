# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Check and display the time difference between the robot and NTP server.

This script queries the robot's NTP (Network Time Protocol) status and prints
the time difference between the robot's system clock and the NTP server.
"""

import tyro

from dexcontrol.robot import Robot


def main() -> None:
    """Query and display the robot's NTP time difference.

    This function initializes a Robot instance, queries the NTP time
    difference (showing the result), and then performs a clean shutdown.

    Raises:
        Exception: If the robot fails to initialize or query NTP.
    """
    robot = Robot()
    try:
        robot.query_ntp(show=True)
    finally:
        robot.shutdown()


if __name__ == "__main__":
    tyro.cli(main)
