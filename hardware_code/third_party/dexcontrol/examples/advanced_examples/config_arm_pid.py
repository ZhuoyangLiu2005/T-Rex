# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to get/set arm PID P-gain multipliers.

This script demonstrates how to query and set the PID P-gain multipliers
for the robot arm joints. The multipliers are applied to the factory
default PID values.

WARNING: Modifying PID values can affect robot stability and performance.
Use with caution and start with small adjustments.
"""

from typing import Literal

import tyro
from loguru import logger

from dexcontrol.robot import Robot


def get_pid(side: Literal["left", "right", "both"] = "both") -> None:
    """Get arm PID P-gain multipliers.

    Args:
        side: Which arm to query ('left', 'right', or 'both').
    """
    with Robot() as bot:
        if side in ("left", "both"):
            result = bot.left_arm.get_pid()
            logger.info(f"Left arm PID: {result}")

        if side in ("right", "both"):
            result = bot.right_arm.get_pid()
            logger.info(f"Right arm PID: {result}")


def set_pid(
    side: Literal["left", "right", "both"] = "left",
    p_multipliers: list[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
) -> None:
    """Set arm PID P-gain multipliers.

    Args:
        side: Which arm to configure ('left', 'right', or 'both').
        p_multipliers: List of 7 P-gain multipliers, one for each joint.
            Values are multiplied with factory defaults (e.g., 1.0 = default,
            0.5 = half, 2.0 = double).
    """
    if len(p_multipliers) != 7:
        raise ValueError("p_multipliers must have exactly 7 values (one per joint)")

    with Robot() as bot:
        if side in ("left", "both"):
            logger.info(f"Setting left arm PID multipliers: {p_multipliers}")
            result = bot.left_arm.set_pid(p_multipliers)
            logger.info(f"Left arm result: {result}")

        if side in ("right", "both"):
            logger.info(f"Setting right arm PID multipliers: {p_multipliers}")
            result = bot.right_arm.set_pid(p_multipliers)
            logger.info(f"Right arm result: {result}")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"get": get_pid, "set": set_pid})
