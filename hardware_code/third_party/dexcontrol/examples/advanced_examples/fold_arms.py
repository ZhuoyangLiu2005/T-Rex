# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script for folding robot arms to a predefined pose.

This script demonstrates how to move both robot arms to a folded position
with an option to compensate for torso pitch. It includes safety prompts
and proper shutdown procedures.
"""

import numpy as np
import tyro
from loguru import logger

from dexcontrol.robot import Robot


def main(comp_pitch: bool = False) -> None:
    """Moves both robot arms to the folded position.

    Args:
        comp_pitch: Whether to compensate for torso pitch angle.
    """
    # CRITICAL SAFETY CONFIRMATION - DOUBLE PROMPT
    logger.warning("[bold red]⚠️  CRITICAL SAFETY WARNING ⚠️[/]")
    logger.warning("[bold red]THIS SCRIPT DOES NOT CHECK FOR SELF-COLLISIONS![/]")
    logger.warning(
        "[bold red]The robot arms may collide with each other or the robot body![/]"
    )
    logger.warning(
        "[bold red]Have your emergency stop (e-stop) button ready at ALL times![/]"
    )
    logger.warning(
        "[yellow]Highly recommend running init_arm_safe.py first to ensure safe starting pose.[/]"
    )
    logger.warning(
        "[yellow]Ensure the arms have sufficient space to move without obstruction.[/]"
    )

    # First confirmation
    first_confirm = input(
        "Do you understand the collision risks and have e-stop ready? [y/N]: "
    ).lower()
    if first_confirm != "y":
        logger.info("[green]Operation cancelled for safety.[/]")
        return

    # Second confirmation - more explicit
    logger.warning("[bold red]⚠️  FINAL SAFETY CHECK ⚠️[/]")
    logger.warning(
        "[bold red]You are about to move robot arms WITHOUT collision detection![/]"
    )
    logger.warning(
        "[bold red]Arms may collide with robot body, each other, or obstacles![/]"
    )
    second_confirm = input(
        "Type 'PROCEED' (all caps) to continue or anything else to cancel: "
    )
    if second_confirm != "PROCEED":
        logger.info("[green]Operation cancelled. Safety first![/]")
        return

    logger.warning("[bold magenta]Proceeding with arm movement - KEEP E-STOP READY![/]")

    target_pose = "folded"

    with Robot() as bot:
        logger.info("Moving both arms to folded position")

        if comp_pitch:
            # Compensate for torso pitch and move arms
            if bot.has_component("torso"):
                torso_pitch = bot.torso.pitch_angle
            else:
                torso_pitch = np.pi / 2  # Default for upper body variants
            logger.debug(f"Current torso pitch: {torso_pitch:.4f} rad")

            left_arm_target_pose = bot.compensate_torso_pitch(
                bot.left_arm.get_predefined_pose(target_pose),
                "left_arm",
            )
            right_arm_target_pose = bot.compensate_torso_pitch(
                bot.right_arm.get_predefined_pose(target_pose),
                "right_arm",
            )

            # Move both arms simultaneously
            bot.set_joint_pos(
                {
                    "left_arm": left_arm_target_pose,
                    "right_arm": right_arm_target_pose,
                },
                wait_time=5.0,
                exit_on_reach=True,
            )
        else:
            # Move both arms to folded position without pitch compensation
            bot.set_joint_pos(
                {
                    "left_arm": bot.left_arm.get_predefined_pose(target_pose),
                    "right_arm": bot.right_arm.get_predefined_pose(target_pose),
                },
                wait_time=5.0,
                exit_on_reach=True,
            )

        logger.info("Arms successfully moved to folded position")


if __name__ == "__main__":
    tyro.cli(main)
