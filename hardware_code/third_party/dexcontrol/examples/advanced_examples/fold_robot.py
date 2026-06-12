# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script demonstrating folding and unfolding robot poses.

This script shows how to safely move the robot between folded and unfolded
configurations using predefined poses. It includes safety checks and proper
error handling.
"""

import os
from typing import Final

import tyro
from loguru import logger

from dexcontrol.robot import Robot
from dexcontrol.utils.compat import supported_models


def is_running_remote() -> bool:
    """Check if the script is running in a remote environment.

    Returns:
        True if running in a remote environment (no display), False otherwise.
    """
    return "DISPLAY" not in os.environ


def unfold_robot() -> None:
    """Unfold the robot from folded position to operational position.

    Raises:
        RuntimeError: If torso or arms fail to reach target positions.
    """
    torso_pose: Final[str] = "crouch45_high"
    arm_pose: Final[str] = "L_shape"
    arm_joints: Final[list[int]] = list(range(1, 7))

    with Robot() as bot:
        # First move torso
        if not bot.torso.is_pose_reached(torso_pose):
            logger.info("Moving torso to operational position")
            bot.set_joint_pos(
                {
                    "torso": bot.torso.get_predefined_pose(torso_pose),
                },
                wait_time=10.0,
                exit_on_reach=True,
            )

        if not bot.torso.is_pose_reached(torso_pose):
            raise RuntimeError("Torso did not reach the target position! Exiting...")

        # Then move arms
        left_arm_ready = bot.left_arm.is_pose_reached(arm_pose, joint_id=arm_joints)
        right_arm_ready = bot.right_arm.is_pose_reached(arm_pose, joint_id=arm_joints)

        if not (left_arm_ready and right_arm_ready):
            logger.info("Moving arms to operational position")
            left_arm_pose = bot.left_arm.get_predefined_pose(arm_pose)
            right_arm_pose = bot.right_arm.get_predefined_pose(arm_pose)
            left_arm_pose = bot.compensate_torso_pitch(left_arm_pose, "left_arm")
            right_arm_pose = bot.compensate_torso_pitch(right_arm_pose, "right_arm")
            bot.set_joint_pos(
                {
                    "left_arm": left_arm_pose,
                    "right_arm": right_arm_pose,
                },
                wait_time=6.0,
                exit_on_reach=True,
            )

        # Verify arms reached target position
        left_arm_ready = bot.left_arm.is_pose_reached(arm_pose, joint_id=arm_joints)
        right_arm_ready = bot.right_arm.is_pose_reached(arm_pose, joint_id=arm_joints)

        if not (left_arm_ready and right_arm_ready):
            raise RuntimeError("Arms did not reach the target position! Exiting...")

        # Set head to home position
        head_home_pose = bot.compensate_torso_pitch(
            bot.head.get_predefined_pose("home"), "head"
        )
        bot.head.set_joint_pos(head_home_pose, wait_time=2.0, exit_on_reach=True)
        logger.info("Robot is unfolded!")


def fold_robot(safe_motion: bool = True) -> None:
    """Fold the robot to its compact storage position.

    Args:
        safe_motion: If True, use collision-aware motion planning.

    Raises:
        RuntimeError: If arms or torso fail to reach folded positions.
    """
    is_remote = is_running_remote()
    arm_desired_pose: Final[str] = "folded_closed_hand"
    l_shape_joints: Final[list[int]] = list(range(1, 7))
    partial_fold_joints: Final[list[int]] = list(range(6))

    with Robot() as bot:
        # Close hands first
        if bot.have_hand("left"):
            logger.info("Closing hands before folding arms")
            bot.left_hand.close_hand()
        if bot.have_hand("right"):
            logger.info("Closing hands before folding arms")
            bot.right_hand.close_hand(wait_time=2.0)

        # Check if arms are already folded
        arms_folded = bot.left_arm.is_pose_reached(
            arm_desired_pose
        ) and bot.right_arm.is_pose_reached(arm_desired_pose)
        if bot.torso.is_pose_reached("folded"):
            logger.info("Robot is already folded!")
            return

        if not arms_folded:
            arms_half_folded = bot.left_arm.is_pose_reached(
                arm_desired_pose, joint_id=partial_fold_joints
            ) and bot.right_arm.is_pose_reached(
                arm_desired_pose, joint_id=partial_fold_joints
            )

            if not arms_half_folded:
                # Check if arms are in L_shape position
                arms_l_shape = bot.left_arm.is_pose_reached(
                    "L_shape", joint_id=l_shape_joints
                ) and bot.right_arm.is_pose_reached("L_shape", joint_id=l_shape_joints)

                # If not in L_shape, move to L_shape first
                if not arms_l_shape:
                    logger.info("Moving arms to intermediate position")
                    if safe_motion:
                        try:
                            from init_arm_safe import ArmSafeInitializer

                            arm_safe_initializer = ArmSafeInitializer(
                                control_hz=250,
                                visualize=not is_remote,
                                bot=bot,
                            )
                            arm_safe_initializer.run(
                                "L_shape", shutdown_after_execution=False
                            )
                        except Exception as e:
                            logger.error(f"Error during safe motion: {e}")
                            # Ask the user if they are fine without collision avoidance
                            user_input = input(
                                "Continue without motion planning? [y/N]: "
                            )
                            if user_input.lower() != "y":
                                raise RuntimeError("Exiting...") from e
                    else:
                        bot.set_joint_pos(
                            {
                                "left_arm": bot.left_arm.get_predefined_pose("L_shape"),
                                "right_arm": bot.right_arm.get_predefined_pose(
                                    "L_shape"
                                ),
                            },
                            wait_time=5.0,
                        )

            # Move arms to folded position
            logger.info("Folding arms")
            bot.set_joint_pos(
                {
                    "left_arm": bot.left_arm.get_predefined_pose(arm_desired_pose),
                    "right_arm": bot.right_arm.get_predefined_pose(arm_desired_pose),
                },
                wait_time=6.0,
                exit_on_reach=True,
            )

            # Verify arms reached folded position
            arms_folded = bot.left_arm.is_pose_reached(
                arm_desired_pose
            ) and bot.right_arm.is_pose_reached(arm_desired_pose)
            if not arms_folded:
                raise RuntimeError("Arms did not reach folded position! Exiting...")

        # Move torso to folded position
        logger.info("Folding torso")
        bot.set_joint_pos(
            {
                "torso": bot.torso.get_predefined_pose("folded"),
            },
            wait_time=8.0,
            exit_on_reach=True,
        )

        if not bot.torso.is_pose_reached("folded"):
            raise RuntimeError("Torso did not reach folded position! Exiting...")

        bot.head.go_to_pose("tucked", wait_time=2.0, exit_on_reach=True)
        logger.info("Robot is folded!")


@supported_models("vega_1", "vega_1p")
def main(
    unfold: bool = False,
    safe_motion: bool = True,
) -> None:
    """Execute robot folding or unfolding sequence.

    Args:
        unfold: If True, unfold the robot; if False, fold the robot.
        safe_motion: If True, use safe motion; if False, use unsafe motion.
    """
    # Safety confirmation
    logger.warning(
        "Warning: Be ready to press e-stop if needed! "
        "This example does not check for self-collisions."
    )
    logger.warning("Please ensure the arms and torso have sufficient space to move.")
    if input("Continue? [y/N]: ").lower() != "y":
        return

    if unfold:
        unfold_robot()
    else:
        fold_robot(safe_motion=safe_motion)


if __name__ == "__main__":
    tyro.cli(main)
