# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Helper utilities for RobotInfo integration."""

from typing import Any

import numpy as np
from dexbot_utils import RobotInfo
from loguru import logger


def get_joint_config(robot_info: RobotInfo, component: str) -> dict[str, Any]:
    """Get joint configuration for a component from RobotInfo.

    Args:
        robot_info: RobotInfo instance.
        component: Component name (e.g., "left_arm").

    Returns:
        Dict with joint_name, joint_limit, joint_vel_limit.
        Returns defaults if component not found or URDF not available.
    """
    # Get joint names from component object
    if component not in robot_info.get_possible_components():
        return {}
    joint_names = robot_info.get_component_joints(component)
    if not joint_names:
        logger.warning(f"No joints defined for '{component}'")
        return _defaults(component)

    # Try to get limits from URDF
    joint_limit = None
    joint_vel_limit = None

    if robot_info.has_urdf:
        try:
            urdf_limits = robot_info.get_joint_limits(joint_names)
            joint_limit = urdf_limits[:, :2].tolist()
            joint_vel_limit = urdf_limits[:, 2].tolist()
            logger.debug(f"Loaded {component} limits from URDF")
        except Exception as e:
            logger.debug(f"Could not get URDF limits for {component}: {e}")

    # Use defaults if URDF not available
    if joint_limit is None:
        joint_limit = [[-np.pi, np.pi]] * len(joint_names)
    if joint_vel_limit is None:
        joint_vel_limit = [2.0] * len(joint_names)

    return {
        "joint_name": joint_names,
        "joint_limit": joint_limit,
        "joint_vel_limit": joint_vel_limit,
    }


def get_component_attrs(robot_info: RobotInfo, component: str) -> dict[str, Any]:
    """Get component attributes from RobotInfo.

    Args:
        robot_info: RobotInfo instance.
        component: Component name (e.g., "chassis", "head").

    Returns:
        Dict with component-specific attributes like pv_mode, dof, etc.
    """
    comp_obj = robot_info.get_component(component)
    if comp_obj is None:
        return {}

    attrs = {
        "dof": comp_obj.dof,
        "pv_mode": getattr(comp_obj, "pv_mode", False),
    }

    # Chassis-specific attributes
    if component == "chassis":
        attrs.update(
            {
                "center_to_wheel_axis_dist": getattr(
                    comp_obj, "center_to_wheel_axis_dist", None
                ),
                "wheels_dist": getattr(comp_obj, "wheels_dist", None),
                "max_linear_speed": getattr(comp_obj, "max_linear_speed", None),
                "steer_joints": getattr(comp_obj, "steer_joints", []),
                "drive_joints": getattr(comp_obj, "drive_joints", []),
            }
        )

    # Hand-specific attributes
    elif "hand" in component:
        attrs["hand_type"] = getattr(comp_obj, "hand_type", "HandF5D6_V2")

    return attrs


def _defaults(component: str) -> dict[str, Any]:
    """Get default config when component not found."""
    if "arm" in component:
        side = "R" if "right" in component else "L"
        return {
            "joint_name": [f"{side}_arm_j{i + 1}" for i in range(7)],
            "joint_limit": [[-np.pi, np.pi] for _ in range(7)],
            "joint_vel_limit": [2.4, 2.4, 2.7, 2.7, 2.7, 2.7, 2.7],
        }
    elif "torso" in component:
        return {
            "joint_name": [f"torso_j{i + 1}" for i in range(3)],
            "joint_limit": [[-np.pi, np.pi] for _ in range(3)],
            "joint_vel_limit": [0.5, 0.5, 0.5],
        }
    elif "head" in component:
        return {
            "joint_name": [f"head_j{i + 1}" for i in range(3)],
            "joint_limit": [[-np.pi, np.pi] for _ in range(3)],
            "joint_vel_limit": [2.0, 2.0, 2.0],
        }
    elif "hand" in component:
        side = "R" if "right" in component else "L"
        return {
            "joint_name": [
                f"{side}_th_j1",
                f"{side}_ff_j1",
                f"{side}_mf_j1",
                f"{side}_rf_j1",
                f"{side}_lf_j1",
                f"{side}_th_j0",
            ],
            "joint_limit": [[0, 1] for _ in range(6)],
            "joint_vel_limit": [1.0] * 6,
        }
    elif "chassis" in component:
        return {
            "joint_name": ["L_wheel_j1", "L_wheel_j2", "R_wheel_j1", "R_wheel_j2"],
            "joint_limit": [[-np.inf, np.inf] for _ in range(4)],
            "joint_vel_limit": [10.0] * 4,
        }

    # Fallback
    return {
        "joint_name": [],
        "joint_limit": [],
        "joint_vel_limit": [],
    }
