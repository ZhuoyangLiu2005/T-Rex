# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Operating system utility functions."""

import os
import re
from importlib.metadata import version
from typing import Any, Final

from loguru import logger

import dexcontrol
from dexcontrol.utils.constants import ROBOT_NAME_ENV_VAR


def resolve_key_name(key: str) -> str:
    """Resolves a key name for zenoh topic by prepending robot name.

    Args:
        key: Original key name (e.g. 'lidar' or '/lidar')

    Returns:
        Resolved key with robot name prepended (e.g. 'robot/lidar')
    """
    # Get robot name from env var or use default
    robot_name: Final[str] = os.getenv(ROBOT_NAME_ENV_VAR, "robot")

    # Remove leading slash if present
    key = key.lstrip("/")

    # Check if robot name is already present at the beginning
    if key.startswith(f"{robot_name}/"):
        return key

    # Combine robot name and key with single slash
    return f"{robot_name}/{key}"


def get_robot_model() -> str:
    """Get the robot model from the environment variable."""
    robot_model_abb_mapping = dict(vg="vega")
    robot_name = os.getenv(ROBOT_NAME_ENV_VAR)
    if robot_name is None:
        raise ValueError(
            f"Robot name is not set, please set the environment variable {ROBOT_NAME_ENV_VAR}"
        )
    if not re.match(r"^dm/[a-zA-Z0-9]{12}-(?:\d+|rc\d+)$", robot_name):
        raise ValueError(f"Robot name is not in the correct format: {robot_name}")

    robot_model_abb = robot_name.split("/")[-1].split("-")[0][:2]
    if robot_model_abb not in robot_model_abb_mapping:
        raise ValueError(f"Unknown robot model: {robot_model_abb}")
    model = robot_model_abb_mapping[robot_model_abb] + "-" + robot_name.split("-")[-1]
    return model


def check_version_compatibility(version_info: dict[str, Any]) -> None:
    """Check version compatibility between client and server.

    This function uses the new JSON-based version interface to:
    1. Compare client library version with server's minimum required version
    2. Check server component versions for compatibility
    3. Provide clear guidance for version mismatches

    Args:
        version_info: Dictionary containing version information from get_version_info()
        show_warnings: Whether to show warning messages (default: True)
    """
    validate_client_version(version_info)
    validate_server_version(version_info)


def validate_client_version(version_info: dict[str, Any]) -> None:
    """Validate client library version against server requirements.

    Args:
        version_info: Dictionary containing server and client version information.
    """

    min_required_version = version_info["min_client_version"]

    if not min_required_version:
        logger.debug("No minimum version requirement from server")
        return

    # Get current library version using importlib.metadata
    current_version = version("dexcontrol")

    if current_version == "unknown":
        logger.warning("Could not determine current library version")
        return

    # Compare versions
    comparison = compare_versions(current_version, min_required_version)

    if comparison < 0:
        show_version_upgrade_warning(current_version, min_required_version)


def show_version_upgrade_warning(current: str, required: str) -> None:
    """Display version upgrade warning to user.

    Args:
        current: Current library version
        required: Required minimum version
    """
    logger.error(
        f"🚨 CLIENT VERSION TOO OLD! 🚨\n"
        f"Current library version: {current}\n"
        f"Minimum required version: {required}\n"
        f"\n"
        f"⚠️  Your dexcontrol library is outdated and may not work correctly!\n"
        f"📦 Please update your library using: pip install --upgrade dexcontrol\n"
    )


def validate_server_version(version_info: dict[str, Any]) -> None:
    """Validate server software versions against minimum requirements.

    Args:
        version_info: Dictionary containing server and client version information.
    """
    server_info = version_info.get("firmware_version", {})

    if not server_info:
        logger.debug("No server version information available")
        return

    # Check each component's software version
    soc_info = server_info.get("soc", {})
    software_version = soc_info.get("software_version", {})
    if software_version is not None:
        software_version_int = int(software_version)
        if software_version_int < dexcontrol.MIN_SOC_SOFTWARE_VERSION:
            show_server_version_warning(
                [("soc", software_version_int)], dexcontrol.MIN_SOC_SOFTWARE_VERSION
            )


def show_server_version_warning(
    components: list[tuple[str, int]], min_version: int
) -> None:
    """Display server version warning to user.

    Args:
        components: List of (component_name, version) tuples for components below minimum.
        min_version: Minimum required server software version.
    """
    components_str = "\n".join(
        f"  - {name}: version {version}" for name, version in components
    )

    logger.error(
        f"🚨 SERVER VERSION TOO OLD! 🚨\n"
        f"The following server components are below minimum version {min_version}:\n"
        f"{components_str}\n"
        f"\n"
        f"⚠️  Your robot's firmware may be outdated and some features may not work correctly!\n"
        f"📦 Please contact your robot admin or check https://github.com/dexmate-ai/vega-firmware.\n"
    )


def compare_versions(version1: str, version2: str) -> int:
    """Compare two semantic version strings.

    Args:
        version1: First version string (e.g., "1.2.3")
        version2: Second version string (e.g., "1.1.0")

    Returns:
        -1 if version1 < version2
         0 if version1 == version2
         1 if version1 > version2
    """
    try:
        # Clean version strings (remove 'v' prefix, handle pre-release suffixes)
        def clean_version(v: str) -> list[int]:
            v = v.strip().lower()
            if v.startswith("v"):
                v = v[1:]
            # Split by dots and take only numeric parts
            parts = v.split(".")
            numeric_parts = []
            for part in parts:
                # Remove any non-numeric suffixes (like -alpha, etc.)
                numeric_part = ""
                for char in part:
                    if char.isdigit():
                        numeric_part += char
                    else:
                        break
                if numeric_part:
                    numeric_parts.append(int(numeric_part))
            return numeric_parts

        parts1 = clean_version(version1)
        parts2 = clean_version(version2)

        # Pad shorter version with zeros
        max_len = max(len(parts1), len(parts2))
        parts1.extend([0] * (max_len - len(parts1)))
        parts2.extend([0] * (max_len - len(parts2)))

        # Compare part by part
        for p1, p2 in zip(parts1, parts2):
            if p1 < p2:
                return -1
            elif p1 > p2:
                return 1

        return 0

    except Exception as e:
        logger.debug(f"Version comparison error: {e}")
        # Fallback to string comparison
        return -1 if version1 < version2 else (1 if version1 > version2 else 0)
