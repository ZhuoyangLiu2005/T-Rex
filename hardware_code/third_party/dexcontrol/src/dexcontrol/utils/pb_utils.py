# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Utility functions for Python data structures and protobuf messages."""

from enum import Enum
from typing import Any, Literal

TYPE_SOFTWARE_VERSION = dict[
    Literal["hardware_version", "software_version", "main_hash", "compile_time"], Any
]


class ComponentStatus(Enum):
    """Enum representing the status of a component."""

    NORMAL = 0
    NA = 1
    ERROR = 2


def status_to_enum(status: int) -> ComponentStatus:
    """Convert a ComponentStatus protobuf message to a ComponentStatus enum.

    Args:
        status: ComponentStatus protobuf message.

    Returns:
        ComponentStatus enum value.

    Raises:
        ValueError: If the status value is not recognized.
    """
    status_map = {
        0: ComponentStatus.NORMAL,
        1: ComponentStatus.NA,
        2: ComponentStatus.ERROR,
    }

    if status not in status_map:
        raise ValueError(f"Invalid status: {status}")

    return status_map[status]


def status_to_dict(
    status_msg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Convert a ComponentStates dictionary to a dictionary.

    Args:
        status_msg: ComponentStates dictionary.

    Returns:
        Dictionary containing status information for each component.
    """
    return {
        name: {
            "connected": state["connection"],
            "enabled": status_to_enum(state["enabled"]),
            "error_state": status_to_enum(state["error_state"]),
            "error_code": state["error"],
        }
        for name, state in status_msg["states"].items()
    }


def convert_enum_to_str(d: dict) -> dict:
    """Convert enum values to their string representations in a dictionary.

    Args:
        d: Dictionary containing enum values.

    Returns:
        Dictionary with enum values converted to their string representations.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = convert_enum_to_str(value)
        elif isinstance(value, Enum):
            d[key] = value.name
    return d
