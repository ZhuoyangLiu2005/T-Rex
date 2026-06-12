# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai
#    See README.md for details

"""DexControl: Robot Control Interface Library.

This package provides interfaces for controlling and monitoring robot systems.
It serves as the primary API for interacting with Dexmate robots.
"""

import os
from pathlib import Path
from typing import Final

from loguru import logger
from rich.logging import RichHandler

# DO NOT REMOVE this following import, it is needed for hydra to find the config
import dexcontrol.config  # pylint: disable=unused-import
from dexcontrol.exceptions import (
    ComponentError,
    ComponentNotAvailableError,
    ConfigurationError,
    DexcontrolError,
    ModelNotSupportedError,
    RobotConnectionError,
    SensorNotAvailableError,
    ServiceUnavailableError,
)
from dexcontrol.robot import Robot
from dexcontrol.utils.constants import COMM_CFG_PATH_ENV_VAR

# Package-level constants
LIB_PATH: Final[Path] = Path(__file__).resolve().parent
CFG_PATH: Final[Path] = LIB_PATH / "config"
MIN_SOC_SOFTWARE_VERSION: int = 419

logger.configure(
    handlers=[
        {"sink": RichHandler(markup=True), "format": "{message}", "level": "INFO"}
    ]
)


def get_comm_cfg_path() -> str | None:
    """Get Zenoh config path.

    Priority:
    1. .dzcfg files directly in ~/.dexmate/comm/zenoh/ or subdirectories
    2. .json5 files in subdirectories of ~/.dexmate/comm/zenoh/
    """
    base_dir = Path("~/.dexmate/comm/zenoh/").expanduser()

    # Priority 1: .dzcfg files (any name, directly in dir or subdirectories)
    dzcfg_files = sorted(base_dir.glob("**/*.dzcfg"))
    if dzcfg_files:
        return dzcfg_files[0].as_posix()

    # Priority 2: .json5 files (subdirectories only)
    json5_files = sorted(base_dir.glob("**/zenoh*config*.json5"))
    if json5_files:
        return json5_files[0].as_posix()

    logger.debug(
        "No Zenoh config found in ~/.dexmate/comm/zenoh/ - will use DexComm defaults"
    )
    return None


if os.getenv(COMM_CFG_PATH_ENV_VAR) is None:
    comm_cfg_path = get_comm_cfg_path()
    if comm_cfg_path is not None:
        os.environ[COMM_CFG_PATH_ENV_VAR] = comm_cfg_path

ROBOT_CFG_PATH: Final[Path] = CFG_PATH

__all__ = [
    "Robot",
    "LIB_PATH",
    "CFG_PATH",
    "ROBOT_CFG_PATH",
    # Exceptions
    "DexcontrolError",
    "ComponentError",
    "ComponentNotAvailableError",
    "ModelNotSupportedError",
    "SensorNotAvailableError",
    "ConfigurationError",
    "RobotConnectionError",
    "ServiceUnavailableError",
]
