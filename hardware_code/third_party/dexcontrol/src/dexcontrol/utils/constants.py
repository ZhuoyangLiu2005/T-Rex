# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Constants used throughout the dexcontrol package."""

from typing import Final

# Environment variable for robot name
ROBOT_NAME_ENV_VAR: Final[str] = "ROBOT_NAME"

# Environment variable for communication config path
COMM_CFG_PATH_ENV_VAR: Final[str] = "ZENOH_CONFIG"

# Environment variable to disable heartbeat monitoring
DISABLE_HEARTBEAT_ENV_VAR: Final[str] = "DEXCONTROL_DISABLE_HEARTBEAT"

# Environment variable to disable estop checking
DISABLE_ESTOP_CHECKING_ENV_VAR: Final[str] = "DEXCONTROL_DISABLE_ESTOP_CHECKING"
