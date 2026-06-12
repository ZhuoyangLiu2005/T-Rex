# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Chassis IMU sensor implementation using DexComm subscribers.

This module provides IMU sensor class for chassis IMU data
using DexComm's Raw API.
"""

import numpy as np
from dexcomm import Node
from dexcomm.codecs import IMUDataCodec


class ChassisIMUSensor:
    """Chassis IMU sensor using DexComm subscriber.

    This sensor provides IMU data from the chassis
    """

    def __init__(
        self,
        name,
        configs,
    ) -> None:
        """Initialize the chassis IMU sensor.

        Args:
            configs: Configuration for the chassis IMU sensor.
        """
        self._name = name
        self._node = Node(name=self._name)
        self._subscriber = self._node.create_subscriber(
            topic=configs.topic,
            callback=None,
            decoder=IMUDataCodec.decode,
        )

    def shutdown(self) -> None:
        """Shutdown the chassis IMU sensor."""
        self._subscriber.shutdown()

    def is_active(self) -> bool:
        """Check if the chassis IMU sensor is actively receiving data.

        Returns:
            True if receiving data, False otherwise.
        """
        return self._subscriber.is_active(0.1)

    def wait_for_active(self, timeout: float = 5.0) -> bool:
        """Wait for the chassis IMU sensor to start receiving data.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if sensor becomes active, False if timeout is reached.
        """
        return self._node.wait_for_active(timeout)

    def get_obs(
        self, obs_keys: list[str] | None = None
    ) -> dict[str, np.ndarray] | None:
        """Get observation data for the chassis IMU sensor.

        Args:
            obs_keys: List of observation keys to retrieve. If None, returns all available data.
                     Valid keys: ['gyro', 'acc', 'quat', 'mag', 'timestamp']

        Returns:
            Dictionary with observation data including all IMU measurements.
            Keys are mapped as follows:
            - 'gyro': Angular velocity from 'gyro'
            - 'acc': Linear acceleration from 'acc'
            - 'quat': Orientation quaternion from 'quat' [w, x, y, z]
            - 'mag': Magnetometer from 'mag' (if available)
            - 'timestamp_ns': Timestamp in nanoseconds
        """
        if obs_keys is None:
            obs_keys = ["gyro", "acc", "quat"]

        data = self._subscriber.get_latest()

        obs_out = {}
        obs_out["timestamp_ns"] = data["timestamp_ns"]

        for key in obs_keys:
            if key == "gyro":
                obs_out[key] = data.get("gyro", np.zeros(3))
            elif key == "acc":
                obs_out[key] = data.get("acc", np.zeros(3))
            elif key == "quat":
                obs_out[key] = data.get("quat", np.array([1.0, 0.0, 0.0, 0.0]))
            elif key == "mag" and "mag" in data:
                obs_out[key] = data["mag"]
        return obs_out

    def get_acc(self) -> np.ndarray | None:
        """Get the latest linear acceleration from chassis IMU.

        Returns:
            Linear acceleration [x, y, z] in m/s² if available, None otherwise.
        """
        data = self._subscriber.get_latest()
        return data.get("acc") if data else None

    def get_gyro(self) -> np.ndarray | None:
        """Get the latest angular velocity from chassis IMU.

        Returns:
            Angular velocity [x, y, z] in rad/s if available, None otherwise.
        """
        data = self._subscriber.get_latest()
        return data.get("gyro") if data else None

    def get_quat(self) -> np.ndarray | None:
        """Get the latest orientation quaternion from chassis IMU.

        Returns:
            Orientation quaternion [w, x, y, z] if available, None otherwise.
            Note: dexcomm uses [w, x, y, z] quaternion format.
        """
        data = self._subscriber.get_latest()
        return data.get("quat") if data else None

    def get_mag(self) -> np.ndarray | None:
        """Get the latest magnetometer reading from chassis IMU.

        Returns:
            Magnetic field [x, y, z] in µT if available, None otherwise.
        """
        data = self._subscriber.get_latest()
        if not data or not isinstance(data, dict):
            return None
        return data.get("mag", None)

    def has_mag(self) -> bool:
        """Check if the chassis IMU has magnetometer data available.

        Returns:
            True if magnetometer data is available, False otherwise.
        """
        data = self._subscriber.get_latest()
        if not data or not isinstance(data, dict):
            return False
        return "mag" in data and data["mag"] is not None

    # Backward compatibility aliases
    get_acceleration = get_acc
    get_angular_velocity = get_gyro
    get_orientation = get_quat
    get_magnetometer = get_mag
    has_magnetometer = has_mag

    @property
    def name(self) -> str:
        """Get the sensor name.

        Returns:
            Sensor name string.
        """
        return self._name
