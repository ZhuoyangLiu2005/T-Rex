# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""ZED camera sensor for multi-stream acquisition.

This module provides a production-ready ZED camera sensor implementation that
receives multiple streams (left_rgb, right_rgb, depth) from dexsensor. RGB
streams support both Zenoh and RTC transports, while depth uses Zenoh only.

Example Usage:
    ```python
    # Mixed transports configuration
    camera = ZedCameraSensor(
        name="head_camera",
        streams={
            "left_rgb": {
                "enable": True,
                "transport": "zenoh",
                "topic": "sensors/head_camera/left_rgb"
            },
            "right_rgb": {
                "enable": True,
                "transport": "rtc",
                "rtc_channel": "sensors/head_camera/right_rgb_rtc",
                "codec": "h264"
                # width/height optional - VideoSubscriber queries from publisher
            },
            "depth": {
                "enable": True,
                "topic": "sensors/head_camera/depth"
            }
        }
    )

    # Get all observations
    obs = camera.get_obs()  # {'left_rgb': np.ndarray, 'right_rgb': np.ndarray, ...}

    # Get specific streams
    obs = camera.get_obs(obs_keys=["left_rgb", "depth"])

    # Get individual streams
    left_img = camera.get_left_rgb()
    depth_map = camera.get_depth()
    ```
"""

from typing import Any, cast

import numpy as np
from dexbot_utils.configs.components.sensors import CameraConfig
from dexcomm.codecs import JsonDataCodec
from loguru import logger

from dexcontrol.sensors.camera.base_camera import (
    BaseCameraSensor,
    StreamType,
)


class ZedCameraSensor(BaseCameraSensor):
    """ZED camera sensor for multi-stream acquisition.

    Manages multiple streams from a ZED camera published by dexsensor:
    - left_rgb: Left RGB image (Zenoh JPEG or RTC video)
    - right_rgb: Right RGB image (Zenoh JPEG or RTC video)
    - depth: Depth map in meters (Zenoh PNG only)

    Each RGB stream can independently use either Zenoh (JPEG compressed,
    reliable) or RTC (hardware accelerated video, low latency). Depth always
    uses Zenoh with lossless PNG compression.

    Attributes:
        name: Unique identifier for this camera sensor.
        available_streams: List of configured stream names.
        active_streams: List of streams currently receiving data.

    Thread Safety:
        All public methods are thread-safe.
    """

    def __init__(
        self,
        name,
        configs
    ) -> None:
        """Initialize the ZED camera sensor.

        Args:
            name: Unique identifier for this camera sensor.

        Raises:
            ServiceUnavailableError: If Node creation fails.
        """
        super().__init__(name=name)

        # Create stream subscribers
        for name in ["left_rgb", "right_rgb", "depth"]:
            cam_config = cast(CameraConfig, getattr(configs, name))
            self._streams[name] = self._create_stream(
                stream_name=name,
                config={
                    "enable": cam_config.enabled,
                    "transport": cam_config.transport,
                    "topic": cam_config.topic,
                    "rtc_channel": cam_config.rtc_channel,
                },
                stream_type=StreamType.RGB if name in ["left_rgb", "right_rgb"] else StreamType.DEPTH,
            )


        # Set up camera info service
        self._camera_info_cache: dict[str, Any] | None = None
        self._setup_camera_info_service(name, configs)

        # Log initialization summary
        enabled = self.available_streams
        if enabled:
            transports = [
                f"{s}({self._streams[s].transport.value})"
                for s in enabled
            ]
            logger.info(
                f"ZED camera '{name}' initialized with streams: {', '.join(transports)}"
            )
        else:
            logger.warning(f"ZED camera '{name}' has no active streams")

    def get_obs(
        self,
        obs_keys: list[str] | None = None,
        include_timestamp: bool = False,
    ) -> dict[str, np.ndarray | dict | None]:
        """Get observations from specified camera streams.

        Args:
            obs_keys: List of stream names to retrieve (e.g., ['left_rgb', 'depth']).
                     If None, retrieves all available streams.
            include_timestamp: If True and using Zenoh, includes timestamp in result.
                              For RTC streams, timestamp is not available and will
                              be None.

        Returns:
            Dictionary mapping stream names to their data:
            - If include_timestamp=False or using RTC:
                np.ndarray for each stream (RGB: uint8, Depth: float32)
            - If include_timestamp=True and using Zenoh:
                dict with 'data' (np.ndarray) and 'timestamp' (int) for each stream
            - None for streams with no data available

        Thread Safety:
            This method is thread-safe.

        Example:
            ```python
            # Get all streams
            obs = camera.get_obs()
            # {'left_rgb': array(...), 'right_rgb': array(...), 'depth': array(...)}

            # Get specific streams
            obs = camera.get_obs(obs_keys=["left_rgb"])
            # {'left_rgb': array(...)}

            # With timestamps (Zenoh only)
            obs = camera.get_obs(include_timestamp=True)
            # {'left_rgb': {'data': array(...), 'timestamp': 1234567890}, ...}
            ```
        """
        keys_to_fetch = obs_keys or self.available_streams
        obs_out = {}

        for key in keys_to_fetch:
            stream = self._streams.get(key)
            if stream is None:
                obs_out[key] = None
                continue

            data = stream.get_latest()
            if data is None:
                obs_out[key] = None
                continue

            # Handle different return formats
            is_dict_with_data = isinstance(data, dict) and "data" in data

            if include_timestamp:
                if is_dict_with_data:
                    obs_out[key] = data  # Zenoh: already correct format
                else:
                    # RTC: wrap in dict with None timestamp
                    obs_out[key] = {"data": data, "timestamp": None}
            else:
                if is_dict_with_data:
                    obs_out[key] = data["data"]  # Zenoh: extract data
                else:
                    obs_out[key] = data  # RTC: already just the array

        return obs_out

    def get_left_rgb(self) -> np.ndarray | None:
        """Get the latest left RGB image.

        Returns:
            Left RGB image as np.ndarray (H, W, 3) uint8, or None if unavailable.

        Thread Safety:
            This method is thread-safe.
        """
        return self._get_stream_data("left_rgb")

    def get_right_rgb(self) -> np.ndarray | None:
        """Get the latest right RGB image.

        Returns:
            Right RGB image as np.ndarray (H, W, 3) uint8, or None if unavailable.

        Thread Safety:
            This method is thread-safe.
        """
        return self._get_stream_data("right_rgb")

    def get_depth(self) -> np.ndarray | None:
        """Get the latest depth map.

        Returns:
            Depth map as np.ndarray (H, W) float32 in meters, or None if unavailable.

        Thread Safety:
            This method is thread-safe.
        """
        return self._get_stream_data("depth")

    def _get_stream_data(self, stream_name: str) -> np.ndarray | None:
        """Internal helper to extract numpy array from stream data.

        Args:
            stream_name: Name of the stream to get data from.

        Returns:
            Numpy array or None if unavailable.
        """
        stream = self._streams.get(stream_name)
        if stream is None:
            return None

        data = stream.get_latest()
        if data is None:
            return None

        # Extract array from dict if needed (Zenoh returns dict, RTC returns array)
        return data["data"] if isinstance(data, dict) and "data" in data else data

    @property
    def height(self) -> dict[str, int]:
        """Get the height of each available stream in pixels.

        Returns:
            Dictionary mapping stream names to heights.
            Example: {'left_rgb': 1080, 'right_rgb': 1080, 'depth': 1080}
        """
        return {
            name: self._get_stream_dimension(name, 0)
            for name in self.available_streams
        }

    @property
    def width(self) -> dict[str, int]:
        """Get the width of each available stream in pixels.

        Returns:
            Dictionary mapping stream names to widths.
            Example: {'left_rgb': 1920, 'right_rgb': 1920, 'depth': 1920}
        """
        return {
            name: self._get_stream_dimension(name, 1)
            for name in self.available_streams
        }

    @property
    def resolution(self) -> dict[str, tuple[int, int]]:
        """Get the resolution of each available stream.

        Returns:
            Dictionary mapping stream names to (height, width) tuples.
            Example: {'left_rgb': (1080, 1920), 'depth': (1080, 1920)}
        """
        return {
            name: (
                self._get_stream_dimension(name, 0),
                self._get_stream_dimension(name, 1),
            )
            for name in self.available_streams
        }

    def _get_stream_dimension(self, stream_name: str, dim_index: int) -> int:
        """Internal helper to get a specific dimension of a stream's data.

        Args:
            stream_name: Name of the stream.
            dim_index: Dimension index (0 for height, 1 for width).

        Returns:
            Dimension size in pixels, or 0 if unavailable.
        """
        data = self._get_stream_data(stream_name)
        if data is not None and len(data.shape) > dim_index:
            return data.shape[dim_index]
        return 0

    def get_stream_transport(self, stream_name: str) -> str | None:
        """Get the transport type used by a specific stream.

        Args:
            stream_name: Name of the stream ('left_rgb', 'right_rgb', 'depth').

        Returns:
            'zenoh', 'rtc', or None if stream not configured.
        """
        stream = self._streams.get(stream_name)
        return stream.transport.value if stream is not None else None

    def _setup_camera_info_service(self, name: str, configs: Any) -> None:
        """Set up camera info service to query dexsensor for camera information.

        Args:
            name: Camera name used to construct the service topic.
            configs: Camera configuration object.
        """
        if self._node is None:
            logger.warning(f"Cannot set up camera info service for '{name}': Node not available")
            return

        # Construct service topic based on camera name
        # Expected format: sensors/{camera_name}/info
        service_topic = f"sensors/{name}/info"

        try:
            # Query camera info immediately on startup
            self._query_camera_info(service_topic)
            logger.info(f"Camera info service initialized for '{name}' at '{service_topic}'")
        except Exception as e:
            logger.warning(f"Failed to query camera info for '{name}': {e}")

    def _query_camera_info(self, service_topic: str) -> dict[str, Any] | None:
        """Query camera info from dexsensor service.

        Args:
            service_topic: Service topic to query.

        Returns:
            Camera information dictionary or None if query fails.
        """
        if self._node is None:
            return None

        try:
            # Call service with empty request
            response = self._node.call_service(service_topic, b"", timeout=2.0)

            if response is None:
                logger.debug(f"No response from camera info service: {service_topic}")
                return None

            # Decode JSON response
            info_dict = JsonDataCodec.decode(response)
            self._camera_info_cache = info_dict
            return info_dict

        except Exception as e:
            logger.debug(f"Failed to query camera info from '{service_topic}': {e}")
            return None

    def get_camera_info(self, force_refresh: bool = False) -> dict[str, Any] | None:
        """Get camera information from dexsensor.

        This method queries the camera info service to get comprehensive information
        about the camera including model, serial number, firmware version, resolution,
        calibration parameters, and current status.

        Args:
            force_refresh: If True, forces a new query to the service.
                          If False, returns cached info if available.

        Returns:
            Dictionary containing camera information with keys:
            - type: Camera type (e.g., "ZED_CAMERA")
            - camera_id: Camera ID or serial number
            - status: Camera status ("running", "error", etc.)
            - model: Camera model name (if available)
            - serial_number: Camera serial number (if available)
            - firmware_version: Firmware version (if available)
            - actual: Dict with actual resolution and FPS
            - configured: Dict with configured settings
            - streams: Dict of enabled streams
            - statistics: Dict with frame count statistics

            Returns None if camera info is not available.

        Thread Safety:
            This method is thread-safe.

        Example:
            ```python
            camera = ZedCameraSensor(name="head_camera", configs=configs)

            # Get cached camera info
            info = camera.get_camera_info()
            if info:
                print(f"Camera model: {info['model']}")
                print(f"Resolution: {info['actual']['width']}x{info['actual']['height']}")
                print(f"Enabled streams: {info['streams']}")

            # Force refresh from service
            info = camera.get_camera_info(force_refresh=True)
            ```
        """
        if force_refresh or self._camera_info_cache is None:
            service_topic = f"sensors/{self._name}/info"
            return self._query_camera_info(service_topic)

        return self._camera_info_cache


class ZedXOneCameraSensor(BaseCameraSensor):
    """ZED X One camera sensor for single RGB stream acquisition.

    Manages a single RGB video stream from a ZED X One monocular camera published
    by dexsensor. Supports both reliable Zenoh transport (JPEG compressed) and
    low-latency RTC transport (hardware accelerated video codecs).

    Attributes:
        name: Unique identifier for this camera sensor.
        resolution: Frame resolution as (height, width) tuple.
        transport: Transport type used ('zenoh' or 'rtc').

    Thread Safety:
        All public methods are thread-safe.
    """

    def __init__(
        self,
        name: str,
        configs,
    ) -> None:
        """Initialize the ZED X One camera sensor.

        Args:
            name: Unique identifier for this camera sensor.
            configs: Configuration object containing stream_config with keys:
                - enable (bool): Whether the stream is enabled.
                - transport (str): 'zenoh' or 'rtc'.
                - topic (str): Zenoh topic (for zenoh transport).
                - rtc_channel (str): RTC signaling channel (for rtc transport).
                - width (int, optional): Frame width (auto-queried if omitted).
                - height (int, optional): Frame height (auto-queried if omitted).
                - codec (str): Video codec (for rtc transport, default: 'h264').
                - buffer_size (int): Frame buffer size (default: 1).

        Raises:
            ServiceUnavailableError: If stream creation fails.
        """
        super().__init__(name=name)
        cam_config = cast(CameraConfig, getattr(configs, "rgb"))

        # Create RGB stream subscriber
        self._streams["rgb"] = self._create_stream(
            stream_name="rgb",
            config={
                    "enable": cam_config.enabled,
                    "transport": cam_config.transport,
                    "topic": cam_config.topic,
                    "rtc_channel": cam_config.rtc_channel,
                },
            stream_type=StreamType.RGB,
        )

        if self._streams["rgb"] is None:
            logger.warning(f"ZED X One camera '{name}' has no active RGB stream")
        else:
            logger.info(
                f"ZED X One camera '{name}' initialized with "
                f"{self._streams['rgb'].transport.value} transport"
            )

    def get_obs(self, include_timestamp: bool = False) -> np.ndarray | dict | None:
        """Get the latest RGB observation from the camera.

        Args:
            include_timestamp: If True and using Zenoh transport, returns dict
                              with 'data' and 'timestamp' keys. For RTC transport,
                              timestamp is not available and only image data is
                              returned regardless of this parameter.

        Returns:
            If include_timestamp=False or using RTC:
                RGB image as np.ndarray (H, W, 3) uint8, or None if unavailable.
            If include_timestamp=True and using Zenoh:
                dict with 'data' (np.ndarray) and 'timestamp' (int) keys, or None.

        Thread Safety:
            This method is thread-safe.

        Example:
            ```python
            # Get just the image
            img = camera.get_obs()  # np.ndarray (1080, 1920, 3)

            # Get image with timestamp (Zenoh only)
            data = camera.get_obs(include_timestamp=True)
            if data is not None:
                img = data['data']  # np.ndarray
                ts = data['timestamp']  # int (nanoseconds)
            ```
        """
        stream = self._streams.get("rgb")
        if stream is None:
            return None

        data = stream.get_latest()
        if data is None:
            return None

        # Handle different return formats based on transport
        is_dict_with_data = isinstance(data, dict) and "data" in data

        if include_timestamp:
            if is_dict_with_data:
                return data  # Zenoh: already has correct format
            else:
                # RTC: wrap in dict with None timestamp
                return {"data": data, "timestamp": None}
        else:
            if is_dict_with_data:
                return data["data"]  # Zenoh: extract data
            else:
                return data  # RTC: already just the array

    def get_rgb(self) -> np.ndarray | None:
        """Get the latest RGB image (convenience method).

        Returns:
            RGB image as np.ndarray (H, W, 3) uint8, or None if unavailable.

        Thread Safety:
            This method is thread-safe.
        """
        return self.get_obs(include_timestamp=False)

    @property
    def height(self) -> int:
        """Get the height of the camera image in pixels.

        Returns:
            Height in pixels, or 0 if no data available.
        """
        img = self.get_rgb()
        return img.shape[0] if img is not None else 0

    @property
    def width(self) -> int:
        """Get the width of the camera image in pixels.

        Returns:
            Width in pixels, or 0 if no data available.
        """
        img = self.get_rgb()
        return img.shape[1] if img is not None else 0

    @property
    def resolution(self) -> tuple[int, int]:
        """Get the resolution of the camera image.

        Returns:
            (height, width) tuple in pixels, or (0, 0) if no data available.
        """
        img = self.get_rgb()
        return (img.shape[0], img.shape[1]) if img is not None else (0, 0)

    @property
    def transport(self) -> str | None:
        """Get the transport type used by the RGB stream.

        Returns:
            'zenoh', 'rtc', or None if stream not configured.
        """
        stream = self._streams.get("rgb")
        return stream.transport.value if stream is not None else None
