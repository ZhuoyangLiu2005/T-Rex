# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""USB camera sensor for single RGB stream acquisition.

This module provides a production-ready USB camera sensor implementation that
receives data from dexsensor via either Zenoh (JPEG compressed, reliable) or
RTC (real-time video, hardware accelerated) transport.

Example Usage:
    ```python
    # Zenoh transport
    camera = USBCameraSensor(
        name="camera_front",
        stream_config={
            "enable": True,
            "transport": "zenoh",
            "topic": "sensors/camera_front/rgb"
        }
    )

    # RTC transport (width/height auto-queried by VideoSubscriber)
    camera = USBCameraSensor(
        name="camera_front",
        stream_config={
            "enable": True,
            "transport": "rtc",
            "rtc_channel": "sensors/camera_front/rgb_rtc",
            "codec": "h264"
            # width/height optional - VideoSubscriber queries from publisher
        }
    )

    # Get latest frame
    frame = camera.get_rgb()  # Returns np.ndarray (H, W, 3) uint8

    # With timestamp (Zenoh only)
    data = camera.get_obs(include_timestamp=True)
    # Returns: {'data': np.ndarray, 'timestamp': int} or None
    ```
"""

import numpy as np
from loguru import logger

from dexcontrol.sensors.camera.base_camera import (
    BaseCameraSensor,
    StreamType,
)


class USBCameraSensor(BaseCameraSensor):
    """USB camera sensor for single RGB stream acquisition.

    Manages a single RGB video stream from a USB camera published by dexsensor.
    Supports both reliable Zenoh transport (JPEG compressed) and low-latency
    RTC transport (hardware accelerated video codecs).

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
        """Initialize the USB camera sensor.

        Args:
            name: Unique identifier for this camera sensor.
            stream_config: RGB stream configuration with keys:
                - enable (bool): Whether the stream is enabled.
                - transport (str): 'zenoh' or 'rtc'.
                - topic (str): Zenoh topic (for zenoh transport).
                - rtc_channel (str): RTC signaling channel (for rtc transport).
                - width (int, optional): Frame width (VideoSubscriber auto-queries if omitted).
                - height (int, optional): Frame height (VideoSubscriber auto-queries if omitted).
                - codec (str): Video codec (for rtc transport, default: 'h264').
                - buffer_size (int): Frame buffer size (default: 1).

        Raises:
            ServiceUnavailableError: If stream creation fails.
        """
        super().__init__(name=name)

        # Handle both CameraConfig and configs with stream_config attribute
        if hasattr(configs, "stream_config"):
            stream_config = configs.stream_config
        else:
            # Convert CameraConfig to stream_config format
            stream_config = {
                "enable": configs.enabled,
                "transport": configs.transport,
                "topic": configs.topic,
                "rtc_channel": configs.rtc_channel,
            }

        # Create RGB stream subscriber
        self._streams["rgb"] = self._create_stream(
            stream_name="rgb",
            config=stream_config,
            stream_type=StreamType.RGB,
        )

        if self._streams["rgb"] is None:
            logger.warning(f"USB camera '{name}' has no active RGB stream")
        else:
            logger.info(
                f"USB camera '{name}' initialized with "
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
