from typing import Any, Protocol

from dexbot_utils import RobotInfo
from dexbot_utils.configs import BaseComponentConfig, BaseRobotConfig
from dexbot_utils.configs.components import sensors, vega_1

from dexcontrol.core.arm import Arm
from dexcontrol.core.chassis import Chassis
from dexcontrol.core.hand import DexGripper, HandF5D6, HandF5D6V2
from dexcontrol.core.head import Head
from dexcontrol.core.misc import Battery, EStop, Heartbeat
from dexcontrol.core.torso import Torso
from dexcontrol.sensors import (
    ChassisIMUSensor,
    Lidar3DSensor,
    UltrasonicSensor,
    USBCameraSensor,
    ZedCameraSensor,
    ZedIMUSensor,
    ZedXOneCameraSensor,
)


def get_robot_config() -> BaseRobotConfig:
    """Get the default robot configuration without loading URDF."""
    return RobotInfo.get_default_config()


class RobotComponentProtocol(Protocol):
    """Protocol for robot components that can be initialized with name and robot_info."""

    def __init__(
        self,
        name: str,
        robot_info: RobotInfo,
    ) -> None:
        """Initialize the component.

        Args:
            name: Component name.
            robot_info: RobotInfo instance.
        """
        ...

    def wait_for_active(self, timeout: float = 5.0) -> bool:
        """Wait for component to become active.

        Args:
            timeout: Maximum time to wait in seconds.

        Returns:
            True if component becomes active, False if timeout is reached.
        """
        ...


def get_component_config_map() -> dict[
    type[BaseComponentConfig], type[RobotComponentProtocol]
]:
    component_mapping = {
        vega_1.Vega1ChassisConfig: Chassis,
        vega_1.Vega1ArmConfig: Arm,
        vega_1.Vega1HeadConfig: Head,
        vega_1.Vega1TorsoConfig: Torso,
        vega_1.DexSGripperConfig: DexGripper,
        vega_1.DexDGripperConfig: DexGripper,
        vega_1.F5D6HandV1Config: HandF5D6,
        vega_1.F5D6HandV2Config: HandF5D6V2,
        vega_1.BatteryConfig: Battery,
        vega_1.EStopConfig: EStop,
        vega_1.HeartbeatConfig: Heartbeat,
    }
    return component_mapping


def get_sensor_mapping() -> dict[type[BaseComponentConfig], Any]:
    sensor_mapping = {
        sensors.ZedXCameraConfig: ZedCameraSensor,
        sensors.ZedXOneCameraConfig: ZedXOneCameraSensor,
        sensors.CameraConfig: USBCameraSensor,
        sensors.ChassisIMUConfig: ChassisIMUSensor,
        sensors.ZedIMUConfig: ZedIMUSensor,
        sensors.Lidar3DConfig: Lidar3DSensor,
        sensors.UltraSonicConfig: UltrasonicSensor,
    }
    return sensor_mapping
