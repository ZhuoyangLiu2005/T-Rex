"""Custom exceptions for dexcontrol.

This module defines a hierarchy of exceptions for better error handling
and user experience when connection or configuration issues occur.
"""


class DexcontrolError(Exception):
    """Base exception for all dexcontrol errors.

    All custom dexcontrol exceptions inherit from this class,
    making it easy to catch all dexcontrol-specific errors.
    """

    pass


class ConfigurationError(DexcontrolError):
    """Raised when there is a configuration problem.

    This includes missing or invalid environment variables,
    missing config files, or invalid configuration content.
    """

    pass


class RobotConnectionError(DexcontrolError):
    """Raised when the robot cannot be reached.

    This typically indicates network issues, robot not powered on,
    or Zenoh routing problems.
    """

    pass


class ServiceUnavailableError(DexcontrolError):
    """Raised when a specific service is not responding.

    This indicates the robot is likely connected but a specific
    service (e.g., hand type query) is not available. This could
    happen during robot initialization or if a component is disabled.
    """

    pass


class ComponentError(DexcontrolError):
    """Raised when a component fails to initialize or activate.

    This indicates that one or more robot components could not be
    started, activated, or are in an invalid state for operation.
    """

    pass


class ComponentNotAvailableError(ComponentError, AttributeError):
    """Raised when accessing a component that is not available on this robot.

    This typically means the component is either not present on this robot model
    or has been disabled in the configuration.
    """

    def __init__(self, component: str, robot_model: str) -> None:
        self.component = component
        self.robot_model = robot_model
        super().__init__(
            f"Component '{component}' is not available on this robot (model: {robot_model}). "
            f"Use robot.has_component('{component}') to check availability before access."
        )


class SensorNotAvailableError(ComponentError, AttributeError):
    """Raised when accessing a sensor that is not available or not initialized.

    This typically means the sensor is either not present on this robot model,
    not enabled in the configuration, or failed to initialize.
    """

    def __init__(self, sensor: str) -> None:
        self.sensor = sensor
        super().__init__(
            f"Sensor '{sensor}' is not available or not initialized. "
            f"Use robot.has_sensor('{sensor}') to check availability before access."
        )


class ModelNotSupportedError(ComponentError):
    """Raised when a method is called on an unsupported robot model.

    This indicates that the called method or feature is not available
    on the current robot model.
    """

    def __init__(
        self, method: str, robot_model: str, supported_models: tuple[str, ...]
    ) -> None:
        self.method = method
        self.robot_model = robot_model
        self.supported_models = supported_models
        super().__init__(
            f"'{method}' is not supported on model '{robot_model}'. "
            f"Supported models: {', '.join(supported_models)}"
        )
