# Changelog

All notable changes to this project will be documented in this file.

## [0.4.4] - 2026-02-18

### Fixed

- **Double-namespaced Zenoh topics in RTC and Arm.** `RTCSubscriber._query_connection_info()` was calling `resolve_key_name()` before passing the topic to `query_json_service()`, which already applies `resolve_key_name()` internally — resulting in the robot name prefix being added twice. Similarly, `Arm` was wrapping `config.ee_pass_through_pub_topic` with `resolve_key_name()` before passing it to `create_publisher()`, which handles namespace resolution at the communication layer. Both call sites now pass the raw topic directly.

## [0.4.3] - 2026-02-17

### Added

- **Robot model compatibility decorators.** New `supported_models` decorator for example scripts and `requires_model` decorator for class methods, both in `dexcontrol.utils.compat`. Scripts decorated with `@supported_models` check the robot model from environment variables before running and exit with a clear error if the model is unsupported — avoiding unnecessary robot connections. Methods decorated with `@requires_model` raise `ModelNotSupportedError` at call time.
- **`ModelNotSupportedError` exception.** New exception type raised when a method or script is called on an unsupported robot model. Includes `method`, `robot_model`, and `supported_models` attributes for programmatic handling.
- **Model annotations on examples.** Examples that require specific hardware (chassis, torso, ultrasonic sensors, chassis cameras) are now annotated with `@supported_models` to prevent confusing errors when run on incompatible robot variants.

### Fixed

- **EE pass-through publish format.** `Arm.send_ee_pass_through_message()` now wraps the message bytes in the expected `{"data": message}` dict format.

### Breaking Changes

- **Removed `variant` parameter from `Robot` constructor.** The robot variant is now always resolved automatically from the `ROBOT_NAME` environment variable. To use a custom configuration, pass `configs=` directly. Callers using `Robot(variant=...)` must remove the argument and rely on the `ROBOT_NAME` environment variable instead, or pass a config object via `Robot(configs=...)`.

### Changed

- **`robot_model` property now returns the base model.** `Robot.robot_model` is derived from `RobotInfo` and returns the core platform type (e.g., `"vega_1"`, `"vega_1p"`, `"vega_1u"`) without hand/config suffixes.

## [0.4.2] - 2026-02-15

### Added

- **Force torque sensor control.** New `activate_force_torque_sensor()` and `get_force_torque_sensor_mode()` methods on `Arm`.
- **PID safety validation.** `set_pid` now rejects P-gain multipliers outside [0.1, 4].
- **Arm tracking benchmark examples** (sine wave and step response) for comparing PID settings.
- **New exception types for component/sensor access.** `ComponentNotAvailableError` raised when accessing unavailable robot components (with helpful message suggesting `has_component()` check). `SensorNotAvailableError` raised when accessing unavailable sensors (suggests `has_sensor()` check).

### Fixed

- **Config overwrite bug.** User-provided `configs` passed to `Robot()` were silently ignored because `RobotInfo` was always constructing its own default config. `Robot` now passes `configs` directly to `RobotInfo`, and when both `robot_model` and `configs` are provided, `configs` takes priority with a warning logged.

### Changed

- **Replaced `RuntimeError` with dexcontrol-specific exceptions throughout.** Arm service methods raise `ServiceUnavailableError`, robot-level methods raise `DexcontrolError`, and component activation raises `ComponentError`.
- **Consistent error handling for service calls.** Read-only queries (`get_pid`, `get_brake_status`, `get_force_torque_sensor_mode`, `get_ee_baud_rate`) raise `ServiceUnavailableError` on timeout. Write operations (`set_pid`, `set_ee_baud_rate`, `release_brake`, `activate_force_torque_sensor`, `DexGripper.set_mode`) return `{"success": False, ...}` fallback dict on timeout.
- **Examples use `has_component()` method** instead of direct attribute checks.
- **`get_robot_config()` now uses `RobotInfo.get_default_config`** instead of instantiating a full `RobotInfo` object, avoiding unnecessary URDF loading.
- **Replay trajectory example** now warns about potential end effector collisions with pre-existing trajectories.
- **Renamed sensor examples for clarity.** `get_lidar_data.py` → `get_2d_lidar_data.py`, `get_head_cam_data.py` → `get_head_zed_x_mini_data.py`, `get_live_lidar_data.py` → `get_live_2d_lidar_data.py`, `get_wrist_camera_data.py` → `get_wrist_zed_x_one_data.py`.

### Dependencies

- Requires `dexcomm >= 0.4.2`.
- Requires `dexbot-utils >= 0.4.3`.

### Version Requirements

- **SOC Minimal Version**: 419 (raised from 360).

## [0.4.1] - 2026-02-10

### Performance Improvements

- **Significantly reduced GIL contention for real-time control loops.** Robot component state updates now happen entirely in Rust without acquiring the Python GIL. Previously, every incoming state message triggered a Python callback that acquired the GIL — this caused latency spikes when running CPU-intensive workloads like neural network inference alongside control loops. With this change, background threads store raw bytes in Rust, and state is only decoded when you read it (with smart caching: <1μs for repeated reads, ~10μs for new data).
- **Heartbeat monitoring is now GIL-free.** The heartbeat safety monitor has been moved from a Python background thread to a Rust-backed `HeartbeatMonitor`. Heartbeat subscription, decoding, and timeout detection all run without GIL involvement, so heartbeat monitoring no longer interferes with Python workloads. The GIL is only briefly acquired if a timeout actually fires (to log the critical error before exiting).

### Added

- **Clearer error messages when things go wrong.** Introduced a hierarchy of specific exceptions to replace generic `RuntimeError`:
  - `ConfigurationError` — raised when `ZENOH_CONFIG` is not set, the config file is missing, or permissions are wrong. The error message tells you exactly what to fix.
  - `RobotConnectionError` — raised when the robot cannot be reached (not powered on, network issues, Zenoh routing problems).
  - `ServiceUnavailableError` — raised when a specific robot service (e.g., hand type query, version info) is not responding. This can happen during robot initialization or if a component is disabled.
  - All exceptions inherit from `DexcontrolError`, so you can catch all dexcontrol errors with a single `except DexcontrolError`.

### Fixed

- **End effector baud rate configuration now works.** The `set_ee_baud_rate` feature was broken because the service client used a hardcoded endpoint name that didn't match the robot's actual service. It now uses the correct service name from the arm configuration.

### Dependencies

- Requires `dexcomm >= 0.4.1` (for Rust-side subscriber storage and `HeartbeatMonitor`).
- Requires `dexbot-utils >= 0.4.1`.

## [0.4.0] - 2026-02-02

### Added
- Unified support for Vega-1U, Vega-1, and Vega-1P robot variants.
- Dex-gripper end effector support.
- Robot arm PID tuning (currently P gain only).
- Arm brake release when joint limits are exceeded.
- Extracted configuration management to `dexbot-utils` package.
- Custom end effector feedback topic for receiving data frames from unrecognized end effectors.
- USB camera support with dedicated `USBCamera` sensor class and configuration.
- Enhanced camera base class with improved RTC and dexcomm integration.
- New robot info helper utilities for better system information management.

### Changed
- **Major dexcomm Migration**: Migrated to dexcomm (>= 0.4.0) with Rust backend for improved performance and reliability.
- Refactored camera architecture with unified base camera implementation supporting both local and RTC modes.
- Simplified subscriber infrastructure by removing redundant subscriber wrapper code.
- Updated all sensor configurations for seamless compatibility with latest dexcomm API.
- Improved ZED camera implementation with better depth handling and timestamp support.
- Improve the error code message parsing for all the components. Now dexcontrol will no longer get any error code, it will directly get the error message for each joint.

### Fixed
- Sensor initialization issues after dexcomm migration.
- Camera streaming bugs with latest dexcomm integration.
- ZED head camera configuration and initialization.
- RTC camera configuration handling.
- IMU, LiDAR, and ultrasonic sensor compatibility with new dexcomm version.
- Error handling and reliability across all sensors.

### Dependencies
- **Breaking**: Requires `dexcomm >= 0.4.0` (migrated from `>= 0.1.18`).
- **Breaking**: Requires `dexbot-utils >= 0.4.0`.
- **Breaking**: Requires firmware >= 0.4.0 on the SOC.
- Removed optional dependencies: `aiortc`, `websockets` (now handled by dexcomm).

### Version Requirements
- **SOC Minimal Version**: 360

## [0.3.3] - 2025-10-09

### Added
- WebRTC camera streaming via `RTCSubscriber` with DexComm-compatible API; added `create_rtc_camera_subscriber`.
- ZED and RGB camera sensors now support RTC mode and DexComm factories; depth stream uses DexComm depth deserializer.
- ZED `get_obs` supports `include_timestamp` to passthrough timestamps when provided by DexComm.
- Core DexComm subscriber utilities: `create_subscriber`, `create_buffered_subscriber`, `create_camera_subscriber`, `create_depth_subscriber`, `create_imu_subscriber`, `create_lidar_subscriber`, `create_generic_subscriber`.
- Ultrasonic sensor now uses typed protobuf deserialization and returns standardized `(4,)` distance array in order `[front_left, front_right, back_left, back_right]`.

### Changed
- Consolidated sensor subscriptions onto DexComm-based factories with consistent topic resolution and config handling.
- Chassis IMU mappings standardized; `timestamp_ns` returned when available for clearer units.

### Dependencies
- Requires `dexcomm >= 0.1.15`.
- Optional WebRTC: `aiortc`, `websockets`; performance: `uvloop` (Unix).

### Version Requirements
- **SOC Minimal Version**: 298

## [0.3.2] - 2025-09-27

### Fixed
- **Arm Mode Switch Query**: Resolved timeout failures in arm mode switch queries, improving reliability during mode transitions.

### Version Requirements
- **SOC Minimal Version**: 298

## [0.3.1] - 2025-09-26

### Added
- **End Effector Control Example**: Added simple example demonstrating end effector control patterns for custom manipulator integration.

### Changed
- **DexComm Integration**: Migrated all Zenoh-based communication to dexcomm library for improved consistency and maintainability.
- **Environment Variable Standardization**: Changed Zenoh config file path environment variable from `DEXMATE_COMM_CFG_PATH` to `ZENOH_CONFIG` for consistency with dexcomm.

### Fixed
- **Hand Type Detection**: Fixed hand type query bug that incorrectly assumed unknown hand types when hands were connected.

### Version Requirements
- **SOC Minimal Version**: 298

## [0.3.0] - 2025-08-29

### Added
- Automatic hand type detection for component initialization. The robot will only have left_hand or right_hand attributes when hands are physically connected.
- Minimum version checking on the server side. Robot firmware now returns a minimum required client version to ensure dexcontrol compatibility.
- RS-485 pass-through support for end-effectors, enabling dexcontrol to send raw bytes directly to unknown end effectors.
- Free-drive motion support with brake release capability for arm motors.
- Support for v2-hand touch sensor readings.
- Unified interface for all motors, providing position, velocity, current/torque, error codes, and timestamps.

### Changed
- Enhanced internal driver communication efficiency. The driver now handles cases where client control frequency (e.g., dexcontrol) exceeds motor limits, preventing communication overload.
- Deployed new internal communication protocol for improved consistency and efficiency.
- Unified arm and head motor enables logic. Both the e-stop button and the software e-stop now disable head motors to prevent head-torso collisions.
- Separated chassis_steer and chassis_wheel into independent modules for better modularity.
- Enhanced logging and debugging capabilities.

### Fixed
- Control Frequency Issue: Fixed bug where control frequency affected reading frequency.
- Display Bug: Resolved false positive issue in display_robot_info function.

### Breaking Changes
- **Protobuf Definition Changes**: Updated protobuf definitions require dexcontrol >= 0.3.0 for all firmware >= 0.3.0.

### Version Requirements
- **SOC Minimal Version**: 286
