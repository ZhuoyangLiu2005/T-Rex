"""Dataclass-based configuration for the teleop stack.

All site/deployment-specific settings (network endpoints, device serials,
default poses, rates, environment geometry) live in a YAML file under
config/ at the repository root and are loaded into the typed dataclasses
below. Algorithmic constants stay in code.

Usage:
    from teleop.config import load_config
    cfg = load_config("config/default.yaml")

main_teleop.py loads the file given by its --config argument and lets a few
common CLI flags (e.g. --data-dir) override the loaded values.
"""

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


@dataclass
class RobotConfig:
    """Default joint positions the robot is reset to before each episode."""

    left_arm_default_joint_pos: list[float]
    right_arm_default_joint_pos: list[float]
    torso_default_joint_pos: list[float]
    head_default_joint_pos: list[float]

    @property
    def default_joint_pos(self) -> dict[str, np.ndarray]:
        return {
            "left_arm": np.array(self.left_arm_default_joint_pos),
            "right_arm": np.array(self.right_arm_default_joint_pos),
            "torso": np.array(self.torso_default_joint_pos),
            "head": np.array(self.head_default_joint_pos),
        }


@dataclass
class HandsConfig:
    """Sharpa Wave hand connection settings."""

    left_serial: str
    right_serial: str
    interpolate: bool = True


@dataclass
class ViveConfig:
    """Vive tracker streamer endpoint (vive_tracker/server.py on the Windows PC)."""

    ip: str
    port: str = "5555"
    left_tracker_name: str = "left_tracker"
    right_tracker_name: str = "right_tracker"
    update_hz: float = 30.0
    timeout_tol_s: float = 0.2


@dataclass
class HandActionConfig:
    """ZMQ endpoint of the Manus retargeting HandAction publisher."""

    address: str = "tcp://localhost:6668"


@dataclass
class TargetsConfig:
    """TeleopTargetSource update loop settings."""

    update_hz: float = 250.0


@dataclass
class CameraStreamConfig:
    """One ZMQ camera stream sender (see camera/README.md)."""

    sender_ip: str
    ports: dict[str, int] = field(default_factory=dict)


@dataclass
class CamerasConfig:
    """Camera streams and the recorded image size."""

    head: CameraStreamConfig
    wrist: CameraStreamConfig
    image_width: int = 640
    image_height: int = 360


@dataclass
class ControlConfig:
    """Control loop rates and safety tolerances."""

    command_hz: float = 30.0
    arm_action_hz: float = 300.0
    reset_dof_err_tol: float = 0.3  # rad


@dataclass
class TactileConfig:
    """Tactile fetching settings."""

    fetch_hz: float = 30.0


@dataclass
class EnvironmentConfig:
    """Collision environment around the robot (None disables an obstacle)."""

    table_height: float = 0.64
    back_wall_distance: typing.Optional[float] = None
    left_wall_distance: typing.Optional[float] = None
    right_wall_distance: typing.Optional[float] = None


@dataclass
class InferenceConfig:
    """Policy inference client settings (eval/eval_trex_async.py).

    The client talks to the T-Rex ZMQ inference server (T-Rex/scripts/test.py)
    using the slow/fast cascaded protocol.
    """

    # ZMQ REQ endpoint of the inference server.
    server_address: str = "tcp://localhost:5678"
    # Language instruction sent with every slow request; can be overridden on
    # the command line with --task-description.
    task_description: str = ""
    # False = right arm + right hand only (31-D actions instead of 62-D).
    dual_arm: bool = True
    # Action chunk length predicted by the policy.
    chunk_size: int = 16
    # Steps executed from each chunk before requesting a new one.
    execute_steps_per_chunk: int = 16
    # Hard cap on control steps per trajectory.
    max_steps: int = 10000
    # Send mode='slow_and_fast' at chunk start and tactile-only mode='fast'
    # refinements at refine_offsets within the chunk.
    use_tactile_refine: bool = True
    refine_offsets: list[int] = field(default_factory=lambda: [4, 8, 12])
    # ACT-style exponential temporal aggregation across received chunks.
    use_temporal_aggregation: bool = True
    temporal_agg_k: float = 0.0  # 0 = uniform average; larger = newer dominates
    # Head image crop (row_start, row_end, col_start, col_end) applied before
    # sending to the server; null disables cropping. Must match training.
    head_crop_box: typing.Optional[list[int]] = field(default_factory=lambda: [0, 300, 140, 540])
    # Live OpenCV dashboard of the inference inputs.
    show_live_viz: bool = True
    # Save per-step camera JPEGs to the working directory (debug only).
    save_debug_images: bool = False


@dataclass
class DataConfig:
    """Episode recording settings."""

    data_dir: str = "./teleop_data"
    # Store DEFORM/RAW tactile maps as lossless grayscale videos instead of
    # uncompressed HDF5 datasets (see data_writer.py).
    tactile_maps_as_video: bool = True
    tactile_video_codec: str = "libx264"  # 'libx264' (-qp 0) or 'ffv1'


@dataclass
class TeleopConfig:
    robot: RobotConfig
    hands: HandsConfig
    vive: ViveConfig
    cameras: CamerasConfig
    hand_action: HandActionConfig = field(default_factory=HandActionConfig)
    targets: TargetsConfig = field(default_factory=TargetsConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    tactile: TactileConfig = field(default_factory=TactileConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def _build_dataclass(cls, data: dict, path: str):
    """Recursively construct dataclass `cls` from a dict, erroring on unknown keys."""
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping for '{path}', got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"Unknown config key(s) under '{path}': {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _build_dataclass(hint, value, f"{path}.{name}")
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> TeleopConfig:
    """Load a TeleopConfig from a YAML file."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        # Also try resolving relative to the repository root, so
        # `--config config/lab.yaml` works regardless of the cwd.
        candidate = PROJECT_ROOT / path
        if candidate.exists():
            path = candidate
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _build_dataclass(TeleopConfig, raw, "config")
