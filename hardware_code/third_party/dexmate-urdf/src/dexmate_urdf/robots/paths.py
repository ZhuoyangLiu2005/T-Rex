import importlib.resources
from pathlib import Path
from typing import Literal

ROBOT_TYPES = Literal["hands", "humanoid", "arms", "assembly"]


def get_robot_dir() -> Path:
    """Get the root directory containing all robot models."""
    f = importlib.resources.files("dexmate_urdf")
    return (f / "robots").resolve()


def get_robot_names(
    robot_type: ROBOT_TYPES,
) -> list[str]:
    """Get all available robots for a given type.

    Args:
        robot_type: Type of robot (hand, humanoid, arm, assembly)

    Returns:
        List of robot names available for the given type
    """
    robot_dir = get_robot_dir() / robot_type
    if not robot_dir.exists():
        raise ValueError(f"Robot type {robot_type} not found")
    return [p.name for p in robot_dir.iterdir() if p.is_dir()]


def get_robot_path(
    robot_type: ROBOT_TYPES,
    robot_name: str,
) -> Path:
    """Get the path to a specific robot's directory.

    Args:
        robot_type: Type of robot (hand, humanoid, arm, assembly)
        robot_name: Name of the robot

    Returns:
        Path to the robot's directory
    """
    robot_dir = get_robot_dir() / robot_type / robot_name
    if not robot_dir.exists():
        raise ValueError(f"Robot {robot_name} of type {robot_type} not found")
    return robot_dir

def get_type_path(
    robot_type: ROBOT_TYPES,
) -> Path:
    """Get the path to a robot type's directory."""
    type_dir = get_robot_dir() / robot_type
    if not type_dir.exists():
        raise ValueError(f"Robot type {robot_type} not found")
    return type_dir


def get_urdf_paths(
    robot_type: ROBOT_TYPES, robot_name: str
) -> list[Path]:
    """Get the path to a robot's URDF file.

    Args:
        robot_type: Type of robot (hand, humanoid, arm, assembly)
        robot_name: Name of the robot

    Returns:
        List of paths to the robot's URDF files
    """
    robot_dir = get_robot_path(robot_type, robot_name)
    urdf_files = list(robot_dir.glob("*.urdf"))
    if not urdf_files:
        raise FileNotFoundError(f"No URDF file found for {robot_name} in {robot_type}")
    return urdf_files


def get_mesh_dir(
    robot_type: ROBOT_TYPES, robot_name: str
) -> Path:
    """Get the path to a robot's mesh directory.

    Args:
        robot_type: Type of robot (hand, humanoid, arm, assembly)
        robot_name: Name of the robot

    Returns:
        Path to the robot's mesh directory
    """
    robot_dir = get_robot_path(robot_type, robot_name)
    mesh_dir = robot_dir / "meshes"
    if not mesh_dir.exists():
        raise FileNotFoundError(
            f"No meshes directory found for {robot_name} in {robot_type}"
        )
    return mesh_dir

class URDFModel:
    """Represents a specific URDF model file within a robot directory."""

    def __init__(self, robot_type: str, robot_name: str, urdf_name: str):
        self._type = robot_type
        self._name = robot_name
        self._urdf_name = urdf_name
        self._parent_dir = get_robot_path(robot_type, robot_name)

    @property
    def urdf(self) -> Path:
        """Get the path to this specific URDF file."""
        return self._parent_dir / f"{self._urdf_name}.urdf"

    @property
    def srdf(self) -> Path:
        """Get the corresponding SRDF file path if it exists."""
        srdf_name = f"{self._urdf_name.rsplit('.', 1)[0]}.srdf"
        path = self._parent_dir / srdf_name
        if path.exists():
            return path
        raise FileNotFoundError(f"SRDF file not found: {path}")

    @property
    def collision_spheres_urdf(self) -> Path:
        """Get the corresponding collision spheres URDF path if it exists."""
        collision_name = f"{self._urdf_name.rsplit('.', 1)[0]}_collision_spheres.collision.urdf"
        path = self._parent_dir / collision_name
        if path.exists():
            return path
        raise FileNotFoundError(f"Collision spheres URDF file not found: {path}")

    def __repr__(self) -> str:
        return str(self.urdf)


class RobotModel:
    """Represents a robot model containing URDF models."""

    def __init__(self, robot_type: str, robot_name: str):
        self._type = robot_type
        self._name = robot_name
        self._urdf_models: dict[str, URDFModel] = {}

        # Initialize URDF models
        for urdf_path in get_urdf_paths(robot_type, robot_name):
            if urdf_path.suffix == ".urdf":
                model_name = urdf_path.stem
                self._urdf_models[model_name] = URDFModel(
                    robot_type, robot_name, urdf_path.stem
                )

    @property
    def model_list(self) -> list[URDFModel]:
        """Get the list of URDF models in this robot."""
        return list(self._urdf_models.values())

    @property
    def mesh_dir(self) -> Path:
        """Get the mesh directory for this robot."""
        return get_mesh_dir(self._type, self._name)

    @property
    def path(self) -> Path:
        """Get the path to this robot directory."""
        return get_robot_path(self._type, self._name)

    def __getattr__(self, name: str) -> URDFModel:
        """Allow accessing URDF models as attributes."""
        if name in self._urdf_models:
            return self._urdf_models[name]
        raise AttributeError(f"URDF model '{name}' not found in {self.path}")

    def __dir__(self) -> list[str]:
        return list(self._urdf_models.keys())

    def __repr__(self) -> str:
        return str(self.path)


class RobotType:
    """Represents a type of robot (e.g., humanoid, hand)."""

    def __init__(self, robot_type: str):
        self._type = robot_type
        self._models: dict[str, RobotModel] = {}

        # Directories to skip (shared assets, not robot models)
        skip_dirs = {"meshes", "internal"}

        # Dynamically load robot models
        type_dir = get_robot_dir() / robot_type
        if type_dir.exists():
            for robot_dir in type_dir.iterdir():
                if robot_dir.is_dir() and robot_dir.name not in skip_dirs:
                    model_name = robot_dir.name
                    self._models[model_name] = RobotModel(robot_type, model_name)

    @property
    def path(self) -> Path:
        """Get the path to this robot type's directory."""
        return get_type_path(self._type)

    def __getattr__(self, name: str) -> RobotModel:
        if name in self._models:
            return self._models[name]
        raise AttributeError(f"Robot model '{name}' not found in {self._type}")

    def __dir__(self) -> list[str]:
        return list(self._models.keys())

    def __repr__(self) -> str:
        return str(self.path)
