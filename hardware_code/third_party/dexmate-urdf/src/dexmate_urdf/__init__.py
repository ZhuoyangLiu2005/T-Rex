"""dexmate_urdf: URDF tools for Dexmate robots."""

from pathlib import Path

from . import robots
from .robots.paths import get_mesh_dir, get_robot_names, get_robot_path, get_urdf_paths

LIB_PATH = Path(__file__).resolve().parent
ASSET_PATH = LIB_PATH.joinpath("robots")

__all__ = [
    "robots",
    "get_robot_names",
    "get_robot_path",
    "get_mesh_dir",
    "get_urdf_paths",
]
