# Copyright 2025 Dexmate Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import subprocess
from pathlib import Path

from scripts.workflows.decomp_urdf_collision_meshes import decompose_urdf

CONVERT_SCRIPT = Path("scripts/workflows/convert_urdf.py")
SRC_ROOT = Path(__file__).parent.parent.parent / "robots"

# Add the robot directories to process here
TARGET_ROBOT_DIRS = [
    "humanoid/vega_1",
    "humanoid/vega_1p",
    "humanoid/vega_1u",
]


def convert_urdf_to_usd(urdf_path: Path, output_dir: Path, python_exe: Path) -> None:
    """Convert URDF file to USD format.

    Args:
        urdf_path: Path to the URDF file
        output_dir: Directory to store the USD output
        python_exe: Path to Python executable

    Raises:
        subprocess.CalledProcessError: If conversion process fails
    """
    decomposed_urdf_path = decompose_urdf(urdf_path)

    robot_name = urdf_path.stem
    usd_dir = output_dir / robot_name
    usd_path = usd_dir / f"{robot_name}.usd"
    usd_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(python_exe),
        str(CONVERT_SCRIPT),
        str(decomposed_urdf_path),
        str(usd_path),
        "--fix-base",
        "--headless",
    ]
    print(f"Converting to USD: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    # Create zip file of the generated USD directory
    zip_path = f"../{robot_name}.zip"
    subprocess.check_call(["zip", "-r", str(zip_path), "."], cwd=str(usd_dir))


def main():
    """Main entry point for the USD workflow."""
    parser = argparse.ArgumentParser(
        description="Generate sub-URDFs and convert to USD in output directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to store all generated USD files.",
    )
    parser.add_argument(
        "--python_exe",
        type=str,
        default="python",
        help="Path to the Python executable to use for subprocess calls (default: python).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    python_exe = Path(args.python_exe)

    # Generate all sub-URDFs
    all_urdf_paths = []
    for robot_model_dir in TARGET_ROBOT_DIRS:
        robot_model_dir = SRC_ROOT / robot_model_dir
        urdf_paths = list(robot_model_dir.rglob("*.urdf"))
        all_urdf_paths.extend(urdf_paths)

    # Convert all URDFs to USD
    for urdf_path in all_urdf_paths:
        urdf_path = urdf_path.resolve()
        assert urdf_path.exists()
        convert_urdf_to_usd(urdf_path, output_dir, python_exe)


if __name__ == "__main__":
    main()
