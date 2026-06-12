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

"""Convert GLB visual meshes to OBJ/STL and update URDF files."""

import re
from pathlib import Path

import trimesh
import tyro
from tqdm import tqdm


def _rename_texture_files(output_dir: Path, mesh_name: str):
    """Rename generic texture files to be unique per mesh.

    Trimesh exports textures as material_0.png, material_1.png, etc.
    This function renames them to {mesh_name}_material_0.png, etc.
    and updates the corresponding MTL file references.

    Args:
        output_dir: Directory containing the exported OBJ/MTL/texture files
        mesh_name: Name of the mesh (stem of the filename)
    """
    mtl_file = output_dir / f"{mesh_name}.mtl"
    if not mtl_file.exists():
        return

    # Read MTL content
    with open(mtl_file, "r") as f:
        mtl_content = f.read()

    # Find all texture references and rename files
    texture_pattern = re.compile(r"map_(\w+)\s+(\S+)")
    updated_content = mtl_content

    for match in texture_pattern.finditer(mtl_content):
        map_type = match.group(1)  # e.g., "Kd" for diffuse
        old_texture_name = match.group(2)  # e.g., "material_0.png"

        old_texture_path = output_dir / old_texture_name
        if old_texture_path.exists():
            # Create unique texture name: {mesh_name}_{material_0.png}
            new_texture_name = f"{mesh_name}_{old_texture_name}"
            new_texture_path = output_dir / new_texture_name

            # Rename the texture file
            old_texture_path.rename(new_texture_path)

            # Update MTL content
            updated_content = updated_content.replace(
                f"map_{map_type} {old_texture_name}",
                f"map_{map_type} {new_texture_name}",
            )

    # Write updated MTL file
    with open(mtl_file, "w") as f:
        f.write(updated_content)


def convert_visual_meshes(
    robot_dir: Path,
    output_format: str = "obj",
) -> dict[str, str]:
    """Convert all GLB files in meshes/visual/ to meshes/{format}/ folder.

    Args:
        robot_dir: Robot directory containing URDF files and meshes folder
        output_format: Output format - 'obj' (ASCII) or 'stl' (binary)

    Returns:
        Dictionary mapping GLB paths to converted paths (relative to robot_dir)
    """
    visual_dir = robot_dir / "meshes" / "visual"
    if not visual_dir.exists():
        print(f"Warning: No meshes/visual directory found in {robot_dir}")
        return {}

    # Create output directory: meshes/obj/ or meshes/stl/
    output_dir = robot_dir / "meshes" / output_format
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all GLB files
    glb_files = list(visual_dir.rglob("*.[gG][lL][bB]"))
    if not glb_files:
        print(f"No GLB files found in {visual_dir}")
        return {}

    print(f"Found {len(glb_files)} GLB files in {visual_dir.relative_to(robot_dir)}")

    # Track conversions for URDF updates
    conversion_map = {}
    successful = 0
    failed = 0

    for glb_file in tqdm(glb_files, desc=f"Converting GLB to {output_format.upper()}"):
        # Get relative path from visual dir to preserve subdirectory structure
        rel_path = glb_file.relative_to(visual_dir)
        out_file = output_dir / rel_path.with_suffix(f".{output_format}")

        # Create parent folders in output dir
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Load GLB mesh using trimesh
        mesh = trimesh.load(str(glb_file), force="mesh")

        # Export with format-specific options
        if output_format == "obj":
            material_name = f"{glb_file.stem}.mtl"
            mesh.export(
                str(out_file),
                file_type="obj",
                include_color=True,
                include_texture=True,
                include_normals=True,
                mtl_name=material_name,
            )

            _rename_texture_files(out_file.parent, glb_file.stem)

        else:
            # STL exports as binary by default
            mesh.export(str(out_file), file_type=output_format)

        # Track conversion: meshes/visual/file.glb -> meshes/obj/file.obj
        old_path = f"meshes/visual/{rel_path}"
        new_path = f"meshes/{output_format}/{rel_path.with_suffix(f'.{output_format}')}"
        conversion_map[str(old_path)] = str(new_path)

        successful += 1

    # Summary
    if failed > 0:
        print(f", ✗ {failed} failed", end="")
    print()

    return conversion_map


def update_urdf_visual_meshes(
    urdf_path: Path,
    conversion_map: dict[str, str],
    output_format: str,
) -> Path:
    """Create new URDF file with updated visual mesh references.

    Args:
        urdf_path: Path to original URDF file
        conversion_map: Dictionary mapping old paths to new paths
        output_format: Output format used ('obj' or 'stl')

    Returns:
        Path to new URDF file
    """
    # Read original URDF
    with open(urdf_path, "r") as f:
        content = f.read()

    # Track changes
    changes = 0

    # Replace GLB references with converted format paths
    def replace_visual_mesh(match):
        nonlocal changes
        full_path = match.group(1)

        # Only replace if it's a GLB file
        if not full_path.lower().endswith(".glb"):
            return match.group(0)

        # Check if we have a conversion for this path
        if full_path in conversion_map:
            new_path = conversion_map[full_path]
            changes += 1

            return f'<mesh filename="{new_path}" />'

        return match.group(0)

    # Use regex to find and replace mesh filenames
    # This preserves XML formatting and only affects GLB meshes
    visual_pattern = re.compile(r'<mesh filename="([^"]*\.glb)"[^>]*/>', re.IGNORECASE)
    updated_content = visual_pattern.sub(replace_visual_mesh, content)

    # Generate output filename
    stem = urdf_path.stem
    output_path = urdf_path.parent / f"{stem}-{output_format}.urdf"

    # Write new URDF
    with open(output_path, "w") as f:
        f.write(updated_content)

    return output_path


def process_robot_folder(
    robot_dir: Path,
    output_format: str = "obj",
):
    """Process a robot folder: convert meshes and update URDFs.

    Args:
        robot_dir: Path to robot directory containing URDF and meshes
        output_format: Output format - 'obj' (ASCII) or 'stl' (binary)
    """
    print(f"\n{'=' * 60}")
    print(f"Processing robot folder: {robot_dir.name}")
    print(f"{'=' * 60}\n")

    # Step 1: Convert visual meshes
    conversion_map = convert_visual_meshes(robot_dir, output_format)

    if not conversion_map:
        print("No meshes converted. Skipping URDF updates.")
        return

    # Step 2: Find and update all URDF files (skip already-converted ones)
    all_urdf_files = list(robot_dir.glob("*.urdf"))
    # Filter out URDFs that are already converted (contain -obj.urdf or -stl.urdf)
    urdf_files = [
        f
        for f in all_urdf_files
        if "-obj.urdf" not in f.name and "-stl.urdf" not in f.name
    ]

    if not urdf_files:
        if all_urdf_files:
            print(
                f"All URDF files in {robot_dir} are already converted. Skipping URDF updates."
            )
        else:
            print(f"Warning: No URDF files found in {robot_dir}")
        return

    print(f"Updating {len(urdf_files)} URDF file(s)...")
    updated_files = []

    for urdf_file in urdf_files:
        new_urdf = update_urdf_visual_meshes(urdf_file, conversion_map, output_format)
        updated_files.append(new_urdf)

    print(f"\n✅ Successfully processed robot: {robot_dir.name}")


def main(
    robot_dir: Path,
    format: str = "obj",
):
    """Convert GLB visual meshes to OBJ/STL and generate updated URDF files.

    Args:
        robot_dir: Path to robot directory containing URDF files and meshes folder
        format: Output format - 'obj' for ASCII OBJ (with materials) or 'stl' for binary STL
    """
    # Validate robot directory
    robot_dir = robot_dir.resolve()
    if not robot_dir.is_dir():
        print(f"Error: {robot_dir} is not a valid directory")
        return 1

    # Validate format
    format = format.lower()
    if format not in ["obj", "stl"]:
        print(f"Error: Unsupported format '{format}'. Use 'obj' or 'stl'")
        return 1

    # Process the robot folder
    process_robot_folder(robot_dir, format)
    return 0


if __name__ == "__main__":
    exit(tyro.cli(main))
