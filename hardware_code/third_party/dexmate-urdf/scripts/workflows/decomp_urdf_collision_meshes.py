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
import xml.etree.ElementTree as ET
from pathlib import Path

import trimesh


def process_collision_mesh(mesh_path: Path) -> list[Path]:
    """Process a collision mesh and return paths to the processed meshes.

    Args:
        mesh_path: Path to the original mesh file

    Returns:
        List of paths to the processed mesh files
    """
    try:
        # Load the mesh using trimesh
        mesh = trimesh.load(str(mesh_path))
        submeshes = mesh.split()

        # If there's only one mesh, return the original path
        if len(submeshes) <= 1:
            return [mesh_path]

        # Create directory for submeshes
        mesh_name = mesh_path.stem
        submesh_dir = mesh_path.parent / "submeshes" / mesh_name
        submesh_dir.mkdir(parents=True, exist_ok=True)

        # Process each submesh
        processed_paths = []
        for i, submesh in enumerate(submeshes):
            if isinstance(submesh, trimesh.Trimesh):
                # Save each submesh
                submesh_path = submesh_dir / f"{mesh_name}_{i}.stl"
                submesh.export(str(submesh_path))
                processed_paths.append(submesh_path)

        return processed_paths

    except Exception as e:
        print(f"Error processing mesh {mesh_path}: {str(e)}")
        return [mesh_path]


def copy_origin_element(source: ET.Element, target: ET.Element):
    """Copy origin element from source to target if it exists.

    Args:
        source: Source element to copy from
        target: Target element to copy to
    """
    origin = source.find("origin")
    if origin is not None:
        # Create a deep copy of the origin element
        new_origin = ET.SubElement(target, "origin")
        # Copy all attributes
        for key, value in origin.attrib.items():
            new_origin.set(key, value)


def decompose_urdf(input_urdf: Path) -> Path:
    """Process a URDF file and handle collision meshes.

    Args:
        input_urdf: Path to input URDF file
    """
    # Create output directory for meshes
    output_urdf = input_urdf.with_name(
        f"{input_urdf.stem}-decomposed{input_urdf.suffix}"
    )

    # Load and parse the URDF
    with open(input_urdf, "r", encoding="utf-8") as f:
        content = f.read()
        # Extract the XML header
        header_end = content.find(">") + 1
        xml_header = content[:header_end]
        # Parse the rest of the content
        root = ET.fromstring(content[header_end:])

    # Process each link's collision meshes
    for link in root.findall("link"):
        # Find all collision elements
        collisions = link.findall("collision")
        for collision in collisions:
            geometry = collision.find("geometry")
            if geometry is None:
                continue

            mesh = geometry.find("mesh")
            if mesh is None:
                continue

            mesh_path = mesh.get("filename")
            if not mesh_path:
                continue

            # Get absolute path of the mesh
            abs_mesh_path = input_urdf.parent / mesh_path

            # Process the mesh
            new_mesh_paths = process_collision_mesh(abs_mesh_path)

            # If only one mesh, just update the path
            if len(new_mesh_paths) == 1:
                rel_path = new_mesh_paths[0].relative_to(output_urdf.parent)
                mesh.set("filename", str(rel_path))
                continue

            # For multiple meshes, create new collision elements
            parent = link
            for i, new_mesh_path in enumerate(new_mesh_paths):
                if i == 0:
                    # Update the first collision element
                    rel_path = new_mesh_path.relative_to(output_urdf.parent)
                    mesh.set("filename", str(rel_path))
                else:
                    # Create new collision elements for additional meshes
                    new_collision = ET.SubElement(parent, "collision")
                    # Copy origin if it exists
                    copy_origin_element(collision, new_collision)

                    new_geometry = ET.SubElement(new_collision, "geometry")
                    new_mesh = ET.SubElement(new_geometry, "mesh")
                    rel_path = new_mesh_path.relative_to(output_urdf.parent)
                    new_mesh.set("filename", str(rel_path))

    # Save the modified URDF with original header
    with open(output_urdf, "w", encoding="utf-8") as f:
        f.write(xml_header)
        f.write(ET.tostring(root, encoding="unicode"))
    print(f"Decomposed URDF saved to {output_urdf}")
    return Path(output_urdf)


def main():
    parser = argparse.ArgumentParser(description="Process URDF collision meshes")
    parser.add_argument("input_urdf", type=Path, help="Path to input URDF file")
    args = parser.parse_args()

    decompose_urdf(args.input_urdf)


if __name__ == "__main__":
    main()
