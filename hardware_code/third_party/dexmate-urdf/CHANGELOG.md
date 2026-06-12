# Changelog

All notable changes to this project will be documented in this file.

## [0.8.0] - 2025-12-18

### New Robot Variants
- Added 9 robot variants with different end-effector configurations:
  - `vega_1`, `vega_1_gripper`, `vega_1_f5d6` (full body, same as before but we rename the urdf name from vega.urdf to vega_1.urdf)
  - `vega_1u`, `vega_1u_gripper`, `vega_1u_f5d6` (upper body only)
  - `vega_1p`, `vega_1p_gripper`, `vega_1p_f5d6` (full body, pro version)

### Folder Structure Reorganization
- Reorganized folder structure: Variants with different end-effectors are now consolidated into parent folders:
  - `vega_1/` contains `vega_1.urdf`, `vega_1_f5d6.urdf`, `vega_1_gripper.urdf`
  - `vega_1u/` contains `vega_1u.urdf`, `vega_1u_f5d6.urdf`, `vega_1u_gripper.urdf`
  - `vega_1p/` contains `vega_1p.urdf`, `vega_1p_f5d6.urdf`, `vega_1p_gripper.urdf`
- Configs are organized in subdirectories: `configs/vega_1/`, `configs/vega_1_f5d6/`, etc.
- Reorganized mesh assets into shared `humanoid/vega_1/meshes` folder for all variants.

- **Connector Visual**: Added connector.glb visual element to gripper_base links in all gripper variants (vega_1_gripper, vega_1p_gripper, vega_1u_gripper) for proper hand-arm interface visualization
- **New Gripper Model**: Added dexd_gripper with complete URDF and mesh files

### Vega Pro Version (vega_1p)
- Added 4 new visual meshes specific to vega_1p:
  - back_lidar.glb
  - front_lidar.glb
  - base.glb (pro version)
  - torso_l3.glb (pro version)
- Enhanced torso_l3 collision mesh with improved geometry (18,102 lines)

### Backward Compatibility
- Maintains backward compatibility with `vega.urdf` and `vega.srdf` symlinks for older versions
- Automated symlink creation via GitHub workflows


## [0.7.1] - 2025-10-13
- Added USD and PyPI release workflows to the public GitHub repository, making the USD generation process transparent.

## [0.7.0] - 2025-10-10
- Simplified the visual meshes, significantly reducing their size. As a result, the generated IsaacSim USD files are now much smaller, saving GPU memory during simulations with visual sensors.
- Added OBJ file script alongside `.glb` files to support legacy visualization software.
