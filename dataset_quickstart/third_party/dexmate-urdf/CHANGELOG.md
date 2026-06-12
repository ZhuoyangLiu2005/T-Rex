# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.3] - 2026-05-07

### Changed
- Updated vega_1u visual meshes for `lift_link` and `torso_link` for better fidelity with the real robot.

### Fixed
- Corrected `base_link` orientation in all vega_1u URDF variants (`vega_1u.urdf`, `vega_1u_f5d6.urdf`, `vega_1u_gripper.urdf`) so the Vega U base frame is consistent with Vega 1 robots; also adjusted the `Lift` joint origin to compensate.

## [0.8.2] - 2026-04-10

### Fixed
- Added `disable_collisions` entry between `lift_link` and `torso_flip_link` in all vega_1u SRDF variants (`vega_1u.srdf`, `vega_1u_f5d6.srdf`, `vega_1u_gripper.srdf`) to prevent always-in-collision warnings.

## [0.8.1] - 2026-03-26

### Changed
- Improved vega_1u modeling to better match the real robot.

### Fixed
- Lidar joint orientations (`back_lidar_mount`, `front_lidar_mount`) in all vega_1p variants (`vega_1p.urdf`, `vega_1p_f5d6.urdf`, `vega_1p_gripper.urdf`) to ensure consistent lidar mounting across all configurations.

## [0.8.0] - 2025-12-18

### Added
- 9 robot variants with different end-effector configurations:
  - `vega_1`, `vega_1_gripper`, `vega_1_f5d6` (full body; renamed from `vega.urdf` to `vega_1.urdf`)
  - `vega_1u`, `vega_1u_gripper`, `vega_1u_f5d6` (upper body only)
  - `vega_1p`, `vega_1p_gripper`, `vega_1p_f5d6` (full body, pro version)
- `connector.glb` visual element on `gripper_base` links in all gripper variants (`vega_1_gripper`, `vega_1p_gripper`, `vega_1u_gripper`) for proper hand-arm interface visualization.
- New `dexd_gripper` model with complete URDF and mesh files.
- 4 new visual meshes specific to vega_1p: `back_lidar.glb`, `front_lidar.glb`, `base.glb` (pro version), `torso_l3.glb` (pro version).
- Backward-compatibility symlinks for `vega.urdf` and `vega.srdf` to support older versions, with automated symlink creation via GitHub workflows.

### Changed
- Reorganized folder structure so variants with different end-effectors are consolidated into parent folders:
  - `vega_1/` contains `vega_1.urdf`, `vega_1_f5d6.urdf`, `vega_1_gripper.urdf`
  - `vega_1u/` contains `vega_1u.urdf`, `vega_1u_f5d6.urdf`, `vega_1u_gripper.urdf`
  - `vega_1p/` contains `vega_1p.urdf`, `vega_1p_f5d6.urdf`, `vega_1p_gripper.urdf`
- Configs are now organized in subdirectories: `configs/vega_1/`, `configs/vega_1_f5d6/`, etc.
- Mesh assets reorganized into shared `humanoid/vega_1/meshes` folder for all variants.
- Enhanced `torso_l3` collision mesh with improved geometry.

## [0.7.1] - 2025-10-13

### Added
- USD and PyPI release workflows to the public GitHub repository, making the USD generation process transparent.

## [0.7.0] - 2025-10-10

### Added
- OBJ export script alongside `.glb` files to support legacy visualization software.

### Changed
- Simplified the visual meshes, significantly reducing their size. As a result, the generated IsaacSim USD files are now much smaller, saving GPU memory during simulations with visual sensors.
