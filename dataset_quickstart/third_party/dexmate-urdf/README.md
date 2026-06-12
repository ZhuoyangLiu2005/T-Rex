<div align="center">
  <h1>🤖 Dexmate URDF Collection</h1>
  <p>
    <strong>High-quality Robot Models for Simulation and Planning</strong>
  </p>
</div>

![Models](https://img.shields.io/badge/Models-URDF%20%7C%20SRDF-blue)
![Python](https://img.shields.io/badge/pypi/wheel/dexmate-urdf)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

## 🎯 Overview

This repository contains high-fidelity robot models in Unified Robot Description Format (URDF) and Semantic Robot Description Format (SRDF). These models are essential for:

- 🎮 Robot Simulation
- 🔄 Motion Planning
- 🎨 Visualization
- 🛠️ Control System Development

The corresponding USD files are be found in the [Release](https://github.com/dexmate-ai/dexmate-urdf/releases).

## 📦 Installation

```shell
pip install dexmate_urdf
```

or install from source
```shell
cp -r robots/* src/dexmate_urdf/robots/
python scripts/workflows/generate_content.py
pip install -e .
```

## 🚀 Quick Start

```python
from dexmate_urdf import robots

# Access different robot models and configurations
# Base variants
vega_1_urdf = robots.humanoid.vega_1.vega_1.urdf
vega_1u_urdf = robots.humanoid.vega_1u.vega_1u.urdf
vega_1p_urdf = robots.humanoid.vega_1p.vega_1p.urdf

# Variants with different end-effectors
vega_1_f5d6_urdf = robots.humanoid.vega_1.vega_1_f5d6.urdf
vega_1_gripper_urdf = robots.humanoid.vega_1.vega_1_gripper.urdf
vega_1u_f5d6_urdf = robots.humanoid.vega_1u.vega_1u_f5d6.urdf

# Access SRDF and collision URDFs
vega_1_srdf = robots.humanoid.vega_1.vega_1.srdf
vega_1_collision = robots.humanoid.vega_1.vega_1.collision_spheres_urdf

# Load into your favorite simulator
load_robot(vega_1_f5d6_urdf)
```

## 🤖 Available Models

| Robot | Visual[^1] | Convex Collision[^2] | Sphere Collision[^3] |
|:-----------:|:----------:|:------------------:|:------------------:|
| Vega-1 | <img src="docs/vega/visual.png" width="400"> | <img src="docs/vega/collision.png" width="400"> | <img src="docs/vega/collision_spheres.png" width="400"> |

[^1]: High quality visual modeling for rendering.
[^2]: Collision modeling composed of convex decomposition meshes with light simplification, which can be used for physical simulation, etc.
[^3]: Collision modeling composed of spheres, which can be an alternative collision representation when speed is more of a concern. These meshes are larger than the real one, which is not desirable to be used in high-fidelity simulation.

## 🔧 Supported Platforms

Our models are tested with popular robotics frameworks:

- 🎮 **IsaacSim/IsaacLab/IsaacGym** - For simulation and RL training
- 🔄 **Pinocchio** - For kinematics and dynamics computation
- 🎯 **SAPIEN** - For simulation and visualization

## 📚 Package Structure

```python
dexmate_urdf.robots
├── humanoid/
│   ├── vega_1/                  # Base variant folder
│   │   ├── vega_1.urdf          # Base variant (no hands)
│   │   ├── vega_1_f5d6.urdf     # With F5D6 hands
│   │   ├── vega_1_gripper.urdf  # With gripper hands
│   │   ├── vega.urdf            # Alias (symlink to vega_1_f5d6)
│   │   └── configs/             # Variant-specific configs
│   ├── vega_1u/                  # Upper body variants
│   │   ├── vega_1u.urdf
│   │   ├── vega_1u_f5d6.urdf
│   │   └── vega_1u_gripper.urdf
│   └── vega_1p/                  # Pro variants
│       ├── vega_1p.urdf
│       ├── vega_1p_f5d6.urdf
│       └── vega_1p_gripper.urdf
└── ... # More robots
```

## 📝 Changelog
For information about recent updates and changes, please refer to the [CHANGELOG.md](CHANGELOG.md).

## 📄 Licensing

### 🔓 Apache License 2.0
This software is licensed under the **Apache License 2.0**. This permissive license allows you to:

- ✅ Use the software commercially
- ✅ Modify the software and create derivatives
- ✅ Distribute copies and modifications
- ✅ Use patent claims (if applicable)

See the [LICENSE](./LICENSE) file for the complete license text.

---

<div align="center">
  <h3>🤝 Need help with robot models?</h3>
  <p>
    <a href="https://dexmate.ai">🌐 Visit Website</a> •
    <a href="mailto:contact@dexmate.ai">📧 Contact Us</a> •
    <a href="./robots/">📚 View Robots</a>
  </p>
</div>
