<div align="center">
  <h1>🤖 Dexmate Robot Control and Sensing API</h1>
</div>

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)

## 📦 Installation

```shell
pip install dexcontrol
```

To run the examples in this repo, you can try:

```shell
pip install dexcontrol[example]
```

## ⚠️ Version Compatibility

**Important:** `dexcontrol >= 0.4.0` requires robot firmware `>= 0.4.0`. Using older firmware with this version will not work.

> **Note:** `dexcontrol 0.4.x` depends on `dexcomm >= 0.4.0`, which is **not compatible** with `dexcontrol 0.3.x`. If you need to stay on `dexcontrol 0.3.x`, do not upgrade `dexcomm` to `0.4.0` or above.

**Before upgrading, check your current firmware version:**
```shell
dextop firmware info
```

If your firmware is outdated, please update it before installing the new version to ensure full compatibility. Please contact the Dexmate team if you do not know how to do it.

**📋 See [CHANGELOG.md](./CHANGELOG.md) for detailed release notes and version history.**

## 📄 Licensing

This project is **dual-licensed**:

### 🔓 Open Source License
This software is available under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
See the [LICENSE](./LICENSE) file for details.

### 💼 Commercial License
For businesses that want to use this software in proprietary applications without the AGPL requirements, commercial licenses are available.

**📧 Contact us for commercial licensing:** contact@dexmate.ai

**Commercial licenses provide:**
- ✅ Right to use in closed-source applications
- ✅ No source code disclosure requirements
- ✅ Priority support options


## 📚 Examples

Explore our comprehensive examples in the `examples/` directory:

- 🎮 **Basic Control** - Simple movement and sensor reading
- 🎯 **Advanced Control** - Complex manipulation tasks
- 📺 **Teleoperation** - Remote control interfaces
- 🔧 **Troubleshooting** - Diagnostic and maintenance tools

---

<div align="center">
  <h3>🤝 Ready to build amazing robots?</h3>
  <p>
    <a href="mailto:contact@dexmate.ai">📧 Contact Us</a> •
    <a href="./examples/">📚 View Examples</a> •
  </p>
</div>
