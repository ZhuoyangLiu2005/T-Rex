"""Dataset constants + canonical joint names for the TRex tactile-play LeRobotDataset v3.0.

`observation.state` / `action` are 58-D, laid out as
`[left_arm(7) | left_hand(22) | right_arm(7) | right_hand(22)]`. The per-dim joint
names here match the Pinocchio model built in `robot.py` (Dexmate arms + Sharpa
ha4 hands), so each state/action dim maps to a URDF joint by name for replay.
"""

from __future__ import annotations

FPS = 30

ARM_DOF = 7
HAND_DOF = 22
N_FINGERS = 5
TACTILE_F6_DIM = 6

STATE_DIM = 2 * (ARM_DOF + HAND_DOF)  # 58
ACTION_DIM = STATE_DIM
TACTILE_FORCE_DIM = 2 * N_FINGERS * TACTILE_F6_DIM  # 60

# Tactile finger order (image sensor axis + f6 force layout).
FINGER_NAMES: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")

# 58-D state/action component slices.
LEFT_ARM = slice(0, ARM_DOF)  # 0:7
LEFT_HAND = slice(ARM_DOF, ARM_DOF + HAND_DOF)  # 7:29
RIGHT_ARM = slice(ARM_DOF + HAND_DOF, 2 * ARM_DOF + HAND_DOF)  # 29:36
RIGHT_HAND = slice(2 * ARM_DOF + HAND_DOF, STATE_DIM)  # 36:58

# ---- Canonical joint names (MUST match robot.py's Pinocchio model.names) ----
# Arms: Dexmate, uppercase `L_`/`R_`.  Hands: Sharpa, lowercase `left_`/`right_` + order.
LEFT_ARM_JOINTS: tuple[str, ...] = tuple(f"L_arm_j{i}" for i in range(1, ARM_DOF + 1))
RIGHT_ARM_JOINTS: tuple[str, ...] = tuple(f"R_arm_j{i}" for i in range(1, ARM_DOF + 1))
SHARPA_HAND_JOINT_ORDER: tuple[str, ...] = (
    "thumb_CMC_FE",
    "thumb_CMC_AA",
    "thumb_MCP_FE",
    "thumb_MCP_AA",
    "thumb_IP",
    "index_MCP_FE",
    "index_MCP_AA",
    "index_PIP",
    "index_DIP",
    "middle_MCP_FE",
    "middle_MCP_AA",
    "middle_PIP",
    "middle_DIP",
    "ring_MCP_FE",
    "ring_MCP_AA",
    "ring_PIP",
    "ring_DIP",
    "pinky_CMC",
    "pinky_MCP_FE",
    "pinky_MCP_AA",
    "pinky_PIP",
    "pinky_DIP",
)
assert len(SHARPA_HAND_JOINT_ORDER) == HAND_DOF


def joint_names() -> list[str]:
    """The 58 joint names in state/action order [L_arm, L_hand, R_arm, R_hand]."""
    return [
        *LEFT_ARM_JOINTS,
        *(f"left_{n}" for n in SHARPA_HAND_JOINT_ORDER),
        *RIGHT_ARM_JOINTS,
        *(f"right_{n}" for n in SHARPA_HAND_JOINT_ORDER),
    ]


def split_state(vec):
    """Split a 58-D state/action vector into {left_arm, left_hand, right_arm, right_hand}."""
    if vec.shape[-1] != STATE_DIM:
        raise ValueError(f"expected last dim {STATE_DIM}, got {vec.shape}")
    return {
        "left_arm": vec[..., LEFT_ARM],
        "left_hand": vec[..., LEFT_HAND],
        "right_arm": vec[..., RIGHT_ARM],
        "right_hand": vec[..., RIGHT_HAND],
    }


# ---- Video feature keys ----
RGB_KEYS: list[str] = [
    "observation.images.head_left",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
]


def tactile_video_key(side: str, kind: str, finger: str) -> str:
    return f"observation.images.tactile_{side}_{kind}_{finger}"


def all_tactile_video_keys() -> list[str]:
    return [
        tactile_video_key(side, kind, finger)
        for side in ("left", "right")
        for kind in ("raw", "deform")
        for finger in FINGER_NAMES
    ]


def all_video_keys() -> list[str]:
    return [*RGB_KEYS, *all_tactile_video_keys()]
