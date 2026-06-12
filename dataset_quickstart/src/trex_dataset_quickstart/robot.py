"Contains information about the robot descriptions and factory methods Pinocchio models."

from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import coal as fcl

from dexmate_urdf import robots

# Dexmate Vega-1 Robot
DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES = {
    "left_arm": [
        "L_arm_j1",
        "L_arm_j2",
        "L_arm_j3",
        "L_arm_j4",
        "L_arm_j5",
        "L_arm_j6",
        "L_arm_j7",
    ],
    "right_arm": [
        "R_arm_j1",
        "R_arm_j2",
        "R_arm_j3",
        "R_arm_j4",
        "R_arm_j5",
        "R_arm_j6",
        "R_arm_j7",
    ],
    "head": ["head_j1", "head_j2", "head_j3"],
    "torso": ["torso_j1", "torso_j2", "torso_j3"],
}
# Additional wheel joint names appearing in Pinocchio
DEXMATE_WHEEL_JOINT_NAMES = [
    "B_wheel_j1",
    "B_wheel_j2",
    "L_wheel_j1",
    "L_wheel_j2",
    "R_wheel_j1",
    "R_wheel_j2",
]
# Default joint positions for the fixed torso and head joints
DEFAULT_TORSO = np.array([0.9, 1.57, 0.1])
DEFAULT_HEAD = np.array([0.28, 0.0, 0.0])

# Sharpa Wave Hand
# MJCF paths
_REPO_ROOT = Path(__file__).resolve().parents[2]  # .../trex_dataset_quickstart
SHARPA_LEFT_HAND_MJCF_PATH = (
    _REPO_ROOT
    / "third_party"
    / "sharpa-urdf-usd-xml"
    / "wave_01"
    / "left_sharpa_wave"
    / "left_sharpa_wave_with_wrist.xml"
).resolve()
SHARPA_RIGHT_HAND_MJCF_PATH = (
    _REPO_ROOT
    / "third_party"
    / "sharpa-urdf-usd-xml"
    / "wave_01"
    / "right_sharpa_wave"
    / "right_sharpa_wave_with_wrist.xml"
).resolve()
if not SHARPA_LEFT_HAND_MJCF_PATH.exists() or not SHARPA_RIGHT_HAND_MJCF_PATH.exists():
    raise FileNotFoundError(
        f"Sharpa hand MJCF not found at {SHARPA_LEFT_HAND_MJCF_PATH} or {SHARPA_RIGHT_HAND_MJCF_PATH}; expected it to be included in the repo at third_party/sharpa-urdf-usd-xml/, or you can change the path constants in this file to point to the correct location if you have it locally."
    )

# Default table height for visualization, not accurate for simulation
TABLE_HEIGHT = 0.64

# joint order, should be consistent with the xml robot descriptions and the order of the Pinocchio model joints
SHARPA_LEFT_HAND_JOINT_ORDER = [
    "left_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
]
SHARPA_RIGHT_HAND_JOINT_ORDER = [
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
]


# Factory methods
def build_reduced_bimanual_robot(
    default_joint_by_component: dict[str, np.ndarray],
) -> tuple[
    RobotWrapper,
    Callable[[dict[str, np.ndarray]], np.ndarray],
    Callable[[np.ndarray], dict[str, np.ndarray]],
]:
    """Build a reduced fixed base Dexmate Vega-1 robot Pinocchio RobotWrapper model (including arms only, freezing head and torso).

    Note that the returned RobotWrapper has joint order [L_arm_j1, ..., L_arm_j7, R_arm_j1, ..., R_arm_j7], and the assemble_qpos and disassemble_qpos functions are consistent with this order.

    Args:
        default_joint_by_component: A dict mapping component name to default joint positions.
            Note that only the head and torso matters since they will be locked.

    Returns:
        reduced_bimanual_robot: A Pinocchio RobotWrapper containing the reduced bimanual robot model.
        assemble_qpos: A function that assembles a full qpos from a dict of component joint positions.
        disassemble_qpos: A function that disassembles a full qpos into a dict of component joint positions.
    """
    assert default_joint_by_component.keys() == {"head", "torso"}
    assert default_joint_by_component["head"].shape == (
        len(DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["head"]),
    )
    assert default_joint_by_component["torso"].shape == (
        len(DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["torso"]),
    )

    # Build the Dexmate bimanual robot and lock torso & head joints, leaving arm joints active.
    bimanual_robot = RobotWrapper.BuildFromURDF(
        robots.humanoid.vega_1.vega_1.urdf, [robots.humanoid.vega_1.vega_1._parent_dir]
    )
    assert list(bimanual_robot.model.names) == (
        ["universe"]
        + DEXMATE_WHEEL_JOINT_NAMES
        + DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["torso"]
        + DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]
        + DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]
        + DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["head"]
    )
    assert all(
        bimanual_robot.model.joints[bimanual_robot.model.getJointId(name)].nq == 1
        for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["head"]
        + DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["torso"]
    )
    bimanual_head_indices = np.array(
        [
            bimanual_robot.model.joints[bimanual_robot.model.getJointId(name)].idx_q
            for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["head"]
        ]
    )
    bimanual_torso_indices = np.array(
        [
            bimanual_robot.model.joints[bimanual_robot.model.getJointId(name)].idx_q
            for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["torso"]
        ]
    )
    bimanual_default_qpos = pin.neutral(bimanual_robot.model)
    # Disregard the wheel joints
    bimanual_default_qpos[bimanual_head_indices] = default_joint_by_component["head"]
    bimanual_default_qpos[bimanual_torso_indices] = default_joint_by_component["torso"]

    unlocked = set(
        DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]
        + DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]
    )
    joints_to_lock = [name for name in bimanual_robot.model.names[1:] if name not in unlocked]
    joints_to_lock_ids = [bimanual_robot.model.getJointId(name) for name in joints_to_lock]

    reduced_bimanual_model, (reduced_bimanual_collision_model, reduced_bimanual_visual_model) = (
        pin.buildReducedModel(
            bimanual_robot.model,
            [bimanual_robot.collision_model, bimanual_robot.visual_model],
            joints_to_lock_ids,
            bimanual_default_qpos,
        )
    )

    # Add valid collision pairs
    reduced_bimanual_collision_model.addAllCollisionPairs()
    pin.removeCollisionPairs(
        reduced_bimanual_model, reduced_bimanual_collision_model, robots.humanoid.vega_1.vega_1.srdf
    )  # NOTE: reusing srdf
    reduced_bimanual_robot = pin.RobotWrapper(
        reduced_bimanual_model, reduced_bimanual_collision_model, reduced_bimanual_visual_model
    )
    reduced_bimanual_robot.collision_data.enable_contact = True  # TODO: Check this

    # Sanity check
    assert list(reduced_bimanual_robot.model.names) == (
        ["universe"]
        + DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]
        + DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]
    )
    assert reduced_bimanual_robot.nq == reduced_bimanual_robot.nv == 14
    assert all(joint.nq == 1 for joint in reduced_bimanual_robot.model.joints)

    # Defining helper functions
    _nq = reduced_bimanual_robot.nq
    left_arm_indices = np.array(
        [
            reduced_bimanual_robot.model.joints[reduced_bimanual_robot.model.getJointId(name)].idx_q
            for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]
        ]
    )
    right_arm_indices = np.array(
        [
            reduced_bimanual_robot.model.joints[reduced_bimanual_robot.model.getJointId(name)].idx_q
            for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]
        ]
    )

    def assemble_qpos(joint_pos_by_component: dict[str, np.ndarray]) -> np.ndarray:
        assert set(joint_pos_by_component.keys()) == {"left_arm", "right_arm"}
        assert joint_pos_by_component["left_arm"].shape == (
            len(DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]),
        )
        assert joint_pos_by_component["right_arm"].shape == (
            len(DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]),
        )
        qpos = np.zeros(_nq)
        qpos[left_arm_indices] = joint_pos_by_component["left_arm"]
        qpos[right_arm_indices] = joint_pos_by_component["right_arm"]
        return qpos

    def disassemble_qpos(qpos: np.ndarray) -> dict[str, np.ndarray]:
        assert qpos.shape == (_nq,)
        return {
            "left_arm": qpos[left_arm_indices].copy(),
            "right_arm": qpos[right_arm_indices].copy(),
        }

    return reduced_bimanual_robot, assemble_qpos, disassemble_qpos


def _prefix_geometry_object_names(geometry_model: pin.GeometryModel, prefix: str) -> None:
    """Prefix every geometry object's name in-place (e.g. 'wrist_visual' -> 'left_wrist_visual').

    pin.appendModel requires geometry object names to be unique across the merged model. The
    left/right Sharpa MJCFs share unprefixed geometry names ('wrist', 'wrist_visual'), so without
    this the first-appended hand's wrist geometry is silently dropped. Joints already carry
    left_/right_ prefixes, so only the geometry models need this. Already-prefixed names are left
    untouched (no double-prefixing). Collision pairs reference geometry by index, not name, so
    renaming is safe.
    """
    for i in range(len(geometry_model.geometryObjects)):
        obj = geometry_model.geometryObjects[i]
        if not obj.name.startswith(prefix):
            obj.name = f"{prefix}{obj.name}"


def build_full_robot(
    default_joint_by_component: dict[str, np.ndarray],
) -> tuple[
    RobotWrapper,
    Callable[[dict[str, np.ndarray]], np.ndarray],
    Callable[[np.ndarray], dict[str, np.ndarray]],
]:
    """Build a full Pinocchio RobotWrapper model for Dexmate + Sharpa morphology.

    The full model consists of the reduced Dexmate bimanual robot (arms unlockedonly, head and torso frozen) and the two attached Sharpa hands. Note that the hand placement is not calibrated and is only used for visualization.

    Also note that the returned RobotWrapper has joint order [L_arm_j1, ..., L_arm_j7, R_arm_j1, ..., R_arm_j7, left_thumb_CMC_FE, ..., left_pinky_DIP, right_thumb_CMC_FE, ..., right_pinky_DIP], and the assemble_qpos and disassemble_qpos functions are consistent with this order.

    Args:
        default_joint_by_component: A dict mapping component name to default joint positions.
            Note that only the head and torso matters since they will be locked.

    Returns:
        combined_robot: A Pinocchio RobotWrapper containing the full combined robot model.
        assemble_qpos: A function that assembles a full qpos from a dict of component joint positions.
        disassemble_qpos: A function that disassembles a full qpos into a dict of component joint positions.
    """

    assert default_joint_by_component.keys() == {"head", "torso"}
    assert default_joint_by_component["head"].shape == (
        len(DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["head"]),
    )
    assert default_joint_by_component["torso"].shape == (
        len(DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["torso"]),
    )

    # Build the Dexmate bimanual robot and lock torso & head joints, leaving arm joints active.
    reduced_bimanual_robot, _, _ = build_reduced_bimanual_robot(default_joint_by_component)

    # Combine the left Shapra hand and the Dexmate bimanual robot.
    left_hand_robot = RobotWrapper.BuildFromMJCF(SHARPA_LEFT_HAND_MJCF_PATH)
    # Make hand geometry names unique before appending (see _prefix_geometry_object_names).
    _prefix_geometry_object_names(left_hand_robot.visual_model, "left_")
    _prefix_geometry_object_names(left_hand_robot.collision_model, "left_")
    left_ee_frame_id = reduced_bimanual_robot.model.getFrameId("L_ee")
    # NOTE: the hand placement is not calibrated
    # the current values are chosen based on rough measurements
    left_hand_pose_in_ee_frame = pin.SE3(
        rotation=np.eye(3),
        translation=np.array([0.0, 0.0, 0.0]),
    )
    _, partially_combined_visual_model = pin.appendModel(
        reduced_bimanual_robot.model,
        left_hand_robot.model,
        reduced_bimanual_robot.visual_model,
        left_hand_robot.visual_model,
        left_ee_frame_id,
        left_hand_pose_in_ee_frame,
    )
    partially_combined_model, partially_combined_collision_model = pin.appendModel(
        reduced_bimanual_robot.model,
        left_hand_robot.model,
        reduced_bimanual_robot.collision_model,
        left_hand_robot.collision_model,
        left_ee_frame_id,
        left_hand_pose_in_ee_frame,
    )

    # Combine the right Sharpa hand and the partially combined model.
    right_hand_robot = RobotWrapper.BuildFromMJCF(SHARPA_RIGHT_HAND_MJCF_PATH)
    # Make hand geometry names unique before appending (see _prefix_geometry_object_names).
    _prefix_geometry_object_names(right_hand_robot.visual_model, "right_")
    _prefix_geometry_object_names(right_hand_robot.collision_model, "right_")
    right_ee_frame_id = partially_combined_model.getFrameId("R_ee")
    # NOTE: the hand placement is not calibrated
    # the current values are chosen based on rough measurements
    right_hand_pose_in_ee_frame = pin.SE3(
        rotation=np.eye(3),
        translation=np.array([0.0, 0.0, 0.0]),
    )
    _, combined_visual_model = pin.appendModel(
        partially_combined_model,
        right_hand_robot.model,
        partially_combined_visual_model,
        right_hand_robot.visual_model,
        right_ee_frame_id,
        right_hand_pose_in_ee_frame,
    )
    combined_model, combined_collision_model = pin.appendModel(
        partially_combined_model,
        right_hand_robot.model,
        partially_combined_collision_model,
        right_hand_robot.collision_model,
        right_ee_frame_id,
        right_hand_pose_in_ee_frame,
    )

    # NOTE: pin.appendModel seesm to deal with collision pairs through the following procedure:
    # 1. copies existing collision pairs
    # 2. add cross A-B pairs for geometry objects with different parentJoint
    # Since the attached bodies (e.g. L_arm_l7_0 & L_arm_l8_0 and left_hand_C_MC_0, same for right)
    # have the same parent joint due to appending, they are not included as collision pairs.
    # This means the current combined_model's collision pairs contain:
    # 1. valid self collision pairs within the bimanual robot
    # 2. left hand - bimanual
    # 3. right hand - bimanual
    # 4. left hand - right hand
    # We want to disable between-hand collision (and also within-hand collision, which is already
    # excluded) due to the need for dexterous in-hand manipulation.
    left_hand_collision_geometry_names = set(
        obj.name for obj in left_hand_robot.collision_model.geometryObjects
    )
    right_hand_collision_geometry_names = set(
        obj.name for obj in right_hand_robot.collision_model.geometryObjects
    )

    def is_between_hand_pair(collision_pair):
        name1 = combined_collision_model.geometryObjects[collision_pair.first].name
        name2 = combined_collision_model.geometryObjects[collision_pair.second].name
        if (
            name1 in left_hand_collision_geometry_names
            and name2 in right_hand_collision_geometry_names
        ):
            return True
        elif (
            name1 in right_hand_collision_geometry_names
            and name2 in left_hand_collision_geometry_names
        ):
            return True
        else:
            return False

    original_collision_pairs = deepcopy(combined_collision_model.collisionPairs)
    for collision_pair in original_collision_pairs:
        if is_between_hand_pair(collision_pair):
            combined_collision_model.removeCollisionPair(collision_pair)

    # Bundle the combined model into a RobotWrapper
    combined_robot = pin.RobotWrapper(
        combined_model, combined_collision_model, combined_visual_model
    )

    # Sanity check
    assert list(combined_robot.model.names) == (
        ["universe"]
        + DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]
        + SHARPA_LEFT_HAND_JOINT_ORDER
        + DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]
        + SHARPA_RIGHT_HAND_JOINT_ORDER
    ), breakpoint()
    assert combined_robot.nq == combined_robot.nv == 2 * 7 + 2 * 22
    assert all(joint.nq == 1 for joint in combined_robot.model.joints)
    # Geometry names must be unique, else appendModel silently drops duplicates (e.g. wrist).
    for geometry_model in (combined_visual_model, combined_collision_model):
        names = [obj.name for obj in geometry_model.geometryObjects]
        assert len(names) == len(set(names)), (
            f"duplicate geometry object names: {sorted(n for n in names if names.count(n) > 1)}"
        )

    # Defining helper functions
    _nq = combined_robot.nq
    left_arm_indices = np.array(
        [
            combined_robot.model.joints[combined_robot.model.getJointId(name)].idx_q
            for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]
        ]
    )
    right_arm_indices = np.array(
        [
            combined_robot.model.joints[combined_robot.model.getJointId(name)].idx_q
            for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]
        ]
    )
    left_hand_indices = np.array(
        [
            combined_robot.model.joints[combined_robot.model.getJointId(name)].idx_q
            for name in SHARPA_LEFT_HAND_JOINT_ORDER
        ]
    )
    right_hand_indices = np.array(
        [
            combined_robot.model.joints[combined_robot.model.getJointId(name)].idx_q
            for name in SHARPA_RIGHT_HAND_JOINT_ORDER
        ]
    )

    def assemble_qpos(joint_pos_by_component: dict[str, np.ndarray]) -> np.ndarray:
        assert set(joint_pos_by_component.keys()) == {
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        }
        assert joint_pos_by_component["left_arm"].shape == (
            len(DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]),
        )
        assert joint_pos_by_component["right_arm"].shape == (
            len(DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]),
        )
        assert joint_pos_by_component["left_hand"].shape == (len(SHARPA_LEFT_HAND_JOINT_ORDER),)
        assert joint_pos_by_component["right_hand"].shape == (len(SHARPA_RIGHT_HAND_JOINT_ORDER),)
        qpos = np.zeros(_nq)
        qpos[left_arm_indices] = joint_pos_by_component["left_arm"]
        qpos[right_arm_indices] = joint_pos_by_component["right_arm"]
        qpos[left_hand_indices] = joint_pos_by_component["left_hand"]
        qpos[right_hand_indices] = joint_pos_by_component["right_hand"]
        return qpos

    def disassemble_qpos(qpos: np.ndarray) -> dict[str, np.ndarray]:
        assert qpos.shape == (_nq,)
        return {
            "left_arm": qpos[left_arm_indices].copy(),
            "right_arm": qpos[right_arm_indices].copy(),
            "left_hand": qpos[left_hand_indices].copy(),
            "right_hand": qpos[right_hand_indices].copy(),
        }

    return combined_robot, assemble_qpos, disassemble_qpos


def add_table_model(
    robot: RobotWrapper,
    default_joint_by_component: dict[str, np.ndarray],
    assemble_qpos: Callable[[dict[str, np.ndarray]], np.ndarray],
    table_height: float,
) -> RobotWrapper:
    """Add a fixed table to the Pinocchio RobotWrapper model.

    The table is added as a fixed body to the visual and collision models.

    Args:
        robot: The input robot wrapper.
        default_joint_by_component: A dict mapping component name to default joint positions.
            Used for sanity check to ensure the robot is not already in collision at the
            default configuration.
        assemble_qpos: A function that assembles a full qpos from a dict of component joint positions.
        table_height: The height of the table (z direction).

    Returns:
        A new RobotWrapper with the combined model.

    Raises:
        ValueError: If the robot is in collision with the table at the default configuration.
    """
    model, collision_model, visual_model = robot.model, robot.collision_model, robot.visual_model
    # NOTE: These object dimensions are hard coded.
    table_obj_shape = fcl.Box(2.0, 4.0, 0.08)
    added_collision_obj_ids = []

    # Add obstacles to the visual and collision models
    assert table_height > 0
    table_pose = pin.SE3(rotation=np.eye(3), translation=np.array([1.1, 0.0, table_height - 0.01]))
    table_obj = pin.GeometryObject("table", 0, table_pose, table_obj_shape)
    table_obj.meshColor = np.array([0.2, 0.2, 1.0, 0.5])
    visual_model.addGeometryObject(table_obj)
    added_collision_obj_ids.append(collision_model.addGeometryObject(table_obj))

    # Add collision between the robot and the added obstacles
    for obj_id in range(collision_model.ngeoms):
        if obj_id in added_collision_obj_ids:
            continue
        for added_obj_id in added_collision_obj_ids:
            collision_model.addCollisionPair(pin.CollisionPair(obj_id, added_obj_id))

    # Create a new RobotWrapper with new data
    robot = RobotWrapper(model, collision_model, visual_model)

    # Sanity check: Ensure the robot is not in collision with the obstacles at the default configuration.
    qpos = assemble_qpos(default_joint_by_component)
    pin.computeCollisions(
        robot.model, robot.data, robot.collision_model, robot.collision_data, qpos, False
    )
    if any(cr.isCollision() for cr in robot.collision_data.collisionResults):
        raise ValueError(
            "The robot is in collision with the added obstacles at the default configuration."
        )

    return robot


def forward_kinematics(
    robot_wrapper: RobotWrapper,
    qpos: np.ndarray,
    frames: list[str],
) -> dict[str, pin.SE3]:
    # NOTE: qpos should be in pinocchio robot_wrapper order
    # e.g. assembled using the assemble_qpos function
    # NOTE: the default name for the left/right wrist ee frame
    # in the combined model is "L_ee" and "R_ee".
    assert qpos.shape == (robot_wrapper.nq,)
    pin.forwardKinematics(robot_wrapper.model, robot_wrapper.data, qpos)
    pin.updateFramePlacements(robot_wrapper.model, robot_wrapper.data)
    result = {}
    for frame in frames:
        fid = robot_wrapper.model.getFrameId(frame)
        result[frame] = robot_wrapper.data.oMf[fid].copy()
    return result


if __name__ == "__main__":
    # Visualization test for the combined robot model, assemble_qpos, and forward kinematics.
    import time
    import viser
    from pinocchio.visualize import ViserVisualizer
    from scipy.spatial.transform import Rotation as sRot

    combined_robot, assemble_qpos, disassemble_qpos = build_full_robot(
        default_joint_by_component={
            "torso": np.array([0.9, 1.57, 0.1]),
            "head": np.array([0.28, 0.0, 0.0]),
        }
    )
    combined_robot = add_table_model(
        robot=combined_robot,
        default_joint_by_component={
            "left_arm": np.array([0.84, 0.51, 0.37, -1.30, -0.65, -0.29, -0.03]),
            "right_arm": np.array([-0.84, -0.51, -0.37, -1.30, 0.65, 0.29, 0.03]),
            "left_hand": np.zeros(
                22,
            ),
            "right_hand": np.zeros(
                22,
            ),
        },
        assemble_qpos=assemble_qpos,
        table_height=0.64,  # Measured
    )

    server = viser.ViserServer()  # defaults to localhost:8080
    # Add a floor
    server.scene.add_box(
        "floor",
        dimensions=(20.0, 20.0, 0.01),
        position=(0, -0.005, 0),
        color=(190, 150, 255),
        opacity=1,
    )

    # Visualize the reduced robot
    viz = ViserVisualizer(
        combined_robot.model,
        combined_robot.collision_model,
        combined_robot.visual_model,
        copy_models=True,
    )
    viz.initViewer(viewer=server, open=False, loadModel=False)
    viz.loadViewerModel(rootNodeName="robot")
    viz.displayCollisions(False)
    viz.displayVisuals(True)

    # Add handles to visualize fk results for wrist ee frames
    left_ee_handle = server.scene.add_frame(
        "left_ee_pose",
        axes_length=0.1,
        axes_radius=0.005,
    )
    right_ee_handle = server.scene.add_frame(
        "right_ee_pose",
        axes_length=0.1,
        axes_radius=0.005,
    )

    q = assemble_qpos(
        {
            "left_arm": np.array([0.84, 0.51, 0.37, -1.30, -0.65, -0.29, -0.03]),
            "right_arm": np.array([-0.84, -0.51, -0.37, -1.30, 0.65, 0.29, 0.03]),
            "left_hand": np.zeros((22,)),
            "right_hand": np.zeros((22,)),
        }
    )
    viz.display(q)
    ee_poses = forward_kinematics(combined_robot, q, frames=["L_ee", "R_ee"])
    left_ee_handle.position = ee_poses["L_ee"].translation
    left_ee_handle.wxyz = sRot.from_matrix(ee_poses["L_ee"].rotation).as_quat(scalar_first=True)
    right_ee_handle.position = ee_poses["R_ee"].translation
    right_ee_handle.wxyz = sRot.from_matrix(ee_poses["R_ee"].rotation).as_quat(scalar_first=True)

    input("Press Enter to start random visualization...")
    try:
        while True:
            time.sleep(1)
            random_q = np.random.uniform(
                low=combined_robot.model.lowerPositionLimit,
                high=combined_robot.model.upperPositionLimit,
            )
            viz.display(random_q)
            ee_poses = forward_kinematics(combined_robot, random_q, frames=["L_ee", "R_ee"])
            left_ee_handle.position = ee_poses["L_ee"].translation
            left_ee_handle.wxyz = sRot.from_matrix(ee_poses["L_ee"].rotation).as_quat(
                scalar_first=True
            )
            right_ee_handle.position = ee_poses["R_ee"].translation
            right_ee_handle.wxyz = sRot.from_matrix(ee_poses["R_ee"].rotation).as_quat(
                scalar_first=True
            )
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
