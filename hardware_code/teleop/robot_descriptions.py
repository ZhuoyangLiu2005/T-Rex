"""Contains information about the robot descriptions and factory methods Pinocchio models."""

from copy import deepcopy
from pathlib import Path
from typing import Callable

import coal as fcl
import numpy as np
import pinocchio as pin
from dexmate_urdf import robots
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation as sRot

# Constants
# Dexmate bimanual robot
# Default velocity limits (rad/s) - retrieved from dexmate docs
DEXMATE_DEFAULT_ARM_VEL_LIMITS = np.array([2.4, 2.4, 2.7, 2.7, 2.7, 2.7, 2.7])
# Names per component
# Note: DEXMATE_WHEEL_JOINT_NAMES are not included, due to dexcontrol api compatibility
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
# Arm joint order (prefixed by "L_" and "R_")
DEXMATE_ARM_JOINT_ORDER = ["arm_j1", "arm_j2", "arm_j3", "arm_j4", "arm_j5", "arm_j6", "arm_j7"]
# Arm qpos limits
DEXMATE_LEFT_ARM_LOWER_QPOS_LIMITS = np.array(
    [-3.071, -0.453, -3.071, -3.071, -3.071, -1.396, -1.378]
)
DEXMATE_RIGHT_ARM_LOWER_QPOS_LIMITS = np.array(
    [-3.071, -1.553, -3.071, -3.071, -3.071, -1.396, -1.117]
)
DEXMATE_LEFT_ARM_UPPER_QPOS_LIMITS = np.array([3.071, 1.553, 3.071, 0.244, 3.071, 1.396, 1.117])
DEXMATE_RIGHT_ARM_UPPER_QPOS_LIMITS = np.array([3.071, 0.453, 3.071, 0.244, 3.071, 1.396, 1.378])

# Sharpa dexterous hand
# MJCF paths
project_root = Path(__file__).resolve().parents[1]  # .../SharpaDexmateTeleop
SHARPA_LEFT_HAND_MJCF_PATH = (
    project_root
    / "third_party/sharpa-urdf-usd-xml/wave_01/left_sharpa_wave/left_sharpa_wave_with_wrist.xml"
).resolve()
SHARPA_RIGHT_HAND_MJCF_PATH = (
    project_root
    / "third_party/sharpa-urdf-usd-xml/wave_01/right_sharpa_wave/right_sharpa_wave_with_wrist.xml"
).resolve()
if not SHARPA_LEFT_HAND_MJCF_PATH.exists() or not SHARPA_RIGHT_HAND_MJCF_PATH.exists():
    raise FileNotFoundError(
        f"Sharpa hand MJCF not found at {SHARPA_LEFT_HAND_MJCF_PATH} or {SHARPA_RIGHT_HAND_MJCF_PATH}; expected it to be included in the repo at third_party/sharpa-urdf-usd-xml/, or you can change the path constants in this file to point to the correct location if you have it locally."
    )
# Hand joint order (prefixed by "left_" and "right_")
SHARPA_HAND_JOINT_ORDER = [
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
]


# Factory methods
def build_reduced_bimanual_robot(
    default_joint_by_component: dict[str, np.ndarray],
) -> tuple[
    RobotWrapper,
    Callable[[dict[str, np.ndarray]], np.ndarray],
    Callable[[np.ndarray], dict[str, np.ndarray]],
]:
    """Build a reduced robot model (including arms only, freezing head and torso) for IK.

    Note the function is hard coded for our robot (Vega) for now.

    Args:
        default_joint_by_component: A dict mapping component name to default joint positions.
            Note that only the head and torso matters since they will be locked.

    Returns:
        A tuple of (reduced_robot, assemble_qpos, disassemble_qpos).
    """
    assert default_joint_by_component.keys() >= {"head", "torso"}
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
    assert (
        set(bimanual_robot.model.names[1:])
        == set(
            DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]
            + DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]
            + DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["head"]
            + DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["torso"]
            + [
                "B_wheel_j1",
                "B_wheel_j2",
                "L_wheel_j1",
                "L_wheel_j2",
                "R_wheel_j1",
                "R_wheel_j2",
            ]
        )
        and bimanual_robot.model.names[0] == "universe"
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

    reduced_bimanual_model, (reduced_bimanual_collision_model, rreduced_bimanual_visual_model) = (
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
        reduced_bimanual_model, reduced_bimanual_collision_model, rreduced_bimanual_visual_model
    )
    reduced_bimanual_robot.collision_data.enable_contact = True

    # Sanity check
    assert (
        set(reduced_bimanual_robot.model.names[1:])
        == set(
            ["L_" + name for name in DEXMATE_ARM_JOINT_ORDER]
            + ["R_" + name for name in DEXMATE_ARM_JOINT_ORDER]
        )
        and reduced_bimanual_robot.model.names[0] == "universe"
    )
    assert (
        reduced_bimanual_robot.nq == reduced_bimanual_robot.nv == 2 * len(DEXMATE_ARM_JOINT_ORDER)
    )
    assert all(joint.nq == 1 for joint in reduced_bimanual_robot.model.joints)

    # Defining helper functions
    _nq = reduced_bimanual_robot.nq
    left_arm_indices = np.array(
        [
            reduced_bimanual_robot.model.joints[
                reduced_bimanual_robot.model.getJointId("L_" + name)
            ].idx_q
            for name in DEXMATE_ARM_JOINT_ORDER
        ]
    )
    right_arm_indices = np.array(
        [
            reduced_bimanual_robot.model.joints[
                reduced_bimanual_robot.model.getJointId("R_" + name)
            ].idx_q
            for name in DEXMATE_ARM_JOINT_ORDER
        ]
    )

    def assemble_qpos(joint_pos_by_component: dict[str, np.ndarray]) -> np.ndarray:
        assert set(joint_pos_by_component.keys()) == {"left_arm", "right_arm"}
        assert joint_pos_by_component["left_arm"].shape == (len(DEXMATE_ARM_JOINT_ORDER),)
        assert joint_pos_by_component["right_arm"].shape == (len(DEXMATE_ARM_JOINT_ORDER),)
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
    """Build a robot model (including arms and hands, freezing head and torso) for IK.

    Note the function is hard coded for our robot (Vega) and Sharpa hands for now.

    Args:
        default_joint_by_component: A dict mapping component name to default joint positions.
            Note that only the head and torso matters since they will be locked.

    Returns:
        A tuple of (combined_robot, assemble_qpos, disassemble_qpos).
    """
    assert default_joint_by_component.keys() >= {"head", "torso"}
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
        translation=np.array([0.0, 0.0, 0.0]),  # with_wrist model already includes the wrist mount
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
        translation=np.array([0.0, 0.0, 0.0]),  # with_wrist model already includes the wrist mount
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
    assert (
        set(combined_robot.model.names[1:])
        == set(
            ["L_" + name for name in DEXMATE_ARM_JOINT_ORDER]
            + ["R_" + name for name in DEXMATE_ARM_JOINT_ORDER]
            + ["left_" + name for name in SHARPA_HAND_JOINT_ORDER]
            + ["right_" + name for name in SHARPA_HAND_JOINT_ORDER]
        )
        and combined_robot.model.names[0] == "universe"
    )
    assert (
        combined_robot.nq
        == combined_robot.nv
        == 2 * len(DEXMATE_ARM_JOINT_ORDER) + 2 * len(SHARPA_HAND_JOINT_ORDER)
    )
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
            combined_robot.model.joints[combined_robot.model.getJointId("L_" + name)].idx_q
            for name in DEXMATE_ARM_JOINT_ORDER
        ]
    )
    right_arm_indices = np.array(
        [
            combined_robot.model.joints[combined_robot.model.getJointId("R_" + name)].idx_q
            for name in DEXMATE_ARM_JOINT_ORDER
        ]
    )
    left_hand_indices = np.array(
        [
            combined_robot.model.joints[combined_robot.model.getJointId("left_" + name)].idx_q
            for name in SHARPA_HAND_JOINT_ORDER
        ]
    )
    right_hand_indices = np.array(
        [
            combined_robot.model.joints[combined_robot.model.getJointId("right_" + name)].idx_q
            for name in SHARPA_HAND_JOINT_ORDER
        ]
    )

    def assemble_qpos(joint_pos_by_component: dict[str, np.ndarray]) -> np.ndarray:
        assert set(joint_pos_by_component.keys()) == {
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        }
        assert joint_pos_by_component["left_arm"].shape == (len(DEXMATE_ARM_JOINT_ORDER),)
        assert joint_pos_by_component["right_arm"].shape == (len(DEXMATE_ARM_JOINT_ORDER),)
        assert joint_pos_by_component["left_hand"].shape == (len(SHARPA_HAND_JOINT_ORDER),)
        assert joint_pos_by_component["right_hand"].shape == (len(SHARPA_HAND_JOINT_ORDER),)
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


def add_env_obstacles(
    robot: RobotWrapper,
    default_joint_by_component: dict[str, np.ndarray],
    assemble_qpos: Callable[[dict[str, np.ndarray]], np.ndarray],
    back_wall_distance: float | None,
    left_wall_distance: float | None,
    right_wall_distance: float | None,
    table_height: float | None,
) -> RobotWrapper:
    """Add environment obstacles (back wall, right wall, table) to the robot's collision model.

    The obstacles are added as fixed bodies to the visual and collision models.

    Args:
        robot: The input robot wrapper.
        default_joint_by_component: A dict mapping component name to default joint positions.
            Used for sanity check to ensure the robot is not already in collision at the
            default configuration.
        assemble_qpos: A function that assembles a full qpos from a dict of component joint positions.
        back_wall_distance: The distance of the back wall from the robot (-x direction).
            If None, the back wall will not be added.
        left_wall_distance: The distance of the left wall from the robot (+y direction).
            If None, the left wall will not be added.
        right_wall_distance: The distance of the right wall from the robot (-y direction).
            If None, the right wall will not be added.
        table_height: The height of the table (z direction). If None, the table will not be added.

    Returns:
        A new RobotWrapper with the combined model.

    Raises:
        ValueError: If the robot is in collision with the obstacles at the default configuration.
    """
    model, collision_model, visual_model = robot.model, robot.collision_model, robot.visual_model
    # NOTE: These object dimensions are hard coded.
    wall_obj_shape = fcl.Box(0.02, 5.0, 3.0)
    table_obj_shape = fcl.Box(2.0, 4.0, 0.08)
    added_collision_obj_ids = []

    # Add obstacles to the visual and collision models
    if back_wall_distance is not None:
        assert back_wall_distance > 0
        back_wall_pose = pin.SE3(
            rotation=np.eye(3), translation=np.array([-back_wall_distance, 0.0, 1.5])
        )
        back_wall_obj = pin.GeometryObject("back_wall", 0, back_wall_pose, wall_obj_shape)
        back_wall_obj.meshColor = np.array([1.0, 0.2, 0.2, 0.5])
        visual_model.addGeometryObject(back_wall_obj)
        added_collision_obj_ids.append(collision_model.addGeometryObject(back_wall_obj))
    if left_wall_distance is not None:
        assert left_wall_distance > 0
        left_wall_pose = pin.SE3(
            rotation=sRot.from_rotvec([0, 0, -np.pi / 2]).as_matrix(),
            translation=np.array([0.0, left_wall_distance, 1.5]),
        )
        left_wall_obj = pin.GeometryObject("left_wall", 0, left_wall_pose, wall_obj_shape)
        left_wall_obj.meshColor = np.array([0.2, 1.0, 0.2, 0.5])
        visual_model.addGeometryObject(left_wall_obj)
        added_collision_obj_ids.append(collision_model.addGeometryObject(left_wall_obj))
    if right_wall_distance is not None:
        assert right_wall_distance > 0
        right_wall_pose = pin.SE3(
            rotation=sRot.from_rotvec([0, 0, np.pi / 2]).as_matrix(),
            translation=np.array([0.0, -right_wall_distance, 1.5]),
        )
        right_wall_obj = pin.GeometryObject("right_wall", 0, right_wall_pose, wall_obj_shape)
        right_wall_obj.meshColor = np.array([0.2, 1.0, 0.2, 0.5])
        visual_model.addGeometryObject(right_wall_obj)
        added_collision_obj_ids.append(collision_model.addGeometryObject(right_wall_obj))
    if table_height is not None:
        assert table_height > 0
        table_pose = pin.SE3(
            rotation=np.eye(3), translation=np.array([1.1, 0.0, table_height - 0.01])
        )
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


if __name__ == "__main__":
    # Visualize the current room setup
    import time

    import viser
    from pinocchio.visualize import ViserVisualizer

    combined_robot, assemble_qpos, disassemble_qpos = build_full_robot(
        default_joint_by_component={
            "torso": np.array([0.9, 1.57, 0.1]),
            "head": np.array([0.28, 0.0, 0.0]),
        }
    )
    combined_robot = add_env_obstacles(
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
        back_wall_distance=0.60,  # Measured
        left_wall_distance=None,  # Not present in the current environment
        right_wall_distance=0.75,  # Measured
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

    # q = pin.neutral(combined_robot.model)
    q = assemble_qpos(
        {
            "left_arm": np.array([0.84, 0.51, 0.37, -1.30, -0.65, -0.29, -0.03]),
            "right_arm": np.array([-0.84, -0.51, -0.37, -1.30, 0.65, 0.29, 0.03]),
            "left_hand": np.zeros((22,)),
            "right_hand": np.zeros((22,)),
        }
    )
    pin.forwardKinematics(combined_robot.model, combined_robot.data, q)
    pin.updateFramePlacements(combined_robot.model, combined_robot.data)
    viz.display(q)
    time.sleep(10)
    server.stop()
