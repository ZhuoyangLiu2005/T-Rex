"""Module for arm and hand control in teleoperation.

Contains ArmIKManager for handling IK solving and warmstarting,  SmoothingAndSafetyManager for
smoothing arm commands and checking for self-collisions, and the full_robot_action_loop which
should run in a separate thread to send commands to the full robot at high frequency.
"""

import threading
import time
from copy import deepcopy
from typing import Any, Callable

import hppfcl
import numpy as np
import pinocchio as pin
from dexcontrol.robot import Robot
from loguru import logger
from loop_rate_limiters import RateLimiter
from ompl import base as ob
from ompl import geometric as og
from pinocchio.robot_wrapper import RobotWrapper
from ruckig import InputParameter, OutputParameter, Result, Ruckig
from sharpa import SharpaWave

from teleop.ik_utils import PinkLocalIK
from teleop.robot_descriptions import (
    DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES,
    DEXMATE_DEFAULT_ARM_VEL_LIMITS,
    DEXMATE_LEFT_ARM_LOWER_QPOS_LIMITS,
    DEXMATE_LEFT_ARM_UPPER_QPOS_LIMITS,
    DEXMATE_RIGHT_ARM_LOWER_QPOS_LIMITS,
    DEXMATE_RIGHT_ARM_UPPER_QPOS_LIMITS,
)

DEXMATE_VEL_LIMIT_SCALE = 0.4


# Safety threshold for arm tracking (in radians)
# If actual joint position diverges from target by more than this, stop
TRACKING_SAFETY_THRESHOLD = 10.0  # rad, in joint space
ARM_DOF = 7
HAND_DOF = 22


class ArmIKManager:
    """Manages IK solving for arm control, possibly including warmstart state."""

    def __init__(
        self,
        pink_ik_solver: PinkLocalIK,
        warmstart_with_actual: bool,
    ):
        """Initialize ArmIKManager.

        Args:
            pink_ik_solver: IK solver instance for solving inverse kinematics.
            warmstart_with_actual: Whether to warmstart IK with current joint positions
                or previous IK solution.
        """
        self._pink_ik_solver = pink_ik_solver
        self._warmstart_with_actual = warmstart_with_actual
        # Initialize IK solution buffer
        self._arm_target_joint_pos: dict[str, np.ndarray] = {
            "left_arm": np.full(7, np.nan),
            "right_arm": np.full(7, np.nan),
        }
        # Reset event
        self._reset_event = threading.Event()
        # On initialization, reset
        self.reset()

    def get_arm_action(
        self,
        arm_absolute_target_poses: dict[str, np.ndarray],
        arm_current_joint_positions: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Process absolute target poses and return raw IK solution (no smoothing).

        Performs IK solving only. Smoothing is done separately via smooth_action().
        Maintains internal state for IK warmstarting.

        Args:
            arm_absolute_target_poses: Dict with "L_ee" and "R_ee" keys, each containing
                a 4x4 homogeneous transformation matrix (np.ndarray) representing
                the absolute target pose in robot base frame.
            arm_current_joint_positions: Dict with "left_arm" and "right_arm" keys, each containing
                current joint positions (np.ndarray of shape (7,)) for IK warmstarting.

        Returns:
            Dict with "left_arm" and "right_arm" keys, each containing raw IK
            solution joint DOFs (np.ndarray of shape (7,)).
        """
        assert set(arm_absolute_target_poses.keys()) == {"L_ee", "R_ee"}
        assert arm_absolute_target_poses["L_ee"].shape == (4, 4)
        assert arm_absolute_target_poses["R_ee"].shape == (4, 4)
        assert set(arm_current_joint_positions.keys()) == {"left_arm", "right_arm"}
        assert arm_current_joint_positions["left_arm"].shape == (7,)
        assert arm_current_joint_positions["right_arm"].shape == (7,)

        # If needs to reset, reinitialize IK warmstart state to current joint positions
        if self._reset_event.is_set():
            self._arm_target_joint_pos = {
                "left_arm": arm_current_joint_positions["left_arm"].copy(),
                "right_arm": arm_current_joint_positions["right_arm"].copy(),
            }
            self._reset_event.clear()

        arm_absolute_target_poses = {
            "L_ee": pin.SE3(arm_absolute_target_poses["L_ee"]),
            "R_ee": pin.SE3(arm_absolute_target_poses["R_ee"]),
        }
        if self._warmstart_with_actual:
            arm_initial_joint_pos = {
                "left_arm": arm_current_joint_positions["left_arm"].copy(),
                "right_arm": arm_current_joint_positions["right_arm"].copy(),
            }
        else:
            arm_initial_joint_pos = {
                "left_arm": self._arm_target_joint_pos["left_arm"].copy(),
                "right_arm": self._arm_target_joint_pos["right_arm"].copy(),
            }
        # Solve IK
        arm_target_joint_pos = self._pink_ik_solver.solve_ik(
            ee_target_poses=arm_absolute_target_poses,
            arm_initial_joint_pos=arm_initial_joint_pos,
        )
        # Update warmstart state for next IK call
        self._arm_target_joint_pos = {
            "left_arm": arm_target_joint_pos["left_arm"].copy(),
            "right_arm": arm_target_joint_pos["right_arm"].copy(),
        }
        return arm_target_joint_pos

    def reset(self):
        """Reset IK warmstart state (i.e. solution buffer) for episode restart."""
        self._reset_event.set()


class InitializationCollisionPlanner:
    """Plan and execute collision-safe arm initialization with OMPL + Pinocchio."""

    # OMPL state order: left_arm (7), right_arm (7), left_hand (22), right_hand (22), (0) if not exist
    # NOTE: This order might be different from qpos order
    ompl_state_component_order = ["left_arm", "right_arm", "left_hand", "right_hand"]
    component_to_nq = {
        "left_arm": ARM_DOF,
        "right_arm": ARM_DOF,
        "left_hand": HAND_DOF,
        "right_hand": HAND_DOF,
    }

    def __init__(
        self,
        pin_full_robot_wrapper: RobotWrapper,
        assemble_qpos: Callable[[dict[str, np.ndarray]], np.ndarray],
        disassemble_qpos: Callable[[np.ndarray], dict[str, np.ndarray]],
        max_edge_joint_step: float = 0.03,
        plan_timeout_s: float = 10.0,
        solve_step_s: float = 0.1,
    ):
        """Initialize InitializationCollisionPlanner.

        Args:
            pin_full_robot_wrapper: Pinocchio RobotWrapper instance for the robot model,
                should be the full model with arms and hands (lock torso and head) and
                added obstacles used for collision checking.
            assemble_qpos: Function to convert dict of joint positions dict to full qpos.
            disassemble_qpos: Function to convert full qpos array back to dict of joint
                positions.
            max_edge_joint_step: Maximum allowed joint step (in radians) for edge collision
                checking and linear interpolation of paths.
            plan_timeout_s: Timeout in seconds for OMPL planner to find a solution.
            solve_step_s: Time in seconds for each call to planner.solve() in the OMPL
                planning loop.
        """
        self.max_edge_joint_step = max_edge_joint_step
        self.plan_timeout_s = plan_timeout_s
        self.solve_step_s = solve_step_s

        self._assemble_qpos = assemble_qpos
        self._disassemble_qpos = disassemble_qpos
        self._pin_full_robot_wrapper = pin.RobotWrapper(
            pin.Model(pin_full_robot_wrapper.model),
            pin.GeometryModel(pin_full_robot_wrapper.collision_model),
            pin.GeometryModel(pin_full_robot_wrapper.visual_model),
        )
        assert (
            self._pin_full_robot_wrapper.nq == 2 * ARM_DOF + 2 * HAND_DOF
        )  # left arm + right arm + left hand + right hand

    def add_collision_box(self, name: str, position: np.ndarray, full_extents: np.ndarray):
        """Add a box obstacle to the internal collision model.

        Args:
            name: Unique name for the obstacle (used to remove it later).
            position: Center position [x, y, z] in the robot base frame.
            full_extents: Full box dimensions [x, y, z] in meters.
        """
        collision_model = self._pin_full_robot_wrapper.collision_model
        shape = hppfcl.Box(*full_extents)
        pose = pin.SE3(np.eye(3), position)
        obj = pin.GeometryObject(name, 0, pose, shape)
        obj_id = collision_model.addGeometryObject(obj)
        for existing_id in range(collision_model.ngeoms):
            if existing_id != obj_id:
                collision_model.addCollisionPair(pin.CollisionPair(existing_id, obj_id))
        self._pin_full_robot_wrapper.collision_data = collision_model.createData()

    def remove_collision_box(self, name: str):
        """Remove a previously added collision box by name."""
        collision_model = self._pin_full_robot_wrapper.collision_model
        collision_model.removeGeometryObject(name)
        self._pin_full_robot_wrapper.collision_data = collision_model.createData()

    def _ompl_state_to_joint_pos_dict(self, state: ob.State) -> dict[str, np.ndarray]:
        joint_pos_dict = {}
        state_idx = 0
        for component in self.ompl_state_component_order:
            nq = self.component_to_nq[component]
            if component in self._enabled_components:
                joint_pos_dict[component] = np.zeros(nq, dtype=np.float64)
                for i in range(nq):
                    joint_pos_dict[component][i] = state[state_idx]
                    state_idx += 1
            else:
                joint_pos_dict[component] = self._disabled_default_joint_pos[component].copy()
        assert state_idx == self._state_space_size
        return joint_pos_dict

    def _joint_pos_dict_to_ompl_array(self, joint_pos_dict: dict[str, np.ndarray]) -> np.ndarray:
        state_array = np.zeros(self._state_space_size, dtype=np.float64)
        state_idx = 0
        for component in self.ompl_state_component_order:
            nq = self.component_to_nq[component]
            if component in self._enabled_components:
                for i in range(nq):
                    state_array[state_idx] = joint_pos_dict[component][i]
                    state_idx += 1
            else:
                # Skip disabled components
                pass
        assert state_idx == self._state_space_size
        return state_array

    def _is_qpos_in_collision(self, qpos: np.ndarray) -> bool:
        assert qpos.shape == (self._pin_full_robot_wrapper.nq,)
        pin.computeCollisions(
            self._pin_full_robot_wrapper.model,
            self._pin_full_robot_wrapper.data,
            self._pin_full_robot_wrapper.collision_model,
            self._pin_full_robot_wrapper.collision_data,
            qpos,
            True,  # stop at first collision
        )
        return any(
            self._pin_full_robot_wrapper.collision_data.collisionResults[idx].isCollision()
            for idx in range(len(self._pin_full_robot_wrapper.collision_data.collisionResults))
        )

    def _is_state_valid_ompl(self, state: Any) -> bool:
        joint_pos_dict = self._ompl_state_to_joint_pos_dict(state)
        qpos = self._assemble_qpos(joint_pos_dict)
        return not self._is_qpos_in_collision(qpos)

    def _is_edge_collision_free(self, qpos1: np.ndarray, qpos2: np.ndarray) -> bool:
        max_delta = float(np.max(np.abs(qpos2 - qpos1)))
        interpolation_steps = max(1, int(np.ceil(max_delta / self.max_edge_joint_step)))
        for step in range(interpolation_steps + 1):
            alpha = step / interpolation_steps
            q = (1.0 - alpha) * qpos1 + alpha * qpos2
            if self._is_qpos_in_collision(q):
                return False
        return True

    def _is_path_collision_free(self, qpos_path: list[np.ndarray]) -> bool:
        if len(qpos_path) < 2:
            assert len(qpos_path) == 1
            return not self._is_qpos_in_collision(qpos_path[0])
        if self._is_qpos_in_collision(qpos_path[0]):
            return False
        for idx in range(1, len(qpos_path)):
            if not self._is_edge_collision_free(qpos_path[idx - 1], qpos_path[idx]):
                return False
        return True

    def _shortcut_path(self, qpos_path: list[np.ndarray]) -> list[np.ndarray]:
        if len(qpos_path) < 3:
            return qpos_path
        shortcut_path = [qpos_path[0].copy()]
        idx = 0
        while idx < len(qpos_path) - 1:
            next_idx = len(qpos_path) - 1
            while next_idx > idx + 1 and not self._is_edge_collision_free(
                qpos_path[idx], qpos_path[next_idx]
            ):
                next_idx -= 1
            shortcut_path.append(qpos_path[next_idx].copy())
            idx = next_idx
        return shortcut_path

    def _densify_path(self, qpos_path: list[np.ndarray]) -> list[np.ndarray]:
        if len(qpos_path) <= 1:
            return qpos_path
        dense: list[np.ndarray] = [qpos_path[0].copy()]
        for idx in range(1, len(qpos_path)):
            start = qpos_path[idx - 1]
            goal = qpos_path[idx]
            max_delta = float(np.max(np.abs(goal - start)))
            interpolation_steps = max(1, int(np.ceil(max_delta / self.max_edge_joint_step)))
            for step in range(1, interpolation_steps + 1):
                alpha = step / interpolation_steps
                dense.append(((1.0 - alpha) * start + alpha * goal).copy())
        return dense

    def plan(
        self,
        start_joint_pos: dict[str, np.ndarray],
        goal_joint_pos: dict[str, np.ndarray],
        enabled_components: set[str],
    ) -> list[dict[str, np.ndarray]]:
        """Plan a collision-free dense waypoint path in joint space from start to goal.

        Only the enabled components move; disabled components stay fixed at their
        start_joint_pos values.

        This function first uses OMPL's implementation of RRTConnect to plan a path
        with timeouts, then postprocesses the path by shortcutting unnecessary waypoints
        and densifying it by linear interpolation in joint space.

        Args:
            start_joint_pos: Dict mapping component name to joint positions (np.ndarray).
            goal_joint_pos: Dict mapping component name to joint positions (np.ndarray).
            enabled_components: Set of component names that are allowed to move during
                planning. Components not in this set should have the same start and goal
                joint positions.

        Returns:
            List of dicts mapping component name to joint positions (np.ndarray)
            representing the planned path from start to goal. The joint space difference
            between each dense waypoint should not exceed self.max_edge_joint_step, the
            path should be collision-free, and the final waypoint should be within a small
            tolerance of the goal.

        Raises:
            AssertionError: If input validation fails.
            RuntimeError: If OMPL fails to find a solution within the timeout, or if the
                postprocessed path is not collision-free.
        """
        # Basic input validation
        assert set(start_joint_pos.keys()) == set(self.ompl_state_component_order)
        assert set(goal_joint_pos.keys()) == set(self.ompl_state_component_order)
        assert len(enabled_components) > 0
        assert enabled_components.issubset(set(self.ompl_state_component_order))
        for component in self.ompl_state_component_order:
            if component not in enabled_components:
                assert np.allclose(
                    start_joint_pos[component],
                    goal_joint_pos[component],
                )
        start_qpos = self._assemble_qpos(start_joint_pos)
        goal_qpos = self._assemble_qpos(goal_joint_pos)
        assert not self._is_qpos_in_collision(start_qpos)
        assert not self._is_qpos_in_collision(goal_qpos)
        # assert np.all(start_qpos <= self._pin_full_robot_wrapper.model.upperPositionLimit)
        # assert np.all(start_qpos >= self._pin_full_robot_wrapper.model.lowerPositionLimit)
        # assert np.all(goal_qpos <= self._pin_full_robot_wrapper.model.upperPositionLimit)
        # assert np.all(goal_qpos >= self._pin_full_robot_wrapper.model.lowerPositionLimit)

        if np.allclose(start_qpos, goal_qpos, atol=1e-6):
            return [self._disassemble_qpos(goal_qpos)]

        # (Re)set OMPL planner and relevant state
        self._enabled_components = deepcopy(enabled_components)
        self._disabled_default_joint_pos = deepcopy(start_joint_pos)
        self._state_space_size = sum(
            self.component_to_nq[component] for component in self._enabled_components
        )
        self._space = ob.RealVectorStateSpace(self._state_space_size)

        bounds = ob.RealVectorBounds(self._state_space_size)
        lower_position_limit_dict = self._disassemble_qpos(
            self._pin_full_robot_wrapper.model.lowerPositionLimit
        )
        upper_position_limit_dict = self._disassemble_qpos(
            self._pin_full_robot_wrapper.model.upperPositionLimit
        )
        ompl_lower_bounds = self._joint_pos_dict_to_ompl_array(lower_position_limit_dict)
        ompl_upper_bounds = self._joint_pos_dict_to_ompl_array(upper_position_limit_dict)
        for i in range(self._state_space_size):
            bounds.setLow(i, float(ompl_lower_bounds[i]))
            bounds.setHigh(i, float(ompl_upper_bounds[i]))
        self._space.setBounds(bounds)

        self._space_information = ob.SpaceInformation(self._space)
        # Classic (<=1.6, Py++) OMPL bindings need the StateValidityCheckerFn wrapper;
        # the nanobind bindings (PyPI ompl >= 1.7) accept a plain Python callable.
        _classic_ompl_bindings = hasattr(ob, "StateValidityCheckerFn")
        self._space_information.setStateValidityChecker(
            ob.StateValidityCheckerFn(self._is_state_valid_ompl)
            if _classic_ompl_bindings
            else self._is_state_valid_ompl
        )
        self._space_information.setStateValidityCheckingResolution(
            0.005
        )  # Keep OMPL's internal discretization tighter than our explicit edge check.
        self._space_information.setup()

        self._problem_definition = ob.ProblemDefinition(self._space_information)
        if _classic_ompl_bindings:
            start_state = ob.State(self._space)
            goal_state = ob.State(self._space)
        else:  # nanobind bindings expose no ob.State constructor; allocate via the space
            start_state = self._space.allocState()
            goal_state = self._space.allocState()
        ompl_start_array = self._joint_pos_dict_to_ompl_array(start_joint_pos)
        ompl_goal_array = self._joint_pos_dict_to_ompl_array(goal_joint_pos)
        for i in range(self._state_space_size):
            start_state[i] = float(ompl_start_array[i])
            goal_state[i] = float(ompl_goal_array[i])
        self._problem_definition.setStartAndGoalStates(start_state, goal_state)

        self._planner = og.RRTConnect(self._space_information)
        self._planner.setProblemDefinition(self._problem_definition)
        self._planner.setup()

        # OMPL planner loop with timeout
        solved = False
        goal_reach_tolerance = max(1e-3, 0.5 * self.max_edge_joint_step)
        best_approx_goal_error = np.inf
        planning_start_time = time.perf_counter()
        while time.perf_counter() - planning_start_time < self.plan_timeout_s:
            status = self._planner.solve(self.solve_step_s)
            if bool(status):
                # Not tolerate approximate solution
                candidate_solution_path = self._problem_definition.getSolutionPath()
                if candidate_solution_path is None or candidate_solution_path.getStateCount() == 0:
                    continue
                candidate_final_joint_pos_dict = self._ompl_state_to_joint_pos_dict(
                    candidate_solution_path.getState(candidate_solution_path.getStateCount() - 1)
                )
                candidate_goal_error = max(
                    np.max(
                        np.abs(
                            candidate_final_joint_pos_dict[component] - goal_joint_pos[component]
                        )
                    )
                    for component in self._enabled_components
                )
                if candidate_goal_error <= goal_reach_tolerance:
                    solved = True
                    break
                best_approx_goal_error = min(best_approx_goal_error, candidate_goal_error)

        if not solved:
            approx_error_info = ""
            if np.isfinite(best_approx_goal_error):
                approx_error_info = f", best_approx_goal_max_error={best_approx_goal_error:.4f} rad"
            raise RuntimeError(
                "OMPL RRTConnect failed to find a collision-free initialization path within constraints "
                f"(timeout={self.plan_timeout_s:.2f}s{approx_error_info})."
            )

        # Postprocess OMPL solution by shortcutting and densification
        solution_path = self._problem_definition.getSolutionPath()
        if solution_path is None or solution_path.getStateCount() < 2:
            raise RuntimeError("OMPL returned an empty initialization path.")

        raw_qpos_path = [
            self._assemble_qpos(self._ompl_state_to_joint_pos_dict(solution_path.getState(i)))
            for i in range(solution_path.getStateCount())
        ]
        shortcut_qpos_path = self._shortcut_path(raw_qpos_path)
        dense_path = self._densify_path(shortcut_qpos_path)
        if not self._is_path_collision_free(dense_path):
            raise RuntimeError(
                "Postprocessed initialization path failed Pinocchio edge collision validation."
            )

        return [self._disassemble_qpos(q) for q in dense_path]


def move_robot_to_position_safe(
    dexmate_bimanual_robot: Robot,
    sharpa_left_hand: SharpaWave,
    sharpa_right_hand: SharpaWave,
    initialization_planner: InitializationCollisionPlanner,
    action_buffer: dict[str, np.ndarray | None],
    action_buf_lock: threading.Lock,
    target_joint_pos: dict[str, np.ndarray],
    hardware_lock: threading.Lock,
    command_hz: float,
    dof_error_tolerance: float,
    hold_time_s: float,
    right_arm_collision_boxes: list[dict] | None = None,
):
    """Move robot to initial pose with collision-aware arm planning.

    Args:
        dexmate_bimanual_robot: dexcontrol robot handle for the arms/torso/head.
        sharpa_left_hand: Left Sharpa Wave hand handle.
        sharpa_right_hand: Right Sharpa Wave hand handle.
        initialization_planner: Collision-aware joint-space planner.
        action_buffer: Shared action buffer consumed by full_robot_action_loop.
        action_buf_lock: Lock guarding action_buffer.
        target_joint_pos: Target joint positions per component (must include
            both arms and both hands; head/torso optional).
        hardware_lock: Lock serializing hardware access.
        command_hz: Rate at which waypoints are streamed.
        dof_error_tolerance: Max per-DOF error (rad) to consider the move done.
        hold_time_s: Time to hold the final pose before returning.
        right_arm_collision_boxes: Optional list of collision boxes to add only during
            right arm planning. Each dict has keys: name (str), position (np.ndarray),
            full_extents (np.ndarray).
    """
    assert set(target_joint_pos.keys()) <= {
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "head",
        "torso",
    }
    assert set(target_joint_pos.keys()) >= {"left_arm", "right_arm", "left_hand", "right_hand"}
    logger.info("Planning collision-safe move to initial position...")
    limiter = RateLimiter(frequency=command_hz, name="initialization_move_limiter", warn=True)
    hold_iters = int(hold_time_s * command_hz)
    with action_buf_lock:
        action_buffer["left_arm"] = None
        action_buffer["right_arm"] = None
        action_buffer["left_hand"] = None
        action_buffer["right_hand"] = None

    # 0. Torso and head
    safe_components = {"head", "torso"}
    safe_goal_joint_pos = {
        component: target_joint_pos.pop(component)
        for component in safe_components
        if component in target_joint_pos
    }
    if safe_goal_joint_pos:
        with hardware_lock:
            dexmate_bimanual_robot.set_joint_pos(safe_goal_joint_pos, relative=False, wait_time=4)

    # 1. Left arm
    with hardware_lock:
        curr_joint_pos_dict = dexmate_bimanual_robot.get_joint_pos_dict(
            component=["left_arm", "right_arm"]
        )
        left_hand_joint_positions = np.array(sharpa_left_hand.get_states().angles, dtype=np.float64)
        right_hand_joint_positions = np.array(
            sharpa_right_hand.get_states().angles, dtype=np.float64
        )
    left_arm_joint_positions = np.array(
        [curr_joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]]
    )
    right_arm_joint_positions = np.array(
        [curr_joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]]
    )

    waypoint_list = initialization_planner.plan(
        start_joint_pos={
            "left_arm": left_arm_joint_positions,
            "right_arm": right_arm_joint_positions,
            "left_hand": left_hand_joint_positions,
            "right_hand": right_hand_joint_positions,
        },
        goal_joint_pos={
            "left_arm": target_joint_pos["left_arm"],
            "right_arm": right_arm_joint_positions,
            "left_hand": left_hand_joint_positions,
            "right_hand": right_hand_joint_positions,
        },
        enabled_components={"left_arm"},
    )
    for waypoint in waypoint_list:
        limiter.sleep()
        with action_buf_lock:
            action_buffer["left_arm"] = waypoint["left_arm"]
            action_buffer["right_arm"] = waypoint["right_arm"]
            action_buffer["left_hand"] = waypoint["left_hand"]
            action_buffer["right_hand"] = waypoint["right_hand"]
    for _ in range(hold_iters):
        limiter.sleep()
        with action_buf_lock:
            action_buffer["left_arm"] = waypoint_list[-1]["left_arm"]
            action_buffer["right_arm"] = waypoint_list[-1]["right_arm"]
            action_buffer["left_hand"] = waypoint_list[-1]["left_hand"]
            action_buffer["right_hand"] = waypoint_list[-1]["right_hand"]

    # 2. Right arm
    with hardware_lock:
        curr_joint_pos_dict = dexmate_bimanual_robot.get_joint_pos_dict(
            component=["left_arm", "right_arm"]
        )
        left_hand_joint_positions = np.array(sharpa_left_hand.get_states().angles, dtype=np.float64)
        right_hand_joint_positions = np.array(
            sharpa_right_hand.get_states().angles, dtype=np.float64
        )
    left_arm_joint_positions = np.array(
        [curr_joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]]
    )
    right_arm_joint_positions = np.array(
        [curr_joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]]
    )

    # Add temporary collision boxes for right arm planning
    if right_arm_collision_boxes:
        for box in right_arm_collision_boxes:
            initialization_planner.add_collision_box(**box)
    try:
        waypoint_list = initialization_planner.plan(
            start_joint_pos={
                "left_arm": left_arm_joint_positions,
                "right_arm": right_arm_joint_positions,
                "left_hand": left_hand_joint_positions,
                "right_hand": right_hand_joint_positions,
            },
            goal_joint_pos={
                "left_arm": left_arm_joint_positions,
                "right_arm": target_joint_pos["right_arm"],
                "left_hand": left_hand_joint_positions,
                "right_hand": right_hand_joint_positions,
            },
            enabled_components={"right_arm"},
        )
    finally:
        if right_arm_collision_boxes:
            for box in right_arm_collision_boxes:
                initialization_planner.remove_collision_box(box["name"])
    for waypoint in waypoint_list:
        limiter.sleep()
        with action_buf_lock:
            action_buffer["left_arm"] = waypoint["left_arm"]
            action_buffer["right_arm"] = waypoint["right_arm"]
            action_buffer["left_hand"] = waypoint["left_hand"]
            action_buffer["right_hand"] = waypoint["right_hand"]
    for _ in range(hold_iters):
        limiter.sleep()
        with action_buf_lock:
            action_buffer["left_arm"] = waypoint_list[-1]["left_arm"]
            action_buffer["right_arm"] = waypoint_list[-1]["right_arm"]
            action_buffer["left_hand"] = waypoint_list[-1]["left_hand"]
            action_buffer["right_hand"] = waypoint_list[-1]["right_hand"]

    # 3. Check if arms are close enough to target before moving hands, for safety
    with hardware_lock:
        curr_joint_pos_dict = dexmate_bimanual_robot.get_joint_pos_dict(
            component=["left_arm", "right_arm"]
        )
    left_arm_joint_positions = np.array(
        [curr_joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]]
    )
    right_arm_joint_positions = np.array(
        [curr_joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]]
    )
    max_arm_error = max(
        np.max(np.abs(left_arm_joint_positions - target_joint_pos["left_arm"])),
        np.max(np.abs(right_arm_joint_positions - target_joint_pos["right_arm"])),
    )
    # NOTE(safety): initialization errors beyond tolerance currently only warn instead of
    # raising, because the hand collision model is unverified and can leave residual error.
    if max_arm_error > dof_error_tolerance:
        logger.warning(
            f"Arm initialization: final joint position error {max_arm_error:.4f} rad exceeds tolerance {dof_error_tolerance:.4f} rad."
        )

    # 4. Left and right hands
    for _ in range(hold_iters):
        limiter.sleep()
        with action_buf_lock:
            action_buffer["left_arm"] = target_joint_pos["left_arm"]
            action_buffer["right_arm"] = target_joint_pos["right_arm"]
            action_buffer["left_hand"] = target_joint_pos["left_hand"]
            action_buffer["right_hand"] = target_joint_pos["right_hand"]

    # 5. Reset to none targets
    with action_buf_lock:
        action_buffer["left_arm"] = None
        action_buffer["right_arm"] = None
        action_buffer["left_hand"] = None
        action_buffer["right_hand"] = None

    # 6. Final check
    with hardware_lock:
        curr_joint_pos_dict = dexmate_bimanual_robot.get_joint_pos_dict(
            component=["left_arm", "right_arm"]
        )
        left_hand_joint_positions = np.array(sharpa_left_hand.get_states().angles, dtype=np.float64)
        right_hand_joint_positions = np.array(
            sharpa_right_hand.get_states().angles, dtype=np.float64
        )
    left_arm_joint_positions = np.array(
        [curr_joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]]
    )
    right_arm_joint_positions = np.array(
        [curr_joint_pos_dict[name] for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]]
    )
    final_pos_error = max(
        np.max(np.abs(left_arm_joint_positions - target_joint_pos["left_arm"])),
        np.max(np.abs(right_arm_joint_positions - target_joint_pos["right_arm"])),
        np.max(np.abs(left_hand_joint_positions - target_joint_pos["left_hand"])),
        np.max(np.abs(right_hand_joint_positions - target_joint_pos["right_hand"])),
    )
    # NOTE(safety): warn-only for the same reason as above (unverified hand collision model).
    if final_pos_error > dof_error_tolerance:
        logger.warning(
            f"Arm and hand initialization: final joint position error {final_pos_error:.4f} rad exceeds tolerance {dof_error_tolerance:.4f} rad."
        )
    else:
        logger.info(f"Reset final max joint position error: {final_pos_error}")


class SmoothingAndSafetyManager:
    """Handles arm control command smoothing and collision checking for full robot."""

    def __init__(
        self,
        pin_full_robot_wrapper: RobotWrapper,
        assemble_qpos: Callable[[dict[str, np.ndarray]], np.ndarray],
        disassemble_qpos: Callable[[np.ndarray], dict[str, np.ndarray]],
        ruckig_smoothing: bool,
        action_hz: float,
    ):
        """Initialize SmoothingAndSafetyManager.

        Args:
            pin_full_robot_wrapper: Pinocchio RobotWrapper instance for the robot model, should be the full model
                with arms and hands.
            assemble_qpos: Function to convert dict of joint positions to full qpos array for collision checking.
            disassemble_qpos: Function to convert full qpos array back to dict of joint positions
            ruckig_smoothing: Whether to use Ruckig for smoothing. If False, uses simple velocity-limited smoothing.
            action_hz: Action loop frequency (Hz) for smoothing velocity limits.
        """
        # Deep copy to avoid modifying original
        # NOTE: this model should contain Sharpa hands and Dexmate arms.
        self._pin_full_robot_wrapper = pin.RobotWrapper(
            pin.Model(pin_full_robot_wrapper.model),
            pin.GeometryModel(pin_full_robot_wrapper.collision_model),
            pin.GeometryModel(pin_full_robot_wrapper.visual_model),
        )
        assert (
            self._pin_full_robot_wrapper.nq == 7 + 7 + 22 + 22
        )  # left arm + right arm + left hand + right hand
        self._assemble_qpos = assemble_qpos
        self._disassemble_qpos = disassemble_qpos

        # Smoothing state for arm joint targets.
        self._ruckig_smoothing = ruckig_smoothing
        self._action_hz = action_hz
        self._smoothing_state = {
            # This must be in collision-free configuration and contain both Sharpa hands
            # and Dexmate arms, initialized on first call to smooth_action.
            "previous_smoothed_safe_full_targets": {
                "left_arm": np.full(7, np.nan),
                "right_arm": np.full(7, np.nan),
                "left_hand": np.full(22, np.nan),
                "right_hand": np.full(22, np.nan),
            },  # type: dict[str, dict[str, np.ndarray]
        }
        if self._ruckig_smoothing:
            # Ruckig setup
            # NOTE: order of Ruckig is "left_arm, right_arm" for all 14 DOFs, no hand DOFs included.
            otg = Ruckig(14, 1 / self._action_hz)
            inp = InputParameter(14)
            out = OutputParameter(14)
            # NOTE: Free version of Ruckig does not natively support tracking and does not support joint position limits
            # inp.current_position = None
            # inp.current_velocity = None
            # inp.current_acceleration = None
            # inp.target_position = None
            inp.target_velocity = np.zeros(14).tolist()
            inp.target_acceleration = np.zeros(14).tolist()
            inp.max_velocity = (
                DEXMATE_DEFAULT_ARM_VEL_LIMITS.tolist() + DEXMATE_DEFAULT_ARM_VEL_LIMITS.tolist()
            )
            inp.max_acceleration = [10.0] * 14
            inp.max_jerk = [50.0] * 14
            self._smoothing_state["otg"] = otg
            self._smoothing_state["inp"] = inp
            self._smoothing_state["out"] = out

        # Reset status
        self._reset_event = threading.Event()
        # On initialization, reset
        self.reset()

    def _collision_check(
        self,
        joint_positions: dict[str, np.ndarray],
    ) -> bool:
        assert set(joint_positions.keys()) == {"left_arm", "right_arm", "left_hand", "right_hand"}
        assert joint_positions["left_arm"].shape == (7,)
        assert joint_positions["right_arm"].shape == (7,)
        assert joint_positions["left_hand"].shape == (22,)
        assert joint_positions["right_hand"].shape == (22,)
        pin.computeCollisions(
            self._pin_full_robot_wrapper.model,
            self._pin_full_robot_wrapper.data,
            self._pin_full_robot_wrapper.collision_model,
            self._pin_full_robot_wrapper.collision_data,
            self._assemble_qpos(joint_positions),
            True,  # Terminate on first collision for efficiency
        )
        return any(
            self._pin_full_robot_wrapper.collision_data.collisionResults[idx].isCollision()
            for idx in range(len(self._pin_full_robot_wrapper.collision_data.collisionResults))
        )

    def smooth_and_check_action(
        self,
        full_target_joint_positions: dict[str, np.ndarray],
        full_current_joint_positions: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Smooth arm joint position targets and check full model for self collision.

        This method should be called at high frequency (e.g., 300Hz) in the action loop.
        Maintains internal state for smoothing continuity.

        Args:
            full_target_joint_positions: Target full joint positions from IK, as dict with
                "left_arm", "right_arm", "left_hand", "right_hand" keys.
            full_current_joint_positions: Current full robot joint positions (for initialization)

        Returns:
            Dict with "left_arm", "right_arm", "left_hand", "right_hand" keys containing smoothed
                joint positions. If the smoothed target causes self-collision, returns the
                previous smoothed safe target instead.
        """
        assert set(full_target_joint_positions.keys()) == {
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        }
        assert set(full_current_joint_positions.keys()) == {
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
        }
        assert full_target_joint_positions["left_arm"].shape == (7,)
        assert full_target_joint_positions["right_arm"].shape == (7,)
        assert full_target_joint_positions["left_hand"].shape == (22,)
        assert full_target_joint_positions["right_hand"].shape == (22,)
        assert full_current_joint_positions["left_arm"].shape == (7,)
        assert full_current_joint_positions["right_arm"].shape == (7,)
        assert full_current_joint_positions["left_hand"].shape == (22,)
        assert full_current_joint_positions["right_hand"].shape == (22,)

        # If needs to reset, reinitialize smoothing state to current joint positions
        if self._reset_event.is_set():
            # Assume the current state is collision free
            self._smoothing_state["previous_smoothed_safe_full_targets"] = {
                "left_arm": full_current_joint_positions["left_arm"].copy(),
                "right_arm": full_current_joint_positions["right_arm"].copy(),
                "left_hand": full_current_joint_positions["left_hand"].copy(),
                "right_hand": full_current_joint_positions["right_hand"].copy(),
            }
            if self._ruckig_smoothing:
                self._smoothing_state["inp"].current_position = np.concatenate(
                    [
                        self._smoothing_state["previous_smoothed_safe_full_targets"]["left_arm"],
                        self._smoothing_state["previous_smoothed_safe_full_targets"]["right_arm"],
                    ],
                    axis=0,
                ).tolist()  # Still concatenate based on convention
                self._smoothing_state["inp"].current_velocity = np.zeros(14).tolist()
                self._smoothing_state["inp"].current_acceleration = np.zeros(14).tolist()
            self._reset_event.clear()

        # Apply smoothing for arm targets only
        full_smoothed_target_joint_positions = {
            "left_hand": full_target_joint_positions["left_hand"].copy(),
            "right_hand": full_target_joint_positions["right_hand"].copy(),
        }
        if self._ruckig_smoothing:
            # Higher order smoothing via Ruckig for arm joints
            # NOTE: Free version of Ruckig does not support tracking.
            otg = self._smoothing_state["otg"]
            inp = self._smoothing_state["inp"]
            out = self._smoothing_state["out"]
            inp.target_position = np.concatenate(
                [full_target_joint_positions["left_arm"], full_target_joint_positions["right_arm"]]
            ).tolist()
            ruckig_res = otg.update(inp, out)
            if ruckig_res != Result.Working and ruckig_res != Result.Finished:
                raise RuntimeError(
                    f"[Arm Action] Ruckig update returned {ruckig_res}, expected Working or Finished"
                )
            # NOTE: Free version of Ruckig does not support joint position hard limits
            full_smoothed_target_joint_positions["left_arm"] = np.clip(
                out.new_position[:7],
                DEXMATE_LEFT_ARM_LOWER_QPOS_LIMITS,
                DEXMATE_LEFT_ARM_UPPER_QPOS_LIMITS,
            )
            full_smoothed_target_joint_positions["right_arm"] = np.clip(
                out.new_position[7:],
                DEXMATE_RIGHT_ARM_LOWER_QPOS_LIMITS,
                DEXMATE_RIGHT_ARM_UPPER_QPOS_LIMITS,
            )
        else:
            # Simple velocity-limited smoothing for arm joints
            for arm_name in ["left_arm", "right_arm"]:
                pos_diff = (
                    full_target_joint_positions[arm_name]
                    - self._smoothing_state["previous_smoothed_safe_full_targets"][arm_name]
                )
                clipped_pos_diff = np.clip(
                    pos_diff,
                    -DEXMATE_VEL_LIMIT_SCALE * DEXMATE_DEFAULT_ARM_VEL_LIMITS / self._action_hz,
                    DEXMATE_VEL_LIMIT_SCALE * DEXMATE_DEFAULT_ARM_VEL_LIMITS / self._action_hz,
                )
                full_smoothed_target_joint_positions[arm_name] = (
                    self._smoothing_state["previous_smoothed_safe_full_targets"][arm_name]
                    + clipped_pos_diff
                )

        # As a precaution, clip to joint limits
        full_smoothed_target_joint_positions = self._disassemble_qpos(
            np.clip(
                self._assemble_qpos(full_smoothed_target_joint_positions),
                self._pin_full_robot_wrapper.model.lowerPositionLimit,
                self._pin_full_robot_wrapper.model.upperPositionLimit,
            )
        )

        # Collision Check and possibly fallback, also update
        if self._collision_check(full_smoothed_target_joint_positions):
            logger.warning(
                "[Arm Action] Smoothed target causes self-collision, reverting to previous safe target"
            )
            full_smoothed_target_joint_positions = {
                "left_arm": self._smoothing_state["previous_smoothed_safe_full_targets"][
                    "left_arm"
                ].copy(),
                "right_arm": self._smoothing_state["previous_smoothed_safe_full_targets"][
                    "right_arm"
                ].copy(),
                "left_hand": self._smoothing_state["previous_smoothed_safe_full_targets"][
                    "left_hand"
                ].copy(),
                "right_hand": self._smoothing_state["previous_smoothed_safe_full_targets"][
                    "right_hand"
                ].copy(),
            }
            if self._ruckig_smoothing:
                self._smoothing_state["inp"].current_position = np.concatenate(
                    [
                        self._smoothing_state["previous_smoothed_safe_full_targets"]["left_arm"],
                        self._smoothing_state["previous_smoothed_safe_full_targets"]["right_arm"],
                    ],
                    axis=0,
                ).tolist()
                self._smoothing_state["inp"].current_velocity = np.zeros(14).tolist()
                self._smoothing_state["inp"].current_acceleration = np.zeros(14).tolist()
        else:
            self._smoothing_state["previous_smoothed_safe_full_targets"] = {
                "left_arm": full_smoothed_target_joint_positions["left_arm"].copy(),
                "right_arm": full_smoothed_target_joint_positions["right_arm"].copy(),
                "left_hand": full_smoothed_target_joint_positions["left_hand"].copy(),
                "right_hand": full_smoothed_target_joint_positions["right_hand"].copy(),
            }
            if self._ruckig_smoothing:
                self._smoothing_state["out"].pass_to_input(self._smoothing_state["inp"])

        return full_smoothed_target_joint_positions

    def reset(self):
        """Reset smoothing state for episode restart."""
        self._reset_event.set()


# NOTE: hand commands are sent at action_hz (300 Hz by default); whether the Sharpa hand
# control interface is rated for this frequency has not been verified.
def full_robot_action_loop(
    terminate_event: threading.Event,
    action_buf_lock: threading.Lock,
    action_buffer: dict[str, np.ndarray | None],
    hardware_lock: threading.Lock,
    dexmate_bimanual_robot: Robot,
    sharpa_left_hand: SharpaWave,
    sharpa_right_hand: SharpaWave,
    smoothing_and_safety_manager: SmoothingAndSafetyManager,
    action_hz: float,
    hand_interpolate: bool,
):
    """High-frequency thread that sends hand and arm commands to robot.

    Reads from action_buffer and sends commands at action_hz frequency.
    When buffer is empty (None), stops sending commands.

    smoothing_and_safety_manager includes safety checks:
    1. Arm joint limits clipping - clamps to safe joint ranges
    2. Arm joint smoothing - smooths arm command updates to prevent sudden jumps
    3. Full Self-collision checking - skips commands that cause collision

    In addition, full_robot_action_loop monitors arm tracking error and raises Exception
    if actual joint positions diverge from target by more than TRACKING_SAFETY_THRESHOLD.
    """
    limiter = RateLimiter(frequency=action_hz, name="full_robot_action_limiter", warn=False)
    logger.info(f"[Full Robot Action] Thread started at {action_hz} Hz")

    while not terminate_event.is_set():
        limiter.sleep()

        # Read from buffer with lock
        with action_buf_lock:
            joint_targets = {
                "left_arm": action_buffer["left_arm"],
                "right_arm": action_buffer["right_arm"],
                "left_hand": action_buffer["left_hand"],
                "right_hand": action_buffer["right_hand"],
            }

        # If buffer is empty/None, skip this iteration
        if any(target is None for target in joint_targets.values()):
            smoothing_and_safety_manager.reset()
            continue

        try:
            # Get current joint positions
            with hardware_lock:
                arm_joint_pos_dict = dexmate_bimanual_robot.get_joint_pos_dict(
                    component=["left_arm", "right_arm"]
                )
                left_hand_joint_positions = np.array(
                    sharpa_left_hand.get_states().angles, dtype=np.float64
                )
                right_hand_joint_positions = np.array(
                    sharpa_right_hand.get_states().angles, dtype=np.float64
                )
            joint_positions = {
                "left_arm": np.array(
                    [
                        arm_joint_pos_dict[name]
                        for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["left_arm"]
                    ]
                ),
                "right_arm": np.array(
                    [
                        arm_joint_pos_dict[name]
                        for name in DEXMATE_COMPONENT_NAME_TO_JOINT_NAMES["right_arm"]
                    ]
                ),
                "left_hand": left_hand_joint_positions,
                "right_hand": right_hand_joint_positions,
            }

            # Safety check: verify current arm position is close to target before sending command
            # This prevents dangerous jumps if something goes wrong with the policy
            tracking_error_violation = False
            for arm_name in ["left_arm", "right_arm"]:
                _arm_joint_positions = joint_positions[arm_name]
                _arm_joint_targets = joint_targets[arm_name]
                if not np.allclose(
                    _arm_joint_positions, _arm_joint_targets, atol=TRACKING_SAFETY_THRESHOLD
                ):
                    logger.error(f"[Arm Action] {arm_name} position mismatch - safety stop!")
                    logger.error(f"  Current: {_arm_joint_positions}")
                    logger.error(f"  Target:  {_arm_joint_targets}")
                    tracking_error_violation = True
                    break
            if tracking_error_violation:
                # Clear buffer to stop sending commands
                with action_buf_lock:
                    action_buffer["left_arm"] = None
                    action_buffer["right_arm"] = None
                    action_buffer["left_hand"] = None
                    action_buffer["right_hand"] = None
                terminate_event.set()
                break

            # Smooth the IK solution using velocity limits
            smoothed_targets = smoothing_and_safety_manager.smooth_and_check_action(
                full_target_joint_positions=joint_targets,
                full_current_joint_positions=joint_positions,
            )

            # Send arm and hand commands
            with hardware_lock:
                dexmate_bimanual_robot.set_joint_pos(
                    joint_pos={
                        "left_arm": smoothed_targets["left_arm"],
                        "right_arm": smoothed_targets["right_arm"],
                    },
                    relative=False,
                    wait_time=0,
                )
                sharpa_left_hand.set_joint_position(
                    list(smoothed_targets["left_hand"]), hand_interpolate
                )
                sharpa_right_hand.set_joint_position(
                    list(smoothed_targets["right_hand"]), hand_interpolate
                )

        except Exception as e:
            logger.warning(f"[Full Robot Action] Safety check error: {e}, skipping command")
            continue

    logger.info("[Full Robot Action] Thread stopped")
