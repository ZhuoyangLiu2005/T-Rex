from abc import ABC, abstractmethod

import numpy as np
import pinocchio as pin
from loguru import logger
from pink import Configuration
from pink import solve_ik as _pink_solve_ik
from pink.tasks import FrameTask, PostureTask

from teleop.robot_descriptions import build_reduced_bimanual_robot


class IKSolver(ABC):
    @abstractmethod
    def solve_ik(
        self,
        ee_target_poses: dict[str, pin.SE3],
        arm_initial_joint_pos: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        pass


class PinkLocalIK(IKSolver):
    def __init__(
        self,
        default_joint_by_component: dict[str, np.ndarray],
    ):
        self.reduced_bimanual_robot, self.assemble_qpos, self.disassemble_qpos = (
            build_reduced_bimanual_robot(default_joint_by_component)
        )
        default_joint_by_component_for_qpos = {
            component: default_joint_by_component[component]
            for component in ["left_arm", "right_arm"]
        }
        self.default_qpos = self.assemble_qpos(default_joint_by_component_for_qpos)
        assert self.default_qpos.shape == (self.reduced_bimanual_robot.nq,)
        assert self.reduced_bimanual_robot.nq == 14  # 7 DOF for each arm

    def fk(
        self, frames: list[str], joint_pos_by_component: dict[str, np.ndarray]
    ) -> dict[str, pin.SE3]:
        assert set(joint_pos_by_component.keys()) == {"left_arm", "right_arm"}
        qpos = self.assemble_qpos(joint_pos_by_component)
        assert qpos.shape == (self.reduced_bimanual_robot.nq,)

        pin.forwardKinematics(
            self.reduced_bimanual_robot.model, self.reduced_bimanual_robot.data, qpos
        )
        pin.updateFramePlacements(
            self.reduced_bimanual_robot.model, self.reduced_bimanual_robot.data
        )
        result = {}
        for frame in frames:
            fid = self.reduced_bimanual_robot.model.getFrameId(frame)
            result[frame] = self.reduced_bimanual_robot.data.oMf[fid].copy()
        return result

    def solve_ik(
        self,
        ee_target_poses: dict[str, pin.SE3],
        arm_initial_joint_pos: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        assert set(ee_target_poses.keys()) == {"L_ee", "R_ee"}
        assert set(arm_initial_joint_pos.keys()) == {"left_arm", "right_arm"}
        initial_qpos = self.assemble_qpos(arm_initial_joint_pos)
        assert initial_qpos.shape == (self.reduced_bimanual_robot.nq,)

        left_ee_task = FrameTask(
            frame="L_ee",
            position_cost=50.0,
            orientation_cost=1.0,
            lm_damping=0,
            gain=0.2,
        )
        right_ee_task = FrameTask(
            frame="R_ee",
            position_cost=50.0,
            orientation_cost=1.0,
            lm_damping=0,
            gain=0.2,
        )
        smoothness_posture_task = PostureTask(
            cost=0.2,
            lm_damping=0.0,
            gain=0.2,
        )
        regularization_posture_task = PostureTask(
            cost=0.05,
            lm_damping=0.0,
            gain=0.2,
        )

        smoothness_posture_task.set_target(initial_qpos)
        regularization_posture_task.set_target(self.default_qpos)
        left_ee_task.set_target(ee_target_poses["L_ee"])
        right_ee_task.set_target(ee_target_poses["R_ee"])

        tasks = [left_ee_task, right_ee_task, smoothness_posture_task, regularization_posture_task]

        configuration = Configuration(
            model=self.reduced_bimanual_robot.model,
            data=self.reduced_bimanual_robot.data,
            q=initial_qpos,
            copy_data=True,
            forward_kinematics=True,
            # NOTE: Adding collision model in configuration slows down pink
            # collision_model=self.reduced_bimanual_robot.collision_model,
            # collision_data=self.reduced_bimanual_robot.collision_data,
        )

        dt = 0.05  # integration step for the velocity returned by pink
        for _ in range(5):
            try:
                velocity = _pink_solve_ik(
                    configuration=configuration,
                    tasks=tasks,
                    dt=dt,
                    solver="daqp",
                    damping=0,
                )
                if np.any(np.isnan(velocity)):
                    velocity = np.zeros_like(configuration.q)
            except Exception as e:
                logger.error(
                    f"[PinkLocalIK] IK solver failed to find solution, returning previous joint pos. Error: {e}"
                )
                velocity = np.zeros_like(configuration.q)
            # print("Current configuration:", configuration.q)
            # print("Current velocity:", velocity)
            configuration.integrate_inplace(velocity, dt)
            clipped_q = np.clip(
                configuration.q,
                self.reduced_bimanual_robot.model.lowerPositionLimit,
                self.reduced_bimanual_robot.model.upperPositionLimit,
            )
            configuration.update(clipped_q)

        assert configuration.q.shape == (14,)
        # NOTE: Collision checking is handled outside
        result = self.disassemble_qpos(configuration.q)
        return result
