import numpy as np
from dexcontrol.robot import Robot

robot = Robot()


robot.set_joint_pos(
    joint_pos={"left_arm": np.array([0.84, 0.51, 0.37, -1.30, -0.65, -0.29, -0.03])}, wait_time=10.0
)
robot.set_joint_pos(
    joint_pos={"right_arm": np.array([-0.84, -0.51, -0.37, -1.30, 0.65, 0.29, 0.03])},
    wait_time=10.0,
)
robot.set_joint_pos(joint_pos={"head": np.array([0.28, 0.0, 0.0])}, wait_time=1.0)
robot.shutdown()
