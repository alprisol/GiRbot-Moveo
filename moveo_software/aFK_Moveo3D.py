import numpy as np
import math

from typing import Union

import SpatialMath as SM

# ANALITICAL FORWARD KINEMATICS OF MOVEO MANIPULATOR


def FKa_Moveo(q: Union[np.ndarray, list], T_tool: np.ndarray = np.eye(4)):
    """
    Computes the forward kinematics of a 5-degree-of-freedom Moveo robotic manipulator.

    Args:
        q (Union[np.ndarray, list]): Joint angles (q1, q2, q3, q4, q5) of the robot.
        T_tool (np.ndarray, optional): Transformation matrix representing the tool frame
                                       relative to the last joint of the robot.
                                       Defaults to the identity matrix.

    Returns:
        np.ndarray: The 4x4 transformation matrix describing the pose (position and orientation)
                    of the end effector with respect to the base frame of the robot.
    """

    # Define robot-specific parameters (link lengths and offsets)
    d1 = 0.25  # Distance from base to the second joint along z-axis
    d4 = 0.5  # Distance from the third joint to the wrist along z-axis
    a2 = 0.1  # Length of the second link along x-axis

    # Helper function to compute the cosine of the sum of joint angles
    def c(idx_angle: Union[int, list, tuple]):
        sum_angle = 0
        if isinstance(idx_angle, int):
            idx_angle = [
                idx_angle
            ]  # If a single index is provided, convert it to a list
        for i in idx_angle:
            sum_angle += q[i - 1]  # Sum the angles from the list of indices
        return math.cos(sum_angle)  # Return the cosine of the summed angles

    # Helper function to compute the sine of the sum of joint angles
    def s(idx_angle: Union[int, list, tuple]):
        sum_angle = 0
        if isinstance(idx_angle, int):
            idx_angle = [
                idx_angle
            ]  # If a single index is provided, convert it to a list
        for i in idx_angle:
            sum_angle += q[i - 1]  # Sum the angles from the list of indices
        return math.sin(sum_angle)  # Return the sine of the summed angles

    # Compute the rotation matrix components using forward kinematic equations
    # nx, ny, nz: Components of the first column of the rotation matrix (X-axis of the end-effector)
    nx = -c(5) * (s(1) * s(4) - c([2, 3]) * c(1) * c(4)) - s([2, 3]) * c(1) * s(5)
    ny = c(5) * (c(1) * s(4) + c([2, 3]) * c(4) * s(1)) - s([2, 3]) * s(1) * s(5)
    nz = c([2, 3]) * s(5) + s([2, 3]) * c(4) * c(5)

    # ox, oy, oz: Components of the second column of the rotation matrix (Y-axis of the end-effector)
    ox = -c(4) * s(1) - c([2, 3]) * c(1) * s(4)
    oy = c(1) * c(4) - c([2, 3]) * s(1) * s(4)
    oz = -s([2, 3]) * s(4)

    # ax, ay, az: Components of the third column of the rotation matrix (Z-axis of the end-effector)
    ax = s(5) * (s(1) * s(4) - c([2, 3]) * c(1) * c(4)) - s([2, 3]) * c(1) * c(5)
    ay = -s(5) * (c(1) * s(4) + c([2, 3]) * c(4) * s(1)) - s([2, 3]) * c(5) * s(1)
    az = c([2, 3]) * c(5) - s([2, 3]) * c(4) * s(5)

    # x, y, z: Position components of the end-effector in the base frame
    x = -c(1) * (d4 * s([2, 3]) - a2 * c(1))
    y = -s(1) * (d4 * s([2, 3]) - a2 * c(2))
    z = d1 + d4 * c([2, 3]) + a2 * s(2)

    # Construct the homogeneous transformation matrix (4x4 matrix) combining rotation and translation
    T = (
        np.array(
            [
                [nx, ox, ax, x],  # Rotation matrix and translation for the x-axis
                [ny, oy, ay, y],  # Rotation matrix and translation for the y-axis
                [nz, oz, az, z],  # Rotation matrix and translation for the z-axis
                [0, 0, 0, 1],
            ]
        )
        @ T_tool  # Multiply by the tool transformation matrix to include tool offset
    )

    return T  # Return the final transformation matrix


if __name__ == "__main__":

    np.set_printoptions(precision=2, suppress=True)

    qp2 = np.array([math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4])
    qp = np.array([math.pi / 2, math.pi / 4, 0, -math.pi, -math.pi / 4])
    qv = np.array([math.pi / 2, math.pi / 2, -math.pi / 2, 0, 0])
    qz = np.array([0, 0, 0, 0, 0])

    tool = SM.GetMatFromPose([0, 0, 0.05, 0, 0, 0])

    Tfm = FKa_Moveo(qz, tool)
    print(Tfm)

    pose = np.concatenate((SM.GetXYZ(Tfm), SM.GetRPY(Tfm)))
    print(f"Pose of this Tfm: {pose}")
