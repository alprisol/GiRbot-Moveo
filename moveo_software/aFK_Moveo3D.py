import numpy as np
import math

from typing import Union

import SpatialMath as SM

# Canvio


def FKa_Moveo(q: Union[np.ndarray, list], T_tool: np.ndarray = np.eye(4)):

    d1 = 0.25
    d4 = 0.5
    a2 = 0.1

    def c(idx_angle: Union[int, list, tuple]):
        sum_angle = 0
        if isinstance(idx_angle, int):
            idx_angle = [idx_angle]
        for i in idx_angle:
            sum_angle += q[i - 1]
        return math.cos(sum_angle)

    def s(idx_angle: Union[int, list, tuple]):
        sum_angle = 0
        if isinstance(idx_angle, int):
            idx_angle = [idx_angle]
        for i in idx_angle:
            sum_angle += q[i - 1]
        return math.sin(sum_angle)

    nx = -c(5) * (s(1) * s(4) - c([2, 3]) * c(1) * c(4)) - s([2, 3]) * c(1) * s(5)
    ny = c(5) * (c(1) * s(4) + c([2, 3]) * c(4) * s(1)) - s([2, 3]) * s(1) * s(5)
    nz = c([2, 3]) * s(5) + s([2, 3]) * c(4) * c(5)

    ox = -c(4) * s(1) - c([2, 3]) * c(1) * s(4)
    oy = c(1) * c(4) - c([2, 3]) * s(1) * s(4)
    oz = -s([2, 3]) * s(4)

    ax = s(5) * (s(1) * s(4) - c([2, 3]) * c(1) * c(4)) - s([2, 3]) * c(1) * c(5)
    ay = -s(5) * (c(1) * s(4) + c([2, 3]) * c(4) * s(1)) - s([2, 3]) * c(5) * s(1)
    az = c([2, 3]) * c(5) - s([2, 3]) * c(4) * s(5)

    x = -c(1) * (d4 * s([2, 3]) - a2 * c(1))
    y = -s(1) * (d4 * s([2, 3]) - a2 * c(2))
    z = d1 + d4 * c([2, 3]) + a2 * s(2)

    T = (
        np.array([[nx, ox, ax, x], [ny, oy, ay, y], [nz, oz, az, z], [0, 0, 0, 1]])
        @ T_tool
    )

    return T


if __name__ == "__main__":

    np.set_printoptions(precision=2, suppress=True)

    qp2 = np.array([math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4, math.pi / 4])
    qp = np.array([math.pi / 2, math.pi / 4, 0, -math.pi, -math.pi / 4])
    qv = np.array([math.pi / 2, math.pi / 2, -math.pi / 2, 0, 0])
    qz = np.array([0, 0, 0, 0, 0])

    tool = SM.GetMatFromPose([0, 0, 0.05, 0, 0, 0])

    Tfm = FKa_Moveo(qv, tool)
    print(Tfm)

    pose = np.concatenate((SM.GetXYZ(Tfm), SM.GetRPY(Tfm)))
    print(f"Pose of this Tfm: {pose}")

    print(f"T. Matrix from pose: \n {SM.GetMatFromPose(pose)}")

    print(math.atan2(1, 0))
