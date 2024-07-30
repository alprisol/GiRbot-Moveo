import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from dataclasses import dataclass
from typing import List, Optional, Union

import AuxiliarFunctions as AF
import SpatialMath as SM
from PiecewiseExpr import Piece, PiecewiseFunction
from nIK_Corke import ikine_LM_Corke as IK


class InterPoint:

    def __init__(
        self,
        t_stamp: float,
        q: Optional[Union[np.ndarray, None]] = None,
        pose: Optional[Union[np.ndarray, None]] = None,
        d_start: Optional[Union[float, None]] = None,
        q_d: Optional[Union[np.ndarray, None]] = None,
    ):

        self.t_stamp = t_stamp
        self.q = q
        self.pose = pose
        self.d_start = d_start
        self.q_d = q_d


class create_LTraj:

    def __init__(
        self,
        StrtPose: np.ndarray,
        TrgtPose: np.ndarray,
        MoveVel: float,
        CartAccel: float,
        n_interp: int,
        n_dofs: int,
    ):

        self.n_dofs = n_dofs
        self.n_interp = n_interp
        self.StrtPose = StrtPose
        self.TrgtPose = TrgtPose
        self.MoveVel = MoveVel
        self.CartAccel = CartAccel

        self.CartDistance = self.calc_CartDistance(StrtPose, TrgtPose)
        self.MoveTime = self.calc_MoveTime()
        self.MaxCartVel = self.calc_MaxVelocity()
        self.CartVelocityExpr = self.create_CartTrapzVelProfExpr()
        self.CartPositionExpr = self.CartVelocityExpr.integrate()

        self.t = np.linspace(0, self.MoveTime, n_interp)
        self.d = self.calc_DistaceFromStart()
        self.pose = self.calc_InterpPoses()

    def calc_CartDistance(self, pose1, pose2):
        """
        Calculate the Euclidean distance between the starting and ending position coordinates using NumPy arrays.
        """
        point1 = np.array(pose1[:3])
        point2 = np.array(pose2[:3])

        dis = np.linalg.norm(point1 - point2)

        return dis

    def calc_MoveTime(self):
        """
        Calculate the t_total required to travel from the start point to the end point with the set velocity.
        """
        if self.MoveVel <= 0:
            raise ValueError("Speed must be greater than zero.")

        t_time = self.CartDistance / self.MoveVel

        return t_time

    def calc_MaxVelocity(self):
        """
        Calculate the Maximum Velocity to then construct the Trapezoidal or Triangular profile.
        """
        t_total = self.MoveTime
        accel = self.CartAccel
        distance = self.CartDistance

        if distance == 0:
            maxVel = 0

        else:

            if accel < (4 * distance / (t_total**2)):

                accel = 4 * distance / (t_total**2)
                print(
                    f" *Given acceleration is not sufficient to complete the movement in t_total, new acceleration set at {accel}"
                )

            discriminant = ((t_total * accel) ** 2) - 4 * (distance * accel)
            discriminant = 0 if AF.floats_equal(0.0, discriminant) else discriminant

            maxVel = ((t_total * accel) - math.sqrt(discriminant)) / 2

        return maxVel

    def create_CartTrapzVelProfExpr(self):

        distance = self.CartDistance
        t_total = self.MoveTime
        accel = self.CartAccel
        vel = self.MaxCartVel

        t_acc = t_dec = abs(vel / accel)

        t = sp.Symbol("t")

        if t_acc == 0 or distance == 0:
            # No movement
            velProfile = [
                Piece(
                    0,
                    sp.core.numbers.NegativeInfinity,
                    sp.core.numbers.Infinity,
                )
            ]

        elif t_acc >= t_total / 2:

            # Triangular Profile
            t_acc = t_dec = t_total / 2
            velProfile = [
                Piece(0, sp.core.numbers.NegativeInfinity, 0),
                Piece(accel * t, 0, t_acc),
                Piece(
                    vel - (accel * (t - t_acc)),
                    t_acc,
                    t_total,
                ),
                Piece(0, t_total, sp.core.numbers.Infinity),
            ]

        else:
            # Trapezoidal Profile
            t_const = t_total - 2 * t_acc

            velProfile = [
                Piece(0, sp.core.numbers.NegativeInfinity, 0),
                Piece(accel * t, 0, t_acc),
                Piece(vel, t_acc, t_acc + t_const),
                Piece(
                    vel - accel * (t - (t_acc + t_const)),
                    t_acc + t_const,
                    t_total,
                ),
                Piece(0, t_total, sp.core.numbers.Infinity),
            ]

        velProfile = PiecewiseFunction(velProfile, "t")

        self.CartTrapzVelProfExpr = velProfile

        return velProfile

    def calc_DistaceFromStart(self):

        d_list = []

        for t_stamp in self.t:

            d_list.append(self.CartPositionExpr.subs_IndepVar(t_stamp))

        return np.array(d_list)

    def calc_InterpPoses(self):

        pose_list = []
        xs, Ys, Zs, Rs, Ps, YWs = self.StrtPose
        Xe, Ye, Ze, Rt, Pt, YWt = self.TrgtPose

        # Convert RPY to quaternions
        strt_Rmat = SM.GetMatFromPose([0, 0, 0, Rs, Ps, YWs])[:3, :3]
        trgt_Rmat = SM.GetMatFromPose([0, 0, 0, Rt, Pt, YWt])[:3, :3]
        rotations = R.from_matrix([strt_Rmat, trgt_Rmat])
        slerp = Slerp([0, 1], rotations)

        for norm_d in self.d / self.CartDistance:

            norm_d = norm_d if norm_d < 1 else 1

            # Linear interpolation for position
            x = xs + norm_d * (Xe - xs)
            y = Ys + norm_d * (Ye - Ys)
            z = Zs + norm_d * (Ze - Zs)

            # Convert interpolated quaternion back to RPY
            interp_Rmat = np.array(slerp([norm_d]).as_matrix()[0], dtype=np.float64)
            rpy = SM.GetRPY(interp_Rmat)

            # Append interpolated pose to the list
            pose_list.append([x, y, z, rpy[0], rpy[1], rpy[2]])

        return np.array(pose_list, dtype=np.float64)

    def plot(self, data: Optional[np.ndarray] = None, skip=1, length=1):

        if data is None:

            data = self.pose
        else:

            data = data

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Extract XYZ coordinates and RPY angles
        X = data[:, 0]
        Y = data[:, 1]
        Z = data[:, 2]
        roll = data[:, 3]
        pitch = data[:, 4]
        yaw = data[:, 5]

        # Length of the CSYS arrwos
        length_quiver = SM.vector_norm(SM.vector(data[0, 0:3], data[-1, 0:3])) / 20

        # Plotting the XYZ points
        ax.scatter(X, Y, Z, c="r", marker="o", label="XYZ Points")

        # Plotting orientation axes for each point every 'skip' points
        colors = ["r", "limegreen", "dodgerblue"]
        for i in range(0, len(X), skip):

            axes = np.eye(3) * length_quiver
            tfm = SM.GetMatFromPose([0, 0, 0, roll[i], pitch[i], yaw[i]])
            transformed_axes = tfm[:3, :3] @ axes

            # Quiver plot for each axis
            for j in range(3):
                ax.quiver(
                    X[i],
                    Y[i],
                    Z[i],
                    transformed_axes[0, j],
                    transformed_axes[1, j],
                    transformed_axes[2, j],
                    color=colors[j],
                    linewidth=1,
                    length=length,
                )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()
        plt.show()


def rmve_Singularities(
    q: np.ndarray,
    t: np.ndarray,
    joint_ranges: Union[np.ndarray, list, tuple],
    joint_type: Union[np.ndarray, list, tuple],
    rel_tol: float = 5e-2,
):

    def relative_deviation(array1, array2, joint_ranges):

        if len(array1) != len(array2):
            raise ValueError("Arrays must be of the same length.")

        # Calculate the relative deviation using angle_distance
        relative_deviation = np.array(
            [
                abs(AF.angle_dist_in_range(angle1, angle2, j_range)) / (abs(j_range[1]-j_range[0])) 
                for angle1, angle2, j_range in zip(array1, array2, joint_ranges)
            ]
        )

        return relative_deviation

    q_new = [q[0]]  # Initialize with the first position
    t_new = [t[0]]  # Initialize with the first time
    inc_t = t[1] - t[0]  # Time increment
    n_interp_points = 0

    for i in range(len(q) - 1):
        q_curr = q[i]
        q_next = q[i + 1]
        t_curr = (i + n_interp_points) * inc_t
        rel_error = relative_deviation(q_curr, q_next,joint_ranges)

        if np.any(rel_error > rel_tol):

            print("SINGULARITY FOUND AT:")
            print(f"Current q: {q_curr}")
            print(f"Next q: {q_next}")
            print(f"Relative Deviation: {rel_error}")
            print(f'This may lead to problems in the trajectory execution')
            print()

            max_rel_error = np.max(rel_error)
            steps = max(5, int(np.ceil(max_rel_error * 50)))
            n_interp_points += steps

            interp_t_start = t_new[-1]
            interpolated_times = np.linspace(
                interp_t_start, interp_t_start + steps * inc_t, steps, endpoint=False
            )
            interpolated_qs = AF.interpolate_q_in_range(
                q_curr, q_next, joint_type, joint_ranges, steps,
            )

            q_new.extend(
                interpolated_qs[1:]
            )  # Append interpolated values, excluding the first which is already included
            t_new.extend(interpolated_times[1:])

        else:
            # Append next values only if not already added by interpolation
            if t[i + 1] != t_new[-1]:
                q_new.append(q_next)
                t_new.append(t_curr)

    return np.array(q_new), np.array(t_new)


if __name__ == "__main__":

    np.set_printoptions(precision=2, suppress=True)

    Traj1 = create_LTraj(
        n_dofs=6,
        n_interp=100,
        StrtPose=np.array([0, 0, 0, 0, 0, 0]),
        TrgtPose=np.array([10, 10, 10, -2.97, -0.52, -2.19]),
        MoveVel=2,
        CartAccel=1,
    )
