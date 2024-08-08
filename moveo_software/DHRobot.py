import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import get_cmap

import AuxiliarFunctions as AF
import SpatialMath as SM
from nIK_Corke import ikine_LM_Corke
import LinealTrajectory as LTraj

from typing import List, Union, Callable, Optional


class RobotTool:
    def __init__(self, tXYZ: list, rXYZ: list):
        """
        Initialize the tool values for a robot's tool.

        Parameters:
        - tXYZ (list): Translation values [x, y, z].
        - rXYZ (list): Rotation angles [rX, rY, rZ].

        Raises:
        - ValueError: If the length of tXYZ or rXYZ is not exactly three.
        """
        self.tXYZ = tXYZ
        self.rXYZ = rXYZ
        self.rel_pose = tXYZ + rXYZ

        # Validate that tXYZ and rXYZ have exactly three elements each
        if len(self.tXYZ) != 3:
            raise ValueError("tXYZ must have exactly three elements")
        if len(self.rXYZ) != 3:
            raise ValueError("rXYZ must have exactly three elements")

        self.Tfm = self.get_Tfm()

    def get_Tfm(self):

        x = self.tXYZ[0]
        y = self.tXYZ[1]
        z = self.tXYZ[2]

        rX = self.rXYZ[0]
        rY = self.rXYZ[1]
        rZ = self.rXYZ[2]

        T = SM.GetMatFromPose(self.rel_pose)

        return T


class JointDH:

    def __init__(
        self, theta: float, d: float, a: float, alpha: float, qlim: list, j_type: int
    ):
        """
        Initialize the DH parameters for a robotic joint.

        Parameters:
        - theta (float): The joint angle (in radians or degrees).
        - d (float): The offset along the previous z to the common normal.
        - a (float): The length of the common normal (i.e., the offset along x).
        - alpha (float): The angle about the common normal (i.e., the offset in z).
        - qlim (list): The joint limits [min, max].
        - j_type (int): Joint type. 0 for Revolute, 1 for Prismatic
        """

        self.theta = theta
        self.d = d
        self.a = a
        self.alpha = alpha
        self.j_type = j_type

        self.qlim = []

        for bound in qlim:

            if abs(abs(bound) - math.pi) > 1e-5:

                self.qlim.append(
                    bound * 0.9999999999
                )  # Avoid having exactly pi, as it brings problems.

            else:

                self.qlim.append(bound)

    def get_Tfm(self):
        """
        Builds the transformation matrix for the joint
        """
        theta = self.theta
        d = self.d
        a = self.a
        alpha = self.alpha

        Rz = SM.RotZ(theta)  # Rotation about z-axis by theta
        Tz = SM.Trans(0, 0, d)  # Translation along z-axis by d
        Tx = SM.Trans(a, 0, 0)  # Translation along x-axis by a
        Rx = SM.RotX(alpha)  # Rotation about x-axis by alpha

        T = Rz @ Tz @ Tx @ Rx

        return T

    def __str__(self):
        return f"DH Parameters: theta={self.theta}, d={self.d}, a={self.a}, alpha={self.alpha}, qlim={self.qlim}"


class RevoluteDH(JointDH):

    def __init__(
        self,
        home: float,
        d: float,
        a: float,
        alpha: float,
        qlim: list = [-math.pi * 0.999, math.pi * 0.999],
    ):
        """
        Initialize a revolute joint with specific Denavit-Hartenberg parameters.

        Parameters:
        - home (float): The home position angle for the joint (typically the angle when the robot is in its standard position).
        - d (float): The offset along the previous z to the common normal.
        - a (float): The length of the common normal (i.e., the offset along x).
        - alpha (float): The angle about the common normal (i.e., the offset in z).
        - qlim (list): The range of motion limits for the joint, defaulting to full rotation.
        """
        super().__init__(theta=home, d=d, a=a, alpha=alpha, qlim=qlim, j_type=0)


class PrismaticDH(JointDH):

    def __init__(
        self,
        theta: float,
        home: float,
        a: float,
        alpha: float,
        qlim: list = [0, 1],
    ):
        """
        Initialize a prismatic joint with specific Denavit-Hartenberg parameters.

        Parameters:
        - theta (float): The joint angle (typically this might be fixed for a prismatic joint).
        - home (float): The home position offset along the previous z-axis (substitutes 'd').
        - a (float): The length of the common normal (i.e., the offset along x).
        - alpha (float): The angle about the common normal (i.e., the offset in z).
        - qlim (list): The range of motion limits for the joint.
        """
        super().__init__(theta=theta, d=home, a=a, alpha=alpha, qlim=qlim, j_type=1)


class DHRobot:

    def __init__(
        self,
        joint_list: List[Union[JointDH, RevoluteDH, PrismaticDH]],
        IK_solver: Callable,
        tool: Optional[RobotTool] = None,
        DH_name: str = "RobotDH",
    ):
        """
        Initialize the DH parameters for a robotic arm.

        Parameters:
        - joint_list (list): A list of joint objects which can be either JointDH, RevoluteDH, or PrismaticDH.
        - tool (RobotTool): A tool object or a 4x4 identity matrix as the default tool configuration.
        - DH_name (string): Name of the Robot Defined
        """
        self.joint_list = joint_list

        if tool is None:
            self.tool = RobotTool([0, 0, 0], [0, 0, 0])
        else:
            self.tool = tool

        self.DH_name = DH_name
        self.IK_solver = IK_solver

        self.n_joint = len(joint_list)
        self.home_q = self.get_JointValues()

    def get_PrismaticJoints(self):
        """
        Returns  list of booleans as size n_joint where the i_th position indicates
        if the i join is prismatic (True) or angular (False).
        """
        listPrism = []
        for j in self.joint_list:
            if j.j_type == 1:
                listPrism.append(True)
            else:
                listPrism.append(False)

        return listPrism

    def get_JointRanges(self):
        """
        Return a list with tuples containing the lower and upper limit of each joint.
        """
        limits = []
        for j in self.joint_list:
            limits.append(j.qlim)

        return limits

    def check_JointInsideRange(self, check_val=None, idx=None):
        """
        Checks if the specified joint values are within their allowable ranges.
        If no values or indices are provided, it checks all current joint positions.

        Parameters:
        - check_val (None, float, list): The joint value(s) to check. Can be a single value, a list of values, or None.
        - idx (None, int, list): The joint index(indices) corresponding to the values to be checked. Can be a single index, a list of indices, or None.

        Raises:
        - ValueError: If any of the provided or current joint values are out of the allowed range.
        - TypeError: If parameters check_val and idx are lists of mismatched lengths, or if one is a list and the other is not.

        Returns:
        - True
        """

        def check_JointValue(val, idx):
            joint = self.joint_list[idx]
            qlim = joint.qlim

            if joint.j_type == 0:
                val = AF.wrap_angle(val)
                in_range = AF.check_angle_in_range(val, qlim)
            else:
                in_range = AF.check_linear_in_range(val, qlim)

            if not in_range:
                raise ValueError(
                    f"Joint value {val} at index {idx} is out of range {qlim}."
                )

            return in_range

        if check_val is not None and idx is not None:

            if isinstance(check_val, list) and isinstance(idx, list):

                if len(check_val) != len(idx):
                    raise ValueError(
                        "'check_val' and 'idx' lists must have the same length."
                    )
                for val, idx in zip(check_val, idx):
                    check_JointValue(val, idx)

            elif isinstance(check_val, list) or isinstance(idx, list):

                raise TypeError(
                    "Both 'check_val' and 'idx' need to be lists or both need to be single values."
                )
            else:
                check_JointValue(check_val, idx)

        elif check_val is not None and isinstance(check_val, list):

            if len(check_val) != len(self.joint_list):
                raise ValueError(
                    "Length of 'check_val' list must match number of joints."
                )
            for val, joint in zip(check_val, self.joint_list):
                check_JointValue(val, self.joint_list.index(joint))

        elif check_val is None:

            for idx, joint in enumerate(self.joint_list):
                current_val = joint.theta if joint.j_type == 0 else joint.d
                check_JointValue(current_val, idx)

        return True

    def check_Collisions(self):

        tfms = [np.eye(4)]
        pos = []

        for j in self.joint_list:

            new_tfm = tfms[-1] @ j.get_Tfm()
            tfms.append(new_tfm)
            pos.append(SM.GetXYZ(tfms[-1]))

        for i, jp in enumerate(pos):

            if jp[2] < 0.05:

                raise ValueError(
                    f"Collision with the floor. Position {i} with values {jp}"
                )

        return False

    def set_JointValues(self, q_values):
        """
        Sets the angular position (theta) or the prismatic position (d) for each joint.
        """
        if len(q_values) != self.n_joint:
            raise ValueError("The q_values list lenght does not match de DoF")

        else:

            for i, j in enumerate(self.joint_list):

                if j.j_type == 0:

                    new_value = AF.wrap_angle(q_values[i])
                    self.check_JointInsideRange(new_value, i)
                    j.theta = new_value

                else:

                    new_value = q_values[i]
                    self.check_JointInsideRange(new_value, i)
                    j.d = new_value

        self.check_Collisions()

    def get_JointValues(self):
        """
        Return the value for each joint. This values will be stores in dh_params THETA or
        D, depending on the joint nature.
        """
        j_values = np.zeros(self.n_joint)

        for i, j in enumerate(self.joint_list):

            if j.j_type == 0:
                j_values[i] = j.theta
            else:
                j_values[i] = j.d

        return j_values

    def get_Tfm_UpTo_Joint(self, i):
        """
        Computes the matrix ^{0}A_{i} that relates the extrem of joint i with the base.
        """

        iTz = np.identity(4)

        for j in self.joint_list[:i]:

            iT = j.get_Tfm()
            iTz = iTz @ iT

        return iTz

    def get_RobotTfm(self):
        """
        Composes all the transformation matrix for joint 0 to n-1 accordint to DH
        """
        rT = self.get_Tfm_UpTo_Joint(self.n_joint) @ self.tool.Tfm

        return rT

    def calc_Reach(self):

        reach = 0

        for j in self.joint_list:

            reach += j.d + j.a

        return reach

    def get_EndEffPosOr(self, T: Optional[np.ndarray] = None):

        if T is None:
            T = self.get_RobotTfm()

        return np.concatenate((SM.GetXYZ(T), SM.GetRPY(T)))

    def get_Jacobian(self):
        """
        Computes the geometric Jacobian matrix evaluated at the current position
        """
        PrismJoints = self.get_PrismaticJoints()

        rT = self.get_RobotTfm()
        rT_pos = rT[0:3, 3]

        J = np.zeros((6, self.n_joint))

        iTz = np.identity(4)

        for i in range(self.n_joint):

            iTz = self.get_Tfm_UpTo_Joint(i)

            iTz_a = iTz[0:3, 2]

            if PrismJoints[i]:

                J[0:3, i] = iTz_a

            else:

                iTz_pos = iTz[0:3, 3]

                J[0:3, i] = np.cross(iTz_a, (rT_pos - iTz_pos))
                J[3:6, i] = iTz_a

        return J

    def get_invJacobian(self):

        J = self.get_Jacobian()
        # Check if the Jacobian is square
        rows, cols = J.shape
        if rows == cols:

            # Square Jacobian: calculate the inverse step by step
            # Start with the identity matrix
            I = np.eye(rows)

            # Augment the Jacobian with the identity matrix
            augJ = np.hstack((J, I))

            # Apply Gaussian elimination to transform the left side to the identity matrix
            for i in range(rows):

                # Make the diagonal element 1 by dividing the row by the diagonal element
                diagElmt = augJ[i, i]

                if diagElmt == 0:
                    raise ValueError("Jacobian is singular and cannot be inverted.")

                augJ[i] = augJ[i] / diagElmt

                # Make the other elements in the current column 0
                for j in range(rows):
                    if i != j:
                        factor = augJ[j, i]

                        augJ[j] = augJ[j] - factor * augJ[i]

            # Extract the inverse matrix from the augmented matrix
            invJ = augJ[:, rows:]

            return invJ

        else:
            # Non-square matrix: calculate the Moore-Penrose pseudoinverse step by step
            # Calculate the pseudoinverse using SVD
            U, S_val, Vt = np.linalg.svd(J)

            S = np.zeros((rows, cols))
            invS = np.zeros((cols, rows))

            # Create the Sigma and Inv.Sigma matrices from the singular values S.
            for i, e in enumerate(S_val):

                S[i, i] = e
                invS[i, i] = 1 / e

            # Calculate the pseudoinverse
            pinvJ = Vt.T @ invS @ U.T

            return pinvJ

    def calc_FK(self, theta):

        self.set_JointValues(theta)

        return self.get_EndEffPosOr()

    def gen_samples(self, n_samples=1000):
        """
        Generates pose of joint configurations and corresponding end-effector positions.

        Args:
            n_samples (int): Number of pose to generate.

        Returns:
            tuple: A tuple containing two numpy arrays. The first array (`q_values`) stores the joint
                configurations, and the second array (`pose`) stores the corresponding end-effector positions.
        """
        q_values_list = []
        pose_list = []

        joint_limits = self.get_JointRanges()

        print(f"Generating {n_samples} samples for {self.DH_name} ...")

        # Correct joint limits if min is greater than max
        corrected_joint_limits = []
        for low, high in joint_limits:
            if low > high:
                high += 2 * np.pi
            corrected_joint_limits.append((low, high))

        for _ in range(n_samples):
            # Generate random joint values within the specified limits
            theta = []
            for low, high in joint_limits:
                if low > high:
                    # Generate value from low to 2π or -2π to high
                    value = (
                        np.random.uniform(low, 2 * np.pi)
                        if np.random.rand() < 0.5
                        else np.random.uniform(-2 * np.pi, high)
                    )
                else:
                    # Generate value from low to high
                    value = np.random.uniform(low, high)
                theta.append(value)

            theta = np.array(theta)

            # Compute the end-effector position for the given joint configuration
            try:
                pos_orientation = self.calc_FK(theta)

                # Store joint values and the corresponding end-effector position
                q_values_list.append(theta)
                pose_list.append(pos_orientation)

            except ValueError:

                pass

        # Convert lists to numpy arrays
        q_values = np.array(q_values_list)
        pose = np.array(pose_list)

        self.samples = q_values, pose

        return q_values, pose

    def calc_IK(
        self,
        trgt_poses: Union[np.ndarray, list],
        q0: Optional[np.ndarray] = None,
        q_lim: Optional[Union[np.ndarray, bool]] = False,
        IK_solver: Optional[Callable] = None,
        mask: Optional[Union[np.ndarray, list, bool]] = False,
    ):
        if isinstance(trgt_poses, list):
            trgt_poses = np.array(trgt_poses)

        if IK_solver is None:
            IK_solver = self.IK_solver

        if q_lim is True:
            q_lim = self.get_JointRanges()

        return IK_solver(self, trgt_poses=trgt_poses, q0=q0, q_lim=q_lim, mask=mask)

    def calc_DerivatedArray(self, q, t):
        # Number of joints and time-stamps
        M = q.shape[0]
        N = q.shape[1]

        # Initialize velocity array
        qd = np.zeros_like(q)

        # Calculate time steps (assuming uniform time steps)
        dt = np.diff(t)

        # Get prismatic joints information
        prismatic_joints = self.get_PrismaticJoints()

        # Get joint ranges
        joint_ranges = self.get_JointRanges()

        # Forward difference for the first time-stamp
        for j in range(N):
            is_linear = prismatic_joints[j]
            valid_range = joint_ranges[j]

            qd[0, j] = (
                -1
                * AF.calc_dist_in_range(q[1, j], q[0, j], is_linear, valid_range)
                / dt[0]
            )
        print()

        # Central difference for the intermediate points
        for i in range(1, M - 1):

            for j in range(N):
                is_linear = prismatic_joints[j]
                valid_range = joint_ranges[j]

                qd[i, j] = (
                    -1
                    * AF.calc_dist_in_range(
                        q[i + 1, j], q[i - 1, j], is_linear, valid_range
                    )
                    / (dt[i] + dt[i - 1])
                )

            # Backward difference for the last time-stamp
        for j in range(N):
            is_linear = prismatic_joints[j]
            valid_range = joint_ranges[j]

            qd[-1, j] = (
                -1
                * AF.calc_dist_in_range(q[-1, j], q[-2, j], is_linear, valid_range)
                / dt[-1]
            )

        # Setting the first and the last velocities to zero as specified
        qd[0, :] = 0
        qd[-1, :] = 0

        return qd

    def plot_RobotStatic(self, q: Optional[Union[np.ndarray, list]] = None):
        """
        Plots the configuration of the robotic arm, including the coordinate axes at the last joint/tool.
        """
        curr_q = self.get_JointValues()

        if q is None:
            q = curr_q

        elif q is not None:
            q = q

        else:
            raise ValueError(
                "Function <plot_RobotStatic> only supports single configurations. For multiples input configurations use <plot_RobotMovement>"
            )

        self.set_JointValues(q)

        # Initialize the base position and orientation
        pos = [np.array([0, 0, 0])]
        ori = [np.array([0, 0, 0])]
        tfms = [np.identity(4)]

        # Compute transformations for each joint
        for joint in self.joint_list:
            tfms.append(tfms[-1] @ joint.get_Tfm())
            pos.append(SM.GetXYZ(tfms[-1]))
            ori.append(SM.GetRPY(tfms[-1]))

        # Include the tool transformation
        tool_tfm = tfms[-1] @ self.tool.Tfm
        pos.append(tool_tfm[:3, 3])
        tfms.append(tool_tfm)

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", facecolor="white")
        Lx, Ly, Lz = zip(*pos)
        ax.plot(Lx, Ly, Lz, marker="o", ls="-", lw=5, c="tomato", ms=2)

        length = SM.vector_norm(SM.vector(pos[0], pos[-1]))

        colors = ["r", "limegreen", "dodgerblue"]

        # Plotting Z-axes for each joint and all axes for the tool
        for i in range(len(tfms)):

            axes = np.eye(3) * (length / 15)
            transformed_axes = tfms[i][:3, :3] @ axes

            if i == len(tfms) - 1:

                for j in range(3):

                    ax.quiver(
                        *pos[i],
                        *(transformed_axes[:, j] * 1.5),
                        color=colors[j],
                        linewidth=3,
                    )

            elif i < (len(tfms) - 2):

                ax.quiver(
                    *pos[i],
                    *transformed_axes[:, 2],
                    color="deepskyblue",
                    linewidth=1.5,
                )
            else:
                pass

        # Grid and pane settings
        ax.grid(True, linestyle="-", color="whitesmoke", linewidth=0.5)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim([-length, length])
        ax.set_ylim([-length, length])
        ax.set_zlim([-length / 10, length])

        plt.suptitle(
            f"{self.DH_name} Configuration",
            fontsize=18,
            fontweight="bold",
        )
        plt.title(
            f"With Joints at: {np.round(self.get_JointValues(), 2)}",
            fontsize=8,
            color="dimgrey",
        )

        plt.tight_layout()
        plt.show()

        self.set_JointValues(curr_q)

    def plot_RobotMovement(
        self,
        q: Optional[np.ndarray] = None,
        movie_name: str = None,
        movie_time: float = 5.0,
        movie_resolution: int = 150,
    ):
        """
        Plots the configuration of the robotic arm, including the coordinate axes at the last joint/tool.
        """

        length = self.calc_Reach()
        curr_q = self.get_JointValues()

        if q is None:
            q = [curr_q]
        elif q.ndim == 1:

            q = [q]

        # Setup plot
        fig = plt.figure()
        plt.suptitle(f"{self.DH_name} Configuration", fontsize=18, fontweight="bold")
        plt.tight_layout()
        ax = fig.add_subplot(111, projection="3d", facecolor="white")

        ax.set_xlim([-length, length])
        ax.set_ylim([-length, length])
        ax.set_zlim([-length / 10, length])

        # Joints Points
        x, y, z = [], [], []
        (joint_points,) = ax.plot(x, y, z, "-", color="tomato", lw=4)
        traj_x, traj_y, traj_z = [], [], []
        (traj_points,) = ax.plot(
            traj_x, traj_y, traj_z, "-", color="darkorchid", lw=1, alpha=0.8
        )

        # Coordinate system axis
        xCSYS_EF = ax.quiver([], [], [], [], [], [], color="r", linewidth=2)
        yCSYS_EF = ax.quiver([], [], [], [], [], [], color="lime", linewidth=2)
        zCSYS_EF = ax.quiver([], [], [], [], [], [], color="dodgerblue", linewidth=2)
        zAxis_j = ax.quiver([], [], [], [], [], [], color="deepskyblue", linewidth=1.5)

        def single_plot(frame, q):

            self.set_JointValues(q[frame])
            tfms = [np.identity(4)]
            pos = [np.array([0, 0, 0])]

            for j in self.joint_list:
                new_tfm = tfms[-1] @ j.get_Tfm()
                tfms.append(new_tfm)
                pos.append(SM.GetXYZ(tfms[-1]))

            tool_tfm = tfms[-1] @ self.tool.Tfm
            tfms.append(tool_tfm)
            pos.append(SM.GetXYZ(tfms[-1]))

            # Update joint points and Trajectory Points
            Lx, Ly, Lz = zip(*pos)
            joint_points.set_data_3d(Lx, Ly, Lz)
            traj_x.append(Lx[-1])
            traj_y.append(Ly[-1])
            traj_z.append(Lz[-1])
            traj_points.set_data_3d(traj_x, traj_y, traj_z)

            # Update axes
            seg_xCSYS_EF, seg_yCSYS_EF, seg_zCSYS_EF, seg_zAxis_j = [], [], [], []

            for i, tfm in enumerate(tfms):

                axes = np.eye(3) * (length / 15)
                transformed_axes = tfm[:3, :3] @ axes

                for j in range(3):
                    v = transformed_axes[:, j]

                    if i == len(tfms) - 1:
                        if j == 0:
                            seg_xCSYS_EF.insert(
                                0,
                                [
                                    [Lx[i], Ly[i], Lz[i]],
                                    [
                                        Lx[i] + v[0] * 1.5,
                                        Ly[i] + v[1] * 1.5,
                                        Lz[i] + v[2] * 1.5,
                                    ],
                                ],
                            )
                        elif j == 1:
                            seg_yCSYS_EF.insert(
                                0,
                                [
                                    [Lx[i], Ly[i], Lz[i]],
                                    [
                                        Lx[i] + v[0] * 2,
                                        Ly[i] + v[1] * 2,
                                        Lz[i] + v[2] * 2,
                                    ],
                                ],
                            )
                        elif j == 2:
                            seg_zCSYS_EF.insert(
                                0,
                                [
                                    [Lx[i], Ly[i], Lz[i]],
                                    [
                                        Lx[i] + v[0] * 2,
                                        Ly[i] + v[1] * 2,
                                        Lz[i] + v[2] * 2,
                                    ],
                                ],
                            )

                    elif i < (len(tfms) - 2):

                        if j == 2:
                            seg_zAxis_j.insert(
                                0,
                                [
                                    [Lx[i], Ly[i], Lz[i]],
                                    [Lx[i] + v[0], Ly[i] + v[1], Lz[i] + v[2]],
                                ],
                            )

                    else:

                        pass

            xCSYS_EF.set_segments(seg_xCSYS_EF)
            yCSYS_EF.set_segments(seg_yCSYS_EF)
            zCSYS_EF.set_segments(seg_zCSYS_EF)
            zAxis_j.set_segments(seg_zAxis_j)

            return xCSYS_EF, yCSYS_EF, zCSYS_EF, zAxis_j, joint_points, traj_points

        ani = FuncAnimation(
            fig, single_plot, frames=len(q), fargs=(q,), blit=True, repeat=False
        )

        if movie_name:
            print("Creating GIF ...")

            ani.save(
                f"{movie_name}.gif",
                writer=PillowWriter(fps=len(q) / movie_time),
                dpi=movie_resolution,
            )

            print("GIF saved at current directory \n")

        # else:

        plt.show()

        self.set_JointValues(curr_q)

    def plot_JointEvolution(self, q, qd, t):

        # Number of joints
        N = q.shape[1]

        # Create figure and two subplots for shared axes: one for position, one for velocity
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), facecolor="white")

        # Set a super title and subtitle for the entire figure
        fig.suptitle("Joint Kinematics Evolution", fontsize=16, fontweight="bold")

        # Titles for each plot
        ax1.set_title("Position")
        ax2.set_title("Velocity")

        # Define the colormap
        cmap = get_cmap("viridis")
        colors = [cmap(i / N) for i in range(N)]  # Generate colors for each joint

        labels = []
        # Plot all joints on the same axes for positions
        for i in range(N):
            ax1.plot(t, q[:, i], label=f"Joint {i+1}", linewidth=2, color=colors[i])
            labels.append(f"Joint {i+1}")
        ax1.set_ylabel("Position (rad)")
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax1.set_facecolor("whitesmoke")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
        ax1.set_ylim([-math.pi, math.pi])  # Set y-axis range from 0 to 2*pi

        # Plot all joints on the same axes for velocities
        for i in range(N):
            ax2.plot(t, qd[:, i], linewidth=2, color=colors[i])
        ax2.set_ylabel("Velocity(rad/s))")
        ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax2.set_facecolor("whitesmoke")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}"))
        ax2.set_xlabel("Time (s)")

        # Add a single legend
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=N,
            frameon=False,
        )

        # Adjust layout to prevent overlap and ensure visibility
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def __str__(self):
        table_str = "\n"
        table_str += f"DH {self.DH_name} \n"
        table_str += f"(home q: {self.home_q}) \n"
        table_str += "-" * 64 + "\n"
        table_str += (
            "Joint |  theta  |    d    |    a    |  alpha  |      qlim      |\n"
        )
        table_str += "-" * 64 + "\n"
        DH_table_format = "{:<5} | {:^7} | {:^7} | {:^7} | {:^7} | [{:^5} ,{:^5}] |\n"
        for idx, joint in enumerate(self.joint_list, 1):
            theta = (
                "{:.2f}".format(joint.theta)
                if joint.j_type == 1
                else "q{:.0f}".format(idx)
            )
            d = "{:.2f}".format(joint.d) if joint.j_type == 0 else "q{:.0f}".format(idx)
            a = "{:.2f}".format(joint.a)
            alpha = "{:.2f}".format(joint.alpha)
            qlim_min = "{:.2f}".format(joint.qlim[0])
            qlim_max = "{:.2f}".format(joint.qlim[1])
            table_str += DH_table_format.format(
                idx, theta, d, a, alpha, qlim_min, qlim_max
            )
        table_str += "-" * 64 + "\n"

        tXYZ = ", ".join("{:.2f}".format(t) for t in self.tool.tXYZ)
        rXYZ = ", ".join("{:.2f}°".format(r) for r in self.tool.rXYZ)
        tool_info = f"tool | xyz = ({tXYZ}) | rpy = ({rXYZ})\n"
        table_str += "-" * 61 + "\n"
        table_str += tool_info
        table_str += "-" * 61 + "\n"

        return table_str


if __name__ == "__main__":

    np.set_printoptions(precision=2, suppress=True)

    d1 = 0.15275  # m
    a2 = 0.22112  # m
    d4 = 0.223  # m
    dt = 0.09  # m

    joints = [
        RevoluteDH(
            home=math.pi / 2,
            d=d1,
            a=0,
            alpha=math.pi / 2,
            qlim=[-math.pi, math.pi],
        ),
        RevoluteDH(
            home=math.pi / 2,
            d=0,
            a=a2,
            alpha=0,
            qlim=[-math.pi / 12, -11 * math.pi / 12],
        ),
        RevoluteDH(
            home=-math.pi / 2,
            d=0,
            a=0,
            alpha=-math.pi / 2,
            qlim=[11 * math.pi / 12, math.pi / 12],
        ),
        RevoluteDH(
            home=0,
            d=d4,
            a=0,
            alpha=math.pi / 2,
            qlim=[-math.pi, math.pi],
        ),
        RevoluteDH(
            home=0,
            d=0,
            a=0,
            alpha=-math.pi / 2,
            qlim=[-math.pi / 2, math.pi / 2],
        ),
    ]

    tool = RobotTool([0, 0, dt], [0, 0, 0])

    robot = DHRobot(
        DH_name="Prova",
        joint_list=joints,
        tool=tool,
        IK_solver=ikine_LM_Corke,
    )

    print(robot)

    # robot.qz = np.array([0, 0, 0, 0, 0])
    # robot.qv = np.array([math.pi / 2, math.pi / 2, -math.pi / 2, 0, 0])
    # robot.qp = np.array([math.pi / 2, math.pi / 4, 0, -math.pi, math.pi / 4])

    # robot.q1 = np.array([math.pi / 2, math.pi / 4, 0, -math.pi, math.pi / 4])
    # robot.q2 = np.array([-math.pi, -math.pi / 13, -math.pi / 4, -math.pi, -math.pi / 4])

    # start_q = robot.qz
    # end_q = robot.qv

    # robot.plot_RobotStatic(start_q)
    # robot.plot_RobotStatic(end_q)

    # start_pose = robot.calc_FK(start_q)
    # end_pose = robot.calc_FK(end_q)

    # # robot.plot_RobotStatic(robot.calc_IK(start_pose, mask=[1, 1, 1, 1, 1, 1]))
    # # robot.plot_RobotStatic(robot.calc_IK(end_pose, mask=[1, 1, 1, 1, 1, 1]))

    # traj = LTraj.create_LTraj(
    #     StrtPose=start_pose,
    #     TrgtPose=end_pose,
    #     MoveVel=0.01,
    #     CartAccel=0.005,
    #     n_interp=100,
    #     n_dofs=robot.n_joint,
    # )

    # traj_name = "LinealProva22_3"

    # n_inertp = len(traj.t)

    # traj.q = robot.calc_IK(
    #     traj.pose,
    #     q0=start_q,
    #     q_lim=True,
    #     mask=[1, 1, 1, 0, 0, 1],
    # )

    # traj.q, traj.t = LTraj.rmve_Singularities(traj.q, traj.t, robot.get_JointRanges())

    # if len(traj.t) == n_inertp:
    #     print("No Singularities were found")
    # else:
    #     print(f"After Treating the Singularities, there are {len(traj.t)} positions.")
    traj_name = (
        "[0.22, 0.0, 0.47, 0.0, -0.0, -0.0]-to-[0.0, 0.22, 0.47, 1.57, -0.0, 0.0]"
    )

    traj_q = np.load("LTraj_files/" + traj_name + "_q.npy")
    traj_qd = np.load("LTraj_files/" + traj_name + "_qd.npy")
    traj_t = np.load("LTraj_files/" + traj_name + "_t.npy")

    robot.plot_RobotMovement(
        traj_q, movie_name="LTraj_files/" + traj_name, movie_time=10
    )

    # traj.qd = robot.calc_DerivatedArray(traj.q, traj.t)

    # robot.plot_JointEvolution(traj.q, traj.qd, traj.t)

    # np.save("LTraj_files/" + traj_name + "_q", traj.q)
    # np.save("LTraj_files/" + traj_name + "_t", traj.t)
