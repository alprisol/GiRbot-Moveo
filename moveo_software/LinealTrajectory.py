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


class LinealTrajectory:
    """
    This class defines a linear motion trajectory between two cartesian poses of a robot
    manipulator using a Cartesian trapezoidal velocity profile.

    Parameters:
        StrtPose (np.ndarray): Starting Cartesian pose.
        TrgtPose (np.ndarray): Target Cartesian pose.
        MoveVel (float): Desired movement velocity.
        CartAccel (float): Desired Cartesian acceleration.
        n_interp (int): Number of interpolation steps for trajectory calculation.
        n_dofs (int): Number of degrees of freedom of the system.
    """

    def __init__(
        self,
        StrtPose: np.ndarray,  # Starting pose of the system, provided as a numpy array.
        TrgtPose: np.ndarray,  # Target pose to reach, provided as a numpy array.
        MoveVel: float,  # Movement velocity in Cartesian space.
        CartAccel: float,  # Cartesian acceleration.
        n_interp: int,  # Number of interpolation points for the trajectory.
        n_dofs: int,  # Number of degrees of freedom of the system.
    ):
        """
        Initializes a LinealTrajectory object which defines a linear motion
        trajectory between two Cartesian poses, using a trapezoidal velocity profile.
        """

        # Store input parameters.
        self.n_dofs = n_dofs  # Number of degrees of freedom (e.g., robot arm joints).
        self.n_interp = n_interp  # Number of interpolation points for trajectory.
        self.StrtPose = StrtPose  # Starting pose in Cartesian coordinates.
        self.TrgtPose = TrgtPose  # Target pose in Cartesian coordinates.
        self.MoveVel = MoveVel  # Desired velocity of movement.
        self.CartAccel = CartAccel  # Desired acceleration.

        # Calculate the total Cartesian distance between the start and target poses.
        self.CartDistance = self.calc_CartDistance(StrtPose, TrgtPose)

        # Calculate the total movement time based on the velocity and distance.
        self.MoveTime = self.calc_MoveTime()

        # Compute the maximum velocity based on the trapezoidal velocity profile.
        self.MaxCartVel = self.calc_MaxVelocity()

        # Create the trapezoidal velocity profile expression for the Cartesian movement.
        self.CartVelocityExpr = self.create_CartTrapzVelProfExpr()

        # Integrate the velocity profile to get the position expression over time.
        self.CartPositionExpr = self.CartVelocityExpr.integrate()

        # Create an array of time steps for interpolation, ranging from 0 to the total move time.
        self.t = np.linspace(0, self.MoveTime, n_interp)

        # Calculate the distance traveled from the start at each time step.
        self.d = self.calc_DistaceFromStart()

        # Interpolate the poses based on the distance traveled and time steps.
        self.pose = self.calc_InterpPoses()

    @classmethod
    def create(cls, StrtPose, TrgtPose, MoveVel, CartAccel, n_interp, n_dofs):
        """
        Class method to create an instance of LinealTrajectory.

        This method provides an alternate way to initialize the LinealTrajectory
        object by calling the class constructor with the specified parameters.

        Parameters:
        StrtPose (np.ndarray): Starting Cartesian pose.
        TrgtPose (np.ndarray): Target Cartesian pose.
        MoveVel (float): Desired movement velocity.
        CartAccel (float): Desired Cartesian acceleration.
        n_interp (int): Number of interpolation steps for trajectory calculation.
        n_dofs (int): Number of degrees of freedom of the system.

        Returns:
        LinealTrajectory: A new instance of LinealTrajectory initialized with
        the provided parameters.
        """
        return cls(StrtPose, TrgtPose, MoveVel, CartAccel, n_interp, n_dofs)

    def calc_CartDistance(self, pose1, pose2):
        """
        Calculate the Euclidean distance between two Cartesian positions (pose1 and pose2).

        This method takes two pose inputs, extracts their first three elements (representing
        the x, y, and z coordinates), and computes the straight-line Euclidean distance
        between them.

        Parameters:
        pose1 (np.ndarray): The starting pose in Cartesian coordinates.
        pose2 (np.ndarray): The target pose in Cartesian coordinates.

        Returns:
        float: The Euclidean distance between the two points.
        """

        # Extract the (x, y, z) coordinates from the poses (first three elements).
        point1 = np.array(pose1[:3])
        point2 = np.array(pose2[:3])

        # Compute the Euclidean distance between the two points.
        dis = np.linalg.norm(point1 - point2)

        return dis

    def calc_MoveTime(self):
        """
        Calculate the total time required to move from the start point to the end point.

        This method calculates the total time (`t_time`) needed to travel the calculated
        Cartesian distance (`self.CartDistance`) at the specified velocity (`self.MoveVel`).
        It ensures that the velocity is a positive value, as movement with zero or negative
        velocity is invalid.

        Returns:
        float: The total time required to complete the motion.

        Raises:
        ValueError: If the specified movement velocity (`MoveVel`) is less than or equal to zero.
        """
        if self.MoveVel <= 0:
            raise ValueError("Speed must be greater than zero.")

        # Calculate the total time by dividing the distance by the velocity.
        t_time = self.CartDistance / self.MoveVel

        return t_time

    def calc_MaxVelocity(self):
        """
        Calculate the maximum velocity required to construct a trapezoidal or triangular
        velocity profile for the movement.

        This method calculates the maximum velocity (`maxVel`) that will be reached
        during the movement. It first checks if the distance is zero, in which case
        the velocity is set to zero. Otherwise, it computes the required velocity based on
        the provided acceleration and movement time (`t_total`). If the given acceleration
        is insufficient to complete the movement within the total time, the method adjusts
        the acceleration to ensure it can meet the movement time constraint.

        Returns:
        float: The maximum velocity that will be achieved during the movement.

        Notes:
        - The method checks for a valid discriminant in the quadratic equation used for
          calculating the maximum velocity. If the discriminant is near zero, it handles
          that case to avoid calculation errors.
        - If the provided acceleration is too small to meet the `t_total` time, it automatically
          adjusts the acceleration and prints a warning.

        Raises:
        ValueError: If the distance is zero, the maximum velocity is set to zero.
        """

        t_total = self.MoveTime  # Total movement time.
        accel = self.CartAccel  # Given Cartesian acceleration.
        distance = self.CartDistance  # Cartesian distance to be traveled.

        # If there is no distance to travel, set maximum velocity to 0.
        if distance == 0:
            maxVel = 0
        else:
            # Check if the acceleration is sufficient to complete the movement in the given time.
            if accel < (4 * distance / (t_total**2)):
                # Adjust acceleration to the minimum value required.
                accel = 4 * distance / (t_total**2)
                print(
                    f" *Given acceleration is not sufficient to complete the movement in t_total, new acceleration set at {accel}"
                )

            # Calculate the discriminant of the quadratic equation for velocity.
            discriminant = ((t_total * accel) ** 2) - 4 * (distance * accel)

            # Handle cases where the discriminant is near zero to avoid numerical issues.
            discriminant = 0 if AF.floats_equal(0.0, discriminant) else discriminant

            # Solve for maximum velocity using the quadratic formula.
            maxVel = ((t_total * accel) - math.sqrt(discriminant)) / 2

        return maxVel

    def create_CartTrapzVelProfExpr(self):
        """
        Create a trapezoidal or triangular velocity profile expression for Cartesian movement.

        This method constructs a velocity profile based on the given parameters (distance,
        acceleration, and maximum velocity). Depending on the total movement time (`t_total`)
        and the acceleration (`CartAccel`), it will either generate a trapezoidal or
        triangular velocity profile.

        The velocity profile is represented as a piecewise function with different phases:
        - Initial acceleration phase.
        - Constant velocity phase (for trapezoidal profiles).
        - Deceleration phase.
        - Zero velocity before and after the motion.

        Returns:
        PiecewiseFunction: A symbolic piecewise velocity function (`velProfile`) defined
        over time (`t`) for the Cartesian motion.

        Notes:
        - If the acceleration time (`t_acc`) is greater than or equal to half the total
          movement time, a triangular velocity profile is created (no constant velocity phase).
        - If `t_acc` is less than half the total time, a trapezoidal profile is created,
          consisting of acceleration, constant velocity, and deceleration phases.
        - If no movement is required (i.e., zero distance or velocity), the velocity profile
          is set to zero for all time steps.
        """

        # Get the necessary parameters.
        distance = self.CartDistance  # Total Cartesian distance to be traveled.
        t_total = self.MoveTime  # Total movement time.
        accel = self.CartAccel  # Acceleration.
        vel = self.MaxCartVel  # Maximum velocity.

        # Time to accelerate and decelerate (assuming symmetric acceleration and deceleration).
        t_acc = t_dec = abs(vel / accel)

        # Symbol for time in the velocity expression.
        t = sp.Symbol("t")

        # Handle the case where no movement is required (zero acceleration or distance).
        if t_acc == 0 or distance == 0:
            # No movement, velocity is zero for all time.
            velProfile = [
                Piece(
                    0,
                    sp.core.numbers.NegativeInfinity,  # For times before movement starts.
                    sp.core.numbers.Infinity,  # For times after movement ends.
                )
            ]

        # Handle the case where the profile becomes triangular (no constant velocity phase).
        elif t_acc >= t_total / 2:
            # Triangular velocity profile (accelerate, then decelerate).
            t_acc = t_dec = t_total / 2
            velProfile = [
                Piece(
                    0, sp.core.numbers.NegativeInfinity, 0
                ),  # Zero velocity before movement starts.
                Piece(accel * t, 0, t_acc),  # Acceleration phase.
                Piece(
                    vel - (accel * (t - t_acc)), t_acc, t_total
                ),  # Deceleration phase.
                Piece(
                    0, t_total, sp.core.numbers.Infinity
                ),  # Zero velocity after movement ends.
            ]

        # Handle the case where the profile becomes trapezoidal (constant velocity phase).
        else:
            # Trapezoidal velocity profile (acceleration, constant velocity, deceleration).
            t_const = t_total - 2 * t_acc  # Time spent at constant velocity.

            velProfile = [
                Piece(
                    0, sp.core.numbers.NegativeInfinity, 0
                ),  # Zero velocity before movement starts.
                Piece(accel * t, 0, t_acc),  # Acceleration phase.
                Piece(vel, t_acc, t_acc + t_const),  # Constant velocity phase.
                Piece(
                    vel - accel * (t - (t_acc + t_const)),
                    t_acc + t_const,
                    t_total,
                ),  # Deceleration phase.
                Piece(
                    0, t_total, sp.core.numbers.Infinity
                ),  # Zero velocity after movement ends.
            ]

        # Create a piecewise function for the velocity profile.
        velProfile = PiecewiseFunction(velProfile, "t")

        # Store the velocity profile expression in the object.
        self.CartTrapzVelProfExpr = velProfile

        return velProfile

    def calc_DistaceFromStart(self):
        """
        Calculate the distance from the starting position at each time step.

        This method evaluates the position expression (`self.CartPositionExpr`) at each
        time step (`self.t`), which represents the position of the object relative to the
        starting point as the trajectory progresses. The results are stored in a list and
        returned as a NumPy array, where each element represents the distance traveled from
        the start at a specific time stamp.

        Returns:
        np.ndarray: An array of distances traveled from the starting position at each
        time step, corresponding to the time points in `self.t`.
        """

        d_list = []  # Initialize an empty list to store distances.

        # Loop over each time stamp in the time array `self.t`.
        for t_stamp in self.t:
            # Evaluate the position expression at the current time stamp and append it to the list.
            d_list.append(self.CartPositionExpr.subs_IndepVar(t_stamp))

        # Convert the list of distances to a NumPy array for easier manipulation and return it.
        return np.array(d_list)

    def calc_InterpPoses(self):
        """
        Calculate interpolated poses along the trajectory from the start to the target pose.

        This method interpolates positions and orientations (in RPY - Roll, Pitch, Yaw format)
        between the starting pose (`self.StrtPose`) and the target pose (`self.TrgtPose`).
        The positions are linearly interpolated, while the orientations are interpolated
        using spherical linear interpolation (SLERP) for smooth transitions between orientations.

        Returns:
        np.ndarray: A NumPy array containing the interpolated poses at each interpolation point.
                    Each pose consists of [x, y, z, roll, pitch, yaw].
        """

        pose_list = []  # Initialize a list to store interpolated poses.

        # Unpack the start and target pose positions (x, y, z) and orientations (roll, pitch, yaw).
        xs, Ys, Zs, Rs, Ps, YWs = (
            self.StrtPose
        )  # Start pose (position and orientation).
        Xe, Ye, Ze, Rt, Pt, YWt = (
            self.TrgtPose
        )  # Target pose (position and orientation).

        # Convert the start and target RPY (roll, pitch, yaw) to rotation matrices.
        strt_Rmat = SM.GetMatFromPose([0, 0, 0, Rs, Ps, YWs])[
            :3, :3
        ]  # Start rotation matrix.
        trgt_Rmat = SM.GetMatFromPose([0, 0, 0, Rt, Pt, YWt])[
            :3, :3
        ]  # Target rotation matrix.

        # Use SLERP (Spherical Linear Interpolation) to interpolate between the rotation matrices.
        rotations = R.from_matrix([strt_Rmat, trgt_Rmat])  # Create rotation objects.
        slerp = Slerp(
            [0, 1], rotations
        )  # Define the SLERP interpolation between start and target.

        # Iterate through each normalized distance (self.d / self.CartDistance) along the trajectory.
        for norm_d in self.d / self.CartDistance:
            norm_d = (
                norm_d if norm_d < 1 else 1
            )  # Ensure normalized distance does not exceed 1.

            # Linear interpolation for the Cartesian positions (x, y, z).
            x = xs + norm_d * (Xe - xs)
            y = Ys + norm_d * (Ye - Ys)
            z = Zs + norm_d * (Ze - Zs)

            # Interpolate the rotation matrix using SLERP and convert back to RPY.
            interp_Rmat = np.array(
                slerp([norm_d]).as_matrix()[0], dtype=np.float64
            )  # Get the interpolated rotation matrix.
            rpy = SM.GetRPY(
                interp_Rmat
            )  # Convert rotation matrix back to roll, pitch, yaw.

            # Append the interpolated pose (x, y, z, roll, pitch, yaw) to the list.
            pose_list.append([x, y, z, rpy[0], rpy[1], rpy[2]])

        # Convert the list of interpolated poses to a NumPy array for easy manipulation and return it.
        return np.array(pose_list, dtype=np.float64)

    def plot(
        self,
        data: Optional[np.ndarray] = None,
        skip=1,
        length=1,
        save_name: Optional[str] = None,
    ):

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
        if save_name:
            plt.savefig(fname=save_name)
        plt.show()


def rmve_ConfigurationJump(
    q: np.ndarray,
    t: np.ndarray,
    joint_ranges: Union[np.ndarray, list, tuple],
    joint_type: Union[np.ndarray, list, tuple],
    rel_tol: float = 5e-2,
):
    """
    Detect and remove configurations jumps in a joint trajectory by interpolating between points
    where the relative deviation exceeds a given tolerance.

    This function checks for large relative deviations in joint positions between consecutive
    time steps in a trajectory, which may indicate potential singularities or problems in
    the trajectory execution. If such deviations are detected, the function inserts additional
    interpolation points to smooth the transition between the two points.

    Parameters:
    q (np.ndarray): Joint positions (array of joint configurations) over time.
    t (np.ndarray): Time values corresponding to the joint positions.
    joint_ranges (Union[np.ndarray, list, tuple]): The joint limits for each degree of freedom,
                                                  specified as [min, max] pairs.
    joint_type (Union[np.ndarray, list, tuple]): The types of joints (e.g., revolute or prismatic).
    rel_tol (float): Relative deviation tolerance for detecting singularities.
                     Default is 0.05 (5%).

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing:
        - np.ndarray: The modified joint positions with interpolated points added.
        - np.ndarray: The modified time values with interpolated time steps.

    Raises:
    ValueError: If the input arrays `q` and `t` have different lengths, or if `array1` and `array2`
                in the `relative_deviation` function have different lengths.

    Notes:
    - If the relative deviation between consecutive joint positions exceeds `rel_tol`,
      additional interpolated points are added to ensure smooth transitions.
    - The interpolation between joint positions is performed using the custom function
      `AF.interpolate_q_in_range`, which interpolates based on the joint types and their ranges.

    """

    def relative_deviation(array1, array2, joint_ranges):
        """
        Calculate the relative deviation between two arrays of joint positions.

        The relative deviation is computed using the angular distance between each pair of
        joint positions, normalized by the range of the joint.

        Parameters:
        array1 (np.ndarray): The first array of joint positions.
        array2 (np.ndarray): The second array of joint positions.
        joint_ranges (list or np.ndarray): The joint ranges for each degree of freedom.

        Returns:
        np.ndarray: The relative deviations for each joint position.

        Raises:
        ValueError: If the lengths of `array1` and `array2` are not the same.
        """
        if len(array1) != len(array2):
            raise ValueError("Arrays must be of the same length.")

        # Calculate the relative deviation using the angular distance between the joints.
        relative_deviation = np.array(
            [
                abs(AF.angle_dist_in_range(angle1, angle2, j_range))
                / (abs(j_range[1] - j_range[0]))
                for angle1, angle2, j_range in zip(array1, array2, joint_ranges)
            ]
        )

        return relative_deviation

    # Initialize new joint positions and time arrays.
    q_new = [q[0]]  # Start with the first joint position.
    t_new = [t[0]]  # Start with the first time point.
    inc_t = t[1] - t[0]  # Time increment between consecutive points.
    n_interp_points = 0  # Counter for the number of interpolation points added.

    # Iterate over the trajectory to check for singularities between consecutive positions.
    for i in range(len(q) - 1):
        q_curr = q[i]
        q_next = q[i + 1]
        t_curr = (i + n_interp_points) * inc_t

        # Calculate the relative deviation between the current and next joint positions.
        rel_error = relative_deviation(q_curr, q_next, joint_ranges)

        # Check if any relative deviation exceeds the tolerance.
        if np.any(rel_error > rel_tol):
            # Print information about the detected singularity.
            print(f"CONFIGURATION JUMP FOUND BETWEEN POSES {i+1} - {i+2}:")
            print(f"Current q: {q_curr}")
            print(f"Next q: {q_next}")
            print(f"Relative Deviation (%): {np.round(rel_error * 100, 2)}")
            print(f"This may lead to problems in the trajectory physical execution")
            print()

            # Determine the number of interpolation steps needed.
            max_rel_error = np.max(rel_error)
            steps = max(
                10, int(np.ceil(max_rel_error * 50))
            )  # Ensure a minimum of 10 steps.
            n_interp_points += steps

            # Generate interpolated times and joint positions between the singularity points.
            interp_t_start = t_new[-1]  # Start time for interpolation.
            interpolated_times = np.linspace(
                interp_t_start, interp_t_start + steps * inc_t, steps, endpoint=False
            )
            interpolated_qs = AF.interpolate_q_in_range(
                q_curr,
                q_next,
                joint_type,
                joint_ranges,
                steps,
            )

            # Append the interpolated joint positions and times (excluding the first point, which is already added).
            q_new.extend(interpolated_qs[1:])
            t_new.extend(interpolated_times[1:])

        else:
            # Append the next position and time if no singularity was detected and not already added.
            if t[i + 1] != t_new[-1]:
                q_new.append(q_next)
                t_new.append(t_curr)

    # Return the new arrays of joint positions and times.
    return np.array(q_new), np.array(t_new)


if __name__ == "__main__":

    np.set_printoptions(precision=2, suppress=True)

    Traj1 = LinealTrajectory.create(
        n_dofs=6,
        n_interp=100,
        StrtPose=np.array([0, 0, 0, 0, 0, 0]),
        TrgtPose=np.array([10, 10, 10, -2.97, -0.52, -2.19]),
        MoveVel=2,
        CartAccel=1,
    )

    Traj1.plot()
