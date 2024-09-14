import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import get_cmap

import AuxiliarFunctions as AF
import SpatialMath as SM
from nIK_Corke import ikine_LM_Corke
from LinealTrajectory import LinealTrajectory, rmve_ConfigurationJump

from typing import List, Union, Callable, Optional


class RobotTool:
    """
    This class represents the position and orientation of the robot's tool in the robot's coordinate system.
    The tool's position is described by `tXYZ` (translation in x, y, z), and the orientation is described by
    `rXYZ` (rotation angles in roll, pitch, yaw).

    Parameters:
    - tXYZ (list): A list of three translation values representing the tool's position [x, y, z].
    - rXYZ (list): A list of three rotation values representing the tool's orientation [roll (rX), pitch (rY), yaw (rZ)].
    """

    def __init__(self, tXYZ: list, rXYZ: list):
        """
        Initialize the tool values for a robot's end-effector (tool).

        Raises:
        - ValueError: If `tXYZ` or `rXYZ` does not contain exactly three elements.
        """
        self.tXYZ = tXYZ  # Tool translation values [x, y, z]
        self.rXYZ = rXYZ  # Tool rotation angles [rX, rY, rZ]
        self.rel_pose = tXYZ + rXYZ  # Combined tool pose [x, y, z, rX, rY, rZ]

        # Validate that tXYZ and rXYZ each contain exactly three elements
        if len(self.tXYZ) != 3:
            raise ValueError("tXYZ must have exactly three elements")
        if len(self.rXYZ) != 3:
            raise ValueError("rXYZ must have exactly three elements")

        # Calculate the transformation matrix for the tool
        self.Tfm = self.get_Tfm()

    def get_Tfm(self):
        """
        Compute the homogeneous transformation matrix for the tool based on its position and orientation.

        This method constructs the tool's transformation matrix from the position and orientation values
        (`tXYZ` and `rXYZ`). The transformation matrix represents the tool's pose in 3D space as a 4x4 matrix.

        Returns:
        - T (np.ndarray): A 4x4 transformation matrix representing the tool's pose.
        """
        # Extract translation values
        x = self.tXYZ[0]
        y = self.tXYZ[1]
        z = self.tXYZ[2]

        # Extract rotation values (roll, pitch, yaw)
        rX = self.rXYZ[0]
        rY = self.rXYZ[1]
        rZ = self.rXYZ[2]

        # Use an external utility function to compute the transformation matrix from the pose
        T = SM.GetMatFromPose(self.rel_pose)

        return T  # Return the transformation matrix


class JointDH:
    """
    This class represents a single joint in a robot based on the Denavit-Hartenberg convention. The joint
    can be either revolute or prismatic, and its transformation matrix is constructed from the provided
    DH parameters: `theta`, `d`, `a`, `alpha`, and joint limits (`qlim`).

    Parameters:
    - theta (float): The joint angle (for revolute joints).
    - d (float): The offset along the previous z-axis to the common normal (for prismatic joints).
    - a (float): The length of the common normal.
    - alpha (float): The angle about the common normal.
    - qlim (list): The joint limits as a list [min, max]. Used to limit the range of joint movement.
    - j_type (int): The type of joint. `0` for Revolute, `1` for Prismatic.
    """

    def __init__(
        self, theta: float, d: float, a: float, alpha: float, qlim: list, j_type: int
    ):
        """
        Initialize the Denavit-Hartenberg (DH) parameters for a single robotic joint.

        Raises:
        - ValueError: If `qlim` is not a list of exactly two elements.
        """
        self.theta = theta  # Joint angle (for revolute) or offset (for prismatic)
        self.d = d  # Offset along the previous z-axis
        self.a = a  # Length of the common normal (along x-axis)
        self.alpha = alpha  # Twist angle (about x-axis)
        self.j_type = j_type  # Joint type (0 for Revolute, 1 for Prismatic)

        # Process and store joint limits, avoiding exact multiples of pi to prevent computation issues
        self.qlim = []
        for bound in qlim:
            if abs(abs(bound) - math.pi) > 1e-5:
                self.qlim.append(bound * 0.9999999999)  # Slightly modify values near pi
            else:
                self.qlim.append(bound)

    def get_Tfm(self):
        """
        Computes the transformation matrix for the joint based on its DH parameters.

        The transformation matrix is constructed using the following transformations:
        - Rotation about the z-axis by `theta`.
        - Translation along the z-axis by `d`.
        - Translation along the x-axis by `a`.
        - Rotation about the x-axis by `alpha`.

        Returns:
        - T (np.ndarray): A 4x4 homogeneous transformation matrix that defines the joint's transformation
          relative to the previous joint.
        """
        # Extract DH parameters
        theta = self.theta
        d = self.d
        a = self.a
        alpha = self.alpha

        # Construct the transformation matrix step by step
        Rz = SM.RotZ(theta)  # Rotation about z-axis by theta
        Tz = SM.Trans(0, 0, d)  # Translation along z-axis by d
        Tx = SM.Trans(a, 0, 0)  # Translation along x-axis by a
        Rx = SM.RotX(alpha)  # Rotation about x-axis by alpha

        # Combine the transformations into a single transformation matrix
        T = Rz @ Tz @ Tx @ Rx

        return T

    def __str__(self):
        """
        Returns a string representation of the joint's DH parameters.

        The string includes the joint's DH parameters (`theta`, `d`, `a`, `alpha`) and its joint limits (`qlim`).

        Returns:
        - str: A string summarizing the joint's DH parameters and joint limits.
        """
        return f"DH Parameters: theta={self.theta}, d={self.d}, a={self.a}, alpha={self.alpha}, qlim={self.qlim}"


class RevoluteDH(JointDH):
    """
    The `RevoluteDH` class represents a revolute joint in a robotic arm, where the joint rotates about the z-axis
    as defined by its DH parameters. This class is a subclass of `JointDH` and specifically sets up a revolute
    joint with the `theta` parameter representing the rotation angle.

    Parameters:
    - home (float): The home position angle for the joint in radians. This is the default or "neutral" position
        when the robot is in its standard pose.
    - d (float): The offset along the previous z-axis to the common normal. This corresponds to the translation
        along the z-axis for prismatic joints, but is constant for revolute joints.
    - a (float): The length of the common normal, which is the offset along the x-axis.
    - alpha (float): The twist angle about the common normal, defining the orientation of the joint's x-axis
        relative to the previous joint's x-axis.
    - qlim (list): The range of motion for the revolute joint, expressed as a list [min, max]. Defaults to full
        rotation (close to [-π, π]) but with a small margin to avoid computational issues with exact multiples of π.

    Notes:
    - In the DH convention, the `theta` parameter for revolute joints represents the joint's angle, which is
        updated during robot movements.
    - This class uses `j_type=0` to indicate that the joint is revolute.

    """

    def __init__(
        self,
        home: float,
        d: float,
        a: float,
        alpha: float,
        qlim: list = [-math.pi * 0.999, math.pi * 0.999],
    ):
        """
        Initialize a revolute joint with specific Denavit-Hartenberg (DH) parameters.
        """
        # Initialize the revolute joint by calling the superclass constructor with j_type=0 for revolute
        super().__init__(theta=home, d=d, a=a, alpha=alpha, qlim=qlim, j_type=0)


class PrismaticDH(JointDH):
    """
    The `PrismaticDH` class represents a prismatic joint in a robotic arm, where the joint translates along
    the z-axis as defined by its DH parameters. This class is a subclass of `JointDH` and specifically sets
    up a prismatic joint with the `d` parameter representing the translation distance.

    Parameters:
    - theta (float): The joint angle, which is typically fixed for a prismatic joint (it defines the orientation).
    - home (float): The home position offset along the z-axis (this substitutes for the `d` parameter, which defines
        the translation distance for prismatic joints).
    - a (float): The length of the common normal, which is the offset along the x-axis.
    - alpha (float): The twist angle about the common normal, defining the orientation of the joint's x-axis relative
        to the previous joint's x-axis.
    - qlim (list): The range of motion for the prismatic joint, expressed as [min, max]. Defaults to [0, 1].

    Notes:
    - In the DH convention, the `d` parameter for prismatic joints represents the joint's translation distance, which
        is updated during robot movements.
    - This class uses `j_type=1` to indicate that the joint is prismatic.
    """

    def __init__(
        self,
        theta: float,
        home: float,
        a: float,
        alpha: float,
        qlim: list = [0, 1],
    ):
        """
        Initialize a prismatic joint with specific Denavit-Hartenberg (DH) parameters.
        """
        # Initialize the prismatic joint by calling the superclass constructor with j_type=1 for prismatic
        super().__init__(theta=theta, d=home, a=a, alpha=alpha, qlim=qlim, j_type=1)


class DHRobot:
    """
    This class generates a theoritical manipulator based on a chain of Denavit-Hartenberd joints.
    It implements several methods in order to generate possible movements of the robot
    """

    def __init__(
        self,
        joint_list: List[Union[JointDH, RevoluteDH, PrismaticDH]],
        IK_solver: Callable,
        tool: Optional[RobotTool] = None,
        DH_name: str = "RobotDH",
    ):
        """
        Initialize the DHRobot class with the provided joint list, tool, and inverse kinematics solver.
        """
        self.joint_list = (
            joint_list  # Stores the list of joints that define the robot's kinematics.
        )

        if tool is None:
            # If no tool is provided, default to a standard RobotTool positioned at [0,0,0] with no orientation.
            self.tool = RobotTool([0, 0, 0], [0, 0, 0])
        else:
            # If a tool is provided, use the provided RobotTool configuration.
            self.tool = tool

        self.DH_name = DH_name  # Stores the robot's name.
        self.IK_solver = IK_solver  # Inverse kinematics solver function.

        self.n_joint = len(
            joint_list
        )  # The number of joints in the robot, calculated from the joint list.
        self.home_q = (
            self.get_JointValues()
        )  # Gets the robot's home configuration (default joint values).

    def get_PrismaticJoints(self):
        """
        Returns a list of booleans representing the type of each joint in the robot.

        For each joint in the robot's joint list, this method checks if the joint is prismatic.
        A prismatic joint allows linear motion, while an angular (revolute) joint allows rotational motion.

        Returns:
        - listPrism (List[bool]): A list of booleans where each element corresponds to a joint in the robot.
          `True` indicates the joint is prismatic, and `False` indicates the joint is revolute.
        """
        listPrism = []
        for j in self.joint_list:
            if j.j_type == 1:  # Assuming 'j_type' == 1 represents a prismatic joint
                listPrism.append(True)
            else:
                listPrism.append(False)

        return listPrism

    def get_JointRanges(self):
        """
        Returns a list of tuples representing the joint limits for each joint in the robot.

        Each tuple contains the lower and upper limits of a joint's movement. For prismatic joints,
        these limits represent the minimum and maximum allowable linear displacement. For revolute joints,
        they represent the minimum and maximum angular rotation in radians.

        Returns:
        - limits (List[Tuple[float, float]]): A list where each element is a tuple (lower_limit, upper_limit)
          corresponding to the allowable range of motion for each joint.
        """
        limits = []
        for j in self.joint_list:
            # 'qlim' is assumed to store the joint's range as a tuple (lower_limit, upper_limit)
            limits.append(j.qlim)

        return limits

    def check_JointInsideRange(self, check_val=None, idx=None):
        """
        Checks if the specified joint values are within their allowable ranges.

        This method verifies that the provided joint values (or the current joint positions) are within
        the defined limits for each joint in the robot. If no joint values or indices are specified,
        the method checks all joints against their current positions.

        Parameters:
        - check_val (None, float, list): The joint value(s) to check. Can be:
            * None: The current joint positions will be checked.
            * A single float: The value of a single joint to be checked.
            * A list of floats: The values of multiple joints to be checked.
        - idx (None, int, list): The index or indices of the joints corresponding to the values in `check_val`. Can be:
            * None: If `check_val` is None, this is ignored, and all joints are checked.
            * A single int: The index of the joint to check if `check_val` is a single value.
            * A list of ints: The indices corresponding to the joint values in `check_val`.

        Raises:
        - ValueError:
            * If any of the provided or current joint values are out of the allowed range.
            * If `check_val` is a list but its length doesn't match the number of joints, or if the length of `check_val` and `idx` don't match.
        - TypeError: If `check_val` and `idx` are mismatched (e.g., one is a list and the other is not).

        Returns:
        - bool: Always returns True if all joint values are within their respective ranges.
        """

        def check_JointValue(val, idx):
            """
            Helper function to check if a single joint value is within its allowed range.

            Parameters:
            - val (float): The joint value to check.
            - idx (int): The index of the joint whose value is being checked.

            Raises:
            - ValueError: If the joint value is outside the allowed range.
            """
            joint = self.joint_list[idx]  # Get the joint based on the index
            qlim = joint.qlim  # Get the joint's limit (lower, upper)

            if joint.j_type == 0:  # If the joint is revolute
                val = AF.wrap_angle(val)  # Normalize the angle value
                in_range = AF.check_angle_in_range(
                    val, qlim
                )  # Check if the angle is within the range
            else:  # If the joint is prismatic
                in_range = AF.check_linear_in_range(
                    val, qlim
                )  # Check if the linear value is within range

            if not in_range:
                raise ValueError(
                    f"Joint value {val} at index {idx} is out of range {qlim}."
                )

            return in_range

        # Case where both check_val and idx are provided
        if check_val is not None and idx is not None:

            if isinstance(check_val, list) and isinstance(idx, list):
                # Check that both lists are of equal length
                if len(check_val) != len(idx):
                    raise ValueError(
                        "'check_val' and 'idx' lists must have the same length."
                    )
                # Check each value with its corresponding joint index
                for val, idx in zip(check_val, idx):
                    check_JointValue(val, idx)

            elif isinstance(check_val, list) or isinstance(idx, list):
                # Raise TypeError if one is a list and the other isn't
                raise TypeError(
                    "Both 'check_val' and 'idx' need to be lists or both need to be single values."
                )
            else:
                # Single value and index case
                check_JointValue(check_val, idx)

        # Case where only check_val is provided (and it's a list)
        elif check_val is not None and isinstance(check_val, list):

            # Check that the number of provided values matches the number of joints
            if len(check_val) != len(self.joint_list):
                raise ValueError(
                    "Length of 'check_val' list must match number of joints."
                )
            # Check each joint value
            for val, joint in zip(check_val, self.joint_list):
                check_JointValue(val, self.joint_list.index(joint))

        # Case where no check_val is provided (check current joint positions)
        elif check_val is None:
            # Loop through each joint and check its current position
            for idx, joint in enumerate(self.joint_list):
                current_val = joint.theta if joint.j_type == 0 else joint.d
                check_JointValue(current_val, idx)

        return True  # Return True if all checks pass

    def check_Collisions(self):
        """
        Checks for potential collisions of the robot with the floor.

        This method calculates the transformation matrices for each joint and the corresponding position of each joint in 3D space.
        It then verifies if any joint's position is below a certain threshold (z-coordinate < 0.05), indicating a potential collision
        with the floor.

        Raises:
        - ValueError: If any joint's position is below the collision threshold (z-coordinate < 0.05), indicating a collision with the floor.

        Returns:
        - bool: Always returns False if no collisions are detected.
        """

        tfms = [
            np.eye(4)
        ]  # Initialize the first transformation matrix as an identity matrix
        pos = []  # List to store the XYZ positions of each joint

        for j in self.joint_list:
            # Compute the transformation matrix for the current joint by multiplying the last matrix in tfms by the joint's transformation
            new_tfm = tfms[-1] @ j.get_Tfm()
            tfms.append(new_tfm)  # Append the new transformation matrix to the list
            pos.append(
                SM.GetXYZ(tfms[-1])
            )  # Extract the XYZ position from the transformation matrix

        # Check if any joint position has a z-coordinate below the collision threshold (0.05)
        for i, jp in enumerate(pos):
            if jp[2] < 0.05:  # Check if the z-coordinate of the position is below 0.05
                raise ValueError(
                    f"Collision with the floor. Position {i} with values {jp}"
                )

        return False  # No collisions detected, so return False

    def set_JointValues(self, q_values):
        """
        Sets the angular (theta) or prismatic (d) position for each joint in the robot.

        This method assigns new values to the joints of the robot based on the type of joint (revolute or prismatic).
        Revolute joints will update their angular position (theta), while prismatic joints will update their linear position (d).
        Before setting the values, the method ensures they are within the allowable joint limits and checks for potential collisions.

        Parameters:
        - q_values (List[float]): A list of joint values corresponding to each joint.
          For revolute joints, these values represent angles (in radians), and for prismatic joints, they represent linear displacements.

        Raises:
        - ValueError: If the length of `q_values` does not match the number of joints in the robot.
        - ValueError: If any of the joint values are outside the allowable range for their respective joints.

        """
        if len(q_values) != self.n_joint:
            raise ValueError(
                "The q_values list length does not match the number of degrees of freedom (DoF)."
            )

        else:
            for i, j in enumerate(self.joint_list):
                if j.j_type == 0:  # Revolute joint (angular motion)
                    new_value = AF.wrap_angle(q_values[i])  # Normalize the angle value
                    self.check_JointInsideRange(
                        new_value, i
                    )  # Check if the value is within the allowed range
                    j.theta = new_value  # Set the new angle for the revolute joint
                else:  # Prismatic joint (linear motion)
                    new_value = q_values[
                        i
                    ]  # Use the value directly for prismatic joints
                    self.check_JointInsideRange(
                        new_value, i
                    )  # Check if the value is within the allowed range
                    j.d = new_value  # Set the new displacement for the prismatic joint

        # After setting the joint values, check for any potential collisions
        self.check_Collisions()

    def get_JointValues(self):
        """
        Returns the current value for each joint, depending on the type of joint (revolute or prismatic).

        For revolute (angular) joints, the method returns the current angular position (theta).
        For prismatic (linear) joints, it returns the current linear displacement (d). The joint values are returned
        as a numpy array where each index corresponds to a joint in the robot.

        Returns:
        - j_values (np.ndarray): A numpy array of size `n_joint`, where each element is the current value of the corresponding joint.
          For revolute joints, the value is theta (in radians), and for prismatic joints, the value is d (linear displacement).
        """
        j_values = np.zeros(self.n_joint)  # Initialize an array to store joint values

        for i, j in enumerate(self.joint_list):
            if j.j_type == 0:  # If the joint is revolute (angular motion)
                j_values[i] = (
                    j.theta
                )  # Assign the angular position (theta) to the array
            else:  # If the joint is prismatic (linear motion)
                j_values[i] = j.d  # Assign the linear displacement (d) to the array

        return j_values  # Return the array containing all joint values

    def get_Tfm_UpTo_Joint(self, i):
        """
        Computes the homogeneous transformation matrix from the base frame (frame 0) to the frame of joint i.

        This method calculates the cumulative transformation matrix, denoted as ^{0}A_{i}, which relates the base
        frame of the robot (frame 0) to the frame of the specified joint i. The transformation is calculated by
        multiplying the individual transformation matrices of each joint up to joint i.

        Parameters:
        - i (int): The index of the joint to which the transformation matrix should be computed. The index starts from 0.

        Returns:
        - iTz (np.ndarray): A 4x4 homogeneous transformation matrix that represents the transformation from the base
          frame to the frame of joint i.
        """
        iTz = np.identity(
            4
        )  # Initialize the transformation matrix as the identity matrix

        # Loop through each joint up to the ith joint and multiply their transformation matrices
        for j in self.joint_list[:i]:
            iT = j.get_Tfm()  # Get the transformation matrix for the current joint
            iTz = (
                iTz @ iT
            )  # Multiply the current transformation matrix with the cumulative matrix

        return iTz  # Return the final transformation matrix

    def get_RobotTfm(self):
        """
        Computes the overall transformation matrix from the base frame to the end-effector frame of the robot.

        This method calculates the cumulative homogeneous transformation matrix for all joints from joint 0 up to joint n-1
        according to the Denavit-Hartenberg (DH) parameters. It then applies the tool transformation matrix to account for
        any additional end-effector configurations. The resulting matrix represents the pose of the robot's end-effector
        relative to the base frame.

        Returns:
        - rT (np.ndarray): A 4x4 homogeneous transformation matrix representing the pose of the robot's end-effector in
          the base coordinate frame.
        """
        # Compute the cumulative transformation matrix up to the last joint
        robot_tfm = self.get_Tfm_UpTo_Joint(self.n_joint)

        # Multiply by the tool transformation matrix to account for the end-effector
        rT = robot_tfm @ self.tool.Tfm

        return rT

    def calc_Reach(self):
        """
        Calculates the maximum reach of the robot.

        This method computes the total reach of the robot by summing the distances contributed by each joint.
        Specifically, it adds the `d` (displacement along the previous z-axis) and `a` (link length along the previous x-axis)
        parameters for each joint, as defined by the Denavit-Hartenberg (DH) convention. The resulting value represents
        the maximum possible reach of the robot from its base to the furthest extent of its joints.

        Returns:
        - reach (float): The total reach of the robot, calculated as the sum of the `d` and `a` parameters for all joints.
        """
        reach = 0  # Initialize the reach variable

        # Iterate through each joint in the joint list and add the d and a parameters
        for j in self.joint_list:
            reach += (
                j.d + j.a
            )  # Sum the prismatic displacement (d) and link length (a) for each joint

        return reach  # Return the total calculated reach

    def get_EndEffPosOr(self, T: Optional[np.ndarray] = None):
        """
        Returns the position and orientation of the robot's end-effector.

        This method computes the position and orientation of the robot's end-effector in the base frame.
        If no transformation matrix `T` is provided, the method calculates it by using the `get_RobotTfm`
        method, which computes the transformation matrix from the base frame to the end-effector frame.
        The position is extracted as the (x, y, z) coordinates, and the orientation is represented as
        roll-pitch-yaw (RPY) angles.

        Parameters:
        - T (Optional[np.ndarray]): A 4x4 homogeneous transformation matrix that represents the transformation
          from the base frame to the end-effector frame. If not provided, the method will compute it.

        Returns:
        - np.ndarray: A 6-element numpy array containing the position and orientation of the end-effector.
          The first three elements are the (x, y, z) coordinates, and the last three elements are the roll-pitch-yaw (RPY) angles.
        """
        if T is None:
            T = (
                self.get_RobotTfm()
            )  # If no transformation matrix is provided, compute it for the end-effector

        # Combine the position (x, y, z) and orientation (roll, pitch, yaw) into a single array and return
        return np.concatenate((SM.GetXYZ(T), SM.GetRPY(T)))

    def get_Jacobian(self):
        """
        Computes the geometric Jacobian matrix evaluated at the current position of the robot.

        The geometric Jacobian is a 6xN matrix that relates the joint velocities to the end-effector velocity
        (both linear and angular). It is computed based on the current joint configuration and describes the
        differential relationship between the joint space and the Cartesian space.

        The Jacobian matrix has two parts:
        - The upper 3 rows represent the linear velocity contribution of each joint.
        - The lower 3 rows represent the angular velocity contribution of each joint.

        The method distinguishes between prismatic (linear) and revolute (rotational) joints. For prismatic joints,
        the linear velocity component is simply the z-axis of the joint's transformation matrix, and the angular
        velocity contribution is zero. For revolute joints, the linear velocity component is computed using the cross
        product between the joint's z-axis and the vector from the joint to the end-effector, while the angular velocity
        is just the z-axis of the joint.

        Returns:
        - J (np.ndarray): A 6xN Jacobian matrix, where N is the number of joints. The upper 3 rows represent the
          linear velocity components, and the lower 3 rows represent the angular velocity components.
        """
        PrismJoints = (
            self.get_PrismaticJoints()
        )  # Get a list indicating whether each joint is prismatic

        rT = (
            self.get_RobotTfm()
        )  # Compute the transformation matrix for the end-effector
        rT_pos = rT[
            0:3, 3
        ]  # Extract the position of the end-effector from the transformation matrix

        J = np.zeros(
            (6, self.n_joint)
        )  # Initialize the Jacobian matrix (6 rows for linear and angular components)

        iTz = np.identity(
            4
        )  # Initialize the transformation matrix up to the current joint as the identity matrix

        # Loop over each joint to compute the corresponding column of the Jacobian matrix
        for i in range(self.n_joint):
            iTz = self.get_Tfm_UpTo_Joint(
                i
            )  # Get the transformation matrix from the base to joint i
            iTz_a = iTz[0:3, 2]  # Extract the z-axis (rotation axis) of joint i

            if PrismJoints[i]:
                # If the joint is prismatic, the linear velocity is along the joint's z-axis, and there is no angular velocity
                J[0:3, i] = iTz_a  # Linear velocity contribution from prismatic joint
            else:
                # If the joint is revolute, calculate the linear and angular velocity contributions
                iTz_pos = iTz[
                    0:3, 3
                ]  # Extract the position of joint i from the transformation matrix

                # Linear velocity contribution for revolute joint (cross product of z-axis and vector from joint to end-effector)
                J[0:3, i] = np.cross(iTz_a, (rT_pos - iTz_pos))
                # Angular velocity contribution for revolute joint (just the z-axis of the joint)
                J[3:6, i] = iTz_a

        return J  # Return the computed Jacobian matrix

    def get_invJacobian(self):
        """
        Computes the inverse or pseudoinverse of the Jacobian matrix.

        This method calculates the inverse of the Jacobian matrix if it is square (i.e., the number of joints matches
        the number of task-space dimensions). If the Jacobian is not square, the method computes the Moore-Penrose
        pseudoinverse, which is often used in cases where there are more or fewer joints than task-space dimensions.

        The method follows two steps:
        - If the Jacobian is square, it uses Gaussian elimination to find the inverse.
        - If the Jacobian is non-square, it computes the Moore-Penrose pseudoinverse using Singular Value Decomposition (SVD).

        Raises:
        - ValueError: If the Jacobian is singular and cannot be inverted.

        Returns:
        - invJ or pinvJ (np.ndarray): The inverse (for square Jacobian) or pseudoinverse (for non-square Jacobian)
          of the Jacobian matrix.
        """
        J = self.get_Jacobian()  # Get the current Jacobian matrix
        # Check the shape of the Jacobian
        rows, cols = J.shape
        if rows == cols:
            # Square Jacobian: Calculate the inverse step by step using Gaussian elimination
            I = np.eye(rows)  # Identity matrix of the same size as the Jacobian

            # Augment the Jacobian with the identity matrix for Gaussian elimination
            augJ = np.hstack((J, I))

            # Apply Gaussian elimination to transform the Jacobian into an identity matrix
            for i in range(rows):
                diagElmt = augJ[i, i]  # Get the diagonal element

                if diagElmt == 0:
                    # If the diagonal element is zero, the matrix is singular
                    raise ValueError("Jacobian is singular and cannot be inverted.")

                # Normalize the current row by the diagonal element to make it 1
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
            # Non-square matrix: Calculate the Moore-Penrose pseudoinverse using SVD
            U, S_val, Vt = np.linalg.svd(
                J
            )  # Perform Singular Value Decomposition (SVD)

            # Create the Sigma (S) and inverse Sigma (invS) matrices
            S = np.zeros((rows, cols))  # Sigma matrix initialized with zeros
            invS = np.zeros((cols, rows))  # Inverse of Sigma initialized with zeros

            # Fill the diagonal elements of Sigma and invS with singular values
            for i, e in enumerate(S_val):
                S[i, i] = e  # Fill Sigma matrix with singular values
                invS[i, i] = (
                    1 / e
                )  # Fill invS with the reciprocal of the singular values

            # Calculate the pseudoinverse using the formula: pinvJ = Vt.T @ invS @ U.T
            pinvJ = Vt.T @ invS @ U.T

            return pinvJ

    def calc_FK(self, theta):
        """
        Calculates the forward kinematics (FK) of the robot based on the given joint values.

        This method computes the end-effector's position and orientation (pose) based on the provided joint values (`theta`).
        Forward kinematics involves determining the pose of the end-effector from the joint positions. The method first
        updates the robot's joint values to those provided in `theta` and then calculates the corresponding end-effector
        pose.

        Parameters:
        - theta (List[float]): A list of joint values (angles or displacements) for each joint. The length of this list
          should match the number of joints in the robot.

        Returns:
        - np.ndarray: A 6-element numpy array where the first three elements represent the (x, y, z) position of the
          end-effector, and the last three elements represent the roll-pitch-yaw (RPY) orientation angles.
        """
        self.set_JointValues(
            theta
        )  # Set the robot's joint values based on the provided theta

        return (
            self.get_EndEffPosOr()
        )  # Return the calculated position and orientation of the end-effector

    def gen_samples(self, n_samples=1000):
        """
        Generates random joint configurations and their corresponding end-effector positions through forward kinematics.

        This method randomly generates joint configurations (within their respective joint limits) and computes the
        corresponding end-effector positions and orientations using forward kinematics. The result is a collection of
        joint configurations and the corresponding end-effector poses.

        Parameters:
        - n_samples (int): The number of random samples (joint configurations) to generate. Defaults to 1000.

        Returns:
        - tuple: A tuple containing two numpy arrays:
            - q_values (np.ndarray): An array of shape (n_samples, n_joints) representing the generated joint configurations.
            - pose (np.ndarray): An array of shape (n_samples, 6) representing the corresponding end-effector poses
              (position and orientation in roll-pitch-yaw format).
        """
        q_values_list = []  # List to store the generated joint configurations
        pose_list = []  # List to store the corresponding end-effector poses

        joint_limits = (
            self.get_JointRanges()
        )  # Get the allowable joint ranges for each joint

        print(f"Generating {n_samples} samples for {self.DH_name} ...")

        # Correct joint limits in case any min value is greater than the max value
        corrected_joint_limits = []
        for low, high in joint_limits:
            if low > high:
                high += 2 * np.pi  # Adjust the joint limit for wraparound issues
            corrected_joint_limits.append((low, high))

        for _ in range(n_samples):
            # Generate random joint values within the corrected limits
            theta = []
            for low, high in joint_limits:
                if low > high:
                    # If limits are misaligned, generate within the range [low, 2π] or [-2π, high]
                    value = (
                        np.random.uniform(low, 2 * np.pi)
                        if np.random.rand() < 0.5
                        else np.random.uniform(-2 * np.pi, high)
                    )
                else:
                    # Generate a value within the valid range [low, high]
                    value = np.random.uniform(low, high)
                theta.append(value)

            theta = np.array(theta)  # Convert the list of joint values to a numpy array

            # Compute the end-effector position and orientation using forward kinematics
            try:
                pos_orientation = self.calc_FK(theta)

                # Store the generated joint configuration and the corresponding end-effector pose
                q_values_list.append(theta)
                pose_list.append(pos_orientation)

            except ValueError:
                # Skip invalid configurations that result in a ValueError (e.g., joint limits exceeded)
                pass

        # Convert the lists to numpy arrays for better performance and usability
        q_values = np.array(q_values_list)
        pose = np.array(pose_list)

        self.samples = (
            q_values,
            pose,
        )  # Store the samples in the object's state for later use

        return (
            q_values,
            pose,
        )  # Return the generated joint configurations and end-effector poses

    def calc_IK(
        self,
        trgt_poses: Union[np.ndarray, list],
        q0: Optional[np.ndarray] = None,
        q_lim: Optional[Union[np.ndarray, bool]] = False,
        IK_solver: Optional[Callable] = None,
        mask: Optional[Union[np.ndarray, list, bool]] = False,
    ):
        """
        Computes the inverse kinematics (IK) to find the joint values that achieve the desired end-effector pose(s).

        This method uses the provided or default inverse kinematics solver to compute the joint values that achieve
        the specified target pose(s) for the end-effector. The user can provide optional initial joint configurations,
        joint limits, and masks for which degrees of freedom to prioritize or constrain during the computation.

        Parameters:
        - trgt_poses (Union[np.ndarray, list]): The target pose(s) for the end-effector. It can be a single pose (as
          a 1D array or list) or multiple poses (as a 2D array or list of poses). Each pose should contain 6 elements
          representing the position (x, y, z) and orientation (roll, pitch, yaw).
        - q0 (Optional[np.ndarray]): Optional initial guess for the joint configuration. If provided, it helps the IK
          solver converge more quickly by starting close to this configuration. If None, the solver starts from the
          robot's default configuration.
        - q_lim (Optional[Union[np.ndarray, bool]]): Joint limits for the IK solver. If False (default), no joint limits
          are applied. If True, the method fetches the robot's joint limits using `get_JointRanges`. If a numpy array is
          provided, it is used as the explicit joint limits for the solver.
        - IK_solver (Optional[Callable]): A custom inverse kinematics solver. If not provided, the method uses the robot's
          default IK solver (`self.IK_solver`).
        - mask (Optional[Union[np.ndarray, list, bool]]): A mask specifying which degrees of freedom to prioritize or
          constrain. For instance, if you only care about the position and not the orientation, you can mask the
          orientation values. Defaults to False, meaning no masking is applied.

        Returns:
        - np.ndarray: The joint values that achieve the target pose(s). This can be a 1D array (for a single target pose)
          or a 2D array (for multiple target poses). If the solver fails to find a valid solution, the behavior depends
          on the specific solver used.
        """
        # Convert target poses to a numpy array if provided as a list
        if isinstance(trgt_poses, list):
            trgt_poses = np.array(trgt_poses)

        # If no IK solver is provided, use the robot's default IK solver
        if IK_solver is None:
            IK_solver = self.IK_solver

        # If q_lim is set to True, fetch the joint limits from the robot
        if q_lim is True:
            q_lim = self.get_JointRanges()

        # Call the IK solver with the appropriate parameters
        return IK_solver(self, trgt_poses=trgt_poses, q0=q0, q_lim=q_lim, mask=mask)

    def calc_DerivatedArray(self, q, t):
        """
        Calculates the numerical derivatives (joint velocities) from an array of joint positions over time.

        This method estimates the joint velocities by calculating the derivative of the joint positions with respect to
        time. The method uses forward, central, and backward differences to compute the velocity for the first,
        intermediate, and last time points, respectively. It handles both prismatic (linear) and revolute (angular) joints
        and ensures that the velocities remain within the valid joint ranges.

        Parameters:
        - q (np.ndarray): A 2D array of joint positions with shape (M, N), where M is the number of time stamps
          and N is the number of joints. Each element `q[i, j]` represents the position of joint `j` at time step `i`.
        - t (np.ndarray): A 1D array of time stamps with length M. It represents the time corresponding to each row
          in `q`. The time steps should be uniform.

        Returns:
        - qd (np.ndarray): A 2D array of the same shape as `q`, representing the joint velocities. The first and last
          velocities are set to zero as specified.

        Notes:
        - The method handles prismatic and revolute joints differently. For revolute joints, distances are calculated
          modulo `2π` to account for angular wrapping.
        """
        # Number of time steps (M) and joints (N)
        M = q.shape[0]
        N = q.shape[1]

        # Initialize velocity array with the same shape as q
        qd = np.zeros_like(q)

        # Calculate time steps (assuming uniform time steps)
        dt = np.diff(t)

        # Get prismatic joints information (True for prismatic, False for revolute)
        prismatic_joints = self.get_PrismaticJoints()

        # Get joint ranges (limits for each joint)
        joint_ranges = self.get_JointRanges()

        # Forward difference for the first time-stamp (i = 0)
        for j in range(N):
            is_linear = prismatic_joints[j]  # Check if the joint is prismatic
            valid_range = joint_ranges[j]  # Get the valid range for the joint

            # Calculate the velocity for the first time step using forward difference
            qd[0, j] = (
                -1
                * AF.calc_dist_in_range(q[1, j], q[0, j], is_linear, valid_range)
                / dt[0]
            )

        # Central difference for the intermediate time-stamps (1 <= i < M-1)
        for i in range(1, M - 1):
            for j in range(N):
                is_linear = prismatic_joints[j]
                valid_range = joint_ranges[j]

                # Calculate the velocity using central difference
                qd[i, j] = (
                    -1
                    * AF.calc_dist_in_range(
                        q[i + 1, j], q[i - 1, j], is_linear, valid_range
                    )
                    / (dt[i] + dt[i - 1])
                )

        # Backward difference for the last time-stamp (i = M-1)
        for j in range(N):
            is_linear = prismatic_joints[j]
            valid_range = joint_ranges[j]

            # Calculate the velocity for the last time step using backward difference
            qd[-1, j] = (
                -1
                * AF.calc_dist_in_range(q[-1, j], q[-2, j], is_linear, valid_range)
                / dt[-1]
            )

        # Set the first and last velocities to zero as specified
        qd[0, :] = 0
        qd[-1, :] = 0

        return qd

    def plot_RobotStatic(self, q: Optional[Union[np.ndarray, list]] = None):
        """
        Plots the static configuration of the robotic arm, including the coordinate axes at the last joint or tool.

        This method visualizes the robot's current configuration in 3D space. It plots the robot's joints, links,
        and coordinate axes for the final joint or tool (end-effector). The method allows for plotting a single
        configuration of the robot and displays it using a 3D plot.

        Parameters:
        - q (Optional[Union[np.ndarray, list]]): The joint configuration to be plotted. If None, the current joint
          values of the robot will be used. If provided, it should be a single configuration (list or numpy array)
          matching the number of joints.

        Raises:
        - ValueError: If multiple configurations are passed. This method supports only a single configuration.

        Notes:
        - The method will reset the robot to its original configuration after plotting.
        """
        curr_q = self.get_JointValues()  # Store the current joint configuration

        # If no configuration is provided, use the current one
        if q is None:
            q = np.array(curr_q)

        elif q is not None:
            q = np.array(q)

        if q.ndim > 1:
            raise ValueError(
                "Function <plot_RobotStatic> only supports single configurations. For multiple input configurations, use <plot_RobotMovement>"
            )

        # Set the robot's joint values to the provided or current configuration
        self.set_JointValues(q)

        # Initialize base position and orientation
        pos = [np.array([0, 0, 0])]
        ori = [np.array([0, 0, 0])]
        tfms = [np.identity(4)]

        # Compute transformations for each joint and store positions and orientations
        for joint in self.joint_list:
            tfms.append(tfms[-1] @ joint.get_Tfm())
            pos.append(SM.GetXYZ(tfms[-1]))
            ori.append(SM.GetRPY(tfms[-1]))

        # Include the tool transformation
        tool_tfm = tfms[-1] @ self.tool.Tfm
        pos.append(tool_tfm[:3, 3])  # Add the tool position
        tfms.append(tool_tfm)  # Add the full tool transformation matrix

        # Plotting the robot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", facecolor="white")
        Lx, Ly, Lz = zip(*pos)  # Extract X, Y, Z coordinates for plotting
        ax.plot(
            Lx, Ly, Lz, marker="o", ls="-", lw=5, c="tomato", ms=2
        )  # Plot the robot's links

        # Compute the total length of the robot for axis scaling
        length = SM.vector_norm(SM.vector(pos[0], pos[-1]))

        # Colors for the coordinate axes
        colors = ["r", "limegreen", "dodgerblue"]

        # Plot Z-axes for each joint and full axes for the tool
        for i in range(len(tfms)):
            axes = np.eye(3) * (
                length / 15
            )  # Scale the axes relative to the robot size
            transformed_axes = (
                tfms[i][:3, :3] @ axes
            )  # Apply the joint transformation to the axes

            # Plot all axes for the tool (end-effector)
            if i == len(tfms) - 1:
                for j in range(3):
                    ax.quiver(
                        *pos[i],
                        *(transformed_axes[:, j] * 1.5),
                        color=colors[j],
                        linewidth=3,
                    )

            # Plot only the Z-axis for intermediate joints
            elif i < (len(tfms) - 2):
                ax.quiver(
                    *pos[i],
                    *transformed_axes[:, 2],
                    color="deepskyblue",
                    linewidth=1.5,
                )

        # Grid and axis labels
        ax.grid(True, linestyle="-", color="whitesmoke", linewidth=0.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim([-length, length])
        ax.set_ylim([-length, length])
        ax.set_zlim([-length / 10, length])

        # Set titles
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

        # Display the plot
        plt.tight_layout()
        plt.show()

        # Reset the robot to its original configuration after plotting
        self.set_JointValues(curr_q)

    def plot_RobotMovement(
        self,
        q: Optional[np.ndarray] = None,
        movie_name: str = None,
        movie_time: float = 5.0,
        movie_resolution: int = 150,
    ):
        """
        Animates the movement of the robotic arm and optionally saves it as a GIF.

        This method visualizes the movement of the robot by animating the given joint configurations over time.
        It can also save the animation as a GIF if a `movie_name` is provided. The robot's movement is plotted
        in 3D space, including the links between joints, the end-effector trajectory, and the coordinate axes
        at the end-effector and joints.

        Parameters:
        - q (Optional[np.ndarray]): A 2D array of joint configurations (shape: time_steps x joints). Each row represents
          a joint configuration at a specific time. If no configuration is provided, the method uses the current configuration.
        - movie_name (str): The name of the GIF file to save the animation. If not provided, the animation is shown but not saved.
        - movie_time (float): The total duration of the GIF in seconds. Default is 5.0 seconds.
        - movie_resolution (int): The resolution (DPI) of the saved GIF. Default is 150 DPI.

        Notes:
        - This method supports dynamic plotting of multiple configurations, unlike `plot_RobotStatic`, which supports static plots.
        - The method resets the robot to its original configuration after the animation.

        Raises:
        - ValueError: If the input `q` is not a valid 1D or 2D numpy array of joint configurations.
        """
        length = (
            self.calc_Reach()
        )  # Calculate the reach of the robot for scaling the plot
        curr_q = self.get_JointValues()  # Store the current joint configuration

        if q is None:
            q = [curr_q]  # Use current configuration if no configurations are provided
        elif q.ndim == 1:
            q = [q]  # Convert a single configuration to a list of one element

        # Setup the 3D plot
        fig = plt.figure()
        plt.suptitle(f"{self.DH_name} Configuration", fontsize=18, fontweight="bold")
        plt.tight_layout()
        ax = fig.add_subplot(111, projection="3d", facecolor="white")

        # Set plot limits based on the robot's reach
        ax.set_xlim([-length, length])
        ax.set_ylim([-length, length])
        ax.set_zlim([-length / 10, length])

        # Initialize lists to store the joint points and trajectory points
        x, y, z = [], [], []
        (joint_points,) = ax.plot(x, y, z, "-", color="tomato", lw=4)
        traj_x, traj_y, traj_z = [], [], []
        (traj_points,) = ax.plot(
            traj_x, traj_y, traj_z, "-", color="darkorchid", lw=1, alpha=0.8
        )

        # Initialize quivers for the coordinate system axes
        xCSYS_EF = ax.quiver([], [], [], [], [], [], color="r", linewidth=2)
        yCSYS_EF = ax.quiver([], [], [], [], [], [], color="lime", linewidth=2)
        zCSYS_EF = ax.quiver([], [], [], [], [], [], color="dodgerblue", linewidth=2)
        zAxis_j = ax.quiver([], [], [], [], [], [], color="deepskyblue", linewidth=1.5)

        def single_plot(frame, q):
            """
            Updates the plot for each frame of the animation based on the current joint configuration.

            Parameters:
            - frame (int): The current frame of the animation.
            - q (np.ndarray): The joint configurations over time.
            """
            self.set_JointValues(
                q[frame]
            )  # Set the robot's joint values to the current frame
            tfms = [np.identity(4)]  # Initialize transformation matrices
            pos = [np.array([0, 0, 0])]  # Initialize position list

            # Compute transformations for each joint
            for j in self.joint_list:
                new_tfm = tfms[-1] @ j.get_Tfm()
                tfms.append(new_tfm)
                pos.append(SM.GetXYZ(tfms[-1]))

            # Compute the tool (end-effector) transformation
            tool_tfm = tfms[-1] @ self.tool.Tfm
            tfms.append(tool_tfm)
            pos.append(SM.GetXYZ(tfms[-1]))

            # Update joint points and trajectory points
            Lx, Ly, Lz = zip(*pos)
            joint_points.set_data_3d(Lx, Ly, Lz)
            traj_x.append(Lx[-1])
            traj_y.append(Ly[-1])
            traj_z.append(Lz[-1])
            traj_points.set_data_3d(traj_x, traj_y, traj_z)

            # Update the coordinate axes for the end-effector and joints
            seg_xCSYS_EF, seg_yCSYS_EF, seg_zCSYS_EF, seg_zAxis_j = [], [], [], []

            for i, tfm in enumerate(tfms):
                axes = np.eye(3) * (length / 15)  # Scale the axes
                transformed_axes = tfm[:3, :3] @ axes  # Transform the coordinate axes

                for j in range(3):
                    v = transformed_axes[:, j]  # Get the axis vector

                    if i == len(tfms) - 1:  # End-effector coordinate system
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
                    elif i < (len(tfms) - 2):  # Joints' Z-axis
                        if j == 2:
                            seg_zAxis_j.insert(
                                0,
                                [
                                    [Lx[i], Ly[i], Lz[i]],
                                    [Lx[i] + v[0], Ly[i] + v[1], Lz[i] + v[2]],
                                ],
                            )

            # Update the quivers for the coordinate system axes
            xCSYS_EF.set_segments(seg_xCSYS_EF)
            yCSYS_EF.set_segments(seg_yCSYS_EF)
            zCSYS_EF.set_segments(seg_zCSYS_EF)
            zAxis_j.set_segments(seg_zAxis_j)

            return xCSYS_EF, yCSYS_EF, zCSYS_EF, zAxis_j, joint_points, traj_points

        # Animate the plot with the provided joint configurations
        ani = FuncAnimation(
            fig, single_plot, frames=len(q), fargs=(q,), blit=True, repeat=False
        )

        # Save the animation as a GIF if movie_name is provided
        if movie_name:
            print("Creating GIF ...")
            ani.save(
                f"{movie_name}.gif",
                writer=PillowWriter(fps=len(q) / movie_time),
                dpi=movie_resolution,
            )
            print("GIF saved at current directory \n")

        # Display the plot
        plt.show()

        # Reset the robot to its original configuration after the animation
        self.set_JointValues(curr_q)

    def plot_JointEvolution(self, q, qd, t, save_name: Optional[str] = None):
        """
        Plots the evolution of joint positions and velocities over time.

        This method generates two plots: one showing the position of each joint over time, and another showing the
        velocity of each joint over time. All joint positions and velocities are plotted on the same respective axes
        with distinct colors for each joint. Optionally, the plot can be saved to a file.

        Parameters:
        - q (np.ndarray): A 2D array of joint positions over time, where each row corresponds to a time step and
          each column represents a joint.
        - qd (np.ndarray): A 2D array of joint velocities over time, with the same shape as `q`. Each row corresponds
          to a time step and each column represents a joint.
        - t (np.ndarray): A 1D array representing the time steps corresponding to the rows of `q` and `qd`.
        - save_name (Optional[str]): An optional string to specify the file name for saving the plot. If not provided,
          the plot is displayed but not saved.

        Notes:
        - The number of joints is determined from the second dimension of `q`.
        - The plot uses different colors for each joint, and a single legend is displayed at the top of the figure.
        """
        # Number of joints
        N = q.shape[1]

        # Create a figure with two subplots: one for position and one for velocity
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), facecolor="white")

        # Set the main title for the entire figure
        fig.suptitle("Joint Kinematics Evolution", fontsize=16, fontweight="bold")

        # Titles for the individual subplots
        ax1.set_title("Position")
        ax2.set_title("Velocity")

        # Define a colormap and generate colors for each joint
        cmap = get_cmap("viridis")
        colors = [cmap(i / N) for i in range(N)]

        # Plot joint positions over time
        labels = []
        for i in range(N):
            ax1.plot(
                t,
                q[:, i],  # Joint positions over time
                label=f"Joint {i+1}",  # Label for the legend
                linewidth=1,  # Fine line width
                linestyle="--",  # Dashed line style
                marker="o",  # Marker style (you can adjust to other markers if needed)
                markersize=4,  # Marker size
                color=colors[i],  # Color for each joint
            )
            labels.append(f"Joint {i+1}")
        ax1.set_ylabel("Position (rad)")
        ax1.grid(True, which="both", linestyle="--", linewidth=0.5)  # Add grid lines
        ax1.set_facecolor("whitesmoke")  # Set background color
        ax1.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.1f}")
        )  # Format y-axis labels
        ax1.set_ylim(
            [-math.pi, math.pi]
        )  # Set the y-axis limits from -π to π for joint positions

        # Plot joint velocities over time
        for i in range(N):
            ax2.plot(
                t,
                qd[:, i],  # Joint velocities over time
                linewidth=1,  # Fine line width
                linestyle="--",  # Dashed line style
                marker="o",  # Marker style
                markersize=4,  # Marker size
                color=colors[i],  # Color for each joint
            )
        ax2.set_ylabel("Velocity (rad/s)")
        ax2.grid(True, which="both", linestyle="--", linewidth=0.5)  # Add grid lines
        ax2.set_facecolor("whitesmoke")  # Set background color
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x:.1f}")
        )  # Format y-axis labels
        ax2.set_xlabel("Time (s)")  # Set x-axis label for time

        # Create a legend for the figure based on joint labels
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),  # Position legend at the top
            ncol=N,  # Arrange labels in a single row
            frameon=False,  # Disable the frame around the legend
        )

        # Adjust layout to prevent overlap between subplots and ensure everything fits
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # If a save_name is provided, save the figure
        if save_name:
            plt.savefig(fname=save_name)

        # Display the figure
        plt.show()

    def __str__(self):
        """
        Returns a formatted string representation of the DHRobot object, including the Denavit-Hartenberg (DH)
        parameters of each joint and the tool's information.

        The string representation includes:
        - The robot's name (`DH_name`) and home configuration (`home_q`).
        - A table with the DH parameters for each joint: `theta`, `d`, `a`, `alpha`, and `qlim` (joint limits).
        - Tool information, including its position (xyz) and orientation (rpy).

        Returns:
        - str: A formatted string summarizing the robot's DH parameters and tool information.
        """
        # Initialize the table string with a header for the robot's name and home configuration
        table_str = "\n"
        table_str += f"DH {self.DH_name} \n"  # Robot name
        table_str += f"(home q: {self.home_q}) \n"  # Home configuration of the robot
        table_str += "-" * 64 + "\n"  # Divider line

        # Add the column headers for the DH parameter table
        table_str += (
            "Joint |  theta  |    d    |    a    |  alpha  |      qlim      |\n"
        )
        table_str += "-" * 64 + "\n"

        # Format string for each row of the DH parameter table
        DH_table_format = "{:<5} | {:^7} | {:^7} | {:^7} | {:^7} | [{:^5} ,{:^5}] |\n"

        # Loop through each joint and format its parameters into the table
        for idx, joint in enumerate(self.joint_list, 1):
            # Format theta and d based on joint type (theta for revolute, d for prismatic)
            theta = (
                "{:.2f}".format(joint.theta)
                if joint.j_type == 1  # Revolute joint
                else "q{:.0f}".format(idx)  # Prismatic joint
            )
            d = "{:.2f}".format(joint.d) if joint.j_type == 0 else "q{:.0f}".format(idx)

            # Format the remaining DH parameters (a, alpha, and qlim)
            a = "{:.2f}".format(joint.a)
            alpha = "{:.2f}".format(joint.alpha)
            qlim_min = "{:.2f}".format(joint.qlim[0])  # Joint limit (min)
            qlim_max = "{:.2f}".format(joint.qlim[1])  # Joint limit (max)

            # Add the formatted row to the table
            table_str += DH_table_format.format(
                idx, theta, d, a, alpha, qlim_min, qlim_max
            )

        # Add another divider after the DH table
        table_str += "-" * 64 + "\n"

        # Format tool information (position xyz and orientation rpy)
        tXYZ = ", ".join("{:.2f}".format(t) for t in self.tool.tXYZ)
        rXYZ = ", ".join("{:.2f}°".format(r) for r in self.tool.rXYZ)
        tool_info = f"tool | xyz = ({tXYZ}) | rpy = ({rXYZ})\n"

        # Add tool information to the table
        table_str += "-" * 61 + "\n"
        table_str += tool_info
        table_str += "-" * 61 + "\n"

        # Return the full formatted string
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
            qlim=[-(6.75 / 10) * math.pi, (6.75 / 10) * math.pi],
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

    robot.qz = np.array([0, 0, 0, 0, 0], dtype=float)
    robot.qv = np.array([math.pi / 2, math.pi / 2, -math.pi / 2, 0, 0], dtype=float)
    robot.q2 = np.array(
        [-(6.75 / 10) * math.pi, -math.pi / 13, -math.pi / 4, -math.pi, -math.pi / 4],
        dtype=float,
    )

    start_q = robot.qz
    end_q = robot.q2

    robot.set_JointValues(start_q)
    start_pose = robot.get_EndEffPosOr()
    print(start_pose)
    robot.plot_RobotStatic()

    print()

    robot.set_JointValues(end_q)
    end_pose = robot.get_EndEffPosOr()
    print(end_pose)
    robot.plot_RobotStatic()

    # JOINT INTERPOLATION
    j_traj = LinealTrajectory.create(start_pose, end_pose, 0.01, 0.1, 2, 5)

    j_traj.q = robot.calc_IK(
        trgt_poses=j_traj.pose,
        q0=start_q,
        q_lim=True,
        mask=[1, 1, 1, 1, 1, 1],
    )

    j_traj.q, j_traj.t = rmve_ConfigurationJump(
        j_traj.q,
        j_traj.t,
        robot.get_JointRanges(),
        robot.get_PrismaticJoints(),
    )

    j_traj.qd = robot.calc_DerivatedArray(j_traj.q, j_traj.t)

    robot.plot_RobotMovement(j_traj.q, "test_JointInterp_Move_Final_2")

    robot.plot_JointEvolution(
        j_traj.q, j_traj.qd, j_traj.t, "test_JointInterp_Values_Final_2"
    )

    # CARTESIAN INTERPOLATION - STRAIGHT LINE
    l_traj = LinealTrajectory.create(start_pose, end_pose, 0.01, 0.1, 100, 5)

    l_traj.q = robot.calc_IK(
        trgt_poses=l_traj.pose,
        q0=start_q,
        q_lim=True,
        mask=[1, 1, 1, 0, 0, 1],
    )

    robot.plot_RobotMovement(
        l_traj.q,
        "test_CartInterp_Move_untreated_Final_2",
    )

    l_traj.qd = robot.calc_DerivatedArray(l_traj.q, l_traj.t)

    robot.plot_JointEvolution(
        l_traj.q,
        l_traj.qd,
        l_traj.t,
        "test_CartInterp_Values_untreated_Final_2",
    )

    l_traj.plot(skip=5, save_name="test_CartInterp_Path_untreated_Final_2")

    l_traj.q, l_traj.t = rmve_ConfigurationJump(
        l_traj.q,
        l_traj.t,
        robot.get_JointRanges(),
        robot.get_PrismaticJoints(),
    )

    l_traj.qd = robot.calc_DerivatedArray(l_traj.q, l_traj.t)

    robot.plot_RobotMovement(
        l_traj.q,
        "test_CartInterp_Move_treated_Final_2",
    )

    robot.plot_JointEvolution(
        l_traj.q,
        l_traj.qd,
        l_traj.t,
        "test_CartInterp_Values_treated_Final_2",
    )
    l_traj.plot(skip=5, save_name="test_CartInterp_Path_treated_Final_2")
