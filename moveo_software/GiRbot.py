import numpy as np
import math

import AuxiliarFunctions as AF
from DHRobot import JointDH, RobotTool, PrismaticDH, RevoluteDH
from PhysicalRobot import PhysicalRobot, StepperMotor
from IsochroneMove import IsochroneMove as isocrone
from LinealTrajectory import LinealTrajectory, rmve_ConfigurationJump

from nIK_Corke import ikine_LM_Corke as IK_solver

from typing import List, Union, Optional, Callable, Type

import moveo_driver as MD
import time


def check_InputArray(data):
    """
    Validates whether the input `data` is both iterable and indexable.

    This function first checks if the input is iterable by attempting to create
    an iterator. If the input is not iterable, it raises a TypeError. It then
    checks if the input supports indexing by accessing the first element. If
    it does not support indexing, another TypeError is raised. If the input is
    empty, the function simply passes without error.

    Parameters:
    data: Any
        The input to check for iterable and indexable properties.

    Raises:
    TypeError: If the input is not iterable or does not support indexing.
    """

    try:
        # Check if `data` is iterable by trying to create an iterator
        iterator = iter(data)
    except TypeError:
        # Raise an error if `data` is not iterable
        raise TypeError("Input must be iterable")

    try:
        # Check if `data` supports indexing by attempting to access the first element
        item = data[0]
    except TypeError:
        # Raise an error if `data` does not support indexing
        raise TypeError("Input must support indexing")
    except IndexError:
        # If `data` is empty, no action is needed
        pass


class GiRbot(PhysicalRobot):
    """
    GiRbot is a subclass of PhysicalRobot designed to represent a robotic system controlled via an Arduino board.

    Attributes:
        connected (bool): Status of the connection to the Arduino board.
        mov_time (float): Time interval between movements, in seconds.
        driver (Type): Driver for interfacing with the Arduino hardware.
        current_q (List): The current configuration of the robot's joints.
    """

    def __init__(
        self,
        name: str,
        ArduinoDriver: Type,
        MotorsPerJoint: List[List[StepperMotor]],
        DH_JointList: List[Union[JointDH, RevoluteDH, PrismaticDH]],
        DH_IKsolver: Callable,
        DH_Tool: Optional[RobotTool] = None,
        connection_port: str = "/dev/ttyACM0",
    ):
        """
        Initializes the GiRbot class with the necessary parameters to control a robotic system.

        Args:
            name (str): The name of the robot.
            ArduinoDriver (Type): Driver class for communicating with the Arduino board.
            MotorsPerJoint (List[List[StepperMotor]]): A list of stepper motors associated with each joint of the robot.
            DH_JointList (List[Union[JointDH, RevoluteDH, PrismaticDH]]): List of Denavit-Hartenberg (DH) joint representations.
            DH_IKsolver (Callable): A function or object used to compute the inverse kinematics for the robot.
            DH_Tool (Optional[RobotTool], optional): The tool attached to the end effector of the robot. Defaults to None.
            connection_port (str, optional): The serial port used to communicate with the Arduino. Defaults to "/dev/ttyACM0".
        """
        super().__init__(
            name=name,
            MotorsPerJoint=MotorsPerJoint,
            DH_JointList=DH_JointList,
            DH_IKsolver=DH_IKsolver,
            DH_Tool=DH_Tool,
            connection_port=connection_port,
        )

        # Indicates whether the robot is connected to the Arduino.
        self.connected = False

        # Movement time interval, defaulted to 0.01 seconds.
        self.mov_time = 0.01

        # Driver class for interfacing with the Arduino.
        self.driver = ArduinoDriver

        # Sets the current joint configuration to the home position.
        self.current_q = self.home_q

        # Error message shown when the Arduino connection is not established.
        self.not_connected_error_msg = "Not connected to the Arduino board! Please make sure to successfully call <cmd_Connect>."

    def cmd_Connect(self):
        """
        Establishes a connection between the robot and the Arduino board.

        This method attempts to connect to the Arduino through the specified connection port.
        If the connection is unsuccessful, it raises an exception. Upon success, the connection
        status is updated, and a short delay is introduced before returning.

        Returns:
            bool: True if the connection is successful.

        Raises:
            Exception: If the connection to the Arduino fails.
        """

        # Print a message indicating that the connection process is starting.
        print("Connecting to Arduino...")

        # Attempt to connect to the Arduino board using the specified port.
        if not self.driver.connect(self.connection_port):
            # Raise an exception if the connection fails.
            raise Exception("Error connecting to Moveo!")

        # Print a success message if the connection is established.
        print("Successfully connected")

        # Update the connection status to True.
        self.connected = True

        # Introduce a short delay to ensure stable connection initialization.
        time.sleep(0.1)

        # Return True to indicate a successful connection.
        return True

    def cmd_setMotorParams(self):
        """
        Sets the default motor parameters (maximum velocity and acceleration) for each motor.

        This method collects the maximum velocity and acceleration parameters from each motor
        and passes them to the Arduino driver to configure the motors accordingly. If the robot
        is not connected to the Arduino, it raises an exception.

        Returns:
            bool: True if the motor parameters are successfully set.

        Raises:
            Exception: If the robot is not connected or if setting motor parameters fails.
        """

        # Check if the robot is connected to the Arduino.
        if self.connected:
            print("Setting default motor parameters...")

            # Initialize lists to store the max velocity and acceleration for each motor.
            m_maxVel = []
            m_maxAccel = []

            # Loop through each joint and motor to collect their max velocity and acceleration.
            for j in self.MotorsPerJoint:
                for m in j:
                    m_maxVel.append(m.maxStepsVel)  # Store max velocity of each motor.
                    m_maxAccel.append(
                        m.maxStepsAccel
                    )  # Store max acceleration of each motor.

            # Combine the velocity and acceleration lists into a single list of motor parameters.
            motor_params = m_maxVel + m_maxAccel

            # Send the motor parameters to the Arduino driver. Raise an exception if it fails.
            if not self.driver.setMotorParams(motor_params):
                raise Exception("Error setting motor parameters!")

            # Print the rounded motor velocity and acceleration values for confirmation.
            print("Max. Velocity in steps/s:", AF.round_list(m_maxVel))
            print("Max. Acceleration in steps/s²:", AF.round_list(m_maxAccel))

            print("Successfully set default motor parameters")

            # Introduce a short delay before finishing.
            time.sleep(0.1)
            return True

        else:
            # If the robot is not connected, raise an exception with an appropriate error message.
            raise Exception(self.not_connected_error_msg)

    def cmd_setMotorRanges(self):
        """
        Sets the default movement limits (range) for each motor.

        This method retrieves the movement range for each motor in steps, flattens the list, and
        sends it to the Arduino driver to configure the motor limits. If the robot is not connected
        to the Arduino, it raises an exception.

        Returns:
            bool: True if the motor ranges are successfully set.

        Raises:
            Exception: If the robot is not connected or if setting motor ranges fails.
        """

        # Check if the robot is connected to the Arduino.
        if self.connected:
            print("Setting default motor limits...")

            # Retrieve the step range for each motor in the robot.
            m_ranges = self.get_MotorStepRange()

            # Print the motor ranges for confirmation.
            print("Motors Ranges in steps:", m_ranges)

            # Flatten the nested list of motor ranges into a single list.
            motor_ranges = AF.flatten_list(m_ranges)

            # Send the motor ranges to the Arduino driver. Raise an exception if it fails.
            if not self.driver.setMotorRanges(motor_ranges):
                raise Exception("Error setting motor Ranges")

            # Print success message after setting the motor ranges.
            print("Successfully set motor limits")

            # Introduce a short delay before finishing.
            time.sleep(0.1)
            return True

        else:
            # If the robot is not connected, raise an exception with the error message.
            raise Exception(self.not_connected_error_msg)

    def cmd_Stop(self):
        """
        Stops all motors of the robot.

        This method sends a stop command to the Arduino driver to halt all motor movements.
        If the robot is not connected to the Arduino, it raises an exception.

        Returns:
            bool: True if the motors are successfully stopped.

        Raises:
            Exception: If the robot is not connected or if the stop command fails.
        """

        # Check if the robot is connected to the Arduino.
        if self.connected:
            print("Stopping motor...")

            # Send a stop command to the Arduino driver. Raise an exception if it fails.
            if not self.driver.stop():
                raise Exception("Error stopping Motors")

            # Print success message after the motors are successfully stopped.
            print("Successfully stopped")

            # Introduce a short delay before finishing.
            time.sleep(0.1)
            return True

        else:
            # If the robot is not connected, raise an exception with the error message.
            raise Exception(self.not_connected_error_msg)

    def cmd_enableMotors(self):
        """
        Enables all motors of the robot.

        This method sends a command to the Arduino driver to enable the motors, making them
        ready for movement. If the robot is not connected to the Arduino, it raises an exception.

        Returns:
            bool: True if the motors are successfully enabled.

        Raises:
            Exception: If the robot is not connected or if enabling the motors fails.
        """

        # Check if the robot is connected to the Arduino.
        if self.connected:
            print("Enabling motors...")

            # Send an enable command to the Arduino driver. Raise an exception if it fails.
            if not self.driver.enable():
                raise Exception("Error enabling Motors")

            # Print success message after the motors are successfully enabled.
            print("Successfully enabled motors.")

            # Introduce a short delay before finishing.
            time.sleep(0.1)
            return True

        else:
            # If the robot is not connected, raise an exception with the error message.
            raise Exception(self.not_connected_error_msg)

    def cmd_disableMotors(self):
        """
        Disables all motors of the robot.

        This method sends a command to the Arduino driver to disable the motors, preventing them
        from moving. If the robot is not connected to the Arduino, it raises an exception.

        Returns:
            bool: True if the motors are successfully disabled.

        Raises:
            Exception: If the robot is not connected or if disabling the motors fails.
        """

        # Check if the robot is connected to the Arduino.
        if self.connected:
            print("Disabling motors...")

            # Send a disable command to the Arduino driver. Raise an exception if it fails.
            if not self.driver.disable():
                raise Exception("Error disabling Motors")

            # Print success message after the motors are successfully disabled.
            print("Successfully disabled motors")

            # Introduce a short delay before finishing.
            time.sleep(0.1)
            return True

        else:
            # If the robot is not connected, raise an exception with the error message.
            raise Exception(self.not_connected_error_msg)

    def cmd_getJointValues(self):
        """
        Retrieves the current joint values of the robot.

        This method communicates with the Arduino to get the current position of all motors in terms of steps,
        then converts those motor steps into joint angles. The joint angles are wrapped to a valid range
        (0 to 2π). If the robot is not connected or an issue occurs during the process, it raises an exception.

        Returns:
            List: A list of joint angles in radians, representing the current position of the robot's joints.

        Raises:
            Exception: If the robot is not connected or if the motor positions cannot be retrieved.
        """

        # Check if the robot is connected to the Arduino.
        if self.connected:
            print("Getting joint values...")

            # Get the current motor positions in steps from the Arduino.
            flat_qm = self.driver.getCurrentPosition()

            # Check if the number of retrieved motor positions matches the expected number.
            if len(flat_qm) != self.n_motors:
                raise Exception("Error getting position!")

            # Print the current motor steps position for verification.
            print("Motor Current Steps Position:", AF.round_list(flat_qm))

            # Unflatten the list of motor positions to match the motor-joint structure.
            qm = AF.unflatten_list(flat_qm, self.n_MotorsPerJoint)

            # Convert motor steps to joint angles based on the robot's kinematics.
            qj = self.get_JointAnglesFromMotorSteps(motor_steps=qm, out_type="Position")

            # Wrap joint angles to stay within the range [0, 2π].
            qj = AF.wrap_list_half_max(qj, 2 * math.pi).tolist()

            # Print a success message indicating joint values have been obtained.
            print("Successfully obtained joint values.")

            # Introduce a short delay before finishing.
            time.sleep(0.1)

            # Return the list of joint angles.
            return qj

        else:

            raise Exception(self.not_connected_error_msg)

    def cmd_getEndEffectorPose(self):
        """
        Retrieves the current pose of the robot's end effector.

        This method first retrieves the current joint values of the robot, then uses forward kinematics (FK)
        to calculate the end effector's pose based on those joint values.

        Returns:
            pose: The current pose of the end effector, typically in the form of a transformation matrix
            or position/orientation data.
        """

        # Get the current joint values (angles) of the robot.
        qj = self.cmd_getJointValues()

        # Calculate the end effector's pose using forward kinematics.
        pose = self.calc_FK(qj)

        # Return the computed pose.
        return pose

    def cmd_setTargetPose(
        self,
        pose: Optional[list] = None,
        q: Optional[list] = None,
        mask=[[1, 1, 1, 1, 0, 0]],
    ):
        """
        Sets the target pose or joint values for the robot.

        This method allows setting the robot's target either by specifying the end-effector pose or
        directly using joint values. It computes the necessary motor steps based on the input and
        sends them to the Arduino driver. Exactly one of 'pose' or 'q' must be provided.

        Args:
            pose (Optional[list]): The target pose for the end effector (as a list or array). If provided,
                the inverse kinematics (IK) solver will be used to compute joint angles.
            q (Optional[list]): The target joint angles. If provided, it will set the joint angles directly.
            mask (list, optional): A mask used in the IK computation to control which degrees of freedom are used.
                Defaults to [[1, 1, 1, 1, 0, 0]].

        Returns:
            bool: True if the target pose or joint values are successfully set.

        Raises:
            ValueError: If neither 'pose' nor 'q' is provided, or if both are provided simultaneously.
            Exception: If the robot is not connected or setting the target position fails.
        """

        # Ensure that exactly one of 'pose' or 'q' is provided, raise an error if not.
        if (pose is None and q is None) or (pose is not None and q is not None):
            raise ValueError("Exactly one of 'pose' or 'q' must be provided")

        # Check if the robot is connected to the Arduino.
        if self.connected:
            print("Setting target position...")

            # Get the current joint values from the robot.
            self.current_q = self.cmd_getJointValues()

            # If a target pose is provided, use inverse kinematics to calculate joint angles.
            if pose is not None:
                check_InputArray(pose)

                # Convert the pose to a NumPy array.
                pose = np.array(pose)

                # Compute the joint angles from the target pose using inverse kinematics (IK).
                qj = self.calc_IK(trgt_poses=pose, mask=mask)

                # Set the joint values and compute the corresponding motor steps.
                self.set_JointValues(qj)
                qm = self.get_MotorStepsFromJointAngles(
                    joint_angles=qj, out_type="Position"
                )
                flat_qm = AF.flatten_list(qm)

            # If target joint values are provided, use them directly.
            elif q is not None:
                check_InputArray(q)

                # Set the joint values and compute the corresponding motor steps.
                self.set_JointValues(q)
                qm = self.get_MotorStepsFromJointAngles(
                    joint_angles=q, out_type="Position"
                )
                flat_qm = AF.flatten_list(qm)

            # Send the target motor steps to the Arduino driver and raise an error if it fails.
            if not self.driver.setTargetPosition(flat_qm):
                raise Exception("Error setting Position")

            # Print the motor target steps for confirmation.
            print("Motor Target Steps Position:", AF.round_list(flat_qm))
            print("Successfully set target position.")

            # Introduce a short delay before finishing.
            time.sleep(0.1)
            return True

        else:
            # If the robot is not connected, raise an exception with the error message.
            raise Exception(self.not_connected_error_msg)

    def cmd_moveToPosition(
        self,
        maxVel,
        Accel,
        trgtPose: Optional[list] = None,
        trgtQ: Optional[list] = None,
    ):
        """
        Moves the robot to a specified target position by either pose or joint angles.

        This method allows for movement by either providing the target pose of the end effector
        or directly specifying the target joint angles. It computes the necessary motor velocities
        and accelerations to reach the target within the provided limits. Exactly one of 'trgtPose'
        or 'trgtQ' must be provided.

        Args:
            maxVel: The maximum velocity for each joint or motor during the movement.
            Accel: The acceleration for each joint or motor during the movement.
            trgtPose (Optional[list]): Target pose of the end effector. If provided, joint values will be computed via IK.
            trgtQ (Optional[list]): Target joint angles. If provided, the movement will be executed directly.

        Returns:
            bool: True if the movement to the target position is successfully set.

        Raises:
            ValueError: If neither 'trgtPose' nor 'trgtQ' is provided, or both are provided simultaneously.
            Exception: If the robot is not connected or if setting the movement to position fails.
        """

        # Ensure that exactly one of 'trgtPose' or 'trgtQ' is provided, raise an error if not.
        if (trgtPose is None and trgtQ is None) or (
            trgtPose is not None and trgtQ is not None
        ):
            raise ValueError("Exactly one of 'trgtPose' or 'trgtQ' must be provided")

        # Check if the robot is connected to the Arduino.
        if self.connected:
            print("Setting movement type to position...")

            # If a target pose is provided, use the current joint values (assume IK is calculated elsewhere).
            if trgtPose is not None:
                check_InputArray(trgtPose)
                trgtValues = self.cmd_getJointValues()

            # If target joint angles are provided, use them directly.
            elif trgtQ is not None:
                check_InputArray(trgtQ)
                trgtValues = list(trgtQ)

            # Retrieve the current joint values.
            crrntValues = self.current_q

            # Print the current and target joint values for verification.
            print("Current Joint values:", crrntValues)
            print("Target Joint values:", trgtValues)

            # Check if the current and target joint values are equal.
            if AF.floats_equal(crrntValues, trgtValues):
                # If the target and current values are the same, set velocity and acceleration to zero.
                flat_m_maxRPM = [0] * self.n_joint
                flat_m_Accel = [0] * self.n_joint
                self.mov_time = 0.01

            else:
                # Calculate the time required and maximum velocity for each joint to reach the target.
                self.mov_time, j_maxVel = isocrone.get_TimeJointVel(
                    currValues=crrntValues,
                    trgtValues=trgtValues,
                    maxVel=maxVel,
                    accel=Accel,
                    isPrism=self.get_PrismaticJoints(),
                    validRange=self.get_JointRanges(),
                )

                # Convert the joint velocities to motor step velocities.
                m_maxVel = self.get_MotorStepsFromJointAngles(
                    joint_angles=j_maxVel,
                    out_type="Velocity",
                )

                # Set constant acceleration for the joints.
                j_Accel = [Accel] * self.n_joint
                m_Accel = self.get_MotorStepsFromJointAngles(
                    joint_angles=j_Accel,
                    out_type="Acceleration",
                )

                # Check if the calculated velocities and accelerations are valid.
                self.check_Vel(m_maxVel)
                self.check_Accel(m_Accel)

                # Flatten the velocity and acceleration lists for packet preparation.
                flat_m_maxVel = AF.flatten_list(m_maxVel)
                flat_m_Accel = AF.flatten_list(m_Accel)

            # Combine the motor velocity and acceleration parameters and ensure all values are positive.
            motor_params = [abs(num) for num in flat_m_maxVel + flat_m_Accel]

            # Send the motor parameters to the Arduino driver to set the movement to the target position.
            if not self.driver.moveToPosition(motor_params):
                raise Exception("Error setting movement to position!")

            # Print success message after the movement is set.
            print("Successfully set movement type to position.")

            # Update the current joint values to the target values.
            self.current_q = trgtValues

            # Introduce a short delay before finishing.
            time.sleep(0.1)
            return True

        else:
            # If the robot is not connected, raise an exception with the error message.
            raise Exception(self.not_connected_error_msg)

    def cmd_trajToPosition(
        self,
        cartVel,
        cartAccel,
        n_interp,
        trgtPose: Optional[list] = None,
        trgtQ: Optional[list] = None,
        mask=[1, 1, 1, 1, 0, 0],
    ):
        """
        Moves the robot along a linear trajectory to a specified target pose or joint configuration.

        This method generates a linear trajectory from the current pose to the target pose, or
        from the current joint configuration to the target joint values, and interpolates the
        motion based on the provided Cartesian velocity, acceleration, and interpolation points.
        Exactly one of 'trgtPose' or 'trgtQ' must be provided.

        Args:
            cartVel: Cartesian velocity for the movement.
            cartAccel: Cartesian acceleration for the movement.
            n_interp: Number of interpolation points for the trajectory.
            trgtPose (Optional[list]): The target end-effector pose for the robot.
            trgtQ (Optional[list]): The target joint configuration for the robot.
            mask (list, optional): Mask used in inverse kinematics to control which degrees of freedom
                are used. Defaults to [1, 1, 1, 1, 0, 0].

        Raises:
            ValueError: If neither 'trgtPose' nor 'trgtQ' is provided, or both are provided simultaneously.
            Exception: If the robot is not connected or if a movement error occurs.
        """

        # Ensure that exactly one of 'trgtPose' or 'trgtQ' is provided, raise an error if not.
        if (trgtPose is None and trgtQ is None) or (
            trgtPose is not None and trgtQ is not None
        ):
            raise ValueError("Exactly one of 'trgtPose' or 'trgtQ' must be provided")

        # Check if the robot is connected to the Arduino.
        if self.connected:
            print("Setting movement type to position...")

            # If a target joint configuration is provided, calculate the corresponding pose using FK.
            if trgtQ is not None:
                check_InputArray(trgtQ)
                trgtPose = self.calc_FK(trgtQ)

            # If a target pose is provided, ensure the input is valid.
            elif trgtPose is not None:
                check_InputArray(trgtPose)

            # Get the current joint configuration and calculate the corresponding pose using FK.
            crrntQ = self.current_q
            crrntPose = self.calc_FK(crrntQ)
            print("Starting Joint Configuration", crrntQ)

            # Create a unique name for the trajectory based on the start and target poses.
            traj_name = (
                str(AF.round_list(list(crrntPose)))
                + "-to-"
                + str(AF.round_list(list(trgtPose)))
            )

            print(
                f"Generating Linear Traj.\nPose1: {AF.round_list(list(crrntPose))}\n"
                f"to Pose2: {AF.round_list(list(trgtPose))} * if possible."
            )

            # Create a linear trajectory from the current pose to the target pose.
            traj = LinealTrajectory.create(
                StrtPose=crrntPose,
                TrgtPose=trgtPose,
                MoveVel=cartVel,
                CartAccel=cartAccel,
                n_interp=n_interp,
                n_dofs=self.n_joint,
            )

            # Solve inverse kinematics to get the joint values along the trajectory.
            traj.q = self.calc_IK(
                trgt_poses=traj.pose,
                q0=crrntQ,
                mask=mask,
                q_lim=True,
            )

            # Remove any singularities from the trajectory.
            traj.q, traj.t = LinealTrajectory.rmve_ConfigurationJump(
                q=traj.q,
                t=traj.t,
                joint_ranges=self.get_JointRanges(),
                joint_type=self.get_PrismaticJoints(),
            )

            # Compute the joint velocities for each interpolation point.
            traj.qd = self.calc_DerivatedArray(traj.q, traj.t)

            # Convert the joint velocities into motor step velocities.
            m_qd = []
            for i, qd in enumerate(traj.qd):
                m_qd_i = self.get_MotorStepsFromJointAngles(
                    joint_angles=qd, out_type="Velocity"
                )

                # Check if the motor velocities are valid, raise an error if not.
                try:
                    self.check_Vel(m_qd_i)
                except ValueError as e:
                    raise ValueError(
                        f"Interpolation point {i}\n"
                        + str(e)
                        + "\nTry reducing the cartesian velocity of the movement."
                    )

                # Flatten the motor velocities and add them to the list.
                m_qd.append(AF.flatten_list(m_qd_i))

            t_inc = traj.t[-1] - traj.t[-2]  # Time increment between trajectory points.

            print(f"Total time of trajectory: {traj.t[-1]}s")
            print(f"Increments of {t_inc}s")

            # Send the motor velocity commands to the driver at each trajectory point.
            for i, qd in enumerate(m_qd):
                print("-")
                print("Steps/s:", AF.round_list(qd))
                print(f"Timestamp: {traj.t[i]}")
                self.driver.trajToPosition(list(qd))
                time.sleep(t_inc)

            # Set the final position after completing the trajectory.
            self.mov_time = 0.1
            self.cmd_setTargetPose(q=traj.q[-1])
            self.current_q = traj.q[-1]

            print("Successfully set movement type to position.")

        else:
            # If the robot is not connected, raise an exception with the error message.
            raise Exception(self.not_connected_error_msg)

    ########################################################################################################################
    #                                            FINAL COMMANDS                                                           #
    ########################################################################################################################

    def cmd_Init(self):
        """
        Initializes the robot by connecting to the Arduino, setting motor parameters, and motor ranges.

        This method is a high-level initialization routine that performs the following steps:
        1. Connects to the robot via the Arduino.
        2. Sets the default motor parameters.
        3. Sets the default motor ranges.

        Returns:
            bool: True if the initialization is successful.
        """

        # Print initialization header.
        print()
        print("------- INITIALIZE ROBOT ----------")

        # Connect to the robot via the Arduino.
        self.cmd_Connect()

        # Set the default motor parameters (velocity and acceleration).
        self.cmd_setMotorParams()

        # Set the default motor ranges (movement limits).
        self.cmd_setMotorRanges()

        # Print initialization footer.
        print("-----------------------------------")
        print()

        return True

    def cmd_openGripper(self):
        """
        Opens the robot's gripper.

        This method sends a command to the Arduino driver to open the gripper. If the robot is not connected,
        it raises an exception.

        Returns:
            bool: True if the gripper is successfully opened.

        Raises:
            Exception: If the robot is not connected or if there is an error opening the gripper.
        """

        # Print gripper opening header.
        print()
        print("------- OPEN GRIPPER ----------")

        # Check if the robot is connected to the Arduino.
        if self.connected:
            print("Opening gripper...")

            # Send the open gripper command to the Arduino driver. Raise an exception if it fails.
            if not self.driver.openGripper():
                raise Exception("Error opening Gripper")

            # Print success message after the gripper is opened.
            print("Successfully opened gripper.")

            # Introduce a short delay to allow the gripper to fully open.
            time.sleep(0.3)

            # Print gripper opening footer.
            print("-----------------------------------")
            print()

            return True

        else:
            # If the robot is not connected, raise an exception with the error message.
            raise Exception(self.not_connected_error_msg)

    def cmd_closeGripper(self):
        """
        Closes the robot's gripper.

        This method sends a command to the Arduino driver to close the gripper. If the robot is not connected,
        it raises an exception.

        Returns:
            bool: True if the gripper is successfully closed.

        Raises:
            Exception: If the robot is not connected or if there is an error closing the gripper.
        """

        # Print gripper closing header.
        print()
        print("------- CLOSING GRIPPER ----------")

        # Check if the robot is connected to the Arduino.
        if self.connected:
            print("Closing gripper...")

            # Send the close gripper command to the Arduino driver. Raise an exception if it fails.
            if not self.driver.closeGripper():
                raise Exception("Error closing Gripper")

            # Print success message after the gripper is closed.
            print("Successfully closed gripper.")

            # Introduce a short delay to allow the gripper to fully close.
            time.sleep(0.3)

            # Print gripper closing footer.
            print("-----------------------------------")
            print()

            return True

        else:
            # If the robot is not connected, raise an exception with the error message.
            raise Exception(self.not_connected_error_msg)

    def cmd_WaitEndMove(self):
        """
        Waits for the end of the current movement.

        This method pauses execution for the duration of the robot's current movement (as specified
        by `self.mov_time`), then stops the motors once the movement is completed.

        Returns:
            bool: True when the movement has successfully ended.
        """

        # Print header for waiting for the movement to end.
        print()
        print("------------ WAIT END MOVE --------------")

        # Print the duration of the wait time for the current movement to complete.
        print(f"Waiting {round(self.mov_time)}s for the end of the movement")

        # Pause execution for the duration of the movement.
        time.sleep(self.mov_time)

        # Stop the robot's movement after waiting.
        self.cmd_Stop()

        # Print confirmation that the movement has ended.
        print("Movement ended")

        # Print footer indicating the end of the process.
        print("-----------------------------------")
        print()

        return True

    def cmd_MoveL(
        self,
        cartVel,
        cartAccel,
        trgtPose: Optional[list] = None,
        trgtQ: Optional[list] = None,
        n_interp=500,
        mask=[1, 1, 1, 1, 0, 0],
    ):
        """
        Moves the robot in a straight line to a specified target pose or joint configuration.

        This method moves the robot along a linear path (MoveL) to either the target pose or the
        target joint configuration by generating a trajectory and executing it.

        Args:
            cartVel: Cartesian velocity for the linear movement.
            cartAccel: Cartesian acceleration for the linear movement.
            trgtPose (Optional[list]): Target end-effector pose for the robot. Must be provided if 'trgtQ' is None.
            trgtQ (Optional[list]): Target joint configuration for the robot. Must be provided if 'trgtPose' is None.
            n_interp (int, optional): Number of interpolation points for the trajectory. Defaults to 500.
            mask (list, optional): Mask used in inverse kinematics to control which degrees of freedom
                are used. Defaults to [1, 1, 1, 1, 0, 0].

        Raises:
            ValueError: If neither 'trgtPose' nor 'trgtQ' is provided, or both are provided simultaneously.
        """

        # Print header indicating the start of the linear movement.
        print()
        print("------------ MOVE LINE --------------")

        # Ensure that exactly one of 'trgtPose' or 'trgtQ' is provided, raise an error if not.
        if (trgtPose is None and trgtQ is None) or (
            trgtPose is not None and trgtQ is not None
        ):
            raise ValueError("Exactly one of 'trgtPose' or 'trgtQ' must be provided")

        # If a target pose is provided, set the target pose and execute a trajectory to that pose.
        if trgtPose is not None:
            self.cmd_setTargetPose(pose=trgtPose)
            self.cmd_trajToPosition(
                trgtPose=trgtPose,
                cartVel=cartVel,
                cartAccel=cartAccel,
                n_interp=n_interp,
                mask=mask,
            )

        # If a target joint configuration is provided, set the target joint angles and move to that configuration.
        elif trgtQ is not None:
            self.cmd_setTargetPose(q=trgtQ)
            self.cmd_trajToPosition(
                trgtQ=trgtQ,
                cartVel=cartVel,
                cartAccel=cartAccel,
                n_interp=n_interp,
                mask=mask,
            )

        # Print footer indicating the end of the linear movement.
        print("-----------------------------------")
        print()

    def cmd_MoveJ(
        self,
        maxVel,
        Accel,
        trgtPose: Optional[list] = None,
        trgtQ: Optional[list] = None,
    ):
        """
        Moves the robot joints to a specified target pose or joint configuration.

        This method allows joint-based movement (MoveJ) to either a target pose or target joint angles.
        It ensures smooth motion using the specified maximum velocity and acceleration.

        Args:
            maxVel: The maximum velocity for the joints during the movement.
            Accel: The acceleration for the joints during the movement.
            trgtPose (Optional[list]): The target end-effector pose for the robot. Must be provided if 'trgtQ' is None.
            trgtQ (Optional[list]): The target joint configuration for the robot. Must be provided if 'trgtPose' is None.

        Raises:
            ValueError: If neither 'trgtPose' nor 'trgtQ' is provided, or both are provided simultaneously.
        """

        # Print header indicating the start of the joint movement.
        print()
        print("------------ MOVE JOINTS --------------")

        # Ensure that exactly one of 'trgtPose' or 'trgtQ' is provided, raise an error if not.
        if (trgtPose is None and trgtQ is None) or (
            trgtPose is not None and trgtQ is not None
        ):
            raise ValueError("Exactly one of 'trgtPose' or 'trgtQ' must be provided")

        # If a target pose is provided, set the target pose and execute a joint movement.
        if trgtPose is not None:
            self.cmd_setTargetPose(pose=trgtPose)
            self.cmd_moveToPosition(trgtPose=trgtPose, maxVel=maxVel, Accel=Accel)

        # If a target joint configuration is provided, set the target joint values and move the robot.
        elif trgtQ is not None:
            self.cmd_setTargetPose(q=trgtQ)
            self.cmd_moveToPosition(trgtQ=trgtQ, maxVel=maxVel, Accel=Accel)

        # Print footer indicating the end of the joint movement.
        print("-----------------------------------")
        print()

    def cmd_MoveHome(self, maxVel=0.5, maxAccel=0.25):
        """
        Moves the robot to its home position.

        This method moves the robot to a predefined 'home' position using joint-based movement (MoveJ).
        The movement is executed with the specified maximum velocity and acceleration.

        Args:
            maxVel (float, optional): The maximum velocity for the joints during the movement. Defaults to 0.5.
            maxAccel (float, optional): The acceleration for the joints during the movement. Defaults to 0.25.
        """

        # Print header indicating the start of the home position movement.
        print()
        print("------------ MOVE HOME --------------")

        # Set the target position to the predefined home joint configuration and execute the movement.
        self.cmd_setTargetPose(q=self.home_q)
        self.cmd_moveToPosition(trgtQ=self.home_q, maxVel=maxVel, Accel=maxAccel)

        # Print footer indicating the end of the home position movement.
        print("-----------------------------------")
        print()


if __name__ == "__main__":

    np.set_printoptions(precision=2, suppress=True)

    d1 = 0.15275  # m
    a2 = 0.22112  # m
    d4 = 0.223  # m
    dt = 0.09  # m

    joints = [
        RevoluteDH(
            home=0,
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

    # Define stepper motors per joint
    motor_configurations = [
        [
            StepperMotor(
                StepsPerRev=200,
                MicroSteps=8,
                maxRPM=1000,
                maxRPMs=1000,
                TransmRatioMotorToJoint=11.0992204,
                PositvJointRottionDirection=1,
            )
        ],
        [
            StepperMotor(
                StepsPerRev=200,
                MicroSteps=8,
                maxRPM=1000,
                maxRPMs=1000,
                TransmRatioMotorToJoint=6.02340957,
                PositvJointRottionDirection=-1,
            ),
            StepperMotor(
                StepsPerRev=200,
                MicroSteps=8,
                maxRPM=1000,
                maxRPMs=1000,
                TransmRatioMotorToJoint=6.02340957,
                PositvJointRottionDirection=1,
            ),
        ],
        [
            StepperMotor(
                StepsPerRev=1028,
                MicroSteps=4,
                maxRPM=200,
                maxRPMs=200,
                TransmRatioMotorToJoint=4.76641221,
                PositvJointRottionDirection=-1,
            )
        ],
        [
            StepperMotor(
                StepsPerRev=200,
                MicroSteps=4,
                maxRPM=1000,
                maxRPMs=1000,
                TransmRatioMotorToJoint=1,
                PositvJointRottionDirection=1,
            )
        ],
        [
            StepperMotor(
                StepsPerRev=200,
                MicroSteps=4,
                maxRPM=1000,
                maxRPMs=1000,
                TransmRatioMotorToJoint=5.23798627,
                PositvJointRottionDirection=-1,
            )
        ],
    ]

    robot = GiRbot(
        name="Moveo3D",
        ArduinoDriver=MD.Moveo(),
        MotorsPerJoint=motor_configurations,
        DH_JointList=joints,
        DH_Tool=tool,
        DH_IKsolver=IK_solver,
    )

    print(robot)

    ##############################################

    robot.cmd_Init()

    robot.cmd_MoveJ(trgtPose=[-0.34, 0.4, 0.25, 0, 0, 0], maxVel=0.5, Accel=0.1)
    robot.cmd_WaitEndMove()

    robot.cmd_closeGripper()

    robot.cmd_MoveL(trgtPose=[0.34, 0.4, 0.25, 0, 0, 0], cartVel=0.0005, cartAccel=0.01)
    robot.cmd_WaitEndMove()

    robot.cmd_openGripper()

    robot.cmd_MoveJ(trgtQ=robot.home_q, maxVel=0.2, Accel=0.05)
    robot.cmd_WaitEndMove()
