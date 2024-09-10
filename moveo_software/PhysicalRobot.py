import numpy as np
import math

import AuxiliarFunctions as AF
from DHRobot import DHRobot, JointDH, RobotTool, PrismaticDH, RevoluteDH

from typing import List, Union, Optional, Callable

from nIK_Corke import ikine_LM_Corke as IK_solver


class StepperMotor:
    """
    The `StepperMotor` class represents a stepper motor and its relationship to a joint, calculating key
    parameters such as maximum velocity and acceleration in terms of motor steps. It also defines the
    transmission ratio between the motor and the joint and the direction of joint rotation.

    Parameters:
    - StepsPerRev (int): The number of steps the motor takes to complete one full revolution (without microstepping).
    - MicroSteps (int): The number of microsteps per full step, increasing precision.
    - TransmRatioMotorToJoint (float): The transmission ratio between the motor and the joint, where a ratio > 1
        indicates that the motor turns faster than the joint.
    - maxRPM (float): The maximum speed of the motor in revolutions per minute (RPM). Default is 1000 RPM.
    - maxRPMs (float): The maximum acceleration of the motor in RPM per second (RPM/s). Default is 1000 RPM/s.
    - PositvJointRottionDirection (int): The direction of positive rotation for the joint relative to the motor.
        Default is `1` (positive).

    Attributes:
    - StepsPerRev (float): The total number of steps the motor takes to complete one revolution, adjusted for
        microstepping.
    - MicroSteps (int): The number of microsteps per full step, allowing for finer control.
    - maxStepsVel (float): The maximum velocity in steps per second, calculated from `maxRPM`.
    - maxStepsAccel (float): The maximum acceleration in steps per second squared, calculated from `maxRPMs`.
    - TransmRatioMotorToJoint (float): The transmission ratio between the motor's output and the joint's rotation.
    - PositvJointRottionDirection (int): The direction of positive rotation for the joint, relative to the motor.
    """

    def __init__(
        self,
        StepsPerRev,
        MicroSteps,
        TransmRatioMotorToJoint,
        maxRPM=1000,
        maxRPMs=1000,
        PositvJointRottionDirection=1,
    ):
        """
        Initialize a stepper motor with specific characteristics and configuration.
        """
        # Total steps per revolution, accounting for microstepping
        self.StepsPerRev = StepsPerRev * MicroSteps

        # Microstepping factor
        self.MicroSteps = MicroSteps

        # Maximum velocity in steps per second (derived from max RPM)
        self.maxStepsVel = maxRPM * (self.StepsPerRev / 60)

        # Maximum acceleration in steps per second squared (derived from max RPM/s)
        self.maxStepsAccel = maxRPMs * (self.StepsPerRev / 60)

        # Transmission ratio between motor and joint
        self.TransmRatioMotorToJoint = TransmRatioMotorToJoint

        # Direction of positive joint rotation (relative to the motor)
        self.PositvJointRottionDirection = PositvJointRottionDirection


class PhysicalRobot(DHRobot):
    """
    The `PhysicalRobot` class represents a physical robotic arm that extends the `DHRobot` class. It incorporates
    stepper motors for each joint and manages the connection with the hardware. This initialization sets up the
    robot's DH parameters and motors.
    """

    def __init__(
        self,
        MotorsPerJoint: List[List[StepperMotor]],
        DH_JointList: List[Union[JointDH, RevoluteDH, PrismaticDH]],
        DH_IKsolver: Callable,
        DH_Tool: Optional[RobotTool] = None,
        connection_port: str = "ttyACM0",
        name: str = "RobotArm",
    ):
        """
        Initialize a physical robot with stepper motors and Denavit-Hartenberg (DH) parameters.

        Parameters:
        - MotorsPerJoint (list of list of StepperMotor): A nested list where each sublist contains the stepper motors
          controlling a joint. Multiple motors can be used per joint if needed.
        - DH_JointList (list of JointDH, RevoluteDH, or PrismaticDH): A list of DH joints (either revolute or prismatic)
          defining the robot's structure.
        - DH_IKsolver (Callable): A function or callable object that acts as the inverse kinematics (IK) solver for the robot.
        - DH_Tool (RobotTool, optional): A tool attached to the robot's end-effector. If not provided, a default tool is used.
        - connection_port (str): The port used for communication with the robot's hardware (e.g., "ttyACM0"). Default is "ttyACM0".
        - name (str): The name of the robot. Default is "RobotArm".

        Attributes:
        - MotorsPerJoint (list of list of StepperMotor): The stepper motors assigned to each joint.
        - connection_port (str): The port used for connecting the robot to the hardware (communication port).
        - n_MotorsPerJoint (list): A list containing the number of motors for each joint.
        - n_motors (int): The total number of motors used in the robot.
        - name (str): The name of the robot, used for identification.
        """
        # Initialize the superclass (DHRobot) with DH parameters and inverse kinematics solver
        super().__init__(
            joint_list=DH_JointList, tool=DH_Tool, IK_solver=DH_IKsolver, DH_name=name
        )

        # Robot's name
        self.name = name

        # Assign the motors to each joint
        self.MotorsPerJoint = MotorsPerJoint

        # Communication port for connecting to the robot's hardware
        self.connection_port = connection_port

        # Count the number of motors per joint and total number of motors
        self.n_MotorsPerJoint = [
            len(j) for j in MotorsPerJoint
        ]  # List of motor counts per joint
        self.n_motors = sum(
            self.n_MotorsPerJoint
        )  # Total number of motors in the robot

    def check_Vel(self, velocities):
        """
        Check if the provided velocities for each motor in each joint are within the allowable maximum velocity limits.

        This method verifies that the velocities specified for each motor in the robot's joints do not exceed the maximum
        velocity limits defined by the motor specifications. If any velocity exceeds the allowable limit, a `ValueError`
        is raised. The method checks each joint and each motor within that joint.

        Parameters:
        - velocities (list of list of float): A nested list where each sublist corresponds to a joint, and each element
        within the sublist represents the velocity of a specific motor in that joint. The velocities are given in steps
        per second.

        Raises:
        - ValueError: If any motor's velocity exceeds its allowable maximum (`maxStepsVel`), or if velocities for a joint
        are missing in the provided list.

        Returns:
        - bool: Returns `True` if all velocities are within the allowable limits.

        Notes:
        - The method iterates over each joint and each motor within that joint, checking the provided velocities against
        the motor's maximum allowable velocity (`maxStepsVel`).
        - If velocities for any joint are missing from the input list, a `ValueError` is raised indicating the missing joint.
        """
        # Iterate over each joint
        for j in range(self.n_joint):

            motors = self.MotorsPerJoint[
                j
            ]  # Get the list of motors for the current joint

            # Check if velocities are provided for the current joint
            if j < len(velocities):

                # Iterate over each motor within the current joint
                for i, m in enumerate(motors):

                    # Check if a velocity is provided for the current motor
                    if i < len(velocities[j]):

                        vel = abs(
                            velocities[j][i]
                        )  # Get the absolute value of the motor's velocity

                        # If the velocity exceeds the motor's maximum allowable velocity, raise an error
                        if vel > m.maxStepsVel:
                            raise ValueError(
                                f"Velocity {velocities[j][i]} for motor {i+1} in joint {j+1} exceeds max limit of {m.maxStepsVel}"
                            )
            else:
                # If no velocities are provided for the current joint, raise an error
                raise ValueError(f"No velocities provided for joint {j}")

        return True  # Return True if all velocities are within the allowable limits

    def check_Accel(self, accelerations):
        """
        Check if the provided accelerations for each motor in each joint are within the allowable maximum acceleration limits.

        This method verifies that the accelerations specified for each motor in the robot's joints do not exceed the maximum
        acceleration limits defined by the motor specifications. If any acceleration exceeds the allowable limit, a `ValueError`
        is raised. The method checks each joint and each motor within that joint.

        Parameters:
        - accelerations (list of list of float): A nested list where each sublist corresponds to a joint, and each element
        within the sublist represents the acceleration of a specific motor in that joint. The accelerations are given in steps
        per second squared.

        Raises:
        - ValueError: If any motor's acceleration exceeds its allowable maximum (`maxStepsAccel`), or if accelerations for a joint
        are missing in the provided list.

        Returns:
        - bool: Returns `True` if all accelerations are within the allowable limits.

        Notes:
        - The method iterates over each joint and each motor within that joint, checking the provided accelerations against
        the motor's maximum allowable acceleration (`maxStepsAccel`).
        - If accelerations for any joint are missing from the input list, a `ValueError` is raised indicating the missing joint.
        """
        # Iterate over each joint
        for j in range(self.n_joint):

            motors = self.MotorsPerJoint[
                j
            ]  # Get the list of motors for the current joint

            # Check if accelerations are provided for the current joint
            if j < len(accelerations):

                # Iterate over each motor within the current joint
                for i, m in enumerate(motors):

                    # Check if an acceleration is provided for the current motor
                    if i < len(accelerations[j]):

                        accel = abs(
                            accelerations[j][i]
                        )  # Get the absolute value of the motor's acceleration

                        # If the acceleration exceeds the motor's maximum allowable acceleration, raise an error
                        if accel > m.maxStepsAccel:
                            raise ValueError(
                                f"Acceleration {accelerations[j][i]} for motor {i+1} in joint {j+1} exceeds max limit of {m.maxStepsAccel}"
                            )
            else:
                # If no accelerations are provided for the current joint, raise an error
                raise ValueError(f"No accelerations provided for joint {j}")

        return True  # Return True if all accelerations are within the allowable limits

    def get_MotorStepsFromJointAngles(self, out_type: str, joint_angles: list):
        """
        Calculate the motor steps required for each motor in each joint to reach the specified joint angles, velocities,
        or accelerations, depending on the specified output type.

        Parameters:
        - joint_angles (list): The target angles (in radians) for each joint. The length of the list should match the
        number of joints in the robot.
        - out_type (str): The type of output to generate. Accepted values are:
        - "Position": Compute the motor steps required to achieve the specified joint positions.
        - "Velocity": Compute the motor steps required for the specified joint velocities.
        - "Acceleration": Compute the motor steps required for the specified joint accelerations.

        Returns:
        - List[List]: A nested list where each sublist contains the motor steps for the motors controlling a specific joint.
        The number of sublists corresponds to the number of joints, and each sublist contains the steps for each motor
        controlling that joint.

        Raises:
        - ValueError: If `out_type` is not one of the accepted values ("Position", "Velocity", "Acceleration").

        Notes:
        - For "Position" calculations, joint angles are wrapped to stay within the joint's valid range using the `wrap_value_half_max()`
        function.
        - For "Velocity" and "Acceleration" calculations, the motor steps are scaled based on the transmission ratio and the motor's
        steps per revolution.
        - The `PositvJointRottionDirection` is used to ensure the motor steps match the desired direction of joint rotation.
        """
        valid_out_types = ["Position", "Velocity", "Acceleration"]

        # Check if the output type is valid
        if out_type not in valid_out_types:
            raise ValueError(
                f"<out_type> {out_type} is not accepted. Valid values are: {valid_out_types}"
            )

        motors_values = []  # List to store motor steps for each joint

        # If the output type is "Position", adjust the joint angles relative to the home position
        if out_type == "Position":
            joint_angles = (
                (np.array(joint_angles) - np.array(self.home_q)) % (2 * math.pi)
            ).tolist()

        # Iterate over each joint and calculate motor steps
        for i, j in enumerate(joint_angles):

            motors = self.MotorsPerJoint[i]  # Get motors for the current joint
            joint_motors = []  # Store motor steps for this joint

            # Iterate over each motor controlling the current joint
            for m in motors:
                # Calculate motor steps based on the joint angle and transmission ratio
                m_steps = (
                    j * (m.StepsPerRev * m.TransmRatioMotorToJoint) / (2 * math.pi)
                )

                if out_type == "Position":
                    # Wrap the steps to handle revolute joints correctly (e.g., wrapping angles within a valid range)
                    m_steps = AF.wrap_value_half_max(
                        m_steps, m.StepsPerRev * m.TransmRatioMotorToJoint
                    )
                    m_steps = int(round(m_steps, 0))  # Round to the nearest integer

                if out_type in ["Position", "Velocity"]:
                    # Adjust steps based on the positive rotation direction
                    m_steps *= m.PositvJointRottionDirection

                joint_motors.append(m_steps)  # Add the motor steps for this motor

            motors_values.append(
                joint_motors
            )  # Add the motor steps for the current joint

        return motors_values  # Return the motor steps for all joints

    def get_JointAnglesFromMotorSteps(self, out_type: str, motor_steps: list):
        """
        Calculate the joint angles, velocities, or accelerations from the specified motor steps.

        Parameters:
        - motor_steps (list of list of int/float): A nested list where each sublist corresponds to the motor steps
        for the motors in a specific joint.
        - out_type (str): The type of input. Accepted values are:
        - "Position": Interpret motor steps as positions to calculate the joint angles.
        - "Velocity": Interpret motor steps as velocities to calculate the joint angular velocities.
        - "Acceleration": Interpret motor steps as accelerations to calculate the joint angular accelerations.

        Returns:
        - list of float: Joint angles, velocities, or accelerations for each joint, depending on the specified output type.

        Raises:
        - ValueError: If `out_type` is not one of the accepted values ("Position", "Velocity", "Acceleration").

        Notes:
        - The method calculates the joint angles, velocities, or accelerations by converting the motor steps based on the
        motor's transmission ratio (`TransmRatioMotorToJoint`) and steps per revolution (`StepsPerRev`).
        - For "Position" calculations, the joint angles are wrapped to stay within the joint's valid range, and adjusted
        relative to the home position (`self.home_q`).
        - The `PositvJointRottionDirection` is used to ensure the motor steps are correctly interpreted with respect to the
        joint's positive rotation direction.
        """
        valid_out_types = ["Position", "Velocity", "Acceleration"]

        # Check if the output type is valid
        if out_type not in valid_out_types:
            raise ValueError(
                f"<out_type> {out_type} is not accepted. Valid values are: {valid_out_types}"
            )

        joint_angles = []  # List to store the calculated joint angles or other values

        # Iterate over each joint and calculate the joint angle based on the motor steps
        for i, motor_vals in enumerate(motor_steps):

            motors = self.MotorsPerJoint[i]  # Get the motors for the current joint
            joint_angle = 0  # Initialize the joint angle for this joint

            # Iterate over each motor in the joint and calculate the corresponding joint angle
            for m, m_steps in zip(motors, motor_vals):

                # Adjust motor steps based on the positive rotation direction if applicable
                if out_type in ["Position", "Velocity"]:
                    m_steps /= m.PositvJointRottionDirection

                # Calculate the joint angle based on motor steps, transmission ratio, and steps per revolution
                j_angle = (m_steps * (2 * math.pi)) / (
                    m.StepsPerRev * m.TransmRatioMotorToJoint
                )
                joint_angle = j_angle  # Update joint angle

            joint_angles.append(
                joint_angle
            )  # Store the joint angle for the current joint

        # If calculating positions, adjust joint angles relative to home position and wrap them
        if out_type == "Position":

            # Adjust joint angles relative to the home position
            joint_angles = (
                (np.array(joint_angles) + np.array(self.home_q)) % (2 * math.pi)
            ).tolist()

            # Wrap the joint angles to keep them within a valid range
            joint_angles = AF.wrap_angle_list(joint_angles)

        return joint_angles  # Return the calculated joint angles or other values

    def get_MotorStepRange(self):
        """
        Calculate the range of motor steps required to cover the full range of joint angles for each motor.

        This method calculates the motor step ranges for each motor in the robot, based on the joint angle limits. It converts
        the joint angle ranges into motor step ranges, ensuring that the motor steps correctly represent the full range of motion
        for each joint.

        Returns:
        - list of list: A nested list where each sublist contains the minimum and maximum motor steps for a specific motor,
        corresponding to the full range of motion for the associated joint.

        Notes:
        - The method first retrieves the joint angle limits using the `get_JointRanges` method.
        - It then converts the joint angle limits to motor steps using the `get_MotorStepsFromJointAngles` method.
        - The method ensures that the motor step ranges are correctly represented by adjusting cases where the low and high values
        are equal (e.g., full revolutions) or where the low value is greater than the high value.
        """
        # Get the joint angle range (limits) for each joint
        joint_range = self.get_JointRanges()

        # Print the rounded joint angle range
        print("Joint angle range:", AF.round_list(joint_range, 4))

        # Extract the lower and upper limits for each joint
        low_joint_range = [j_range[0] for j_range in joint_range]
        high_joint_range = [j_range[1] for j_range in joint_range]

        # Convert the joint angle limits to motor steps (for the low and high limits)
        low_motor_range = self.get_MotorStepsFromJointAngles(
            joint_angles=low_joint_range,
            out_type="Position",
        )
        high_motor_range = self.get_MotorStepsFromJointAngles(
            joint_angles=high_joint_range,
            out_type="Position",
        )

        # Flatten the nested lists of motor steps for easier processing
        low_motor_range_flat = AF.flatten_list(low_motor_range)
        high_motor_range_flat = AF.flatten_list(high_motor_range)

        motor_ranges = []  # List to store the motor step ranges

        # Iterate over the low and high motor steps for each motor and adjust ranges
        for low, high in zip(low_motor_range_flat, high_motor_range_flat):
            # If the low and high values are equal (e.g., full revolutions), adjust the high limit
            if low == high:
                high = -low
            # Ensure that the low value is less than the high value; if not, swap them
            if low > high:
                motor_ranges.append([high, low])
            else:
                motor_ranges.append([low, high])

        return motor_ranges  # Return the motor step ranges for all motors

    def __str__(self):
        """
        Generate a formatted string representation of the robot's motor configuration and Denavit-Hartenberg parameters.

        This method overrides the `__str__` method from the `DHRobot` superclass to include additional information about the
        motor configuration for each joint. It creates a table that summarizes important motor properties such as the number
        of motors, microstepping, steps per revolution, maximum velocity, acceleration, transmission ratio, and direction.

        Returns:
        - str: A formatted string representing the motor configuration and the Denavit-Hartenberg parameters of the robot.

        Notes:
        - The method first calls the `__str__` method of the superclass to include the Denavit-Hartenberg parameters in the output.
        - The motor configuration for each joint is then added to the output, including detailed properties for each motor in
        a joint.
        - The table is neatly aligned for readability, showing various motor attributes for each joint.
        """
        # Start with the string representation of the superclass (DHRobot)
        output_str = super().__str__() + "\n"

        # Add the header for the motor configuration table
        output_str += f"{self.name} Motor Configuration: \n"
        output_str += "-" * 141 + "\n"
        output_str += (
            "Joint | Nº Motors |  Microsteps  |    Steps/Rev      |      Max Steps/s      "
            "|      Max Steps/s^2      |      Transm. Ratio      | Direction \n"
        )
        output_str += "-" * 141 + "\n"

        # Define the format for each row of the motor configuration table
        tableMotorConfig = (
            "{:<5} | {:^9} | {:^12} | {:^17} | {:^21} | {:^23} | {:^23} | {:^9}\n"
        )

        # Iterate over each joint's motors to gather their properties
        for idx, m_joint in enumerate(self.MotorsPerJoint, 1):

            n_motors = len(m_joint)  # Number of motors for the current joint
            microsteps = []
            steps_rev = []
            max_vel = []
            max_accel = []
            trans_ratio = []
            direct = []

            # Gather motor properties for each motor in the current joint
            for motor in m_joint:
                microsteps.append("{:.1f}".format(motor.MicroSteps))
                steps_rev.append("{:.1f}".format(motor.StepsPerRev))
                max_vel.append("{:.1f}".format(motor.maxStepsVel))
                max_accel.append("{:.1f}".format(motor.maxStepsAccel))
                trans_ratio.append("{:.3f}".format(motor.TransmRatioMotorToJoint))
                direct.append("{:.1f}".format(motor.PositvJointRottionDirection))

            # Format the data into the table
            output_str += tableMotorConfig.format(
                idx,
                n_motors,
                ", ".join(microsteps),
                ", ".join(steps_rev),
                ", ".join(max_vel),
                ", ".join(max_accel),
                ", ".join(trans_ratio),
                ", ".join(direct),
            )

        # Add the closing line for the table
        output_str += "-" * 141 + "\n"

        return output_str


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

    robot = PhysicalRobot(
        name="Moveo3D",
        MotorsPerJoint=motor_configurations,
        DH_JointList=joints,
        DH_Tool=tool,
        DH_IKsolver=IK_solver,
    )

    print(robot)

    print(robot.n_motors)

    robot.get_MotorStepRange()
