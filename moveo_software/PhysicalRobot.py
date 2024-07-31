import numpy as np
import math

import AuxiliarFunctions as AF
from DHRobot import DHRobot, JointDH, RobotTool, PrismaticDH, RevoluteDH

from typing import List, Union, Optional, Callable

from nIK_Corke import ikine_LM_Corke as IK_solver


class StepperMotor:

    def __init__(
        self,
        StepsPerRev,
        MicroSteps,
        TransmRatioMotorToJoint,
        maxRPM=1000,
        maxRPMs=1000,
        PositvJointRottionDirection=1,
    ):

        self.StepsPerRev = StepsPerRev * MicroSteps
        self.MicroSteps = MicroSteps
        self.maxStepsVel = maxRPM * (self.StepsPerRev / 60)
        self.maxStepsAccel = maxRPMs * (self.StepsPerRev / 60)
        self.TransmRatioMotorToJoint = TransmRatioMotorToJoint
        self.PositvJointRottionDirection = PositvJointRottionDirection


class PhysicalRobot(DHRobot):
    def __init__(
        self,
        MotorsPerJoint: List[List[StepperMotor]],
        DH_JointList: List[Union[JointDH, RevoluteDH, PrismaticDH]],
        DH_IKsolver: Callable,
        DH_Tool: Optional[RobotTool] = None,
        connection_port: str = "ttyACM0",
        name: str = "RobotArm",
    ):

        super().__init__(
            joint_list=DH_JointList, tool=DH_Tool, IK_solver=DH_IKsolver, DH_name=name
        )
        self.name = name
        self.MotorsPerJoint = MotorsPerJoint
        self.connection_port = connection_port

        self.n_MotorsPerJoint = [len(j) for j in MotorsPerJoint]
        self.n_motors = sum(self.n_MotorsPerJoint)

    def check_Vel(self, velocities):
        """
        Check if the provided velocities for each motor in each joint are within the maximum velocity limits.
        """
        for j in range(self.n_joint):

            motors = self.MotorsPerJoint[j]

            if j < len(velocities):

                for i, m in enumerate(motors):

                    if i < len(velocities[j]):

                        vel = abs(velocities[j][i])

                        if vel > m.maxStepsVel:

                            raise ValueError(
                                f"Velocity {velocities[j][i]} for motor {i+1} in joint {j+1} exceeds max limit of {m.maxStepsVel}"
                            )
            else:

                raise ValueError(f"No velocities provided for joint {j}")

        return True

    def check_Accel(self, accelerations):
        """
        Check if the provided accelerations for each motor in each joint are within the maximum acceleration limits.
        """
        # Assuming self.MotorsPerJoint is a list of lists of motors for each joint
        for j in range(self.n_joint):

            motors = self.MotorsPerJoint[j]

            if j < len(accelerations):

                for i, m in enumerate(motors):

                    if i < len(accelerations[j]):

                        accel = abs(accelerations[j][i])

                        if accel > m.maxStepsAccel:

                            raise ValueError(
                                f"Acceleration {accelerations[j][i]} for motor {i+1} in joint {j+1} exceeds max limit of {m.maxStepsAccel}"
                            )
            else:
                # Handle case where no accelerations are provided for a joint
                raise ValueError(f"No accelerations provided for joint {j}")

        return True

    def get_MotorStepsFromJointAngles(self, out_type: str, joint_angles: list):
        """
        Calculate the motor steps required for each joint to reach the specified joint_angles.

        Parameters:
        - joint_angles (list) Target angles for each joint.
        - out_type (string): Type of output: 'Position','Velocity','Acceleration'.

        Returns:
        - List[List]: Motor steps for each motor in each joint.
        """
        valid_out_types = ["Position", "Velocity", "Acceleration"]

        if out_type not in valid_out_types:

            raise ValueError(
                f"<out_type> {out_type} is not accepted. Valid values are: {valid_out_types}"
            )

        motors_values = []

        if out_type == "Position":

            joint_angles = (
                (np.array(joint_angles) - np.array(self.home_q)) % (2 * math.pi)
            ).tolist()

        for i, j in enumerate(joint_angles):

            motors = self.MotorsPerJoint[i]
            joint_motors = []

            for m in motors:

                m_steps = (
                    j * (m.StepsPerRev * m.TransmRatioMotorToJoint) / (2 * math.pi)
                )
                if out_type == "Position":
                    m_steps = AF.wrap_value_half_max(
                        m_steps, m.StepsPerRev * m.TransmRatioMotorToJoint
                    )
                    m_steps = int(round(m_steps, 0))

                if out_type in ["Position", "Velocity"]:
                    m_steps *= m.PositvJointRottionDirection
                joint_motors.append(m_steps)
            motors_values.append(joint_motors)

        return motors_values

    def get_JointAnglesFromMotorSteps(self, out_type: str, motor_steps: list):
        """
        Calculate the joint angles from the specified motor steps.

        Parameters:
        - motor_steps (list): Motor steps for each motor in each joint.
        - out_type (string): Type of input: 'Position','Velocity','Acceleration'.

        Returns:
        - List: Joint angles for each joint.
        """
        valid_out_types = ["Position", "Velocity", "Acceleration"]

        if out_type not in valid_out_types:

            raise ValueError(
                f"<out_type> {out_type} is not accepted. Valid values are: {valid_out_types}"
            )

        joint_angles = []

        for i, motor_vals in enumerate(motor_steps):

            motors = self.MotorsPerJoint[i]
            joint_angle = 0

            for m, m_steps in zip(motors, motor_vals):

                if out_type in ["Position", "Velocity"]:
                    m_steps /= m.PositvJointRottionDirection

                j_angle = (m_steps * (2 * math.pi)) / (
                    m.StepsPerRev * m.TransmRatioMotorToJoint
                )
                joint_angle = j_angle

            joint_angles.append(joint_angle)

        if out_type == "Position":

            joint_angles = (
                (np.array(joint_angles) + np.array(self.home_q)) % (2 * math.pi)
            ).tolist()

            joint_angles = AF.wrap_angle_list(joint_angles)

        return joint_angles

    def get_MotorStepRange(self):

        joint_range = self.get_JointRanges()

        print('Joint angle range:', AF.round_list(joint_range,4))

        low_joint_range = [j_range[0] for j_range in joint_range]
        high_joint_range = [j_range[1] for j_range in joint_range]

        low_motor_range = self.get_MotorStepsFromJointAngles(
            joint_angles=low_joint_range,
            out_type="Position",
        )
        high_motor_range = self.get_MotorStepsFromJointAngles(
            joint_angles=high_joint_range,
            out_type="Position",
        )

        low_motor_range_flat = AF.flatten_list(low_motor_range)
        high_motor_range_flat = AF.flatten_list(high_motor_range)

        motor_ranges = []
        for low, high in zip(low_motor_range_flat, high_motor_range_flat):
            # If the cases where the range is full revolution, or almost,
            # make sure to hace different sign limits.
            if low == high:
                high = -low
            # Check if low is greater than high, if so swap them
            if low > high:
                motor_ranges.append([high, low])
            else:
                motor_ranges.append([low, high])

        return motor_ranges

    def __str__(self):

        output_str = super().__str__() + "\n"

        output_str += f"{self.name} Motor Configuration: \n"
        output_str += "-" * 141 + "\n"
        output_str += "Joint | Nº Motors |  Microsteps  |    Steps/Rev      |      Max Steps/s      |      Max Steps/s^2      |      Transm. Ratio      | Direction \n"
        output_str += "-" * 141 + "\n"

        tableMotorConfig = (
            "{:<5} | {:^9} | {:^12} | {:^17} | {:^21} | {:^23} | {:^23} | {:^9}\n"
        )

        for idx, m_joint in enumerate(self.MotorsPerJoint, 1):

            n_motors = len(m_joint)
            microsteps = []
            steps_rev = []
            max_vel = []
            max_accel = []
            trans_ratio = []
            direct = []

            for motor in m_joint:

                steps_rev.append("{:.1f}".format(motor.StepsPerRev))
                microsteps.append("{:.1f}".format(motor.MicroSteps))
                max_vel.append("{:.1f}".format(motor.maxStepsVel))
                max_accel.append("{:.1f}".format(motor.maxStepsAccel))
                trans_ratio.append("{:.3f}".format(motor.TransmRatioMotorToJoint))
                direct.append("{:.1f}".format(motor.PositvJointRottionDirection))

            # Format each motor's data into the output string
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
