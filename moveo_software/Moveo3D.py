from GiRbot import GiRbot
import math
import numpy as np
from DHRobot import RevoluteDH, RobotTool
from PhysicalRobot import StepperMotor
from nIK_Corke import ikine_LM_Corke as IK_solver

import moveo_driver as MD

# Configure NumPy to print arrays with 2 decimal precision and suppress scientific notation
np.set_printoptions(precision=2, suppress=True)

# Define the Denavit-Hartenberg (DH) parameters for the Moveo robot.
d1 = 0.15275  # Base height (meters)
a2 = 0.22112  # Length of second arm (meters)
d4 = 0.223  # Offset to the wrist (meters)
dt = 0.09  # Tool offset (meters)

# Define the robot's joint configuration using RevoluteDH (revolute joints with DH parameters).
joints = [
    RevoluteDH(
        home=0,  # Home position of the joint
        d=d1,  # Link offset along the z-axis
        a=0,  # Link length along the x-axis
        alpha=math.pi / 2,  # Link twist
        qlim=[-(6.75 / 10) * math.pi, (6.75 / 10) * math.pi],  # Joint limits in radians
    ),
    RevoluteDH(
        home=math.pi / 2,
        d=0,
        a=a2,
        alpha=0,
        qlim=[-math.pi / 12, -11 * math.pi / 12],  # Joint limits in radians
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

# Define the tool attached to the robot's end-effector.
tool = RobotTool([0, 0, dt], [0, 0, 0])  # Tool offset in translation and orientation.

# Define stepper motors configurations for each joint.
motor_configurations = [
    [
        StepperMotor(
            StepsPerRev=200,
            MicroSteps=8,  # Number of microsteps per full step
            maxRPM=1000,  # Maximum rotational speed in RPM
            maxRPMs=1000,  # Maximum sustained RPM
            TransmRatioMotorToJoint=11.0992204,  # Transmission ratio from motor to joint
            PositvJointRottionDirection=1,  # Positive rotation direction
        )
    ],
    [
        StepperMotor(
            StepsPerRev=200,
            MicroSteps=8,
            maxRPM=1000,
            maxRPMs=1000,
            TransmRatioMotorToJoint=6.02340957,
            PositvJointRottionDirection=-1,  # Inverse rotation direction
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
            maxRPM=200,  # Lower RPM for this joint
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
            TransmRatioMotorToJoint=1,  # Direct drive (1:1 ratio)
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

# Create an instance of the GiRbot class, representing the Moveo robot with the defined configuration.
Moveo = GiRbot(
    name="Moveo3D",  # Name of the robot
    ArduinoDriver=MD.Moveo(),  # Driver for controlling the robot via Arduino
    MotorsPerJoint=motor_configurations,  # List of motor configurations per joint
    DH_JointList=joints,  # DH joint configuration list
    DH_Tool=tool,  # The tool attached to the end effector
    DH_IKsolver=IK_solver,  # Inverse kinematics solver
)

# Main execution block (typically used when running the script directly).
if __name__ == "__main__":
    # Print the Moveo object to verify its configuration.
    print(Moveo)
