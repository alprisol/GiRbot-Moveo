from GiRbot import GiRbot
import math
import numpy as np
from DHRobot import RevoluteDH, RobotTool
from PhysicalRobot import StepperMotor
from nIK_Corke import ikine_LM_Corke as IK_solver
import moveo_driver as MD

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

Moveo = GiRbot(
    name="Moveo3D",
    ArduinoDriver=MD.Moveo(),
    MotorsPerJoint=motor_configurations,
    DH_JointList=joints,
    DH_Tool=tool,
    DH_IKsolver=IK_solver,
)
