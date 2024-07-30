import numpy as np
import math

import AuxiliarFunctions as AF
from DHRobot import JointDH, RobotTool, PrismaticDH, RevoluteDH
from PhysicalRobot import PhysicalRobot, StepperMotor
from IsochroneMove import IsochroneMove as isocrone
import LinealTrajectory as LinealTrajectory

from nIK_Corke import ikine_LM_Corke as IK_solver

from typing import List, Union, Optional, Callable, Type

import moveo_driver as md
import time


def check_InputArray(data):

    try:
        iterator = iter(data)
    except TypeError:
        raise TypeError("Input must be iterable")

    try:
        item = data[0]
    except TypeError:
        raise TypeError("Input must support indexing")
    except IndexError:
        pass


class GiRbot(PhysicalRobot):

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

        super().__init__(
            name=name,
            MotorsPerJoint=MotorsPerJoint,
            DH_JointList=DH_JointList,
            DH_IKsolver=DH_IKsolver,
            DH_Tool=DH_Tool,
            connection_port=connection_port,
        )

        self.connected = False
        self.mov_time = 0.01
        self.driver = ArduinoDriver
        self.current_q = self.home_q

        self.not_connected_error_msg = "Not connected to the Arduino board! Please make sure previously succesfully call <cmd_Connect>. "

    def cmd_Connect(self):

        print("Connecting to Arduino...")

        if not self.driver.connect(self.connection_port):
            raise Exception("Error connecting to Moveo!")

        print("Succesfully connected")
        print()

        self.connected = True

        time.sleep(0.1)
        return True

    def cmd_setMotorParams(self):

        if self.connected:

            print("Setting default motor parameters...")

            m_maxVel = []
            m_maxAccel = []

            for j in self.MotorsPerJoint:
                for m in j:
                    m_maxVel.append(m.maxStepsVel)
                    m_maxAccel.append(m.maxStepsAccel)

            motor_params = m_maxVel + m_maxAccel

            if not self.driver.setMotorParams(motor_params):
                raise Exception("Error moving to position!")

            print("Max. Velocity in steps/s:", m_maxVel)
            print("Max. Acceleration in steps/s2:", m_maxAccel)
            print("Succesfully set default motor parameters")
            print()

            time.sleep(0.1)
            return True

        else:

            raise Exception(self.not_connected_error_msg)

    def cmd_setMotorRanges(self):

        if self.connected:

            print("Setting default motor limits...")

            m_ranges = self.get_MotorStepRange()

            print("Motors Ranges in steps:", m_ranges)

            motor_ranges = AF.flatten_list(m_ranges)

            if not self.driver.setMotorRanges(motor_ranges):
                raise Exception("Error setting motor Ranges")

            print("Succesfully set motor limits")
            print()

            time.sleep(0.1)
            return True

        else:

            raise Exception(self.not_connected_error_msg)

    def cmd_Stop(self):

        if self.connected:

            print("Stopping motor...")

            if not self.driver.stop():
                raise Exception("Error stopping Motors")

            print("Succesfully stopped")
            print()

            time.sleep(0.1)
            return True

        else:

            raise Exception(self.not_connected_error_msg)

    def cmd_enableMotors(self):

        if self.connected:

            print("Enabling motors...")

            if not self.driver.enable():
                raise Exception("Error enabling Motors")

            print("Succesfully enabled motors.")
            print()

            time.sleep(0.1)
            return True

        else:

            raise Exception(self.not_connected_error_msg)

    def cmd_disableMotors(self):

        if self.connected:

            print("Disabling motors...")

            if not self.driver.disable():
                raise Exception("Error disabling Motors")

            print("Succesfully disabled motors")
            print()

            time.sleep(0.1)
            return True

        else:

            raise Exception(self.not_connected_error_msg)

    def cmd_openGripper(self):

        if self.connected:

            print("Opening gripper...")

            if not self.driver.openGripper():
                raise Exception("Error opening Gripper")

            print("Succesfully opened gripper.")
            print()

            time.sleep(0.3)
            return True

        else:

            raise Exception(self.not_connected_error_msg)

    def cmd_closeGripper(self):

        if self.connected:

            print("Closing gripper...")

            if not self.driver.closeGripper():
                raise Exception("Error closing Gripper")

            print("Succesfully closed gripper.")
            print()

            time.sleep(0.3)
            return True

        else:

            raise Exception(self.not_connected_error_msg)

    def cmd_getJointValues(self):

        if self.connected:

            print("Getting joint values...")

            flat_qm = self.driver.getCurrentPosition()
            if len(flat_qm) != self.n_motors:
                raise Exception("Error getting position!")
            print("Motor Steps Position:", flat_qm)
            qm = AF.unflatten_list(flat_qm, self.n_MotorsPerJoint)
            qj = self.get_JointAnglesFromMotorSteps(motor_steps=qm, out_type="Position")

            qj = AF.wrap_list_half_max(qj, 2 * math.pi).tolist()

            print("Succesfully obtained joint values.")
            print()

            time.sleep(0.1)
            return qj

        else:

            raise Exception(self.not_connected_error_msg)

    def cmd_getEndEffectorPose(self):

        qj = self.cmd_getJointValues()
        pose = self.calc_FK(qj)

        return pose

    def cmd_setTargetPose(
        self,
        pose: Optional[list] = None,
        q: Optional[list] = None,
        mask=[[1, 1, 1, 1, 0, 0]],
    ):

        # Ensure that exactly one of pose or q is provided
        if (pose is None and q is None) or (pose is not None and q is not None):
            raise ValueError("Exactly one of 'pose' or 'q' must be provided")

        if self.connected:

            print("Setting target position...")

            self.current_q = self.cmd_getJointValues()

            if pose is not None:

                check_InputArray(pose)

                pose = np.array(pose)
                qj = self.calc_IK(trgt_poses=pose, mask=[1, 1, 1, 0, 0, 1])

                self.set_JointValues(qj)
                qm = self.get_MotorStepsFromJointAngles(
                    joint_angles=qj, out_type="Position"
                )
                flat_qm = AF.flatten_list(qm)

            elif q is not None:

                check_InputArray(q)

                self.set_JointValues(q)
                qm = self.get_MotorStepsFromJointAngles(
                    joint_angles=q, out_type="Position"
                )
                flat_qm = AF.flatten_list(qm)

            if not self.driver.setTargetPosition(flat_qm):

                raise Exception("Error setting Position")

            print("Target Steps Motor Position:", flat_qm)
            print("Succesfully set target position.")
            print()

            time.sleep(0.1)
            return True

        else:

            raise Exception(self.not_connected_error_msg)

    def cmd_moveToPosition(
        self,
        maxVel,
        Accel,
        trgtPose: Optional[list] = None,
        trgtQ: Optional[list] = None,
    ):
        # Ensure that exactly one of trgtPose or trgtQ is provided
        if (trgtPose is None and trgtQ is None) or (
            trgtPose is not None and trgtQ is not None
        ):
            raise ValueError("Exactly one of 'trgtPose' or 'trgtQ' must be provided")

        if self.connected:

            print("Setting movement type to position...")

            if trgtPose is not None:

                check_InputArray(trgtPose)
                trgtValues = self.cmd_getJointValues()

            elif trgtQ is not None:

                check_InputArray(trgtQ)
                trgtValues = list(trgtQ)

            crrntValues = self.current_q

            print("Current q values:", crrntValues)
            print("Target q values:", trgtValues)

            if AF.floats_equal(crrntValues, trgtValues):

                flat_m_maxRPM = [0] * self.n_joint
                flat_m_Accel = [0] * self.n_joint
                self.mov_time = 0.01

            else:

                # Maximum velocity to reach for each motor
                self.mov_time, j_maxVel = isocrone.get_TimeJointVel(
                    currValues=crrntValues,
                    trgtValues=trgtValues,
                    maxVel=maxVel,
                    accel=Accel,
                    isPrism=self.get_PrismaticJoints(),
                    validRange=self.get_JointRanges(),
                )
                m_maxVel = self.get_MotorStepsFromJointAngles(
                    joint_angles=j_maxVel,
                    out_type="Velocity",
                )
                # Constant acceleration of joint adapted to motor
                j_Accel = [Accel] * self.n_joint
                m_Accel = self.get_MotorStepsFromJointAngles(
                    joint_angles=j_Accel,
                    out_type="Acceleration",
                )

                self.check_Vel(m_maxVel)
                self.check_Accel(m_Accel)

                # Flatten the lists to prepare the packet
                flat_m_maxVel = AF.flatten_list(m_maxVel)
                flat_m_Accel = AF.flatten_list(m_Accel)

            motor_params = [abs(num) for num in flat_m_maxVel + flat_m_Accel]

            if not self.driver.moveToPosition(motor_params):
                raise Exception("Error setting movement to position!")

            print("Succesfully set movement type to position.")
            print()

            self.current_q = trgtValues

            time.sleep(0.1)
            return True

        else:

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

        # Ensure that exactly one of trgtPose or trgtQ is provided
        if (trgtPose is None and trgtQ is None) or (
            trgtPose is not None and trgtQ is not None
        ):
            raise ValueError("Exactly one of 'trgtPose' or 'trgtQ' must be provided")

        if self.connected:

            print("Setting movement type to position...")

            if trgtQ is not None:

                check_InputArray(trgtQ)
                trgtPose = self.calc_FK(trgtQ)

            elif trgtPose is not None:

                check_InputArray(trgtPose)

            crrntQ = self.current_q
            crrntPose = self.calc_FK(crrntQ)
            print("Starting Joint Configuration", crrntQ)

            traj_name = (
                str(AF.round_list(list(crrntPose)))
                + "-to-"
                + str(AF.round_list(list(trgtPose)))
            )

            print(
                f"Generating Linear Traj.\nPose1: {AF.round_list(list(crrntPose))}\nto Pose2:{AF.round_list(list(trgtPose))} * if possible. "
            )

            traj = LinealTrajectory.create_LTraj(
                StrtPose=crrntPose,
                TrgtPose=trgtPose,
                MoveVel=cartVel,
                CartAccel=cartAccel,
                n_interp=n_interp,
                n_dofs=self.n_joint,
            )

            traj.q = self.calc_IK(
                trgt_poses=traj.pose,
                q0=crrntQ,
                mask=mask,
                q_lim=True,
            )

            traj.q, traj.t = LinealTrajectory.rmve_Singularities(
                q=traj.q,
                t=traj.t,
                joint_ranges=self.get_JointRanges(),
                joint_type=self.get_PrismaticJoints(),
            )

            traj.qd = self.calc_DerivatedArray(traj.q, traj.t)

            m_qd = []

            for i, qd in enumerate(traj.qd):

                m_qd_i = self.get_MotorStepsFromJointAngles(
                    joint_angles=qd, out_type="Velocity"
                )

                try:
                    self.check_Vel(m_qd_i)

                except ValueError as e:
                    raise ValueError(
                        f"Interpolation point {i}\n"
                        + str(e)
                        + "\nTry reducing the cartesian velocity of the movement."
                    )

                m_qd.append(AF.flatten_list(m_qd_i))

            np.save("LTraj_files/" + traj_name + "_q", traj.q)
            np.save("LTraj_files/" + traj_name + "_qd", traj.qd)
            np.save("LTraj_files/" + traj_name + "_t", traj.t)

            m_qd = m_qd

            t_inc = traj.t[-1] - traj.t[-2]

            print(f"Total time of trajectory: {traj.t[-1]}s")
            print(f"Increments of {t_inc}s")

            for i, qd in enumerate(m_qd):

                print("Steps/s:", qd)
                self.driver.trajToPosition(list(qd))
                time.sleep(t_inc)
                print()

            self.mov_time = 0.1  # Final position already reached

            self.cmd_setTargetPose(q=traj.q[-1])
            self.current_q = traj.q[-1]

            print("Succesfully set movement type to position.")
            print()

        else:

            raise Exception(self.not_connected_error_msg)

    ########################################################################################################################
    #                                            FINAL COMMANDS                                                           #
    ########################################################################################################################

    def cmd_Init(self):

        self.cmd_Connect()
        self.cmd_setMotorParams()
        self.cmd_setMotorRanges()

        return True

    def cmd_WaitEndMove(self):

        print(f"Waiting {round(self.mov_time)}s for the end of the movement")
        time.sleep(self.mov_time)
        self.cmd_Stop()
        print("Movement completed")
        print()

        return True

    def cmd_MoveL(
        self,
        cartVel,
        cartAccel,
        trgtPose: Optional[list] = None,
        trgtQ: Optional[list] = None,
        n_interp=500,
        mask=[1, 1, 1, 0, 0, 1],
    ):
        # Ensure that exactly one of trgtPose or trgtQ is provided
        if (trgtPose is None and trgtQ is None) or (
            trgtPose is not None and trgtQ is not None
        ):
            raise ValueError("Exactly one of 'trgtPose' or 'trgtQ' must be provided")

        if trgtPose is not None:
            self.cmd_setTargetPose(pose=trgtPose)
            self.cmd_trajToPosition(
                trgtPose=trgtPose,
                cartVel=cartVel,
                cartAccel=cartAccel,
                n_interp=n_interp,
                mask=mask,
            )

        elif trgtQ is not None:
            self.cmd_setTargetPose(q=trgtQ)
            self.cmd_trajToPosition(
                trgtQ=trgtQ,
                cartVel=cartVel,
                cartAccel=cartAccel,
                n_interp=n_interp,
                mask=mask,
            )

    def cmd_MoveJ(
        self,
        maxVel,
        Accel,
        trgtPose: Optional[list] = None,
        trgtQ: Optional[list] = None,
    ):

        # Ensure that exactly one of trgtPose or trgtQ is provided
        if (trgtPose is None and trgtQ is None) or (
            trgtPose is not None and trgtQ is not None
        ):
            raise ValueError("Exactly one of 'trgtPose' or 'trgtQ' must be provided")

        if trgtPose is not None:
            self.cmd_setTargetPose(pose=trgtPose)
            self.cmd_moveToPosition(trgtPose=trgtPose, maxVel=maxVel, Accel=Accel)

        elif trgtQ is not None:
            self.cmd_setTargetPose(q=trgtQ)
            self.cmd_moveToPosition(trgtQ=trgtQ, maxVel=maxVel, Accel=Accel)


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
        ArduinoDriver=md.Moveo(),
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
