import math
import numpy as np
from typing import Union
import roboticstoolbox as rtb
import SpatialMath as SM
import AuxiliarFunctions as AF
from typing import Union, Callable, Optional, List

# from DHRobot import RobotTool, JointDH, RevoluteDH, PrismaticDH, DHRobot


def adjust_angular_range(range_limits):
    # Ensure the input is a list or tuple and contains two elements
    if not isinstance(range_limits, (list, tuple)) or len(range_limits) != 2:
        return "Invalid input. Please provide a tuple or list of two elements."

    low, high = range_limits
    if high - low < 0:
        # Calculate distance of each element from pi
        distances = [(abs(math.pi - abs(x)), x) for x in [low, high]]
        # Find the element closest to pi
        closest = min(distances, key=lambda x: x[0])[1]

        # Change the sign of the closest element to pi and set it to pi
        if closest < 0:
            new_value = math.pi
        else:
            new_value = -math.pi

        # Replace the original value with the new pi value in the range
        if closest == low:
            low = new_value
        else:
            high = new_value

    return (low, high)


def ikine_LM_Corke(
    DHRobot,
    trgt_poses: np.ndarray,
    q0: Optional[np.ndarray] = None,
    q_lim: Optional[np.ndarray] = None,
    mask: Optional[Union[np.ndarray, list, bool]] = False,
):
    print(" *Using Corke's nIK method some joint range of movement is lost.")

    # Initialize the list to store solutions
    solutions = []

    # Ensure trgt_poses is a 2D array (even for a single pose)
    if trgt_poses.ndim == 1:
        trgt_poses = np.array([trgt_poses])

    # Build the robot model for each pose
    Corke_joints = []

    for j in DHRobot.joint_list:
        if j.j_type == 0:
            Corke_joints.append(
                rtb.RevoluteDH(
                    d=j.d,
                    a=j.a,
                    alpha=j.alpha,
                    qlim=adjust_angular_range(j.qlim),
                )
            )
        else:
            Corke_joints.append(
                rtb.PrismaticDH(theta=j.theta, a=j.a, alpha=j.alpha, qlim=j.qlim)
            )

    CorkeRobot = rtb.DHRobot(
        links=Corke_joints, name="Corke Robot", tool=DHRobot.tool.Tfm
    )

    if mask is True:

        mask = np.concatenate((np.ones(DHRobot.n_joint), np.zeros(6 - DHRobot.n_joint)))

    elif mask is False:

        mask = None

    print(f'Mask set at {mask}. Change it if necessary with attribute <mask>')

    # Loop over each target pose
    for i, trgt_pose in enumerate(trgt_poses):

        # print(f"IK of Pose {i+1}")
        Tt = SM.GetMatFromPose(trgt_pose)

        # Use the previous solution as the initial guess for the current pose
        if i > 0 and solutions[-1] is not None:
            q_closer = solutions[-1]
        elif q0 is not None:
            q_closer = q0
        else:
            try:
                q_samples, pose_samples = DHRobot.samples
            except AttributeError:
                q_samples, pose_samples = DHRobot.gen_samples(1000)
            q_closer = q_samples[AF.nearest_neighbor(pose_samples, trgt_pose, True)]

        # Solve inverse kinematics for the current pose
        good_solution = False
        attempt_count = 0
        max_attempts = 10

        while not good_solution and attempt_count < max_attempts:
            try:
                ik_sol = CorkeRobot.ikine_LM(
                    Tep=Tt,
                    joint_limits=q_lim,
                    q0=q_closer,
                    tol=1e-6,
                    ilimit=100,
                    slimit=150,
                    mask=mask if i != 0 else [1,1,1,1,1,1],
                )

                q_sol = ik_sol.q
                DHRobot.check_JointInsideRange(q_sol)

                if i == 0 and not AF.floats_equal(list(q_sol),list(q0),tol=1e-3):

                    print('Initial solution no equal given starting configuration')
                    good_solution = False
                    attempt_count += 1

                if i == 0 or i == len(trgt_poses) - 1:
                    print(f"Corke IK sol {i+1}: {q_sol}")

                elif i == 1: 
                    print('... calculating IK solutions for intermediate points ...')

                good_solution = True
                
            except ValueError as e:
                print(f"Attempt {attempt_count + 1}: ValueError encountered - {e}")
                attempt_count += 1

        if not good_solution:
            raise RuntimeError(f"Failed to find a valid IK solution after {max_attempts} attempts.")

        if ik_sol.success:
            solutions.append(q_sol)
        else:
            print(
                f"Exact solution not found for pose {i+1}. Current error {ik_sol.residual}"
            )
            print()
            solutions.append(q_sol)

    # Convert solutions list to a 2D NumPy array if all elements are not None
    valid_solutions = [s for s in solutions if s is not None]
    if valid_solutions and len(valid_solutions) > 2:
        return np.array(valid_solutions)
    elif valid_solutions and len(valid_solutions) <= 2:
        return np.array(valid_solutions[-1])
    else:
        return None


if __name__ == "__main__":

    pass

    # np.set_printoptions(precision=2, suppress=True)

    # d1 = 0.25
    # d4 = 0.5
    # a2 = 0.1

    # joints = [
    #     RevoluteDH(
    #         home=math.pi / 2,
    #         d=d1,
    #         a=0,
    #         alpha=math.pi / 2,
    #     ),
    #     RevoluteDH(
    #         home=math.pi / 2,
    #         d=0,
    #         a=a2,
    #         alpha=0,
    #     ),
    #     RevoluteDH(
    #         home=-math.pi / 2,
    #         d=0,
    #         a=0,
    #         alpha=-math.pi / 2,
    #     ),
    #     RevoluteDH(
    #         home=0,
    #         d=d4,
    #         a=0,
    #         alpha=math.pi / 2,
    #     ),
    #     RevoluteDH(
    #         home=0,
    #         d=0,
    #         a=0,
    #         alpha=-math.pi / 2,
    #     ),
    # ]

    # tool = RobotTool([0, 0, 0.05], [0, 0, 0])

    # DHRobot = DH(
    #     name="Prova",
    #     joint_list=joints,
    #     tool=tool,
    # )

    # DHRobot.set_JointValues(np.zeros(DHRobot.n_joint))
    # print(DHRobot)

    # T = DHRobot.get_RobotTfm()
    # pose = DHRobot.get_EndEffPosOr()
    # print(f"Target Pose: {pose}")
    # Tp = SM.GetMatFromPose(pose)

    # ik_q = ikine_LM_Corke(DHRobot, pose)

    # DHRobot.set_JointValues(ik_q)
    # print(f"IK Solution Pose: {DHRobot.get_EndEffPosOr()}")
    # DHRobot.plot_Robot()
