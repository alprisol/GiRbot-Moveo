import math
import numpy as np
from typing import Union
import roboticstoolbox as rtb
import SpatialMath as SM
import AuxiliarFunctions as AF
from typing import Union, Callable, Optional, List


def adjust_angular_range(range_limits):
    """
    Adjust the angular range if the high limit is less than the low limit.

    This function ensures that the angular limits provided are valid. If
    the high limit is less than the low limit, it adjusts one of the
    limits to be equal to ±π, ensuring the range is valid for joint limits.

    Parameters:
    -----------
    range_limits : tuple or list
        A tuple or list containing two elements representing the lower and upper
        limits of the angular range.

    Returns:
    --------
    tuple :
        The corrected angular range limits.
    str :
        Error message if input is invalid.
    """
    # Ensure the input is a list or tuple and contains two elements
    if not isinstance(range_limits, (list, tuple)) or len(range_limits) != 2:
        return "Invalid input. Please provide a tuple or list of two elements."

    low, high = range_limits

    # If the high limit is less than the low limit
    if high - low < 0:
        # Calculate the distance of each limit from π
        distances = [(abs(math.pi - abs(x)), x) for x in [low, high]]
        # Find the limit closest to π
        closest = min(distances, key=lambda x: x[0])[1]

        # Set the closest limit to π or -π based on its sign
        new_value = math.pi if closest < 0 else -math.pi

        # Replace the original closest limit with the new value
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
    """
    Perform inverse kinematics using Corke's method with the Levenberg-Marquardt algorithm.

    This function solves the inverse kinematics problem using the
    Levenberg-Marquardt method, applied to a robot defined by Denavit-Hartenberg (DH)
    parameters. It supports multiple poses and provides solutions within joint limits.

    Parameters:
    -----------
    DHRobot : DHRobot
        A robot defined with Denavit-Hartenberg parameters.
    trgt_poses : np.ndarray
        A numpy array of target poses, where each row represents a 6D pose (position + orientation).
    q0 : np.ndarray, optional
        Initial guess for the joint angles, by default None.
    q_lim : np.ndarray, optional
        Joint limits for the robot, by default None.
    mask : Union[np.ndarray, list, bool], optional
        A mask specifying which degrees of freedom to consider in the inverse kinematics,
        by default False.

    Returns:
    --------
    np.ndarray :
        Array of solutions for the joint angles corresponding to the target poses.
    None :
        If no valid solution is found.
    """
    print(" *Using Corke's nIK method some joint range of movement is lost.")

    # Initialize a list to store solutions for each pose
    solutions = []

    # Ensure trgt_poses is a 2D array (handle single pose as well)
    if trgt_poses.ndim == 1:
        trgt_poses = np.array([trgt_poses])

    # Build a DHRobot model based on Corke's revolute and prismatic joints
    Corke_joints = []

    for j in DHRobot.joint_list:
        if j.j_type == 0:
            # Revolute joint with adjusted angular range limits
            Corke_joints.append(
                rtb.RevoluteDH(
                    d=j.d,
                    a=j.a,
                    alpha=j.alpha,
                    qlim=adjust_angular_range(j.qlim),
                )
            )
        else:
            # Prismatic joint
            Corke_joints.append(
                rtb.PrismaticDH(theta=j.theta, a=j.a, alpha=j.alpha, qlim=j.qlim)
            )

    # Create the robot model using Corke's DHRobot class
    CorkeRobot = rtb.DHRobot(
        links=Corke_joints, name="Corke Robot", tool=DHRobot.tool.Tfm
    )

    # Configure the mask for the degrees of freedom if needed
    if mask is True:
        # Full mask for all joint angles
        mask = np.concatenate((np.ones(DHRobot.n_joint), np.zeros(6 - DHRobot.n_joint)))
    elif mask is False:
        mask = None

    print(f"Mask set at {mask}. Change it if necessary with attribute <mask>")
    print(
        f"Valid Ranges for Corke's robot:\n{list(CorkeRobot.qlim[0])}\n{list(CorkeRobot.qlim[1])}"
    )

    # Loop over each target pose and solve the inverse kinematics
    for i, trgt_pose in enumerate(trgt_poses):
        Tt = SM.GetMatFromPose(trgt_pose)  # Get transformation matrix from the pose

        # Determine the initial guess for joint angles
        if i > 0 and solutions[-1] is not None:
            q_closer = solutions[-1]
        elif q0 is not None:
            q_closer = q0
        else:
            try:
                # Get samples of joint angles and corresponding poses
                q_samples, pose_samples = DHRobot.samples
            except AttributeError:
                q_samples, pose_samples = DHRobot.gen_samples(1000)
            # Find the nearest neighbor in the sample set
            q_closer = q_samples[AF.nearest_neighbor(pose_samples, trgt_pose, True)]

        # Try solving the IK with up to max_attempts
        good_solution = False
        attempt_count = 0
        max_attempts = 50

        while not good_solution and attempt_count < max_attempts:
            try:
                # Solve inverse kinematics using Levenberg-Marquardt method
                ik_sol = CorkeRobot.ikine_LM(
                    Tep=Tt,
                    joint_limits=q_lim,
                    q0=q_closer,
                    tol=1e-6,
                    ilimit=100,
                    slimit=150,
                    mask=mask if i != 0 else [1, 1, 1, 1, 1, 1],
                )

                # Extract solution for joint angles
                q_sol = list(ik_sol.q)

                # Check if the solution is within joint limits
                DHRobot.check_JointInsideRange(check_val=q_sol)

                # Validate the solution for the first pose if initial guess q0 is provided
                if (
                    q0 is not None
                    and i == 0
                    and not AF.floats_equal(list(q_sol), list(q0), tol=1e-3)
                ):
                    print("Initial solution not equal to given starting configuration")
                    good_solution = False
                    attempt_count += 1

                if i == 0 or i == len(trgt_poses) - 1:
                    print(f"Corke IK sol {i+1}: {q_sol}")
                elif i == 1:
                    print("... calculating IK solutions for intermediate points ...")

                good_solution = True

            except ValueError as e:
                print(f"Attempt {attempt_count + 1}: ValueError encountered - {e}")
                # If no initial guess, try next sample for initial guess
                if q0 is None:
                    q_samples = np.delete(
                        q_samples,
                        AF.nearest_neighbor(pose_samples, trgt_pose, True),
                        axis=0,
                    )
                    q_closer = q_samples[
                        AF.nearest_neighbor(pose_samples, trgt_pose, True)
                    ]
                attempt_count += 1

        if not good_solution:
            raise RuntimeError(
                f"Failed to find a valid IK solution after {max_attempts} attempts."
            )

        if ik_sol.success:
            solutions.append(q_sol)
        else:
            print(
                f"Exact solution not found for pose {i+1}. Current error {ik_sol.residual}"
            )
            solutions.append(q_sol)

    # Convert the list of solutions to a 2D NumPy array
    valid_solutions = [s for s in solutions if s is not None]
    if valid_solutions and len(valid_solutions) >= 2:
        return np.array(valid_solutions)
    elif valid_solutions and len(valid_solutions) < 2:
        return np.array(valid_solutions[-1])
    else:
        return None
