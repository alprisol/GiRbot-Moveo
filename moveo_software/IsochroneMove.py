import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from PiecewiseExpr import PiecewiseFunction, Piece
import AuxiliarFunctions as AF


class IsochroneMove:
    """
    The `IsochroneMove` class is responsible for generating an isochronous movement, which ensures that each joint
    reaches its target value in the same amount of time while respecting velocity and acceleration limits.
    It handles both prismatic and revolute joints, with special handling for angular wrapping in revolute joints.

    Parameters:
    - currValues (list): The current joint positions (either angles for revolute joints or displacements for prismatic joints).
    - trgtValues (list): The target joint positions.
    - maxVel (list): The maximum allowable velocities for each joint.
    - accel (list): The maximum allowable accelerations for each joint.
    - isPrism (list): A list of booleans indicating whether each joint is prismatic (`True`) or revolute (`False`).
    - validRange (list): A list of valid ranges (limits) for each joint.

    Attributes:
    - mMovDist (np.ndarray): The calculated distance that each joint must move to reach the target.
    - mMovDire (np.ndarray): The direction of motion for each joint (+1 for positive, -1 for negative).
    - moveTime (float): The total time required for the movement to complete.
    - mABSMaxVel (np.ndarray): The absolute maximum velocity for each joint, constrained by `maxVel`.
    - mMaxVel (np.ndarray): The actual maximum velocity for each joint.
    - mVelProfilesExpr (Any): The velocity profile expression for each joint over time.
    - mPositionExpr (Any): The position expression for each joint over time.
    - mAccelExpr (Any): The acceleration expression for each joint over time.
    - mProfileType (str): A string representing the type of motion profile (e.g., "Still" if there is no movement).

    Notes:
    - If the current and target joint values are the same, the movement is classified as "Still", and no motion profile is generated.
    """

    def __init__(
        self,
        currValues,
        trgtValues,
        maxVel,
        accel,
        isPrism,
        validRange,
    ):
        """
        Initialize the IsochroneMove object to compute a smooth motion profile between the current and target joint values.
        """
        self.maxVel = maxVel  # Maximum allowable velocity for each joint
        self.accel = accel  # Maximum allowable acceleration for each joint
        self.isPrism = (
            isPrism  # Boolean list indicating whether each joint is prismatic
        )
        self.validRange = validRange  # Valid range for each joint (min, max)
        self.trgtValues = (
            []
        )  # List to store processed target values (wrapped if revolute)
        self.currValues = (
            []
        )  # List to store processed current values (wrapped if revolute)

        # Process target and current values, wrapping angles for revolute joints
        for i, is_p in enumerate(isPrism):
            if is_p:  # If prismatic, use values as is
                self.trgtValues.append(trgtValues[i])
                self.currValues.append(currValues[i])
            else:  # If revolute, wrap angles to avoid issues with angular discontinuities
                self.trgtValues.append(AF.wrap_angle(trgtValues[i]))
                self.currValues.append(AF.wrap_angle(currValues[i]))

        # Calculate the movement distance for each joint
        self.mMovDist = self.calc_MotorMovDist()

        # If no movement is required (current and target values are the same)
        if AF.floats_equal(self.mMovDist.tolist(), [0.0] * len(self.mMovDist.tolist())):
            print(" *Start and Target values for IsochroneMove are equal")
            self.mMovDire = np.array([1] * len(trgtValues))  # Default direction
            self.moveTime = 0  # No time required if there's no movement
            self.mABSMaxVel = np.array([0] * len(trgtValues))  # No velocity
            self.mMaxVel = np.array([0] * len(trgtValues))  # No velocity
            self.mProfileType = "Still"  # Movement profile is "Still"
            self.mVelProfilesExpr = None  # No velocity profile
            self.mPositionExpr = None  # No position expression
            self.mAccelExpr = None  # No acceleration expression

        # If movement is required, calculate the necessary parameters
        else:
            self.mMovDire = self.find_MotorMovDire()  # Determine direction of motion
            self.moveTime = (
                self.calc_MoveTime()
            )  # Calculate the time required for the move
            self.mABSMaxVel = (
                self.calc_JointsMaxVel()
            )  # Calculate the maximum velocities for each joint
            self.mMaxVel = self.calc_JointVel()  # Adjust velocities if necessary
            self.mVelProfilesExpr, self.mProfileType = (
                self.create_VelProfileExpr()
            )  # Create velocity profiles
            self.mPositionExpr = self.create_PositionExpr()  # Create position profiles
            self.mAccelExpr = self.create_AccelExpr()  # Create acceleration profiles

    def calc_MotorMovDist(self):
        """
        Calculate the movement distance for each joint between the current and target values.

        This method computes the distance each joint needs to move to reach its target position, taking into account
        whether the joint is prismatic (linear) or revolute (angular). For revolute joints, the distance is calculated
        while respecting angular wrapping (i.e., ensuring the movement occurs in the shortest direction around the circle).

        Returns:
        - np.ndarray: An array containing the movement distances for each joint, either as linear or angular distances.

        Notes:
        - For prismatic joints, the distance is simply the difference between the current and target values.
        - For revolute joints, the distance is computed as the shortest angular distance (considering wrapping around ±π).
        """
        # Retrieve current and target values, as well as joint types and valid ranges
        currValues = self.currValues  # Current joint positions (or angles)
        trgtValues = self.trgtValues  # Target joint positions (or angles)
        isPrism = (
            self.isPrism
        )  # Boolean list indicating whether joints are prismatic or revolute
        validRange = self.validRange  # Valid range for each joint's motion

        # Calculate the movement distances using an external utility function
        distances = AF.calc_dist_in_range(
            value1=currValues,
            value2=trgtValues,
            is_linear=isPrism,  # Indicate whether each joint is prismatic (linear) or revolute
            valid_range=validRange,  # Pass the valid range of motion for each joint
        )

        return np.array(distances)  # Return the calculated distances as a numpy array

    def find_MotorMovDire(self):
        """
        Determine the movement direction for each joint based on the calculated movement distances.

        This method calculates the direction in which each joint should move to reach its target value. The movement
        direction is represented as +1 for positive movement (increasing values) and -1 for negative movement (decreasing values).
        If a joint does not need to move (i.e., the movement distance is 0), it defaults to +1.

        Returns:
        - np.ndarray: An array where each element is either +1 (positive movement) or -1 (negative movement), representing
        the movement direction for each joint.

        Notes:
        - If the calculated movement distance is zero for a joint, the direction is set to +1 by default to avoid undefined
        behavior.
        """
        # Compute the sign of the movement distances (positive for increasing, negative for decreasing)
        sign_arr = np.sign(self.mMovDist)

        # If the movement distance is zero, set the direction to +1 by default
        sign_arr[sign_arr == 0] = 1

        return sign_arr  # Return the movement directions as an array

    def calc_MoveTime(self):
        """
        Calculate the total time required to complete the movement for all joints.

        This method computes the time needed for the joints to move from their current positions to the target positions,
        considering acceleration, deceleration, and possible constant velocity phases. The movement can follow either a
        triangular velocity profile (no constant velocity phase) or a trapezoidal velocity profile (with a constant velocity
        phase).

        Returns:
        - total_time (float): The total time required to complete the movement, considering acceleration, constant velocity,
        and deceleration phases.

        Notes:
        - The method handles two cases:
        1. **Triangular Profile**: If the maximum distance to cover is too small to reach the maximum velocity before
            deceleration, the movement follows a triangular profile (accelerating and decelerating without reaching max velocity).
        2. **Trapezoidal Profile**: If the maximum distance allows reaching the maximum velocity, the movement follows a
            trapezoidal profile (accelerating, maintaining constant velocity, then decelerating).
        - The method stores the maximum velocity reached during the movement in `self.fastestVel`.
        """
        accel = self.accel  # Maximum allowable acceleration for the joints
        maxVel = self.maxVel  # Maximum allowable velocity for the joints

        # Find the maximum distance that needs to be covered across all joints
        dis = abs(self.mMovDist)
        maxDis = np.max(dis)  # The maximum distance that any joint must travel

        # Calculate the time needed to reach maximum velocity (acceleration phase)
        t_accel = maxVel / accel

        # Calculate the distance covered during acceleration and deceleration phases
        d_accel = 0.5 * accel * t_accel**2

        # Check if the movement distance is small enough to avoid a constant velocity phase
        if maxDis < 2 * d_accel:
            # If the distance is too small to reach max velocity, use a triangular velocity profile
            t_accel = np.sqrt(
                maxDis / accel
            )  # Recalculate acceleration time for the triangular profile
            total_time = (
                2 * t_accel
            )  # Total time is twice the acceleration time (acceleration + deceleration)
            self.fastestVel = (
                t_accel * accel
            )  # The highest velocity reached is during the acceleration phase

        else:
            # If there is enough distance, use a trapezoidal velocity profile (with a constant velocity phase)
            d_const = (
                maxDis - 2 * d_accel
            )  # Distance covered during the constant velocity phase
            t_const = d_const / maxVel  # Time spent at constant velocity
            total_time = (
                2 * t_accel + t_const
            )  # Total time includes acceleration, constant velocity, and deceleration phases
            self.fastestVel = (
                maxVel  # The maximum velocity reached is the allowable max velocity
            )

        return total_time  # Return the total time required for the movement

    def calc_JointsMaxVel(self):
        """
        Calculate the maximum velocity for each joint based on the movement distance and time constraints.

        This method computes the maximum velocity for each joint required to complete the movement within the specified
        total time (`self.moveTime`). If the joint needs to travel a shorter distance, it may reach a lower maximum
        velocity than the fastest joint. Additionally, the method adjusts the acceleration if it is too low to complete
        the movement within the given time.

        Returns:
        - np.ndarray: An array containing the maximum velocity for each joint, ensuring that all joints finish the movement
        at the same time.

        Notes:
        - If the acceleration is insufficient to complete the movement in the specified time, it is adjusted accordingly.
        - For joints with zero movement distance, the maximum velocity is set to zero.
        - Joints that need to travel the farthest will have their maximum velocity set to `self.fastestVel`.
        """
        time = self.moveTime  # Total time allowed for the movement
        accel = self.accel  # Maximum allowable acceleration for the joints
        dis = np.abs(self.mMovDist)  # Absolute movement distances for each joint

        motors_velocities = (
            []
        )  # List to store the calculated maximum velocities for each joint

        # Loop through each joint's movement distance and calculate the required velocity
        for i, d in enumerate(dis):
            if d == 0:
                # If the movement distance is zero, the velocity is zero
                mVel = 0

            elif d == np.max(dis):
                # If the joint has the maximum movement distance, use the pre-calculated fastest velocity
                mVel = self.fastestVel

            else:
                # Check if the given acceleration is sufficient to complete the movement in the specified time
                if accel < (4 * d / (time**2)):
                    # Adjust the acceleration if it's too low for the movement
                    accel = 4 * d / (time**2)
                    print(
                        f"Given acceleration is not sufficient to complete the movement in time, new acceleration set at {accel}"
                    )

                # Calculate the maximum velocity required for the joint
                mVel = (
                    (time * accel) - math.sqrt(((time * accel) ** 2) - 4 * (d * accel))
                ) / 2

            # Append the calculated maximum velocity to the list
            motors_velocities.append(mVel)

        return np.array(
            motors_velocities
        )  # Return the maximum velocities as a numpy array

    def calc_JointVel(self):
        """
        Calculate the velocity for each joint by combining the maximum velocity and movement direction.

        This method computes the actual velocity for each joint by multiplying the maximum velocity (absolute value)
        by the direction of movement. The direction ensures that the velocity is positive or negative based on whether
        the joint is moving forward or backward.

        Returns:
        - np.ndarray: An array containing the signed velocities for each joint, where the sign indicates the direction
        of movement and the magnitude indicates the speed.
        """
        mVel = self.mABSMaxVel  # The absolute maximum velocity for each joint
        mDir = self.mMovDire  # The direction of movement for each joint (+1 or -1)

        # Multiply the absolute velocity by the direction to get the signed velocity for each joint
        return mVel * mDir  # Return the signed velocities as a numpy array

    @classmethod
    def get_TimeJointVel(
        self, currValues, trgtValues, maxVel, accel, isPrism, validRange
    ):
        """
        Class method to compute the total movement time and joint velocities for a given set of parameters.

        This method initializes an `IsochroneMove` instance with the provided parameters and returns the calculated
        movement time and joint velocities. It provides a convenient way to get these values without manually creating
        an instance of the class.

        Parameters:
        - currValues (list): The current joint positions (either angles for revolute joints or displacements for prismatic joints).
        - trgtValues (list): The target joint positions to be reached.
        - maxVel (list): The maximum allowable velocities for each joint.
        - accel (list): The maximum allowable accelerations for each joint.
        - isPrism (list): A boolean list indicating whether each joint is prismatic (`True`) or revolute (`False`).
        - validRange (list): A list of valid ranges (limits) for each joint's movement.

        Returns:
        - tuple: A tuple containing:
        1. **moveTime (float)**: The total time required for the movement.
        2. **jointVel (np.ndarray)**: An array of velocities for each joint, considering both direction and speed.
        """
        # Create an instance of the IsochroneMove class using the provided parameters
        instance = self(
            currValues=currValues,
            trgtValues=trgtValues,
            maxVel=maxVel,
            accel=accel,
            isPrism=isPrism,
            validRange=validRange,
        )

        # Return the calculated move time and joint velocities
        return instance.calc_MoveTime(), instance.calc_JointVel()

    def create_VelProfileExpr(self):
        """
        Create velocity profile expressions for each joint over time, based on the total movement time and acceleration.

        This method generates symbolic velocity profiles for each joint, defining how the velocity changes over time
        during the movement. The velocity profile can be one of two types:
        - **Triangular Profile**: Used when the joint accelerates and decelerates without reaching a constant velocity.
        - **Trapezoidal Profile**: Used when the joint accelerates, maintains a constant velocity, and then decelerates.

        Returns:
        - tuple:
        1. **mVelProfilesExpr (list)**: A list of symbolic velocity profile expressions for each joint, represented as
            a `PiecewiseFunction` object.
        2. **profileType (list)**: A list of strings indicating the type of profile ("Still", "Triangular", or "Trapezoidal")
            for each joint.

        Notes:
        - **Still Profile**: If the joint does not move, the velocity profile is a constant zero ("Still").
        - The velocity profiles are created using symbolic expressions for time (`t`), allowing for dynamic analysis or
        evaluation at different time points.
        """
        t_end = self.moveTime  # The total time available for the movement
        accel = self.accel  # The maximum allowable acceleration

        profileType = []  # List to store the type of profile for each joint
        mVelProfilesExpr = []  # List to store the symbolic velocity profile expressions

        # Iterate over the maximum velocities and movement directions for each joint
        for maxVel, sign in zip(self.mMaxVel, self.mMovDire):

            # Calculate the time to accelerate and decelerate based on maximum velocity and acceleration
            t_acc = t_dec = abs(maxVel / accel)

            if t_acc == 0:
                # If the joint does not move, create a "Still" profile with zero velocity
                velProfile = [
                    Piece(
                        0,  # Constant zero velocity
                        sp.core.numbers.NegativeInfinity,  # Start time (negative infinity)
                        sp.core.numbers.Infinity,  # End time (infinity)
                    )
                ]
                profileType.append("Still")

            elif t_acc >= t_end / 2:
                # Triangular velocity profile: Acceleration and deceleration phases without reaching constant velocity
                t_acc = t_dec = t_end / 2
                velProfile = [
                    Piece(
                        0, sp.core.numbers.NegativeInfinity, 0
                    ),  # No velocity before t=0
                    Piece(
                        sign * accel * sp.Symbol("t"), 0, t_acc
                    ),  # Linear acceleration phase
                    Piece(
                        maxVel - (sign * accel * (sp.Symbol("t") - t_acc)),
                        t_acc,
                        t_end,
                    ),  # Linear deceleration phase
                    Piece(
                        0, t_end, sp.core.numbers.Infinity
                    ),  # No velocity after the end of the movement
                ]
                profileType.append("Triangular")

            else:
                # Trapezoidal velocity profile: Acceleration, constant velocity, and deceleration phases
                t_const = t_end - 2 * t_acc
                velProfile = [
                    Piece(
                        0, sp.core.numbers.NegativeInfinity, 0
                    ),  # No velocity before t=0
                    Piece(
                        maxVel * sp.Symbol("t") / t_acc, 0, t_acc
                    ),  # Acceleration phase
                    Piece(maxVel, t_acc, t_acc + t_const),  # Constant velocity phase
                    Piece(
                        maxVel * (1 - (sp.Symbol("t") - (t_acc + t_const)) / (t_dec)),
                        t_acc + t_const,
                        t_end,
                    ),  # Deceleration phase
                    Piece(
                        0, t_end, sp.core.numbers.Infinity
                    ),  # No velocity after the end of the movement
                ]
                profileType.append("Trapezoidal")

            # Create a PiecewiseFunction for the velocity profile using symbolic time "t"
            velProfile = PiecewiseFunction(velProfile, "t")
            mVelProfilesExpr.append(velProfile)

        # Return the velocity profiles and their respective types
        return mVelProfilesExpr, profileType

    def create_PositionExpr(self):
        """
        Create position profile expressions for each joint by integrating the velocity profiles.

        This method generates symbolic position profiles for each joint over time by integrating the corresponding velocity
        profiles. The initial position of each joint is set to its current value (`self.currValues[i]`), and the position
        profile describes how the joint's position changes over time during the movement.

        Returns:
        - list: A list of symbolic position profile expressions for each joint, derived by integrating the joint's
        velocity profile.

        Notes:
        - The position expressions are created by integrating the velocity profile over time, starting from the initial
        position (`self.currValues`).
        - These symbolic expressions can be evaluated at any time during the movement to determine the joint's position.
        """
        mPositionExpr = []  # List to store the symbolic position profile expressions

        # Iterate over each joint's velocity profile
        for i, profile in enumerate(self.mVelProfilesExpr):
            # Integrate the velocity profile to obtain the position profile
            position = profile.integrate(first_integration_ctt=self.currValues[i])
            mPositionExpr.append(position)  # Store the position profile

        return mPositionExpr  # Return the list of position profiles

    def create_AccelExpr(self):
        """
        Create acceleration profile expressions for each joint by differentiating the velocity profiles.

        This method generates symbolic acceleration profiles for each joint by taking the derivative of the corresponding
        velocity profiles. The acceleration profile describes how the joint's acceleration changes over time during the movement.

        Returns:
        - list: A list of symbolic acceleration profile expressions for each joint, derived by differentiating the joint's
        velocity profile.

        Notes:
        - The acceleration expressions are created by differentiating the velocity profile over time.
        - These symbolic expressions can be evaluated at any time during the movement to determine the joint's acceleration.
        """
        mAccelExpr = []  # List to store the symbolic acceleration profile expressions

        # Iterate over each joint's velocity profile
        for i, profile in enumerate(self.mVelProfilesExpr):
            # Differentiate the velocity profile to obtain the acceleration profile
            acceleration = (
                profile.derive()
            )  # 'derive()' differentiates the symbolic velocity profile
            mAccelExpr.append(acceleration)  # Store the acceleration profile

        return mAccelExpr  # Return the list of acceleration profiles

    def plot_MotorProfiles(
        self,
        axis_start,
        axis_end,
        n_points=500,
        profile_label="Profile",
        colormap="viridis",
        save=False,
    ):
        """
        Plot the acceleration, velocity, and position profiles for each joint.

        This method generates and plots the symbolic acceleration, velocity, and position profiles for each joint over
        the specified time range. Each profile is plotted on a separate figure, with different colors used for each joint.
        The method uses a colormap to assign colors and can save the generated plots to files if specified.

        Parameters:
        - axis_start (float): The starting time (x-axis) for the plot.
        - axis_end (float): The ending time (x-axis) for the plot.
        - n_points (int): The number of points to use for plotting the profiles. Default is 500.
        - profile_label (str): A label for the profiles (used to differentiate between different joint profiles in the legend).
        Default is "Profile".
        - colormap (str): The colormap to use for coloring the different joint profiles. Default is "viridis".
        - save (bool): Whether to save the generated plots as files. Default is False.

        Notes:
        - The method plots three sets of profiles: acceleration, velocity, and position. Each joint's profile is shown
        in a different color, with labels indicating the joint number.
        - If `save=True`, the method saves each plot as an image file with a filename indicating the profile type
        (e.g., "test_IsocroneMove_Acceleration", "test_IsocroneMove_Velocity", and "test_IsocroneMove_Position").
        """
        # Loop over each profile type: Acceleration, Velocity, and Position
        for domain, domain_label in zip(
            [self.mAccelExpr, self.mVelProfilesExpr, self.mPositionExpr],
            ["Acceleration", "Velocity", "Position"],
        ):

            plt.figure(figsize=(10, 6))  # Create a new figure for each profile type

            # Generate colors from the colormap
            cmap = plt.get_cmap(colormap)
            colors = [
                cmap(i) for i in np.linspace(0, 1, len(domain))
            ]  # Assign colors for each joint

            # Loop over each joint's profile and color
            for idx, (profile, color) in enumerate(zip(domain, colors)):
                label = (
                    f"{profile_label} {idx+1}"  # Create a label for the joint's profile
                )
                if domain_label == "Position":
                    # Plot position profile with wrapping for revolute joints
                    profile.plot(
                        axis_start,
                        axis_end,
                        num_points=n_points,
                        label=label,
                        show=False,
                        wrap=True,  # Wrap the position for angular limits (e.g., [-pi, pi])
                        wrap_limits=[-math.pi, math.pi],
                        ylabel="Position (fraction of revolution)",
                        xlabel="Time (s)",
                        color=color,
                    )
                elif domain_label == "Acceleration":
                    # Plot acceleration profile
                    profile.plot(
                        axis_start,
                        axis_end,
                        num_points=n_points,
                        label=label,
                        show=False,
                        ylabel="Acceleration (rad/s²)",
                        xlabel="Time (s)",
                        color=color,
                    )
                else:
                    # Plot velocity profile
                    profile.plot(
                        axis_start,
                        axis_end,
                        num_points=n_points,
                        label=label,
                        show=False,
                        ylabel="Velocity (rad/s)",
                        xlabel="Time (s)",
                        color=color,
                    )

            # Add gridlines and reference axes to the plot
            plt.axhline(0, color="black", linewidth=0.5)
            plt.axvline(0, color="black", linewidth=0.5)
            plt.grid(color="gray", linestyle="--", linewidth=0.5)
            plt.title(
                domain_label
            )  # Set the title for the plot based on the profile type
            plt.legend()  # Add the legend

            if save:
                # Save the plot to a file if 'save' is True
                plt.savefig(fname=f"test_IsocroneMove_{domain_label}")

            plt.show()  # Display the plot


if __name__ == "__main__":

    np.set_printoptions(precision=2, suppress=True)

    Prova = IsochroneMove(
        currValues=[0, 0, 0, 0, 0],
        trgtValues=[
            (6.75 / 10) * math.pi,
            -math.pi / 13,
            -math.pi / 4,
            -math.pi,
            -math.pi / 4,
        ],
        maxVel=10,
        accel=5,
        isPrism=[False, False, False, False, False],
        validRange=[
            (-(6.75 / 10) * math.pi, (6.75 / 10) * math.pi),
            (-math.pi / 12, -11 * math.pi / 12),
            (11 * math.pi / 12, math.pi / 12),
            (-math.pi, math.pi),
            (-math.pi / 2, math.pi / 2),
        ],
    )

    accel = Prova.accel
    total_time = Prova.moveTime
    start = Prova.currValues
    end = Prova.trgtValues
    dis = Prova.mMovDist
    dir = Prova.mMovDire
    vel = Prova.mMaxVel
    type = Prova.mProfileType

    print(f"Movement time: \n{total_time}")
    print(f"Acceleration rate: \n{accel}")
    print(f"Starting Positions: \n{start}")
    print(f"End Positions: \n{end}")
    print(f"Distance to cover: \n{dis}")
    print(f"Direction of the movement: \n{dir}")
    print(f"Maximum velocity reached: \n{list(vel)}")
    print(f"Type of velocity profile: \n{type}")
    print()

    final_q = []
    for e in Prova.mPositionExpr:

        final_q.append(e.subs_IndepVar(total_time))

    print(f"Position at time {round(total_time,2)}:\n{final_q}")

    sp.pprint(Prova.mPositionExpr[0].expr)

    Prova.plot_MotorProfiles(0, Prova.moveTime, profile_label="Joint", save=True)
