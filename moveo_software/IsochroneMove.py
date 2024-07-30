import numpy as np
import sympy as sp
import math
import matplotlib.pyplot as plt
from PiecewiseExpr import PiecewiseFunction, Piece
import AuxiliarFunctions as AF


class IsochroneMove:

    def __init__(
        self,
        currValues,
        trgtValues,
        maxVel,
        accel,
        isPrism,
        validRange,
    ):

        self.maxVel = maxVel
        self.accel = accel
        self.isPrism = isPrism
        self.validRange = validRange
        self.trgtValues = []
        self.currValues = []

        for i, is_p in enumerate(isPrism):
            if is_p == True:
                self.trgtValues.append(trgtValues[i])
                self.currValues.append(currValues[i])
            else:
                self.trgtValues.append(AF.wrap_angle(trgtValues[i]))
                self.currValues.append(AF.wrap_angle(currValues[i]))

        self.mMovDist = self.calc_MotorMovDist()

        if AF.floats_equal(self.mMovDist.tolist(), [0.0] * len(self.mMovDist.tolist())):

            print(" *Start and Target values for IsochroneMove are equal")
            self.mMovDire = np.array([1] * len(trgtValues))
            self.moveTime = 0
            self.mABSMaxVel = np.array([0] * len(trgtValues))
            self.mMaxVel = np.array([0] * len(trgtValues))
            self.mProfileType = "Still"
            self.mVelProfilesExpr = None
            self.mPositionExpr = None
            self.mAccelExpr = None

        else:
            self.mMovDire = self.find_MotorMovDire()
            self.moveTime = self.calc_MoveTime()
            self.mABSMaxVel = self.calc_JointsMaxVel()
            self.mMaxVel = self.calc_JointVel()
            self.mVelProfilesExpr, self.mProfileType = self.create_VelProfileExpr()
            self.mPositionExpr = self.create_PositionExpr()
            self.mAccelExpr = self.create_AccelExpr()

    def calc_MotorMovDist(self):

        currValues = self.currValues
        trgtValues = self.trgtValues
        isPrism = self.isPrism
        validRange = self.validRange

        distances = AF.calc_dist_in_range(
            value1=currValues,
            value2=trgtValues,
            is_linear=isPrism,
            valid_range=validRange,
        )

        return np.array(distances)

    def find_MotorMovDire(self):

        sign_arr = np.sign(self.mMovDist)
        sign_arr[sign_arr == 0] = 1

        return sign_arr

    def calc_MoveTime(self):

        accel = self.accel
        maxVel = self.maxVel

        # Find the max distance to cover
        dis = abs(self.mMovDist)
        maxDis = np.max(dis)

        # Calculate time to reach max velocity
        t_accel = maxVel / accel

        # Calculate distance covered during acceleration and deceleration
        d_accel = 0.5 * accel * t_accel**2

        if maxDis < 2 * d_accel:
            # No constant velocity phase (triangle profile)
            t_accel = np.sqrt(maxDis / accel)
            total_time = 2 * t_accel
            self.fastestVel = t_accel * accel

        else:
            # With constant velocity phase (trapezoidal profile)
            d_const = maxDis - 2 * d_accel
            t_const = d_const / maxVel
            total_time = 2 * t_accel + t_const
            self.fastestVel = maxVel

        return total_time

    def calc_JointsMaxVel(self):

        time = self.moveTime
        accel = self.accel
        dis = np.abs(self.mMovDist)

        motors_velocities = []

        for i, d in enumerate(dis):
            if d == 0:
                mVel = 0

            elif d == np.max(dis):
                mVel = self.fastestVel

            else:

                if accel < (4 * d / (time**2)):

                    accel = 4 * d / (time**2)
                    print(
                        f"Given acceleration is not sufficient to complete the movement in time, new acceleration set at {accel}"
                    )

                mVel = (
                    (time * accel) - math.sqrt(((time * accel) ** 2) - 4 * (d * accel))
                ) / 2

            motors_velocities.append(mVel)

        return np.array(motors_velocities)

    def calc_JointVel(self):

        mVel = self.mABSMaxVel
        mDir = self.mMovDire

        return mVel * mDir

    @classmethod
    def get_TimeJointVel(self, currValues, trgtValues, maxVel, accel, isPrism, validRange):

        instance = self(
            currValues=currValues,
            trgtValues=trgtValues,
            maxVel=maxVel,
            accel=accel,
            isPrism=isPrism,
            validRange=validRange,
        )

        
        return instance.calc_MoveTime(), instance.calc_JointVel()

    def create_VelProfileExpr(self):

        t_end = self.moveTime
        accel = self.accel

        profileType = []

        mVelProfilesExpr = []

        for maxVel, sign in zip(self.mMaxVel, self.mMovDire):

            t_acc = t_dec = abs(maxVel / accel)

            if t_acc == 0:

                # No movement
                velProfile = [
                    Piece(
                        0,
                        sp.core.numbers.NegativeInfinity,
                        sp.core.numbers.Infinity,
                    )
                ]
                profileType.append("Still")

            elif t_acc >= t_end / 2:
                # Triangular Profile
                t_acc = t_dec = t_end / 2
                velProfile = [
                    Piece(0, sp.core.numbers.NegativeInfinity, 0),
                    Piece(sign * accel * sp.Symbol("t"), 0, t_acc),
                    Piece(
                        maxVel - (sign * accel * (sp.Symbol("t") - t_acc)),
                        t_acc,
                        t_end,
                    ),
                    Piece(0, t_end, sp.core.numbers.Infinity),
                ]
                profileType.append("Triangular")

            else:
                # Trapezoidal Profile
                t_const = t_end - 2 * t_acc
                velProfile = [
                    Piece(0, sp.core.numbers.NegativeInfinity, 0),
                    Piece(maxVel * sp.Symbol("t") / t_acc, 0, t_acc),
                    Piece(maxVel, t_acc, t_acc + t_const),
                    Piece(
                        maxVel * (1 - (sp.Symbol("t") - (t_acc + t_const)) / (t_dec)),
                        t_acc + t_const,
                        t_end,
                    ),
                    Piece(0, t_end, sp.core.numbers.Infinity),
                ]
                profileType.append("Trapezoidal")

            velProfile = PiecewiseFunction(velProfile, "t")
            mVelProfilesExpr.append(velProfile)

        return mVelProfilesExpr, profileType

    def create_PositionExpr(self):

        mPositionExpr = []

        for i, profile in enumerate(self.mVelProfilesExpr):

            position = profile.integrate(first_integration_ctt=self.currValues[i])
            mPositionExpr.append(position)

        return mPositionExpr

    def create_AccelExpr(self):

        mAccelExpr = []

        for i, profile in enumerate(self.mVelProfilesExpr):

            position = profile.derive()
            mAccelExpr.append(position)

        return mAccelExpr

    def plot_MotorProfiles(
        self,
        axis_start,
        axis_end,
        n_points=500,
        profile_label="Profile",
        colormap="viridis",
    ):

        for domain, domain_label in zip(
            [self.mAccelExpr, self.mVelProfilesExpr, self.mPositionExpr],
            ["Acceleration", "Velocity", "Position"],
        ):

            plt.figure(figsize=(10, 6))

            # Generate colors from the colormap
            cmap = plt.get_cmap(colormap)
            colors = [cmap(i) for i in np.linspace(0, 1, len(domain))]

            for idx, (profile, color) in enumerate(zip(domain, colors)):
                label = f"{profile_label} {idx+1}"
                if domain_label == "Position":
                    profile.plot(
                        axis_start,
                        axis_end,
                        num_points=n_points,
                        label=label,
                        show=False,
                        wrap=True,
                        wrap_limits=[-math.pi, math.pi],
                        ylabel="Position (fraction of revolution)",
                        xlabel="Time (s)",
                        color=color,
                    )
                elif domain_label == "Acceleration":
                    profile.plot(
                        axis_start,
                        axis_end,
                        num_points=n_points,
                        label=label,
                        show=False,
                        ylabel="Acceleration (steps/s^2)",
                        xlabel="Time (s)",
                        color=color,
                    )
                else:
                    profile.plot(
                        axis_start,
                        axis_end,
                        num_points=n_points,
                        label=label,
                        show=False,
                        ylabel="Velocity (steps/s)",
                        xlabel="Time (s)",
                        color=color,
                    )

            plt.axhline(0, color="black", linewidth=0.5)
            plt.axvline(0, color="black", linewidth=0.5)
            plt.grid(color="gray", linestyle="--", linewidth=0.5)
            plt.title(domain_label)
            plt.legend()
            plt.show()


if __name__ == "__main__":

    np.set_printoptions(precision=2, suppress=True)

    Prova = IsochroneMove(
        currValues=[-math.pi / 4, math.pi / 4, 0, math.pi / 2, math.pi / 4],
        trgtValues=[
            1.5709066582210909,
            0.00023719941311650672,
            4.1113662971881126e-05,
            0.0,
            0.0,
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

    Prova.plot_MotorProfiles(0, Prova.moveTime, profile_label="Motor")
