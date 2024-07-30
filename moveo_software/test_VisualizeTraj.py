import math
import numpy as np
import matplotlib.pyplot as plt


def plot_JointEvolution(q, qd, t, save_path=None):
    N = q.shape[1]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), facecolor="white")
    fig.suptitle("Joint Kinematics Evolution", fontsize=16, fontweight="bold")
    ax1.set_title("Position")
    ax2.set_title("Velocity")
    labels = []
    for i in range(N):
        ax1.plot(t, q[:, i], label=f"Joint {i+1}", linewidth=2)
        labels.append(f"Joint {i+1}")
    ax1.set_ylabel("Position (steps)")
    ax1.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax1.set_facecolor("whitesmoke")

    for i in range(N):
        ax2.plot(t, qd[:, i], linewidth=2)
    ax2.set_ylabel("Velocity (steps/s)")
    ax2.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax2.set_facecolor("whitesmoke")

    ax2.set_xlabel("Time (s)")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=N,
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":

    traj_name = (
        "LTraj_files/"
        + "[0.22, 0.0, 0.47, 0.0, -0.0, -0.0]-to-[-0.0, 0.0, 0.69, 1.57, -0.0, 0.0]"
    )

    traj_q = np.load(traj_name + '_q.npy')
    traj_qd = np.load(traj_name + "_qd.npy")
    traj_t = np.load(traj_name + '_t.npy')

    print(traj_q.shape)

    plot_JointEvolution(traj_q, traj_qd, traj_t, traj_name + "_plot.png")
