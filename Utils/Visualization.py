"""
Matplotlib 可视化工具集。

Author: Little Si
Date:   2026-04
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 非交互后端，适合服务器/CI 环境
matplotlib.use("Agg")


class Visualization:
    """可视化工具集，所有方法均为静态方法，可直接调用。"""

    # ------------------------------------------------------------------
    # 通用绘图
    # ------------------------------------------------------------------

    @staticmethod
    def plot_trajectory(trajectory, title="2D 无人机轨迹", save_path=None, show=False):
        """绘制单条参考轨迹。"""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(trajectory[:, 0], trajectory[:, 1], "b-", linewidth=2, label="参考轨迹")
        ax.scatter(trajectory[0, 0], trajectory[0, 1], c="green", s=100, label="起点")
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c="red", s=100, label="终点")
        ax.grid(True)
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.legend()
        Visualization._finish(fig, save_path, show)

    @staticmethod
    def plot_uav_trajectory(reference, uav, title="UAV 轨迹跟踪", save_path=None, show=False):
        """绘制参考轨迹与无人机实际轨迹的对比。"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(reference[:, 0], reference[:, 1], "b-", linewidth=2, label="参考轨迹")
        ax.plot(uav[:, 0], uav[:, 1], "r-", linewidth=2, label="无人机轨迹")
        ax.scatter(reference[0, 0], reference[0, 1], c="green", s=100, label="起点")
        ax.scatter(reference[-1, 0], reference[-1, 1], c="red", s=100, label="终点")
        ax.grid(True)
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.legend()
        Visualization._finish(fig, save_path, show)

    @staticmethod
    def plot_error_curve(errors, title="距离误差曲线", save_path=None, show=False):
        """绘制距离误差随时间步变化的曲线。"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(errors)), errors, "r-", linewidth=1, label="距离误差")
        ax.axhline(y=np.mean(errors), color="b", linestyle="--",
                    label=f"平均误差: {np.mean(errors):.2f}")
        ax.grid(True)
        ax.set_title(title)
        ax.set_xlabel("时间步")
        ax.set_ylabel("距离误差 (m)")
        ax.legend()
        Visualization._finish(fig, save_path, show)

    # ------------------------------------------------------------------
    # ModelEvaluator 专用
    # ------------------------------------------------------------------

    @staticmethod
    def plot_rewards(rewards, save_path=None, show=False):
        """绘制各 episode 奖励柱状/折线图。"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(rewards) + 1), rewards, "o-")
        ax.set_title("Episode Rewards")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True)
        Visualization._finish(fig, save_path, show)

    @staticmethod
    def plot_distance_errors(all_distances, save_path=None, show=False):
        """绘制多个 episode 的距离误差曲线。"""
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, distances in enumerate(all_distances):
            ax.plot(range(len(distances)), distances, label=f"Episode {i + 1}")
        ax.set_title("Distance Errors")
        ax.set_xlabel("Step")
        ax.set_ylabel("Distance (m)")
        ax.grid(True)
        ax.legend()
        Visualization._finish(fig, save_path, show)

    @staticmethod
    def plot_trajectories(reference, uav_trajectories, save_path=None, show=False):
        """绘制参考轨迹与多条无人机轨迹。"""
        colors = ["blue", "green", "red", "purple", "orange",
                  "brown", "pink", "gray", "olive", "cyan"]
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(reference[:, 0], reference[:, 1], "b-", linewidth=2, label="Reference")
        ax.scatter(reference[0, 0], reference[0, 1], c="green", s=100, label="Start")
        ax.scatter(reference[-1, 0], reference[-1, 1], c="red", s=100, label="End")

        for i, traj in enumerate(uav_trajectories):
            traj = np.array(traj)
            ax.plot(traj[:, 0], traj[:, 1], "-", color=colors[i % len(colors)],
                    label=f"UAV {i + 1}")

        ax.set_title("UAV Trajectories")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.grid(True)
        ax.legend()
        Visualization._finish(fig, save_path, show)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _finish(fig, save_path, show):
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path)
            print(f"图形已保存到：{save_path}")
        if show:
            plt.show()
        plt.close(fig)
