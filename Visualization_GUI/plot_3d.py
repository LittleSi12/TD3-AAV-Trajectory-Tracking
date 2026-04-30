"""
3D 视角绘图 — 与 TrajectoryGenerator.plot() 样式完全一致，
所有内容在 Z=0 平面。
字体：中文宋体，英文/数字 Times New Roman。
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["SimSun", "Times New Roman"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["mathtext.fontset"] = "stix"


def draw_3d(gen) -> plt.Figure:
    fig = plt.figure(figsize=(8, 8), dpi=120)
    ax = fig.add_subplot(111, projection="3d")

    z0 = 0.0

    # ---- 受灾区域（与原版 plot() 完全一致） ----
    rng_plot = np.random.RandomState(gen.seed)

    for x, y, radius, dl, nc in gen.parent_points:
        angles = 2 * np.pi * rng_plot.rand(nc)
        dists = radius * np.sqrt(rng_plot.rand(nc))
        cx = x + dists * np.cos(angles)
        cy = y + dists * np.sin(angles)
        ax.scatter(cx, cy, z0, alpha=0.6, s=15,
                   label=f"({x},{y}) DL:{dl}")
        ax.scatter(x, y, z0, c="red", marker="x", s=100, zorder=5)

        # 覆盖圆
        theta = np.linspace(0, 2 * np.pi, 80)
        ax.plot(
            x + radius * np.cos(theta),
            y + radius * np.sin(theta),
            z0, color="blue", linestyle="--", linewidth=2,
        )

    # ---- 轨迹路径 ----
    sp = gen.smooth_path
    ax.plot(sp[:, 0], sp[:, 1], z0,
            "g--", linewidth=2, label="运动路径")

    # ---- 起始点 / 终点 ----
    ax.scatter(gen.path[0][0], gen.path[0][1], z0,
               c="green", marker="o", s=150, zorder=6, label="起始点")
    ax.scatter(gen.path[-1][0], gen.path[-1][1], z0,
               c="red", marker="o", s=150, zorder=6, label="终点")

    ax.set_xlim(-100, 1100)
    ax.set_ylim(-100, 1100)
    ax.set_zlim(-10, 100)
    ax.set_zticks([])

    ax.set_xlabel("X坐标 (m)", labelpad=10)
    ax.set_ylabel("Y坐标 (m)", labelpad=10)
    ax.set_zlabel("")
    ax.set_title("搜救轨迹图 — 3D 视角")

    ax.view_init(elev=35, azim=-50)
    ax.grid(True)
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig
