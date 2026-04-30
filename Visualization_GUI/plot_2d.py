"""
2D 视角绘图 — 与 TrajectoryGenerator.plot() 样式完全一致。
字体：中文宋体，英文/数字 Times New Roman。
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["SimSun", "Times New Roman"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["mathtext.fontset"] = "stix"


def draw_2d(gen) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)

    # ---- 受灾区域（与原版 plot() 完全一致） ----
    rng_plot = np.random.RandomState(gen.seed)

    for x, y, radius, dl, nc in gen.parent_points:
        angles = 2 * np.pi * rng_plot.rand(nc)
        dists = radius * np.sqrt(rng_plot.rand(nc))
        ax.scatter(
            x + dists * np.cos(angles),
            y + dists * np.sin(angles),
            alpha=0.6,
            label=f"({x},{y}) DL:{dl}",
        )
        ax.scatter(x, y, c="red", marker="x", s=100)
        ax.add_artist(
            plt.Circle(
                (x, y), radius, color="blue",
                fill=False, linestyle="--", linewidth=2,
            )
        )

    # ---- 轨迹路径 ----
    ax.plot(
        gen.smooth_path[:, 0], gen.smooth_path[:, 1],
        "g--", linewidth=2, label="运动路径",
    )

    # 方向箭头（每隔 50 个点）
    sp = gen.smooth_path
    for i in range(1, len(sp), 50):
        ax.arrow(
            sp[i - 1, 0], sp[i - 1, 1],
            sp[i, 0] - sp[i - 1, 0],
            sp[i, 1] - sp[i - 1, 1],
            head_width=10, head_length=15,
            fc="green", ec="green",
        )

    # 起始点 / 终点
    ax.scatter(*gen.path[0], c="green", marker="o", s=150,
               zorder=6, label="起始点")
    ax.scatter(*gen.path[-1], c="red", marker="o", s=150,
               zorder=6, label="终点")

    ax.set_xlim(-100, 1100)
    ax.set_ylim(-100, 1100)
    ax.set_xlabel("X坐标 (m)")
    ax.set_ylabel("Y坐标 (m)")
    ax.set_title("搜救轨迹图")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig
