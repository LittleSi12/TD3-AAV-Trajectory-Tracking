"""
搜救场景轨迹生成器。

Author: Little Si
Date:   2026-04
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from configs.env_config import EnvConfig


class TrajectoryGenerator:
    """搜救场景轨迹生成器，生成受灾区域并规划无人机访问路径。"""

    def __init__(self, config=None):
        cfg = config or EnvConfig()
        self.region_width = cfg.REGION_WIDTH
        self.region_height = cfg.REGION_HEIGHT
        self.num_zones = cfg.NUM_DISASTER_ZONES
        self.num_points = cfg.NUM_POINTS
        self.seed = cfg.TRAJECTORY_SEED

        self.parent_points = []
        self.path = []
        self.smooth_path = None

    # ------------------------------------------------------------------
    # 核心生成流程
    # ------------------------------------------------------------------

    def generate(self):
        """一键生成：受灾区域 → 贪心路径 → 平滑轨迹。"""
        rng = np.random.RandomState(self.seed)
        self._generate_disaster_zones(rng)
        self._plan_path()
        self._smooth_trajectory(rng)
        return self.smooth_path

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _is_overlapping(new_point, existing_points):
        """检查新圆是否与已有圆相交。"""
        x1, y1, r1 = new_point
        for x2, y2, r2, _, _ in existing_points:
            if np.hypot(x1 - x2, y1 - y2) < (r1 + r2):
                return True
        return False

    def _generate_disaster_zones(self, rng):
        """随机生成互不重叠的受灾区域（父点）。

        子点数量与覆盖面积成正比，受灾等级作为密度系数：
            num_child = clamp( base × (area / ref_area) × (dl / 5), 4, 150 )
        其中 ref_area 为中等半径 (125) 的面积，base=30 为基准子点数。
        """
        self.parent_points = []
        max_attempts = 10000
        attempts = 0
        ref_area = np.pi * 125.0 ** 2   # 中等半径面积作为基准
        base_count = 30                  # 基准子点数

        # 显示边界：覆盖圆不得超出 [-100, region+100]
        margin = 100

        while len(self.parent_points) < self.num_zones and attempts < max_attempts:
            attempts += 1
            radius = rng.randint(50, 201)
            # 限制中心坐标，使 center ± radius 不超出 [-margin, region+margin]
            x_lo = max(0, -margin + radius)
            x_hi = min(self.region_width, self.region_width + margin - radius)
            y_lo = max(0, -margin + radius)
            y_hi = min(self.region_height, self.region_height + margin - radius)
            x = rng.randint(x_lo, x_hi + 1)
            y = rng.randint(y_lo, y_hi + 1)
            disaster_level = rng.randint(1, 11)
            # 子点数量 = 基准数 × 面积比 × 受灾等级系数，限制在 [4, 150]
            area = np.pi * radius ** 2
            num_child = int(np.clip(
                round(base_count * (area / ref_area) * (disaster_level / 5.0)),
                4, 150,
            ))
            if not self._is_overlapping((x, y, radius), self.parent_points):
                self.parent_points.append((x, y, radius, disaster_level, num_child))

    def _plan_path(self):
        """基于受灾密度与距离的贪心路径规划。"""
        w1, w2 = 1.0, 0.1
        unvisited = set(range(len(self.parent_points)))
        self.path = [(0, 0)]  # 起点

        while unvisited:
            cx, cy = self.path[-1]
            best_score, best_idx = -np.inf, -1
            for idx in unvisited:
                x, y, r, dl, nc = self.parent_points[idx]
                density = (dl * nc) / (np.pi * r ** 2)
                dist = np.hypot(x - cx, y - cy)
                score = w1 * density - w2 * dist
                if score > best_score:
                    best_score, best_idx = score, idx
            if best_idx != -1:
                self.path.append(self.parent_points[best_idx][:2])
                unvisited.remove(best_idx)

    def _smooth_trajectory(self, rng):
        """在路径关键点间插值并用三次样条平滑。"""
        path_arr = np.array(self.path, dtype=np.float64)

        # 在相邻关键点间插入带扰动的中间点
        enhanced = []
        for i in range(len(path_arr) - 1):
            enhanced.append(path_arr[i])
            for t in [0.25, 0.5, 0.75]:
                mid = path_arr[i] + t * (path_arr[i + 1] - path_arr[i])
                mid += rng.normal(0, 10, 2)
                mid = np.clip(mid, 0, max(self.region_width, self.region_height))
                enhanced.append(mid)
        enhanced.append(path_arr[-1])
        enhanced = np.array(enhanced)

        # 三次样条插值
        cs = CubicSpline(range(len(enhanced)), enhanced, bc_type='natural')
        smooth_t = np.linspace(0, len(enhanced) - 1, self.num_points)
        smooth = cs(smooth_t)

        # 轻微移动平均
        win = 3
        if len(smooth) > win:
            kernel = np.ones(win) / win
            smooth[:, 0] = np.convolve(smooth[:, 0], kernel, mode='same')
            smooth[:, 1] = np.convolve(smooth[:, 1], kernel, mode='same')
            smooth[0] = self.path[0]
            smooth[-1] = self.path[-1]

        self.smooth_path = smooth.astype(np.float32)

    # ------------------------------------------------------------------
    # 保存 & 可视化
    # ------------------------------------------------------------------

    def save_csv(self, filepath):
        """将平滑轨迹保存为 CSV。"""
        if self.smooth_path is None:
            raise RuntimeError("请先调用 generate() 生成轨迹")
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        pd.DataFrame(self.smooth_path, columns=['x', 'y']).to_csv(filepath, index=False)
        print(f"轨迹点已保存到：{filepath}")

    def plot(self, save_path=None, show=False):
        """绘制受灾区域与规划路径。"""
        if self.smooth_path is None:
            raise RuntimeError("请先调用 generate() 生成轨迹")

        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(8, 8))

        # 绘制受灾区域
        rng_plot = np.random.RandomState(self.seed)
        for x, y, radius, dl, nc in self.parent_points:
            angles = 2 * np.pi * rng_plot.rand(nc)
            dists = radius * np.sqrt(rng_plot.rand(nc))
            ax.scatter(x + dists * np.cos(angles), y + dists * np.sin(angles),
                       alpha=0.6, label=f'({x},{y}) DL:{dl}')
            ax.scatter(x, y, c='red', marker='x', s=100)
            ax.add_artist(plt.Circle((x, y), radius, color='blue',
                                     fill=False, linestyle='--', linewidth=2))

        # 绘制路径
        ax.plot(self.smooth_path[:, 0], self.smooth_path[:, 1],
                'g--', linewidth=2, label='运动路径')
        ax.scatter(*self.path[0], c='green', marker='o', s=150, label='起始点')
        ax.scatter(*self.path[-1], c='red', marker='o', s=150, label='终点')

        ax.set_xlim(0, self.region_width)
        ax.set_ylim(0, self.region_height)
        ax.set_xlabel('X坐标 (m)')
        ax.set_ylabel('Y坐标 (m)')
        ax.set_title('搜救轨迹图')
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend(fontsize=8)

        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            fig.savefig(save_path)
            print(f"轨迹图已保存到：{save_path}")
        if show:
            plt.show()
        plt.close(fig)


# ------------------------------------------------------------------
# 直接运行此脚本时执行生成
# ------------------------------------------------------------------
if __name__ == "__main__":
    gen = TrajectoryGenerator()
    gen.generate()

    env_dir = os.path.dirname(os.path.abspath(__file__))
    gen.save_csv(os.path.join(env_dir, 'trajectory_points.csv'))
    gen.plot(save_path=os.path.join(env_dir, 'rescue_path_plot.png'), show=True)
