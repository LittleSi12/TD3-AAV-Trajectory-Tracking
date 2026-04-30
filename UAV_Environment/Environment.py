"""
2D 无人机轨迹跟随环境。

设计原则：
1. 相对坐标观测 → 泛化到任意轨迹
2. 每 episode 随机轨迹 → 避免过拟合
3. 前瞻点 → 提前规划转弯
4. 高斯型跟随奖励 → 平滑梯度引导

Author: Little Si
Date:   2026-04
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from configs.env_config import EnvConfig


class UAV2DEnv(gym.Env):
    """
    观测空间 (7 + 2*NUM_LOOKAHEAD):
        [dx, dy, dist, speed, yaw_err, vx_local, vy_local,
         la_dx_1, la_dy_1, ..., la_dx_N, la_dy_N]

        - dx, dy:      当前目标点相对偏移（局部坐标系）
        - dist:        到目标点的距离（归一化）
        - speed:       当前速度大小（归一化）
        - yaw_err:     航向误差 [-1, 1]
        - vx, vy:      局部坐标系速度
        - la_dx/dy:    前瞻点相对偏移（局部坐标系）

    动作空间 (2):
        [速度 0~MAX_VEL, 偏航率 -MAX_YAW_RATE~MAX_YAW_RATE]
    """

    metadata = {"render_modes": ["human"]}

    # 用于归一化观测的参考尺度
    _DIST_SCALE = 100.0   # 距离归一化参考值
    _VEL_SCALE = 20.0     # 速度归一化参考值

    def __init__(self, config=EnvConfig):
        super().__init__()
        self.cfg = config()
        self.n_lookahead = self.cfg.NUM_LOOKAHEAD

        obs_dim = 7 + 2 * self.n_lookahead  # 7 基础 + 2*N 前瞻
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # 动作空间：速度 [0, MAX_VEL]，偏航率 [-MAX_YAW_RATE, MAX_YAW_RATE]
        self.action_space = spaces.Box(
            low=np.array([0.0, -self.cfg.MAX_YAW_RATE], dtype=np.float32),
            high=np.array([self.cfg.MAX_VEL, self.cfg.MAX_YAW_RATE], dtype=np.float32),
            dtype=np.float32,
        )

        if self.cfg.RANDOMIZE_TRAJECTORY:
            self.ref = self._random_trajectory()
        else:
            self.ref = self._load_trajectory(self.cfg.TRAJ_FILE)
        self.N = len(self.ref)

        # 初始化状态变量（reset 会重新生成轨迹，这里只设默认值）
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.yaw = 0.0
        self.p = 0
        self.step_cnt = 0
        self._prev_action = np.zeros(2, dtype=np.float32)

    # ------------------------------------------------------------------
    # 轨迹池（类级别缓存，所有实例共享）
    # ------------------------------------------------------------------
    _traj_pool = []
    _POOL_SIZE = 100

    @staticmethod
    def _load_trajectory(path):
        df = pd.read_csv(path)
        return df[["x", "y"]].values.astype(np.float32)

    @classmethod
    def _ensure_pool(cls):
        """首次调用时预生成轨迹池。"""
        if len(cls._traj_pool) >= cls._POOL_SIZE:
            return
        from UAV_Environment.TrajectoryGenerator import TrajectoryGenerator
        print(f"预生成 {cls._POOL_SIZE} 条轨迹...")
        for i in range(cls._POOL_SIZE):
            gen = TrajectoryGenerator()
            gen.seed = np.random.randint(0, 2**31)
            gen.generate()
            cls._traj_pool.append(gen.smooth_path)
        print(f"轨迹池就绪 ({cls._POOL_SIZE} 条)")

    def _random_trajectory(self):
        self._ensure_pool()
        idx = np.random.randint(0, len(self._traj_pool))
        return self._traj_pool[idx].copy()

    # ------------------------------------------------------------------
    # Gym 接口
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.cfg.RANDOMIZE_TRAJECTORY:
            self.ref = self._random_trajectory()
            self.N = len(self.ref)

        self.x = float(self.ref[0, 0])
        self.y = float(self.ref[0, 1])
        self.vx = 0.0
        self.vy = 0.0

        if self.N > 1:
            dx = self.ref[1, 0] - self.ref[0, 0]
            dy = self.ref[1, 1] - self.ref[0, 1]
            self.yaw = float(np.arctan2(dy, dx))
        else:
            self.yaw = 0.0

        self.p = 1
        self.step_cnt = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        return self._obs(), {}

    def _obs(self):
        """构建观测向量（局部坐标系 + 归一化）。"""
        cos_y = np.cos(-self.yaw)
        sin_y = np.sin(-self.yaw)

        # 当前目标点
        tp = min(self.p, self.N - 1)
        tx, ty = self.ref[tp]
        dx_w = tx - self.x
        dy_w = ty - self.y

        # 局部坐标系偏移
        dx_l = (dx_w * cos_y - dy_w * sin_y) / self._DIST_SCALE
        dy_l = (dx_w * sin_y + dy_w * cos_y) / self._DIST_SCALE

        # 距离（归一化）
        dist = np.hypot(dx_w, dy_w) / self._DIST_SCALE

        # 速度
        speed = np.hypot(self.vx, self.vy) / self._VEL_SCALE
        vx_l = (self.vx * cos_y - self.vy * sin_y) / self._VEL_SCALE
        vy_l = (self.vx * sin_y + self.vy * cos_y) / self._VEL_SCALE

        # 航向误差 [-1, 1]
        if np.hypot(dx_w, dy_w) > 0.1:
            target_yaw = np.arctan2(dy_w, dx_w)
            yaw_err = self._angle_diff(self.yaw, target_yaw) / np.pi
        else:
            yaw_err = 0.0

        obs = [dx_l, dy_l, dist, speed, yaw_err, vx_l, vy_l]

        # 前瞻点
        for k in range(1, self.n_lookahead + 1):
            idx = min(self.p + k, self.N - 1)
            lx, ly = self.ref[idx]
            ldx_w = lx - self.x
            ldy_w = ly - self.y
            obs.append((ldx_w * cos_y - ldy_w * sin_y) / self._DIST_SCALE)
            obs.append((ldx_w * sin_y + ldy_w * cos_y) / self._DIST_SCALE)

        return np.array(obs, dtype=np.float32)

    def step(self, a):
        a = np.clip(a, self.action_space.low, self.action_space.high)
        v, yr = float(a[0]), float(a[1])
        self.step_cnt += 1

        # 运动学更新
        self.yaw += yr * self.cfg.DT
        self.vx = v * np.cos(self.yaw)
        self.vy = v * np.sin(self.yaw)
        self.x += self.vx * self.cfg.DT
        self.y += self.vy * self.cfg.DT

        # 当前目标点距离
        tp = min(self.p, self.N - 1)
        tx, ty = self.ref[tp]
        dist = np.hypot(self.x - tx, self.y - ty)

        # ================================================================
        # 奖励计算
        # ================================================================
        reward = 0.0

        # 1) 高斯型跟随奖励：sigma=3.0，鼓励靠近目标
        sigma = self.cfg.FOLLOW_DIST
        follow_reward = np.exp(-(dist / sigma) ** 2)
        reward += 2.0 * follow_reward

        # 2) 航向对齐奖励
        dx = tx - self.x
        dy = ty - self.y
        heading_reward = 0.0
        if dist > 0.1:
            target_angle = np.arctan2(dy, dx)
            angle_err = abs(self._angle_diff(self.yaw, target_angle))
            heading_reward = 0.5 * (1.0 - angle_err / np.pi)
        reward += heading_reward

        # 3) 动作平滑惩罚（惩罚动作变化量，而非动作本身）
        action_delta = np.array([v, yr]) - self._prev_action
        reward -= self.cfg.W_ACT * np.sum(action_delta ** 2)
        self._prev_action = np.array([v, yr], dtype=np.float32)

        terminated = False
        truncated = False

        # 到达当前目标点
        if dist < self.cfg.ARRIVE_DIST:
            reward += self.cfg.R_ARRIVE
            self.p += 1
            if self.p >= self.N:
                terminated = True
                reward += self.cfg.R_FINISH

        # 飞出边界
        if abs(self.x) > self.cfg.BOUND or abs(self.y) > self.cfg.BOUND:
            terminated = True
            reward += self.cfg.R_OUT

        # 超时
        if self.step_cnt >= self.cfg.MAX_STEPS:
            truncated = True

        info = {
            "point": self.p,
            "dist": dist,
            "x": self.x,
            "y": self.y,
            "progress": self.p / self.N,
            # 奖励分项，用于日志分析
            "r_follow": 2.0 * follow_reward,
            "r_heading": heading_reward,
            "r_smooth": -self.cfg.W_ACT * float(np.sum(action_delta ** 2)),
            "r_total": reward,
        }
        return self._obs(), reward, terminated, truncated, info

    @staticmethod
    def _angle_diff(a, b):
        """两个角度的最小差值，范围 [-pi, pi]。"""
        d = a - b
        return (d + np.pi) % (2 * np.pi) - np.pi
