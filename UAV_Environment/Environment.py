import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from configs.env_config import EnvConfig


class UAV2DEnv(gym.Env):
    """2D 无人机轨迹跟踪环境。

    观测空间 (8,):  [x, y, vx, vy, tx, ty, dx, dy]
    动作空间 (2,):  [速度, 偏航率]

    terminated: 完成所有轨迹点 / 飞出边界
    truncated:  超过最大步数
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config=EnvConfig):
        super().__init__()
        self.cfg = config()
        self.load_trajectory()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-self.cfg.MAX_VEL, -self.cfg.MAX_YAW_RATE]),
            high=np.array([self.cfg.MAX_VEL, self.cfg.MAX_YAW_RATE]),
            dtype=np.float32,
        )
        self.reset()

    def load_trajectory(self):
        df = pd.read_csv(self.cfg.TRAJ_FILE)
        self.ref = df[["x", "y"]].values.astype(np.float32)
        self.N = len(self.ref)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = 0.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.yaw = 0.0
        self.p = 0
        self.step_cnt = 0
        return self._obs(), {}

    def _obs(self):
        tx, ty = self.ref[self.p]
        dx = tx - self.x
        dy = ty - self.y
        return np.array(
            [self.x, self.y, self.vx, self.vy, tx, ty, dx, dy],
            dtype=np.float32,
        )

    def step(self, a):
        a = np.clip(a, self.action_space.low, self.action_space.high)
        v, yr = a
        self.step_cnt += 1

        # 运动学更新
        self.yaw += yr * self.cfg.DT
        self.vx = v * np.cos(self.yaw)
        self.vy = v * np.sin(self.yaw)
        self.x += self.vx * self.cfg.DT
        self.y += self.vy * self.cfg.DT

        tx, ty = self.ref[self.p]
        dist = np.hypot(self.x - tx, self.y - ty)
        rew = -self.cfg.W_DIST * dist - self.cfg.W_ACT * np.sum(np.square(a))

        terminated = False
        truncated = False

        # 到达当前目标点
        if dist < self.cfg.ARRIVE_DIST:
            rew += self.cfg.R_ARRIVE
            self.p += 1
            if self.p >= self.N:
                terminated = True
                rew += 100.0

        # 飞出边界 → terminated
        if abs(self.x) > self.cfg.BOUND or abs(self.y) > self.cfg.BOUND:
            terminated = True
            rew -= 100.0

        # 超过最大步数 → truncated（而非 terminated）
        if self.step_cnt >= self.cfg.MAX_STEPS:
            truncated = True

        info = {"point": self.p, "dist": dist, "x": self.x, "y": self.y}
        return self._obs(), rew, terminated, truncated, info
