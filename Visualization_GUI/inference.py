"""
推理模块：加载训练好的 TD3 模型，对给定地面轨迹运行无人机跟随仿真。
返回无人机轨迹点和每步距离。

Author: Little Si
Date:   2026-04
"""

import os
import numpy as np
from stable_baselines3 import TD3
from UAV_Environment.Environment import UAV2DEnv
from configs.rl_config import RLConfig


def load_model(path=None):
    """加载训练好的模型。"""
    path = path or RLConfig.SAVE_PATH
    if not os.path.exists(path + ".zip"):
        raise FileNotFoundError(f"模型文件不存在: {path}.zip\n请先运行 Train.py 训练模型")
    model = TD3.load(path)
    return model


def run_follow(model, ground_trajectory):
    """
    用训练好的模型跟随给定的地面轨迹。

    Parameters
    ----------
    model : TD3 model
    ground_trajectory : np.ndarray, shape (N, 2)
        地面搜救队的轨迹点

    Returns
    -------
    dict with keys:
        ground_pts:  地面轨迹点 (与输入相同)
        uav_pts:     无人机轨迹点 list of [x, y]
        distances:   每步的跟随距离
        progress:    最终完成进度
        steps:       总步数
    """
    env = UAV2DEnv()

    # 将地面轨迹注入环境（覆盖随机生成的轨迹）
    env.ref = ground_trajectory.astype(np.float32)
    env.N = len(ground_trajectory)

    # 手动 reset 状态（不重新生成轨迹）
    env.x = float(env.ref[0, 0])
    env.y = float(env.ref[0, 1])
    env.vx = 0.0
    env.vy = 0.0
    if env.N > 1:
        dx = env.ref[1, 0] - env.ref[0, 0]
        dy = env.ref[1, 1] - env.ref[0, 1]
        env.yaw = float(np.arctan2(dy, dx))
    else:
        env.yaw = 0.0
    env.p = 1
    env.step_cnt = 0
    env._prev_action = np.zeros(2, dtype=np.float32)

    obs = env._obs()

    uav_pts = []
    distances = []
    ground_indices = []  # 记录每步对应的地面轨迹点索引
    terminated = False
    truncated = False

    while not (terminated or truncated):
        uav_pts.append([env.x, env.y])
        ground_indices.append(min(env.p, env.N - 1))

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        distances.append(info["dist"])

    env.close()

    return {
        "ground_pts": ground_trajectory,
        "uav_pts": np.array(uav_pts),
        "distances": np.array(distances),
        "ground_indices": ground_indices,
        "progress": info.get("progress", 0),
        "steps": len(uav_pts),
    }
