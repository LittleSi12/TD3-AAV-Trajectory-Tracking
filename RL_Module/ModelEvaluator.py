import numpy as np
import os

from UAV_Environment.Environment import UAV2DEnv
from Utils.Visualization import Visualization
from Utils.DataIO import DataIO
from configs.rl_config import RLConfig


class ModelEvaluator:
    """使用训练好的模型评估无人机轨迹跟踪效果。"""

    def __init__(self, model, vec_env=None):
        """
        Parameters
        ----------
        model : stable_baselines3 model
        vec_env : VecNormalize env (可选)，用于保持与训练时一致的观测归一化。
                  如果不传则直接使用原始环境。
        """
        self.model = model
        self.vec_env = vec_env
        # 用于获取参考轨迹等原始数据
        self.raw_env = UAV2DEnv()

        os.makedirs(RLConfig.FIGURE_DIR, exist_ok=True)

    def evaluate(self, episodes=RLConfig.EVAL_EPISODES):
        """运行多轮评估，返回统计指标并保存可视化结果。"""
        total_rewards = []
        total_distances = []
        trajectories = []

        for episode in range(episodes):
            # 根据是否有 VecNormalize 选择不同的交互方式
            if self.vec_env is not None:
                ep_rew, ep_dist, ep_traj = self._run_episode_vec()
            else:
                ep_rew, ep_dist, ep_traj = self._run_episode_raw()

            total_rewards.append(ep_rew)
            total_distances.append(ep_dist)
            trajectories.append(ep_traj)
            print(f"Episode {episode + 1}: Reward = {ep_rew:.2f}")

        avg_reward = np.mean(total_rewards)
        avg_distance = np.mean([np.mean(d) for d in total_distances])
        max_distance = np.max([np.max(d) for d in total_distances])

        print(f"\n评估结果:")
        print(f"  平均奖励:     {avg_reward:.2f}")
        print(f"  平均距离误差: {avg_distance:.2f}")
        print(f"  最大距离误差: {max_distance:.2f}")

        self._save_results(total_rewards, total_distances, trajectories)

        return {
            "avg_reward": avg_reward,
            "avg_distance": avg_distance,
            "max_distance": max_distance,
        }

    # ------------------------------------------------------------------
    # 内部运行方法
    # ------------------------------------------------------------------

    def _run_episode_vec(self):
        """通过 VecNormalize 环境运行一个 episode。"""
        obs = self.vec_env.reset()
        reward_sum = 0.0
        distances = []
        trajectory = []
        done = False

        while not done:
            # VecEnv 的 obs 是 (1, obs_dim)
            raw_obs = self.vec_env.get_original_obs()[0]
            x, y = raw_obs[0], raw_obs[1]
            tx, ty = raw_obs[4], raw_obs[5]
            trajectory.append([x, y])
            distances.append(np.hypot(x - tx, y - ty))

            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, dones, infos = self.vec_env.step(action)
            reward_sum += reward[0]
            done = dones[0]

        return reward_sum, distances, trajectory

    def _run_episode_raw(self):
        """直接通过原始环境运行一个 episode（无归一化）。"""
        obs, _ = self.raw_env.reset()
        reward_sum = 0.0
        distances = []
        trajectory = []
        terminated = False
        truncated = False

        while not (terminated or truncated):
            x, y = obs[0], obs[1]
            tx, ty = obs[4], obs[5]
            trajectory.append([x, y])
            distances.append(np.hypot(x - tx, y - ty))

            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.raw_env.step(action)
            reward_sum += reward

        return reward_sum, distances, trajectory

    # ------------------------------------------------------------------
    # 可视化保存（委托给 Utils.Visualization）
    # ------------------------------------------------------------------

    def _save_results(self, total_rewards, total_distances, trajectories):
        """保存奖励图、距离误差图、轨迹图。"""
        fig_dir = RLConfig.FIGURE_DIR

        # 奖励图
        Visualization.plot_rewards(
            total_rewards,
            save_path=os.path.join(fig_dir, "rewards.png"),
        )

        # 距离误差图
        Visualization.plot_distance_errors(
            total_distances,
            save_path=os.path.join(fig_dir, "distance_errors.png"),
        )

        # 轨迹图
        Visualization.plot_trajectories(
            reference=self.raw_env.ref,
            uav_trajectories=trajectories,
            save_path=os.path.join(fig_dir, "trajectories.png"),
        )

        # 保存评估数据到 CSV
        DataIO.save_results(
            {"avg_reward": [np.mean(total_rewards)],
             "avg_distance": [np.mean([np.mean(d) for d in total_distances])],
             "max_distance": [np.max([np.max(d) for d in total_distances])]},
            filename=os.path.join(fig_dir, "eval_results.csv"),
        )

    def close(self):
        self.raw_env.close()
        if self.vec_env is not None:
            self.vec_env.close()
