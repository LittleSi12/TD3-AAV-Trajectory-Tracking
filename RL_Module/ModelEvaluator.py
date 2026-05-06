"""
模型评估器，在随机轨迹上评估跟随效果并生成可视化。

Author: Little Si
Date:   2026-04
"""

import numpy as np
import os

from UAV_Environment.Environment import UAV2DEnv
from Utils.Visualization import Visualization
from configs.rl_config import RLConfig
from configs.env_config import EnvConfig


class ModelEvaluator:
    """评估训练好的模型在新轨迹上的跟随效果。"""

    def __init__(self, model):
        self.model = model
        self.follow_dist = EnvConfig.FOLLOW_DIST
        os.makedirs(RLConfig.FIGURE_DIR, exist_ok=True)

    def evaluate(self, episodes=RLConfig.EVAL_EPISODES):
        """在随机生成的新轨迹上评估多轮。"""
        all_rewards = []
        all_distances = []
        all_trajectories = []
        all_references = []
        all_progress = []
        all_end_reasons = []
        all_max_dists = []

        for ep in range(episodes):
            rew, dists, traj, ref, prog, end_reason = self._run_episode()

            all_rewards.append(rew)
            all_distances.append(dists)
            all_trajectories.append(traj)
            all_references.append(ref)
            all_progress.append(prog)
            all_end_reasons.append(end_reason)
            all_max_dists.append(np.max(dists))

            within = np.mean(np.array(dists) <= self.follow_dist) * 100
            print(
                f"Episode {ep+1}: Reward={rew:.1f}, "
                f"Progress={prog:.1%}, "
                f"AvgDist={np.mean(dists):.2f}m, "
                f"MaxDist={np.max(dists):.2f}m, "
                f"Within {self.follow_dist}m: {within:.1f}%, "
                f"End: {end_reason}"
            )

        avg_reward = np.mean(all_rewards)
        avg_dist = np.mean([np.mean(d) for d in all_distances])
        max_dist = np.max([np.max(d) for d in all_distances])
        avg_progress = np.mean(all_progress)
        avg_within = np.mean([
            np.mean(np.array(d) <= self.follow_dist) for d in all_distances
        ]) * 100

        print(f"\n评估结果 ({episodes} 轮):")
        print(f"  平均奖励:       {avg_reward:.2f}")
        print(f"  平均进度:       {avg_progress:.1%}")
        print(f"  平均跟随距离:   {avg_dist:.2f} m")
        print(f"  最大跟随距离:   {max_dist:.2f} m")
        print(f"  {self.follow_dist}m 内比例: {avg_within:.1f}%")

        self._save_results(all_rewards, all_distances,
                           all_trajectories, all_references)
        self._save_eval_csv(all_rewards, all_distances, all_progress, all_end_reasons)

        return {
            "avg_reward": avg_reward,
            "avg_distance": avg_dist,
            "max_distance": max_dist,
            "avg_progress": avg_progress,
            "within_follow_pct": avg_within,
        }

    def _run_episode(self):
        """直接通过原始环境运行一个 episode。"""
        env = UAV2DEnv()
        obs, _ = env.reset()
        ref = env.ref.copy()
        reward_sum = 0.0
        distances = []
        trajectory = []
        terminated = False
        truncated = False
        progress = 0.0
        end_reason = "unknown"

        while not (terminated or truncated):
            trajectory.append([env.x, env.y])

            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            progress = info.get("progress", progress)
            distances.append(info["dist"])
            if info.get("end_reason"):
                end_reason = info["end_reason"]

        env.close()
        return reward_sum, distances, trajectory, ref, progress, end_reason

    def _save_results(self, rewards, distances, trajectories, references):
        fig_dir = RLConfig.FIGURE_DIR

        Visualization.plot_rewards(rewards,
                                   save_path=os.path.join(fig_dir, "rewards.png"))
        Visualization.plot_distance_errors(distances,
                                           save_path=os.path.join(fig_dir, "distance_errors.png"))
        if references:
            Visualization.plot_trajectories(
                reference=references[0],
                uav_trajectories=trajectories,
                save_path=os.path.join(fig_dir, "trajectories.png"),
            )

    def _save_eval_csv(self, rewards, distances, progress_list, end_reasons):
        """保存每轮详细评估数据到 CSV。"""
        import pandas as pd
        rows = []
        for i in range(len(rewards)):
            dists = np.array(distances[i])
            rows.append({
                "Episode": i + 1,
                "Reward": round(rewards[i], 2),
                "Steps": len(dists),
                "Progress": round(progress_list[i], 4),
                "AvgDist": round(np.mean(dists), 2),
                "MaxDist": round(np.max(dists), 2),
                f"Within{self.follow_dist}m%": round(np.mean(dists <= self.follow_dist) * 100, 1),
                "EndReason": end_reasons[i],
            })
        df = pd.DataFrame(rows)
        path = os.path.join(RLConfig.FIGURE_DIR, "eval_results.csv")
        df.to_csv(path, index=False)
        print(f"评估详情已保存到：{path}")

    def close(self):
        pass
