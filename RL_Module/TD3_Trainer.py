"""
TD3 训练器，封装训练、断点续训与回调日志。

Author: Little Si
Date:   2026-04
"""

import os
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from configs.rl_config import RLConfig
from UAV_Environment.Environment import UAV2DEnv


class TrainingCallback(BaseCallback):
    """训练回调：实时日志（含奖励分项）+ 定期保存 checkpoint。"""

    def __init__(self, log_dir, save_path, save_interval, log_interval=10, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_interval = save_interval
        self.log_interval = log_interval

        self.episode_rewards = []
        self.episode_steps = []
        self.episode_progress = []
        self.episode_count = 0
        self._current_episode_reward = 0.0
        self._current_episode_steps = 0
        self._ep_r_follow = 0.0
        self._ep_r_heading = 0.0
        self._ep_r_smooth = 0.0
        self._ep_dist_sum = 0.0
        self._ep_max_dist = 0.0
        self._ep_within_count = 0
        self._ep_speed_sum = 0.0
        self._last_save_step = 0

        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, "training_log.csv")
        self.log_file = open(self._log_path, "w")
        self.log_file.write(
            "Episode,Reward,Steps,Progress,AvgDist,MaxDist,Within5m%,AvgSpeed,"
            "R_Follow,R_Heading,R_Smooth,EndReason,TotalSteps\n"
        )

    def _on_step(self) -> bool:
        if "rewards" in self.locals:
            rewards = self.locals["rewards"]
            if isinstance(rewards, (list, np.ndarray)):
                self._current_episode_reward += rewards[0] if len(rewards) > 0 else 0
            else:
                self._current_episode_reward += rewards
        self._current_episode_steps += 1

        # 累计奖励分项
        if "infos" in self.locals:
            infos = self.locals["infos"]
            if isinstance(infos, list) and infos:
                info = infos[0]
                self._ep_r_follow += info.get("r_follow", 0)
                self._ep_r_heading += info.get("r_heading", 0)
                self._ep_r_smooth += info.get("r_smooth", 0)
                d = info.get("dist", 0)
                self._ep_dist_sum += d
                self._ep_max_dist = max(self._ep_max_dist, d)
                self._ep_speed_sum += info.get("speed", 0)
                if d <= 5.0:
                    self._ep_within_count += 1

        # 定期保存 checkpoint
        if self.num_timesteps - self._last_save_step >= self.save_interval:
            self._last_save_step = self.num_timesteps
            ckpt_path = f"{self.save_path}_ckpt_{self.num_timesteps}"
            self.model.save(ckpt_path)
            print(f"\n💾 Checkpoint saved at step {self.num_timesteps}: {ckpt_path}")

        # episode 结束时记录
        if "dones" in self.locals:
            dones = self.locals["dones"]
            done = np.any(dones) if isinstance(dones, (list, np.ndarray)) else dones

            if done:
                self.episode_count += 1
                ep_rew = self._current_episode_reward
                ep_step = self._current_episode_steps
                self.episode_rewards.append(ep_rew)
                self.episode_steps.append(ep_step)

                info = {}
                if "infos" in self.locals:
                    infos = self.locals["infos"]
                    if isinstance(infos, list) and infos:
                        info = infos[0]

                progress = info.get("progress", 0)
                self.episode_progress.append(progress)

                avg_dist = self._ep_dist_sum / max(ep_step, 1)
                within_pct = self._ep_within_count / max(ep_step, 1) * 100
                avg_speed = self._ep_speed_sum / max(ep_step, 1)
                end_reason = info.get("end_reason", "unknown")

                self.log_file.write(
                    f"{self.episode_count},{ep_rew:.2f},{ep_step},"
                    f"{progress:.4f},{avg_dist:.2f},{self._ep_max_dist:.2f},{within_pct:.1f},{avg_speed:.2f},"
                    f"{self._ep_r_follow:.2f},{self._ep_r_heading:.2f},"
                    f"{self._ep_r_smooth:.4f},{end_reason},"
                    f"{self.num_timesteps}\n"
                )
                self.log_file.flush()

                if self.episode_count % self.log_interval == 0 or self.episode_count <= 3:
                    recent = min(self.log_interval, len(self.episode_rewards))
                    avg_rew = np.mean(self.episode_rewards[-recent:])
                    avg_prog = np.mean(self.episode_progress[-recent:])
                    print(
                        f"Ep {self.episode_count:>5d} | "
                        f"Steps: {self.num_timesteps:>8d} | "
                        f"Reward: {avg_rew:>8.1f} | "
                        f"Progress: {avg_prog:>5.1%} | "
                        f"AvgDist: {avg_dist:>5.1f}m | "
                        f"MaxDist: {self._ep_max_dist:>5.1f}m | "
                        f"In5m: {within_pct:>5.1f}% | "
                        f"End: {end_reason}"
                    )

                self._current_episode_reward = 0.0
                self._current_episode_steps = 0
                self._ep_r_follow = 0.0
                self._ep_r_heading = 0.0
                self._ep_r_smooth = 0.0
                self._ep_dist_sum = 0.0
                self._ep_max_dist = 0.0
                self._ep_within_count = 0
                self._ep_speed_sum = 0.0
        return True

    def on_training_end(self):
        self.log_file.close()
        print(f"Training log saved to {self._log_path}")


class TD3Trainer:
    """封装 TD3 训练流程。不使用 VecNormalize，环境内置归一化。"""

    N_ENVS = 4  # 并行环境数

    def __init__(self):
        self.cfg = RLConfig()
        os.makedirs(os.path.dirname(self.cfg.SAVE_PATH), exist_ok=True)
        os.makedirs(self.cfg.LOG_PATH, exist_ok=True)

        # 预生成轨迹池（避免每次 reset 重新计算）
        UAV2DEnv._ensure_pool()

        self.venv = DummyVecEnv([lambda: UAV2DEnv() for _ in range(self.N_ENVS)])

        n_actions = self.venv.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),
        )

        self.model = TD3(
            self.cfg.POLICY,
            self.venv,
            learning_rate=self.cfg.LR,
            gamma=self.cfg.GAMMA,
            tau=self.cfg.TAU,
            batch_size=self.cfg.BATCH,
            buffer_size=self.cfg.BUFFER,
            learning_starts=self.cfg.START_STEPS,
            policy_delay=self.cfg.POLICY_DELAY,
            target_policy_noise=self.cfg.TARGET_NOISE,
            target_noise_clip=self.cfg.NOISE_CLIP,
            action_noise=action_noise,
            verbose=1,
            tensorboard_log=None,
        )

    def train(self):
        print("Start training...")
        print(f"Training for {self.cfg.TRAIN_TIMESTEPS} timesteps")
        print(f"Model will be saved to: {self.cfg.SAVE_PATH}")
        self._save_config_snapshot()

        try:
            callback = TrainingCallback(
                log_dir=self.cfg.LOG_PATH,
                save_path=self.cfg.SAVE_PATH,
                save_interval=self.cfg.SAVE_INTERVAL,
                log_interval=self.cfg.LOG_INTERVAL,
            )
            self.model.learn(total_timesteps=self.cfg.TRAIN_TIMESTEPS, callback=callback)
            print("Learning completed successfully")

            self.model.save(self.cfg.SAVE_PATH)
            print(f"Model saved to: {self.cfg.SAVE_PATH}")

            if callback.episode_rewards:
                print("\nTraining Summary:")
                print(f"  Total Episodes: {len(callback.episode_rewards)}")
                print(f"  Average Reward: {np.mean(callback.episode_rewards):.2f}")
                print(f"  Average Steps:  {np.mean(callback.episode_steps):.2f}")
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()

    def resume_train(self, path=None, additional_timesteps=None):
        """从 checkpoint 恢复继续训练。"""
        path = path or self._find_latest_checkpoint()
        if path is None:
            print("没有找到 checkpoint，将从头开始训练")
            self.train()
            return

        steps = additional_timesteps or self.cfg.TRAIN_TIMESTEPS

        UAV2DEnv._ensure_pool()
        self.venv = DummyVecEnv([lambda: UAV2DEnv() for _ in range(self.N_ENVS)])
        self.model = TD3.load(path, env=self.venv)
        prev_steps = self.model.num_timesteps
        target_steps = prev_steps + steps
        print(f"Model loaded from: {path} (previously trained {prev_steps} steps)")
        print(f"Resuming for {steps} additional steps (target: {target_steps})...")

        try:
            callback = TrainingCallback(
                log_dir=self.cfg.LOG_PATH,
                save_path=self.cfg.SAVE_PATH,
                save_interval=self.cfg.SAVE_INTERVAL,
                log_interval=self.cfg.LOG_INTERVAL,
            )
            self.model.learn(
                total_timesteps=target_steps,
                callback=callback,
                reset_num_timesteps=False,
            )
            print("Resume training completed")
            self.model.save(self.cfg.SAVE_PATH)
            print(f"Model saved to: {self.cfg.SAVE_PATH}")

            if callback.episode_rewards:
                print("\nTraining Summary:")
                print(f"  Total Episodes: {len(callback.episode_rewards)}")
                print(f"  Average Reward: {np.mean(callback.episode_rewards):.2f}")
                print(f"  Average Steps:  {np.mean(callback.episode_steps):.2f}")
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()

    def load(self, path=None):
        path = path or self.cfg.SAVE_PATH
        self.model = TD3.load(path, env=self.venv)
        print(f"Model loaded from: {path}")

    def _find_latest_checkpoint(self):
        model_dir = os.path.dirname(self.cfg.SAVE_PATH)
        if not os.path.exists(model_dir):
            return None
        ckpts = []
        for f in os.listdir(model_dir):
            if f.startswith("td3_uav_ckpt_") and f.endswith(".zip"):
                try:
                    step = int(f.replace("td3_uav_ckpt_", "").replace(".zip", ""))
                    ckpts.append((step, os.path.join(model_dir, f.replace(".zip", ""))))
                except ValueError:
                    continue
        final = self.cfg.SAVE_PATH
        if os.path.exists(final + ".zip"):
            ckpts.append((float("inf"), final))
        if not ckpts:
            return None
        ckpts.sort(key=lambda x: x[0], reverse=True)
        best = ckpts[0][1]
        print(f"Found latest checkpoint: {best}")
        return best

    def _save_config_snapshot(self):
        import json
        from configs.env_config import EnvConfig
        def _class_attrs(cls):
            return {k: v for k, v in vars(cls).items()
                    if not k.startswith("_") and not callable(v)}
        snapshot = {
            "rl_config": _class_attrs(RLConfig),
            "env_config": _class_attrs(EnvConfig),
            "obs_space": str(self.venv.observation_space),
            "act_space": str(self.venv.action_space),
        }
        path = os.path.join(self.cfg.LOG_PATH, "config_snapshot.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        print(f"Config snapshot saved to: {path}")

    def get_model(self):
        return self.model

    def get_env(self):
        return self.venv
