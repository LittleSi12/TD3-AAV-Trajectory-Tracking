import os
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from configs.rl_config import RLConfig
from UAV_Environment.Environment import UAV2DEnv


class TrainingCallback(BaseCallback):
    """训练过程中记录每个 episode 的指标到 CSV 日志。"""

    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_count = 0
        self._current_episode_reward = 0.0
        self._current_episode_steps = 0

        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, "training_log.csv")
        self.log_file = open(self._log_path, "w")
        self.log_file.write("Episode,Reward,Steps,UAV_X,UAV_Y,Target_Index,Distance\n")

    def _on_step(self) -> bool:
        # 累计奖励
        if "rewards" in self.locals:
            rewards = self.locals["rewards"]
            if isinstance(rewards, (list, np.ndarray)):
                self._current_episode_reward += rewards[0] if len(rewards) > 0 else 0
            else:
                self._current_episode_reward += rewards
        self._current_episode_steps += 1

        # episode 结束
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
                    elif isinstance(infos, dict):
                        info = infos

                uav_x = info.get("x", 0)
                uav_y = info.get("y", 0)
                target_idx = info.get("point", 0)
                distance = info.get("dist", 0)

                print(
                    f"Episode {self.episode_count}: "
                    f"Reward: {ep_rew:.2f}, Steps: {ep_step}, "
                    f"UAV: ({uav_x:.2f}, {uav_y:.2f}), "
                    f"Target: {target_idx}, Dist: {distance:.2f}"
                )
                self.log_file.write(
                    f"{self.episode_count},{ep_rew:.2f},{ep_step},"
                    f"{uav_x:.2f},{uav_y:.2f},{target_idx},{distance:.2f}\n"
                )
                self.log_file.flush()

                self._current_episode_reward = 0.0
                self._current_episode_steps = 0
        return True

    def on_training_end(self):
        self.log_file.close()
        print(f"Training log saved to {self._log_path}")


class TD3Trainer:
    """封装 TD3 训练流程，包含 VecNormalize 观测归一化。"""

    VECNORM_FILENAME = "vec_normalize.pkl"

    def __init__(self):
        self.cfg = RLConfig()
        os.makedirs(os.path.dirname(self.cfg.SAVE_PATH), exist_ok=True)
        os.makedirs(self.cfg.LOG_PATH, exist_ok=True)

        # 使用 DummyVecEnv + VecNormalize 进行观测归一化
        self.venv = DummyVecEnv([lambda: UAV2DEnv()])
        self.venv = VecNormalize(self.venv, norm_obs=True, norm_reward=False)

        self.model = TD3(
            self.cfg.POLICY,
            self.venv,
            learning_rate=self.cfg.LR,
            gamma=self.cfg.GAMMA,
            tau=self.cfg.TAU,
            batch_size=self.cfg.BATCH,
            buffer_size=self.cfg.BUFFER,
            learning_starts=self.cfg.START_STEPS,
            verbose=1,
            tensorboard_log=self.cfg.LOG_PATH,
        )

    @property
    def _vecnorm_path(self):
        return os.path.join(os.path.dirname(self.cfg.SAVE_PATH), self.VECNORM_FILENAME)

    def train(self):
        print("Start training...")
        print(f"Training for {self.cfg.TRAIN_TIMESTEPS} timesteps")
        print(f"Model will be saved to: {self.cfg.SAVE_PATH}")
        try:
            callback = TrainingCallback(log_dir=self.cfg.LOG_PATH)
            self.model.learn(total_timesteps=self.cfg.TRAIN_TIMESTEPS, callback=callback)
            print("Learning completed successfully")

            # 保存模型和归一化统计
            self.model.save(self.cfg.SAVE_PATH)
            self.venv.save(self._vecnorm_path)
            print(f"Model saved to: {self.cfg.SAVE_PATH}")
            print(f"VecNormalize saved to: {self._vecnorm_path}")

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
        """加载已训练的模型和归一化统计。"""
        path = path or self.cfg.SAVE_PATH
        vecnorm_path = os.path.join(os.path.dirname(path), self.VECNORM_FILENAME)

        # 重建 VecNormalize（加载统计量，评估时不更新）
        if os.path.exists(vecnorm_path):
            base_venv = DummyVecEnv([lambda: UAV2DEnv()])
            self.venv = VecNormalize.load(vecnorm_path, base_venv)
            self.venv.training = False
            self.venv.norm_reward = False
            print(f"VecNormalize loaded from: {vecnorm_path}")
        else:
            print(f"Warning: VecNormalize file not found at {vecnorm_path}, using unnormalized env")

        self.model = TD3.load(path, env=self.venv)
        print(f"Model loaded from: {path}")

    def get_model(self):
        return self.model

    def get_env(self):
        """返回底层的 VecNormalize 环境。"""
        return self.venv
