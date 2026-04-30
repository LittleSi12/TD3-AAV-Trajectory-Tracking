"""
强化学习（TD3）超参数配置。

Author: Little Si
Date:   2026-04
"""


class RLConfig:
    # 策略网络类型
    POLICY = "MlpPolicy"

    # 学习率
    LR = 1e-4               # 降低学习率，更稳定

    # 折扣因子
    GAMMA = 0.99

    # 目标网络更新系数
    TAU = 0.005

    # 批量大小
    BATCH = 256

    # 经验回放缓冲区大小
    BUFFER = 1000000         # 增大 buffer，因为每个 episode 轨迹不同

    # 学习开始前的步数
    START_STEPS = 5000       # 探索步数（测试时可调小）

    # TD3 特有参数
    POLICY_DELAY = 2         # 策略延迟更新频率
    TARGET_NOISE = 0.2       # 目标策略平滑噪声
    NOISE_CLIP = 0.5         # 噪声裁剪范围

    # 训练参数
    TRAIN_TIMESTEPS = 2000000  # 200 万步（随机轨迹需要更多训练）
    SAVE_INTERVAL = 100000     # 每 10 万步保存一次 checkpoint
    LOG_INTERVAL = 10          # 每 10 个 episode 输出一次控制台日志
    SAVE_PATH = "./results/models/td3_uav"
    LOG_PATH = "./results/logs/"
    FIG_PATH = "./results/figures/"
    FIGURE_DIR = "./results/figures/"
    EVAL_EPISODES = 5
