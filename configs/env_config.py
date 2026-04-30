"""
环境配置参数。

Author: Little Si
Date:   2026-04
"""


class EnvConfig:
    # 轨迹参数
    TRAJECTORY_TYPE = "custom"
    NUM_POINTS = 500
    TRAJECTORY_RADIUS = 10.0
    SQUARE_SIZE = 20.0
    TRAJECTORY_SEED = 42

    # 搜救轨迹参数
    REGION_WIDTH = 1000
    REGION_HEIGHT = 1000
    NUM_DISASTER_ZONES = 5

    # 环境边界
    X_MIN = -20.0
    X_MAX = 20.0
    Y_MIN = -20.0
    Y_MAX = 20.0

    # 无人机参数
    MAX_VEL = 10.0              # 最大速度 (m/s)，每步最大位移1m
    MAX_YAW_RATE = 1.5          # 最大偏航率 (rad/s)，低速下转弯更灵活
    DT = 0.1                    # 时间步长 (s)

    # 跟随参数
    FOLLOW_DIST = 5.0           # 目标跟随距离 (m)
    ARRIVE_DIST = 5.0           # 到达目标点的阈值 (m)

    # 奖励参数
    R_ARRIVE = 5.0              # 到达目标点的奖励
    R_FINISH = 200.0            # 完成全部轨迹的奖励
    R_OUT = -50.0               # 飞出边界的惩罚
    W_ACT = 0.001               # 动作平滑惩罚权重

    # 其他参数
    MAX_STEPS = 10000
    BOUND = 1200
    NUM_LOOKAHEAD = 3

    # 训练时是否每个 episode 随机生成新轨迹
    RANDOMIZE_TRAJECTORY = True

    # 轨迹文件路径（RANDOMIZE_TRAJECTORY=False 时使用）
    TRAJ_FILE = "./UAV_Environment/trajectory_points.csv"
