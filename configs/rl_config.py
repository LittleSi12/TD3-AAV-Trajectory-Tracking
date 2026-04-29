class RLConfig:
    # 策略网络类型
    POLICY = "MlpPolicy"  # 使用多层感知器作为策略网络
    
    # 学习率
    LR = 3e-4  # 策略和价值网络的学习率
    
    # 折扣因子
    GAMMA = 0.99  # 未来奖励的折扣因子
    
    # 目标网络更新系数
    TAU = 0.005  # 软更新目标网络的系数
    
    # 批量大小
    BATCH = 256  # 每次训练的批量大小
    
    # 经验回放缓冲区大小
    BUFFER = 300000  # 经验回放缓冲区的最大容量
    
    # 学习开始前的步数
    START_STEPS = 1000  # 开始学习前随机探索的步数

    # 训练参数
    TRAIN_TIMESTEPS = 500000  # 总训练时间步（从 200k 提升到 500k 以获得更好的收敛）
    SAVE_PATH = "./results/models/td3_uav"  # 模型保存路径
    LOG_PATH = "./results/logs/"  # 日志保存路径
    FIG_PATH = "./results/figures/"  # 图表保存路径
    FIGURE_DIR = "./results/figures/"  # 评估结果图表保存路径
    EVAL_EPISODES = 5  # 评估模型的轮次数量