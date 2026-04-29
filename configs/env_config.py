# 环境配置参数

class EnvConfig:
    # 轨迹参数
    TRAJECTORY_TYPE = "custom"  # 轨迹类型：circle, square, custom
    NUM_POINTS = 500  # 轨迹点数量（插值后的密集点数）
    TRAJECTORY_RADIUS = 10.0  # 圆形轨迹半径
    SQUARE_SIZE = 20.0  # 方形轨迹边长
    TRAJECTORY_SEED = 42  # 随机种子，保证轨迹可复现

    # 搜救轨迹参数（仅 custom 类型使用）
    REGION_WIDTH = 1000  # 搜救区域宽度
    REGION_HEIGHT = 1000  # 搜救区域高度
    NUM_DISASTER_ZONES = 5  # 受灾区域数量

    # 环境边界
    X_MIN = -20.0
    X_MAX = 20.0
    Y_MIN = -20.0
    Y_MAX = 20.0
    
    # 无人机参数
    MAX_VEL = 15.0  # 最大速度
    MAX_YAW_RATE = 0.8  # 最大偏航率
    DT = 0.1  # 时间步长
    
    # 奖励参数
    W_DIST = 1.0  # 距离奖励权重
    W_ACT = 0.005  # 动作奖励权重
    R_ARRIVE = 12.0  # 到达目标点的奖励
    
    # 其他参数
    MAX_STEPS = 2000  # 每个episode的最大步数
    ARRIVE_DIST = 8.0  # 到达目标点的阈值
    BOUND = 1100  # 边界
    
    # 轨迹文件路径
    TRAJ_FILE = "./UAV_Environment/trajectory_points.csv"
