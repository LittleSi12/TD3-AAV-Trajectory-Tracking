# UAV TD3 轨迹跟踪项目

## 项目简介

本项目使用深度强化学习算法 TD3 (Twin Delayed DDPG) 实现无人机的轨迹跟踪任务。通过训练智能体学习如何控制无人机，使其能够准确跟踪预设的轨迹。
注：当前项目并未实现，属于个人学习的半成品项目，上传此项目的原因为了解Github项目管理以及更新方法。

## 目录结构

```
UAV_TD3/
├── README.md                  # 项目说明文档
├── requirements.txt           # 依赖包清单
├── configs/                   # 配置文件目录
│   ├── __init__.py
│   ├── env_config.py          # 环境参数（轨迹类型、点数量、边界等）
│   └── rl_config.py           # RL参数（TD3超参、训练步数、奖励权重等）
├── UAV_Environment/           # 环境目录
│   ├── __init__.py
│   ├── Environment.py         # 核心Gym环境（2D无人机+轨迹跟踪逻辑）
│   └── TrajectoryGenerator.py # 轨迹生成模块
├── RL_Module/                 # 强化学习模块
│   ├── __init__.py
│   ├── TD3_Trainer.py         # TD3训练器
│   └── ModelEvaluator.py      # 模型评估器
├── Utils/                     # 工具模块
│   ├── __init__.py
│   ├── DataIO.py              # CSV读写、轨迹加载工具
│   └── Visualization.py       # 可视化工具
├── Train.py                   # 训练入口脚本
├── Test.py                    # 测试入口脚本
├── trajectory_points.csv      # 生成的轨迹文件
└── results/                   # 结果保存目录
    ├── models/                # 训练好的TD3模型
    ├── logs/                  # TensorBoard日志
    └── figures/               # 生成的轨迹图、误差图
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行方法

### 训练模型

```bash
python Train.py
```

### 测试模型

```bash
python Test.py
```

## 核心功能

1. **轨迹生成**：支持多种轨迹类型（圆形、方形、自定义）
2. **环境模拟**：2D无人机运动模型，包含位置、速度、加速度等状态
3. **强化学习训练**：使用SB3的TD3算法进行训练
4. **模型评估**：计算跟踪误差，生成可视化结果
5. **数据可视化**：轨迹图、误差曲线、动画展示

## 配置说明

- **env_config.py**：环境相关参数，如轨迹类型、点数量、边界等
- **rl_config.py**：强化学习相关参数，如学习率、训练步数、奖励权重等

## 结果说明

- **models/**：保存训练好的TD3模型
- **logs/**：保存TensorBoard日志，可通过 `tensorboard --logdir=results/logs` 查看
- **figures/**：保存生成的轨迹图、误差图等

## 技术栈

- Python 3.8+
- Gymnasium
- Stable Baselines3
- NumPy
- Matplotlib
- Pandas

## 注意事项

- 确保安装了所有依赖包
- 首次运行时会自动生成轨迹文件
- 训练过程中会自动保存模型和日志
- 测试时会加载最新的模型进行评估
