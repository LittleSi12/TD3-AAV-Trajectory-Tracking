# UAV TD3 轨迹跟踪项目

基于深度强化学习算法 **TD3 (Twin Delayed DDPG)** 的无人机轨迹跟踪系统。  
项目模拟搜救场景：地面搜救队沿规划路径行进，无人机智能体学习实时跟随其轨迹。

---

## 目录

- [项目简介](#项目简介)
- [目录结构](#目录结构)
- [环境要求](#环境要求)
- [安装与运行](#安装与运行)
- [核心设计](#核心设计)
  - [搜救轨迹生成](#搜救轨迹生成)
  - [Gymnasium 环境](#gymnasium-环境)
  - [TD3 训练流程](#td3-训练流程)
  - [奖励函数](#奖励函数)
- [可视化 GUI](#可视化-gui)
- [配置说明](#配置说明)
- [结果目录](#结果目录)
- [技术栈](#技术栈)
- [已知问题与改进方向](#已知问题与改进方向)
- [许可证](#许可证)

---

## 项目简介

在搜救场景中，无人机需要跟随地面搜救队的行进路线，保持一定距离进行空中侦察。  
本项目将该问题建模为强化学习任务：

- **智能体**：2D 无人机（控制速度与偏航率）
- **环境**：随机生成的搜救轨迹 + 简化运动学模型
- **目标**：学习一个策略，使无人机能准确跟随任意未见过的轨迹

训练采用 [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) 的 TD3 实现，支持从头训练、断点续训和多轮评估。  
项目还提供了一个基于 CustomTkinter 的交互式 GUI，可以直观地观察无人机跟随效果。

---

## 目录结构

```
UAV_TD3/
├── README.md                       # 项目说明文档
├── requirements.txt                # Python 依赖清单
├── .gitignore                      # Git 忽略规则
│
├── Train.py                        # 训练入口脚本
├── Test.py                         # 测试/评估入口脚本
├── RunGUI.py                       # 可视化 GUI 启动脚本
│
├── configs/                        # 配置文件
│   ├── env_config.py               #   环境参数（轨迹、边界、奖励权重等）
│   └── rl_config.py                #   RL 超参数（学习率、训练步数等）
│
├── UAV_Environment/                # Gymnasium 仿真环境
│   ├── Environment.py              #   核心环境类 UAV2DEnv
│   ├── TrajectoryGenerator.py      #   搜救轨迹生成器
│   └── trajectory_points.csv       #   预生成的轨迹文件（可选）
│
├── RL_Module/                      # 强化学习模块
│   ├── TD3_Trainer.py              #   TD3 训练器（含回调、断点续训）
│   └── ModelEvaluator.py           #   模型评估器
│
├── Utils/                          # 工具模块
│   ├── DataIO.py                   #   CSV 读写工具
│   └── Visualization.py            #   Matplotlib 可视化工具
│
├── Visualization_GUI/              # 交互式可视化 GUI
│   ├── app.py                      #   主窗口（三阶段交互）
│   ├── inference.py                #   推理模块（加载模型 + 仿真）
│   ├── plot_2d.py                  #   2D 俯视图绘制
│   └── plot_3d.py                  #   3D 视角绘制
│
└── results/                        # 输出目录（训练产物）
    ├── models/                     #   模型文件 (.zip) 与 checkpoint
    ├── logs/                       #   训练日志 (CSV) 与配置快照
    └── figures/                    #   评估图表（轨迹图、误差图等）
```

---

## 环境要求

- Python 3.8+（开发环境为 Python 3.13）
- 操作系统：Windows / macOS / Linux
- GUI 功能需要图形界面环境（CustomTkinter + Matplotlib TkAgg 后端）

---

## 安装与运行

### 1. 克隆仓库

```bash
git clone https://github.com/<your-username>/UAV_TD3.git
cd UAV_TD3
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 训练模型

```bash
# 从头训练（默认 200 万步）
python Train.py

# 指定训练步数
python Train.py --steps 500000

# 从最新 checkpoint 断点续训
python Train.py --resume

# 从指定 checkpoint 续训
python Train.py --resume --ckpt results/models/td3_uav_ckpt_300000

# 续训并指定额外步数
python Train.py --resume --steps 500000
```

训练过程中会：
- 每 10 万步自动保存 checkpoint
- 每 10 个 episode 输出控制台日志（含奖励分项）
- 训练结束后保存最终模型到 `results/models/td3_uav.zip`
- 训练日志保存到 `results/logs/training_log.csv`

### 5. 测试/评估模型

```bash
python Test.py
```

默认在 5 条随机新轨迹上评估，输出平均跟随距离、5m 内比例等指标，并生成评估图表到 `results/figures/`。

### 6. 启动可视化 GUI

```bash
python RunGUI.py
```

---

## 核心设计

### 搜救轨迹生成

`TrajectoryGenerator` 模拟搜救场景中的路径规划：

1. **灾害区域生成**：在 1000×1000m 区域内随机放置 5 个互不重叠的圆形灾害区域
   - 半径 50~200m，灾害等级 1~10
   - 子点数量与面积和灾害等级成正比
2. **贪心路径规划**：综合考虑灾害密度和距离，依次访问各灾害区域
3. **轨迹平滑**：在关键点间插入扰动中间点，经三次样条插值 + 移动平均生成 500 个平滑轨迹点

训练时每个 episode 随机生成新轨迹（`RANDOMIZE_TRAJECTORY=True`），避免策略过拟合到单一路径。

### Gymnasium 环境

`UAV2DEnv` 是一个标准的 Gymnasium 环境：

**观测空间** — 13 维连续向量（局部坐标系 + 归一化）：

| 维度 | 含义 | 归一化 |
|------|------|--------|
| 0-1 | 当前目标点相对偏移 (dx, dy) | ÷ 100 |
| 2 | 到目标点的距离 | ÷ 100 |
| 3 | 当前速度大小 | ÷ 20 |
| 4 | 航向误差 | ÷ π → [-1, 1] |
| 5-6 | 局部坐标系速度 (vx, vy) | ÷ 20 |
| 7-12 | 3 个前瞻点的相对偏移 | ÷ 100 |

所有空间偏移量都转换到以无人机航向为基准的**局部坐标系**，使策略不依赖绝对位置，提升泛化能力。

**动作空间** — 2 维连续：

| 维度 | 含义 | 范围 |
|------|------|------|
| 0 | 速度 | [0, 20] m/s |
| 1 | 偏航率 | [-1.5, 1.5] rad/s |

**运动学模型**：

```
yaw  += yaw_rate × dt
vx    = speed × cos(yaw)
vy    = speed × sin(yaw)
x    += vx × dt
y    += vy × dt
```

**终止条件**：
- 完成全部轨迹点 → 成功终止，获得完成奖励
- 飞出边界（±1200m）→ 失败终止，受到惩罚
- 超过 6000 步 → 截断

### TD3 训练流程

```
┌─────────────────────────────────────────────────┐
│  TD3Trainer                                     │
│                                                 │
│  1. 创建 DummyVecEnv(UAV2DEnv)                  │
│  2. 配置 TD3 (MlpPolicy + NormalActionNoise)    │
│  3. 保存配置快照 → config_snapshot.json          │
│  4. model.learn(2M steps, callback)             │
│     │                                           │
│     ├── TrainingCallback                        │
│     │   ├── 每步记录奖励分项                      │
│     │   ├── 每 10 万步保存 checkpoint             │
│     │   └── 每 10 episode 输出控制台日志           │
│     │                                           │
│  5. 保存最终模型 → td3_uav.zip                   │
└─────────────────────────────────────────────────┘
```

关键超参数：

| 参数 | 值 | 说明 |
|------|----|------|
| 策略网络 | MlpPolicy | 全连接网络 |
| 学习率 | 1e-4 | Actor 和 Critic 共用 |
| 折扣因子 γ | 0.99 | 长期回报权重 |
| 软更新系数 τ | 0.005 | 目标网络更新速率 |
| 批量大小 | 256 | 每次梯度更新的样本数 |
| 经验回放 | 100 万 | 大 buffer 适应随机轨迹 |
| 探索噪声 | N(0, 0.1) | 正态动作噪声 |
| 策略延迟 | 2 | 每 2 次 Critic 更新后更新 1 次 Actor |
| 目标噪声 | 0.2 (clip 0.5) | 目标策略平滑正则化 |

### 奖励函数

奖励由四部分组成，引导无人机学习精确跟随行为：

```
R_total = R_follow + R_heading + R_smooth + R_event
```

1. **高斯型跟随奖励** `R_follow`：
   - `2.0 × exp(-(dist/3.0)²)` — 距离越近奖励越高，σ=3.0 提供平滑梯度
   - 超过 5m 时额外惩罚：`-0.5 × min(dist - 5, 20)`

2. **航向对齐奖励** `R_heading`：
   - `0.5 × (1 - |angle_err| / π)` — 鼓励无人机朝向目标点

3. **动作平滑惩罚** `R_smooth`：
   - `-0.001 × Σ(Δaction²)` — 惩罚动作突变，使飞行更平稳

4. **事件奖励** `R_event`：
   - 到达目标点：+5.0
   - 完成全部轨迹：+200.0
   - 飞出边界：-50.0

---

## 可视化 GUI

基于 CustomTkinter + Matplotlib 的交互式仿真界面，分三个阶段操作：

### 阶段一：环境设置

点击「① 环境设置」按钮，随机生成灾害区域并在 2D/3D 视图中展示。  
每个灾害区域显示类型（火灾/洪水/滑坡等）、位置、半径和灾害等级。

### 阶段二：轨迹规划

点击「② 轨迹规划」按钮，生成并展示搜救路径（带闪烁动画）。  
日志面板显示轨迹点数、路径长度等信息。

### 阶段三：任务执行

点击「③ 任务开始」按钮，加载训练好的 TD3 模型，逐帧动画展示无人机跟随过程：

- **2D/3D 标签页**：实时绘制地面轨迹（绿色虚线）和无人机轨迹（橙色实线），3D 视角下无人机固定在 z=20m 高度飞行
- **指标卡**：任务进度、当前距离、平均距离、5m 内比例
- **运行日志**：富文本彩色日志，分阶段输出环境信息、轨迹参数和实时跟随数据
- 无人机轨迹延后 30 步显示，直观呈现"跟随"效果
- 任务执行中可点击「⏹ 结束任务」提前停止

> 注意：任务执行需要先通过 `Train.py` 训练并保存模型。

---

## 配置说明

所有配置集中在 `configs/` 目录下，以类属性形式定义，修改后无需额外操作即可生效。

### 环境配置 `configs/env_config.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `REGION_WIDTH/HEIGHT` | 1000 | 搜救区域尺寸 (m) |
| `NUM_DISASTER_ZONES` | 5 | 灾害区域数量 |
| `NUM_POINTS` | 500 | 轨迹插值点数 |
| `MAX_VEL` | 20.0 | 无人机最大速度 (m/s) |
| `MAX_YAW_RATE` | 1.5 | 最大偏航率 (rad/s) |
| `DT` | 0.1 | 仿真时间步长 (s) |
| `FOLLOW_DIST` | 5.0 | 目标跟随距离 (m) |
| `ARRIVE_DIST` | 5.0 | 到达目标点阈值 (m) |
| `MAX_STEPS` | 6000 | 单 episode 最大步数 |
| `BOUND` | 1200 | 飞行边界 (m) |
| `NUM_LOOKAHEAD` | 3 | 前瞻点数量 |
| `RANDOMIZE_TRAJECTORY` | True | 每 episode 是否随机生成新轨迹 |
| `R_ARRIVE` | 5.0 | 到达目标点奖励 |
| `R_FINISH` | 200.0 | 完成全部轨迹奖励 |
| `R_OUT` | -50.0 | 飞出边界惩罚 |
| `W_ACT` | 0.001 | 动作平滑惩罚权重 |

### RL 配置 `configs/rl_config.py`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `POLICY` | MlpPolicy | 策略网络类型 |
| `LR` | 1e-4 | 学习率 |
| `GAMMA` | 0.99 | 折扣因子 |
| `TAU` | 0.005 | 目标网络软更新系数 |
| `BATCH` | 256 | 批量大小 |
| `BUFFER` | 1000000 | 经验回放缓冲区大小 |
| `START_STEPS` | 5000 | 学习开始前的随机探索步数 |
| `POLICY_DELAY` | 2 | 策略延迟更新频率 |
| `TARGET_NOISE` | 0.2 | 目标策略平滑噪声 |
| `NOISE_CLIP` | 0.5 | 噪声裁剪范围 |
| `TRAIN_TIMESTEPS` | 2000000 | 总训练步数 |
| `SAVE_INTERVAL` | 100000 | Checkpoint 保存间隔 |
| `EVAL_EPISODES` | 5 | 评估轮数 |

---

## 结果目录

训练和评估产物保存在 `results/` 下：

```
results/
├── models/
│   ├── td3_uav.zip                 # 最终训练模型
│   ├── td3_uav_ckpt_100000.zip     # 10 万步 checkpoint
│   ├── td3_uav_ckpt_200000.zip     # 20 万步 checkpoint
│   ├── td3_uav_ckpt_300000.zip     # 30 万步 checkpoint
│   └── vec_normalize.pkl           # 归一化参数（如使用 VecNormalize）
├── logs/
│   ├── training_log.csv            # 逐 episode 训练日志
│   └── config_snapshot.json        # 训练时的配置快照
└── figures/
    ├── rewards.png                 # 评估奖励图
    ├── distance_errors.png         # 距离误差曲线
    ├── trajectories.png            # 轨迹对比图
    └── eval_results.csv            # 评估指标汇总
```

`training_log.csv` 包含以下字段：

| 字段 | 说明 |
|------|------|
| Episode | 轮次编号 |
| Reward | 该轮总奖励 |
| Steps | 该轮步数 |
| Progress | 轨迹完成进度 |
| AvgDist | 平均跟随距离 |
| Within5m% | 5m 内比例 |
| R_Follow | 跟随奖励分项 |
| R_Heading | 航向奖励分项 |
| R_Smooth | 平滑惩罚分项 |
| TotalSteps | 累计训练步数 |

---

## 技术栈

| 类别 | 技术 | 版本 |
|------|------|------|
| 语言 | Python | 3.8+ |
| RL 框架 | Stable-Baselines3 | 2.2.1 |
| 环境接口 | Gymnasium | 0.29.1 |
| 数值计算 | NumPy | 1.26.4 |
| 数据处理 | Pandas | 2.2.2 |
| 插值平滑 | SciPy | 1.12.0 |
| 可视化 | Matplotlib | 3.8.4 |
| GUI 框架 | CustomTkinter | 5.2.2 |
| 训练监控 | TensorBoard | 2.16.2 |

---

## 已知问题与改进方向

- **运动学模型简化**：当前为理想 2D 运动学，无惯性、无风扰、无传感器噪声，与真实无人机差距较大
- **环境维度**：仅 2D 平面，未考虑高度维度
- **TensorBoard 未启用**：训练器中 `tensorboard_log=None`，日志仅通过 CSV 记录
- **GUI 绘图冗余**：`plot_2d.py` 和 `plot_3d.py` 未被 `app.py` 直接调用，存在重复代码
- **配置风格不统一**：`EnvConfig` / `RLConfig` 为普通类，可改为 dataclass 或 pydantic 模型

可能的改进方向：

1. 引入 3D 环境和更真实的无人机动力学模型
2. 添加风扰动和传感器噪声以提升鲁棒性
3. 启用 TensorBoard 实时监控训练曲线
4. 引入课程学习（Curriculum Learning），从简单轨迹逐步过渡到复杂轨迹
5. 尝试其他算法（SAC、PPO）进行对比实验

---

## 许可证

本项目仅供学习交流使用。
