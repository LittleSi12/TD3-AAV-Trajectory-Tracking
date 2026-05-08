# 使用文档

本文档说明各脚本的功能与运行方式。

---

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 直接运行 GUI（使用已训练好的模型）
python RunGUI.py

# 3. 或者运行评估查看模型性能
python Test.py
```

---

## 脚本说明

| 脚本 | 功能 | 说明 |
|------|------|------|
| `Train.py` | 训练模型 | 从头训练或断点续训 TD3 模型 |
| `Test.py` | 评估模型 | 在随机轨迹上测试模型性能 |
| `RunGUI.py` | 可视化 GUI | 交互式界面展示无人机跟随效果 |
| `Utils/plot_training.py` | 训练分析 | 从训练日志生成各类分析图表 |

---

## 详细用法

### Train.py — 训练模型

```bash
# 从头训练（默认 200 万步）
python Train.py

# 指定训练步数
python Train.py --steps 500000

# 从最新 checkpoint 断点续训
python Train.py --resume

# 从指定 checkpoint 续训
python Train.py --resume --ckpt results/models/td3_uav_ckpt_500000

# 续训并指定额外步数
python Train.py --resume --steps 1000000
```

**输出：**
- 模型文件 → `results/models/td3_uav.zip`
- Checkpoint → `results/models/td3_uav_ckpt_XXXXXX.zip`（每 10 万步）
- 训练日志 → `results/logs/training_log.csv`
- 配置快照 → `results/logs/config_snapshot.json`

**控制台输出示例：**
```
Ep    10 | Steps:    50000 | Reward:  3800.0 | Progress: 100.0% | AvgDist:  4.2m | MaxDist: 12.3m | In5m: 65.0% | End: completed
```

---

### Test.py — 评估模型

```bash
python Test.py
```

在 5 条随机新轨迹上评估训练好的模型，输出每轮详情和汇总指标。

**输出：**
- 评估详情 → `results/figures/eval_results.csv`
- 奖励图 → `results/figures/rewards.png`
- 距离误差图 → `results/figures/distance_errors.png`
- 轨迹对比图 → `results/figures/trajectories.png`

**控制台输出示例：**
```
Episode 1: Reward=4574.2, Progress=100.0%, AvgDist=1.88m, MaxDist=5.66m, Within 5.0m: 99.5%, End: completed
Episode 2: Reward=4769.5, Progress=100.0%, AvgDist=2.09m, MaxDist=5.34m, Within 5.0m: 99.9%, End: completed
...
评估完成
  平均跟随距离: 2.10 m
  最大跟随距离: 9.15 m
  5.0m 内比例: 99.8%
```

---

### RunGUI.py — 可视化 GUI

```bash
python RunGUI.py
```

启动交互式可视化界面，分三个阶段操作：

1. **① 环境设置** — 随机生成灾害区域，展示在 2D/3D 视图中
2. **② 轨迹规划** — 生成搜救路径并展示（带闪烁动画）
3. **③ 任务开始** — 加载模型，动画展示无人机跟随过程

界面功能：
- 左侧：2D/3D 标签页切换
- 右侧：运行日志（富文本彩色输出）
- 顶部：实时指标卡（任务进度、当前距离、平均距离、8m 内比例）
- 任务执行中可点击「⏹ 结束任务」提前停止
- 关闭窗口自动退出程序

> 注意：需要先有训练好的模型（`results/models/td3_uav.zip`）

---

### Utils/plot_training.py — 训练过程分析

```bash
python -m Utils.plot_training
```

从 `results/logs/training_log.csv` 读取训练日志，生成以下图表：

| 图表文件 | 内容 |
|----------|------|
| `reward_curve.png` | 奖励曲线（原始 + 滑动平均） |
| `distance_curve.png` | 平均跟随距离变化 |
| `max_distance_curve.png` | 最大偏离距离变化 |
| `within5m_curve.png` | 5m 内比例变化 |
| `progress_curve.png` | 轨迹完成进度变化 |
| `reward_components.png` | 奖励分项（跟随/航向/平滑） |
| `end_reason_stats.png` | 终止原因分布饼图 |
| `episode_length.png` | Episode 步数变化 |
| `training_dashboard.png` | 综合仪表盘（4 合 1） |

**输出目录：** `results/figures/training_analysis/`

---

## 配置修改

所有参数在 `configs/` 目录下，修改后直接生效（需重新训练）：

- `configs/env_config.py` — 环境参数（速度、偏航率、奖励权重、轨迹设置等）
- `configs/rl_config.py` — 训练参数（学习率、训练步数、buffer 大小等）

---

## 典型工作流

```
1. 修改参数      → configs/env_config.py 或 rl_config.py
2. 训练模型      → python Train.py --steps 500000
3. 评估性能      → python Test.py
4. 分析训练过程  → python -m Utils.plot_training
5. 可视化展示    → python RunGUI.py
```
