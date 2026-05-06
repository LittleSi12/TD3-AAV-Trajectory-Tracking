"""
训练结果分析与可视化脚本。
从 training_log.csv 读取数据，生成多种图表用于汇报展示。

用法：python -m Utils.plot_training
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["SimSun", "Times New Roman"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "results", "figures", "training_analysis")
LOG_PATH = os.path.join(_PROJECT_ROOT, "results", "logs", "training_log.csv")


def smooth(data, window=10):
    """滑动平均平滑。"""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


def load_log(path=LOG_PATH):
    df = pd.read_csv(path)
    print(f"加载日志: {path}")
    print(f"  总记录: {len(df)} 条, 步数范围: {df['TotalSteps'].iloc[0]}~{df['TotalSteps'].iloc[-1]}")
    return df


def plot_reward_curve(df, save_dir):
    """奖励曲线（原始 + 平滑）"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Episode"], df["Reward"], alpha=0.3, color="blue", label="原始奖励")
    if len(df) > 10:
        sm = smooth(df["Reward"].values)
        ax.plot(range(5, 5 + len(sm)), sm, color="blue", linewidth=2, label="滑动平均(10)")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("累积奖励", fontsize=12)
    ax.set_title("训练奖励曲线", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "reward_curve.png"), dpi=150)
    plt.close(fig)
    print("  ✓ reward_curve.png")


def plot_distance_curve(df, save_dir):
    """平均跟随距离曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Episode"], df["AvgDist"], alpha=0.3, color="orange")
    if len(df) > 10:
        sm = smooth(df["AvgDist"].values)
        ax.plot(range(5, 5 + len(sm)), sm, color="orange", linewidth=2, label="滑动平均")
    ax.axhline(y=5.0, color="red", linestyle="--", linewidth=1.5, label="目标距离 5m")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("平均跟随距离 (m)", fontsize=12)
    ax.set_title("跟随距离变化曲线", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "distance_curve.png"), dpi=150)
    plt.close(fig)
    print("  ✓ distance_curve.png")


def plot_within5m_curve(df, save_dir):
    """5m 内比例曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Episode"], df["Within5m%"], alpha=0.3, color="green")
    if len(df) > 10:
        sm = smooth(df["Within5m%"].values)
        ax.plot(range(5, 5 + len(sm)), sm, color="green", linewidth=2, label="滑动平均")
    ax.axhline(y=80, color="red", linestyle="--", linewidth=1.5, label="目标 80%")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("5m 内比例 (%)", fontsize=12)
    ax.set_title("跟随精度曲线（5m 内比例）", fontsize=14)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "within5m_curve.png"), dpi=150)
    plt.close(fig)
    print("  ✓ within5m_curve.png")


def plot_progress_curve(df, save_dir):
    """轨迹完成进度曲线"""
    fig, ax = plt.subplots(figsize=(10, 5))
    progress_pct = df["Progress"] * 100
    ax.plot(df["Episode"], progress_pct, alpha=0.3, color="purple")
    if len(df) > 10:
        sm = smooth(progress_pct.values)
        ax.plot(range(5, 5 + len(sm)), sm, color="purple", linewidth=2, label="滑动平均")
    ax.axhline(y=100, color="red", linestyle="--", linewidth=1.5, label="目标 100%")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("轨迹完成进度 (%)", fontsize=12)
    ax.set_title("轨迹完成进度曲线", fontsize=14)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "progress_curve.png"), dpi=150)
    plt.close(fig)
    print("  ✓ progress_curve.png")


def plot_reward_components(df, save_dir):
    """奖励分项对比"""
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(df["Episode"], df["R_Follow"], color="green", alpha=0.6)
    axes[0].set_ylabel("跟随奖励", fontsize=11)
    axes[0].set_title("奖励分项分析", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["Episode"], df["R_Heading"], color="blue", alpha=0.6)
    axes[1].set_ylabel("航向奖励", fontsize=11)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["Episode"], df["R_Smooth"], color="red", alpha=0.6)
    axes[2].set_ylabel("平滑惩罚", fontsize=11)
    axes[2].set_xlabel("Episode", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "reward_components.png"), dpi=150)
    plt.close(fig)
    print("  ✓ reward_components.png")


def plot_loss_curves(df, save_dir):
    """Actor/Critic Loss 曲线"""
    if "ActorLoss" not in df.columns or "CriticLoss" not in df.columns:
        print("  ⚠ 日志中无 Loss 数据，跳过")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(df["Episode"], df["ActorLoss"], color="blue", alpha=0.6)
    if len(df) > 10:
        sm = smooth(df["ActorLoss"].values)
        ax1.plot(range(5, 5 + len(sm)), sm, color="blue", linewidth=2)
    ax1.set_ylabel("Actor Loss", fontsize=11)
    ax1.set_title("损失函数曲线", fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(df["Episode"], df["CriticLoss"], color="red", alpha=0.6)
    if len(df) > 10:
        sm = smooth(df["CriticLoss"].values)
        ax2.plot(range(5, 5 + len(sm)), sm, color="red", linewidth=2)
    ax2.set_ylabel("Critic Loss", fontsize=11)
    ax2.set_xlabel("Episode", fontsize=12)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "loss_curves.png"), dpi=150)
    plt.close(fig)
    print("  ✓ loss_curves.png")


def plot_max_distance_curve(df, save_dir):
    """最大跟随距离曲线"""
    if "MaxDist" not in df.columns:
        print("  ⚠ 日志中无 MaxDist 数据，跳过")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Episode"], df["MaxDist"], alpha=0.3, color="crimson")
    if len(df) > 10:
        sm = smooth(df["MaxDist"].values)
        ax.plot(range(5, 5 + len(sm)), sm, color="crimson", linewidth=2, label="滑动平均")
    ax.axhline(y=10.0, color="red", linestyle="--", linewidth=1.5, label="警戒线 10m")
    ax.axhline(y=5.0, color="orange", linestyle="--", linewidth=1.5, label="目标 5m")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("最大跟随距离 (m)", fontsize=12)
    ax.set_title("最大偏离距离变化曲线", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "max_distance_curve.png"), dpi=150)
    plt.close(fig)
    print("  ✓ max_distance_curve.png")


def plot_end_reason_stats(df, save_dir):
    """终止原因统计饼图"""
    if "EndReason" not in df.columns:
        print("  ⚠ 日志中无 EndReason 数据，跳过")
        return

    counts = df["EndReason"].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()

    # 颜色映射
    color_map = {
        "completed": "#10b981",
        "truncated": "#f59e0b",
        "out_of_bound": "#ef4444",
        "unknown": "#94a3b8",
    }
    colors = [color_map.get(l, "#64748b") for l in labels]

    # 中文标签
    label_map = {
        "completed": "完成",
        "truncated": "超时截断",
        "out_of_bound": "飞出边界",
        "unknown": "未知",
    }
    display_labels = [f"{label_map.get(l, l)} ({v})" for l, v in zip(labels, values)]

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        values, labels=display_labels, colors=colors,
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 12}
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_fontweight("bold")
    ax.set_title("Episode 终止原因分布", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "end_reason_stats.png"), dpi=150)
    plt.close(fig)
    print("  ✓ end_reason_stats.png")


def plot_combined_dashboard(df, save_dir):
    """综合仪表盘（4 合 1）"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 奖励
    ax = axes[0, 0]
    ax.plot(df["Episode"], df["Reward"], alpha=0.3, color="blue")
    if len(df) > 10:
        ax.plot(range(5, 5 + len(smooth(df["Reward"].values))),
                smooth(df["Reward"].values), color="blue", linewidth=2)
    ax.set_title("累积奖励", fontsize=13)
    ax.set_xlabel("Episode"); ax.grid(True, alpha=0.3)

    # 距离
    ax = axes[0, 1]
    ax.plot(df["Episode"], df["AvgDist"], alpha=0.3, color="orange")
    if len(df) > 10:
        ax.plot(range(5, 5 + len(smooth(df["AvgDist"].values))),
                smooth(df["AvgDist"].values), color="orange", linewidth=2)
    ax.axhline(y=5.0, color="red", linestyle="--")
    ax.set_title("平均跟随距离 (m)", fontsize=13)
    ax.set_xlabel("Episode"); ax.grid(True, alpha=0.3)

    # In5m
    ax = axes[1, 0]
    ax.plot(df["Episode"], df["Within5m%"], alpha=0.3, color="green")
    if len(df) > 10:
        ax.plot(range(5, 5 + len(smooth(df["Within5m%"].values))),
                smooth(df["Within5m%"].values), color="green", linewidth=2)
    ax.axhline(y=80, color="red", linestyle="--")
    ax.set_title("5m 内比例 (%)", fontsize=13)
    ax.set_xlabel("Episode"); ax.set_ylim(0, 105); ax.grid(True, alpha=0.3)

    # Progress
    ax = axes[1, 1]
    progress_pct = df["Progress"] * 100
    ax.plot(df["Episode"], progress_pct, alpha=0.3, color="purple")
    if len(df) > 10:
        ax.plot(range(5, 5 + len(smooth(progress_pct.values))),
                smooth(progress_pct.values), color="purple", linewidth=2)
    ax.axhline(y=100, color="red", linestyle="--")
    ax.set_title("轨迹完成进度 (%)", fontsize=13)
    ax.set_xlabel("Episode"); ax.set_ylim(0, 110); ax.grid(True, alpha=0.3)

    fig.suptitle("TD3 无人机轨迹跟随 — 训练过程总览", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_dashboard.png"), dpi=150)
    plt.close(fig)
    print("  ✓ training_dashboard.png")


def plot_episode_length(df, save_dir):
    """Episode 长度变化"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["Episode"], df["Steps"], alpha=0.3, color="teal")
    if len(df) > 10:
        sm = smooth(df["Steps"].values)
        ax.plot(range(5, 5 + len(sm)), sm, color="teal", linewidth=2, label="滑动平均")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode 步数", fontsize=12)
    ax.set_title("Episode 长度变化", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "episode_length.png"), dpi=150)
    plt.close(fig)
    print("  ✓ episode_length.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*50}")
    print("  TD3 训练结果分析")
    print(f"{'='*50}\n")

    df = load_log()
    print(f"\n生成图表到: {OUTPUT_DIR}/\n")

    plot_reward_curve(df, OUTPUT_DIR)
    plot_distance_curve(df, OUTPUT_DIR)
    plot_max_distance_curve(df, OUTPUT_DIR)
    plot_within5m_curve(df, OUTPUT_DIR)
    plot_progress_curve(df, OUTPUT_DIR)
    plot_reward_components(df, OUTPUT_DIR)
    plot_loss_curves(df, OUTPUT_DIR)
    plot_end_reason_stats(df, OUTPUT_DIR)
    plot_combined_dashboard(df, OUTPUT_DIR)
    plot_episode_length(df, OUTPUT_DIR)

    # 输出统计摘要
    print(f"\n{'='*50}")
    print("  训练过程概况")
    print(f"{'='*50}")
    print(f"  总 Episode 数:    {len(df)}")
    print(f"  总训练步数:       {df['TotalSteps'].iloc[-1]}")
    print(f"  最高单轮奖励:     {df['Reward'].max():.1f}")
    print(f"  最低平均距离:     {df['AvgDist'].min():.2f} m")
    print(f"  最高 5m 内比例:   {df['Within5m%'].max():.1f}%")

    # 读取评估结果展示最终模型性能
    eval_path = os.path.join(_PROJECT_ROOT, "results", "figures", "eval_results.csv")
    if os.path.exists(eval_path):
        eval_df = pd.read_csv(eval_path)
        print(f"\n{'='*50}")
        print("  最终模型性能（评估结果）")
        print(f"{'='*50}")
        print(f"  评估轮数:         {len(eval_df)}")
        print(f"  平均跟随距离:     {eval_df['AvgDist'].mean():.2f} m")
        print(f"  最大偏离距离:     {eval_df['MaxDist'].max():.2f} m")
        # 动态查找 Within 列名
        within_col = [c for c in eval_df.columns if "Within" in c]
        if within_col:
            print(f"  {within_col[0]}:  {eval_df[within_col[0]].mean():.1f}%")
        print(f"  平均完成进度:     {eval_df['Progress'].mean()*100:.1f}%")
        completed = (eval_df['EndReason'] == 'completed').sum() if 'EndReason' in eval_df.columns else '-'
        print(f"  完成轮数:         {completed}/{len(eval_df)}")
    else:
        print(f"\n  ⚠ 未找到评估结果，请先运行 python Test.py")

    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
