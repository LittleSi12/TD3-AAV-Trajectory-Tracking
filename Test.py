"""
测试/评估入口脚本。

Author: Little Si
Date:   2026-04
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3 import TD3
from UAV_Environment.Environment import UAV2DEnv
from RL_Module.ModelEvaluator import ModelEvaluator
from configs.rl_config import RLConfig

if __name__ == "__main__":
    model_path = RLConfig.SAVE_PATH

    if not os.path.exists(model_path + ".zip"):
        print("没有找到模型文件，请先运行 Train.py 生成模型")
        exit()

    model = TD3.load(model_path)
    print(f"Model loaded from: {model_path}")

    evaluator = ModelEvaluator(model)
    results = evaluator.evaluate(episodes=5)
    print(f"\n评估完成")
    print(f"  平均跟随距离: {results['avg_distance']:.2f} m")
    print(f"  5m 内比例: {results['within_follow_pct']:.1f}%")

    evaluator.close()
