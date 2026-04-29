import os

# 设置环境变量解决 OpenMP 库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from UAV_Environment.Environment import UAV2DEnv
from RL_Module.ModelEvaluator import ModelEvaluator
from configs.rl_config import RLConfig

if __name__ == "__main__":
    model_path = RLConfig.SAVE_PATH
    vecnorm_path = os.path.join(os.path.dirname(model_path), "vec_normalize.pkl")

    if not os.path.exists(model_path + ".zip"):
        print("没有找到模型文件，请先运行 Train.py 生成模型")
        exit()

    # 重建 VecNormalize 环境（评估模式，不更新统计量）
    vec_env = None
    if os.path.exists(vecnorm_path):
        base_venv = DummyVecEnv([lambda: UAV2DEnv()])
        vec_env = VecNormalize.load(vecnorm_path, base_venv)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"VecNormalize loaded from: {vecnorm_path}")

    # 直接加载模型，不再创建 TD3Trainer（避免初始化无用模型）
    model = TD3.load(model_path, env=vec_env)
    print(f"Model loaded from: {model_path}")

    # 评估
    evaluator = ModelEvaluator(model, vec_env=vec_env)
    results = evaluator.evaluate()
    print(f"\n评估完成，平均奖励: {results['avg_reward']:.2f}")

    evaluator.close()
