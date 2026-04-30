"""
训练入口脚本。

用法：
    python Train.py                     # 从头训练
    python Train.py --resume            # 从最新 checkpoint 继续训练
    python Train.py --resume --steps 500000   # 继续训练 50 万步
    python Train.py --resume --ckpt results/models/td3_uav_ckpt_300000  # 从指定 checkpoint 继续

Author: Little Si
Date:   2026-04
"""

import os
import sys
import argparse
import traceback

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from RL_Module.TD3_Trainer import TD3Trainer


def main():
    parser = argparse.ArgumentParser(description="TD3 无人机轨迹跟随训练")
    parser.add_argument("--resume", action="store_true",
                        help="从最新 checkpoint 继续训练")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="指定 checkpoint 路径（不含 .zip）")
    parser.add_argument("--steps", type=int, default=None,
                        help="训练步数（默认使用配置文件中的值）")
    args = parser.parse_args()

    try:
        print("Creating trainer...")
        trainer = TD3Trainer()
        print(f"Observation space: {trainer.venv.observation_space}")
        print(f"Action space: {trainer.venv.action_space}")

        if args.resume or args.ckpt:
            print("\n=== 断点续训模式 ===")
            trainer.resume_train(
                path=args.ckpt,
                additional_timesteps=args.steps,
            )
        else:
            print("\n=== 全新训练模式 ===")
            if args.steps:
                trainer.cfg.TRAIN_TIMESTEPS = args.steps
            trainer.train()

        print("Done.")
        trainer.venv.close()
        del trainer

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
