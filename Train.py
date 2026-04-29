import os
import traceback

# 设置环境变量解决 OpenMP 库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("Starting Train.py...")
try:
    from RL_Module.TD3_Trainer import TD3Trainer

    if __name__ == "__main__":
        # 创建训练器（内部自动创建 VecNormalize 环境）
        print("Creating trainer...")
        trainer = TD3Trainer()
        print("Trainer created successfully")

        # 开始训练
        print("Starting training...")
        trainer.train()
        print("Training completed")

        # 关闭环境
        trainer.venv.close()
        del trainer
        print("Trainer deleted")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
