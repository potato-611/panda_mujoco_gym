import gymnasium as gym
import panda_mujoco_gym
from stable_baselines3 import PPO
import time
import os
import glob

def visualize():
    # 1. 设置环境 ID
    env_id = "FrankaPushDense-v0"
    
    # 获取最新的模型路径
    models_dir = "models"
    model_pattern = os.path.join(models_dir, f"ppo_{env_id}_*.zip")
    model_files = glob.glob(model_pattern)
    
    if model_files:
        # 按修改时间排序，选择最新的
        latest_model = max(model_files, key=os.path.getmtime)
        model_path = latest_model.replace(".zip", "")
        print(f"找到最新模型: {latest_model}")
    else:
        model_path = f"ppo_{env_id}_model"
        print(f"未在 {models_dir} 中找到匹配模型，尝试默认路径: {model_path}")

    # 2. 创建环境，设置 render_mode="human" 用于实时可视化
    print(f"正在启动环境 {env_id} (Human 模式)...")
    env = gym.make(env_id, render_mode="human")

    # 3. 加载训练好的模型
    if os.path.exists(f"{model_path}.zip"):
        print(f"加载模型: {model_path}")
        model = PPO.load(model_path)
    else:
        print(f"警告: 未找到模型文件 {model_path}.zip，将使用随机动作。")
        model = None

    # 4. 运行可视化
    obs, info = env.reset()
    print("开始实时可视化运行 (按 Ctrl+C 退出)...")
    
    try:
        while True:
            if model:
                action, _states = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
                
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 适当等待以匹配现实时间（MuJoCo 默认步长通常较快）
            time.sleep(0.02)
            
            if terminated or truncated:
                obs, info = env.reset()
                print("回合结束，重置环境...")
    except KeyboardInterrupt:
        print("\n正在停止...")
    finally:
        env.close()
        print("环境已关闭。")

if __name__ == "__main__":
    visualize()
