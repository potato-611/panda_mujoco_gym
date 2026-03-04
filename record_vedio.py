import gymnasium as gym
import panda_mujoco_gym
from stable_baselines3 import PPO
import time
import os
from datetime import datetime

def enjoy():
    # 1. 设置环境 ID 和模型路径
    env_id = "FrankaPushDense-v0"
    model_path = f"ppo_{env_id}_model"
    
    # 获取当前时间作为子目录名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_folder = os.path.join("videos", timestamp)

    # 2. 创建环境，设置 render_mode="rgb_array" 用于录制
    print(f"正在启动环境 {env_id}...")
    env = gym.make(env_id, render_mode="rgb_array")

    # 3. 添加视频录制包装器
    # trigger=lambda x: True 表示录制所有回合，这里我们只录制 1 个完整回合
    env = gym.wrappers.RecordVideo(
        env, 
        video_folder=video_folder, 
        name_prefix=f"eval_{env_id}",
        episode_trigger=lambda x: x == 0
    )

    # 4. 加载训练好的模型
    print(f"加载模型: {model_path}")
    model = PPO.load(model_path)

    # 5. 运行模型并录制内容
    obs, info = env.reset()
    print(f"开始录制视频，结果将保存至 {video_folder} 目录...")
    
    count = 0
    while count < 1:  # 录制 1 个完整回合
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
            count += 1
            
    # 必须关闭环境以确保视频文件正常保存
    env.close()
    print(f"录制完成！视频已保存到: {os.path.abspath(video_folder)}")

if __name__ == "__main__":
    enjoy()
