import gymnasium as gym
import panda_mujoco_gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime

def make_env(env_id):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)  # ⭐ 记录 episode reward/len
        return env
    return _init

def train():
    env_id = "FrankaPushDense-v0"

    # VecEnv + Monitor
    env = DummyVecEnv([make_env(env_id)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_panda_tensorboard/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    print(f"开始训练 {env_id}...")
    model.learn(total_timesteps=1_000_000)

    # 保存模型和归一化参数
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/ppo_{env_id}_{timestamp}"
    model.save(model_path)
    env.save(f"{model_path}_vecnormalize.pkl")

    print(f"模型已保存至: {model_path}")

    # ===== 测试阶段（单独环境 + render）=====
    test_env = gym.make(env_id, render_mode="human")

    obs, _ = test_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action)
        if terminated or truncated:
            obs, _ = test_env.reset()

if __name__ == "__main__":
    train()