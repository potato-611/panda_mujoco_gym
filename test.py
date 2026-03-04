import sys
import time
import gymnasium as gym
import panda_mujoco_gym

## 这是一个简单的测试脚本，用于验证环境是否能够正确运行并进行交互。

if __name__ == "__main__":
    env = gym.make("FrankaPickAndPlaceSparse-v0", render_mode="human")

    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        time.sleep(0.2)

    env.close()