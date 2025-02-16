# train_rl.py
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

# Define AI Mixing Environment
class AudioMixingEnv(gym.Env):
    def __init__(self):
        super(AudioMixingEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))  # Volume, Panning, Reverb
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(3,))  

    def step(self, action):
        reward = self.evaluate_mixing(action)
        done = False  # Ensure done is always returned
        truncated = False  # Gymnasium now expects "truncated"
        return self._get_obs(), reward, done, truncated, {}

    def reset(self, seed=None, options=None):  # ✅ Ensure reset() returns (obs, info)
        super().reset(seed=seed)
        return self._get_obs(), {}  # Return observation and empty info dict

    def _get_obs(self):
        return np.random.rand(3)  # Simulated observation

    def evaluate_mixing(self, action):
        return -np.sum(np.abs(action))  # Reward = minimize deviation

# Train PPO Model
env = DummyVecEnv([lambda: AudioMixingEnv()])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# Save the trained model in the correct location
os.makedirs("adaptive_mixing_model", exist_ok=True)
model.save("adaptive_mixing_model/ppo_mixing_model")
print("✅ Model training complete. Model saved to adaptive_mixing_model/ppo_mixing_model")

