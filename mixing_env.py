# mixing_env.py
import gymnasium as gym
import numpy as np

class AudioMixingEnv(gym.Env):
    def __init__(self):
        super(AudioMixingEnv, self).__init__()

        # Action space: Gain adjustments for 4 stems (vocals, drums, bass, other)
        self.action_space = gym.spaces.Box(low=0.5, high=2.0, shape=(4,), dtype=np.float32)

        # Observation space: Mean amplitude values of each stem
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)

        # Dummy initial state
        self.state = np.array([0.5, 0.5, 0.5, 0.5])

    def step(self, action):
        # Apply gain factors
        new_state = self.state * action

        # Compute reward based on SDR improvement
        reward = self.compute_sdr_reward(new_state)
        
        terminated = False  # If episode should end based on logic, set to True
        truncated = False  # For Gymnasium, if an episode is forcefully stopped

        return new_state, reward, terminated, truncated, {}

    def compute_sdr_reward(self, adjusted_state):
        return np.mean(adjusted_state)  # Example: Higher gain = higher reward (You need a real SDR function)

    def reset(self, seed=None, options=None):
        self.state = np.random.rand(4)  # Random initial state
        return self.state, {}

        