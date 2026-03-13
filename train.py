"""
train.py — SAC Training Script
================================
Trains a Soft Actor-Critic agent for the MysteryControlEnv.
Run on Kaggle/Colab with GPU for fastest results (~30 min).
Also works on CPU within ~45-60 min.

Requirements:
    pip install stable-baselines3 gymnasium numpy

After training, run extract_weights.py to get sac_weights.npz for submission.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import time
import json

from environment import MysteryControlEnv


# ── Reward Shaping Wrapper ──────────────────────────────────
class ShapedRewardWrapper(gym.Wrapper):
    """
    Custom reward shaping to accelerate SAC training.

    Design:
    - Primary: negative absolute error (aligned with hackathon metric)
    - Shaping: potential-based bonus for reducing error each step
    - Closeness bonus: extra reward when near target
    - Safety margin: soft penalty starting at 85 (hard limit is 95)
    - Termination: heavy penalty for safety violations
    """

    def __init__(self, env):
        super().__init__(env)
        self.prev_p_error = None
        self.prev_t_error = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        pressure, temp, target_p, target_t = obs
        self.prev_p_error = abs(pressure - target_p)
        self.prev_t_error = abs(temp - target_t)
        return obs, info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        new_pressure, new_temp, target_p, target_t = obs

        p_error = abs(new_pressure - target_p)
        t_error = abs(new_temp - target_t)

        # Primary signal
        reward = -(p_error + t_error)

        # Potential-based shaping
        reward += 2.0 * ((self.prev_p_error - p_error) + (self.prev_t_error - t_error))

        # Closeness bonuses
        if p_error < 2.0:
            reward += 3.0
        if t_error < 2.0:
            reward += 3.0
        if p_error < 0.5 and t_error < 0.5:
            reward += 10.0

        # Safety margin
        if new_pressure > 85:
            reward -= 5.0 * (new_pressure - 85)
        if new_temp > 85:
            reward -= 5.0 * (new_temp - 85)

        # Termination
        if terminated:
            reward -= 200.0

        self.prev_p_error = p_error
        self.prev_t_error = t_error

        return obs, reward, terminated, truncated, info


# ── Observation Normalization ───────────────────────────────
class NormalizedObsWrapper(gym.ObservationWrapper):
    """Scale all observations from [0, 100] to [0, 1]."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

    def observation(self, obs):
        return obs / 100.0


# ── Environment Factory ─────────────────────────────────────
def make_env(rank=0, seed=0):
    def _init():
        env = MysteryControlEnv()
        env = ShapedRewardWrapper(env)
        env = NormalizedObsWrapper(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# ── Training ────────────────────────────────────────────────
def train():
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    TOTAL_TIMESTEPS = 500_000 if device == "cuda" else 300_000
    N_ENVS = 4

    print(f"Device: {device} | Timesteps: {TOTAL_TIMESTEPS:,}")

    train_envs = DummyVecEnv([make_env(rank=i, seed=42) for i in range(N_ENVS)])

    model = SAC(
        "MlpPolicy",
        train_envs,
        learning_rate=3e-4,
        buffer_size=200_000,
        learning_starts=5_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        target_entropy="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=0,
        device=device,
        seed=42,
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    model.save("best_sac_agent")
    print("Saved: best_sac_agent.zip")
    print("Now run: python extract_weights.py")


if __name__ == "__main__":
    train()
