"""
train.py — SAC Training Script
================================
Trains a Soft Actor-Critic agent for the MysteryControlEnv.
Run on CPU for consistent local training.

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

    class DetailedProgressCallback(BaseCallback):
        """Fast console logger for detailed progress without slowing training."""

        def __init__(self, total_timesteps, log_every_steps=2000, verbose=0):
            super().__init__(verbose)
            self.total_timesteps = total_timesteps
            self.log_every_steps = log_every_steps
            self.last_logged_step = 0
            self.episodes_completed = 0
            self.start_time = None

        def _on_training_start(self):
            self.start_time = time.time()

        def _on_step(self) -> bool:
            dones = self.locals.get("dones")
            if dones is not None:
                self.episodes_completed += int(np.sum(dones))

            if self.num_timesteps - self.last_logged_step >= self.log_every_steps:
                elapsed = max(time.time() - self.start_time, 1e-9)
                percent = 100.0 * self.num_timesteps / self.total_timesteps
                fps = self.num_timesteps / elapsed
                remaining_steps = max(self.total_timesteps - self.num_timesteps, 0)
                eta_sec = remaining_steps / max(fps, 1e-9)

                mean_ep_rew = None
                mean_ep_len = None
                if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
                    mean_ep_rew = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
                    mean_ep_len = float(np.mean([ep["l"] for ep in self.model.ep_info_buffer]))

                eta_min = eta_sec / 60.0
                msg = (
                    f"[Progress] {self.num_timesteps:,}/{self.total_timesteps:,} "
                    f"({percent:5.1f}%) | episodes={self.episodes_completed:,} "
                    f"| fps={fps:,.0f} | eta={eta_min:5.1f} min"
                )
                if mean_ep_rew is not None and mean_ep_len is not None:
                    msg += f" | mean_ep_rew={mean_ep_rew:7.2f} | mean_ep_len={mean_ep_len:6.1f}"

                print(msg)
                self.last_logged_step = self.num_timesteps

            return True

    device = "cpu"
    TOTAL_TIMESTEPS = 300_000
    N_ENVS = 4
    LOG_EVERY_STEPS = 2000

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

    print(f"Detailed progress: every {LOG_EVERY_STEPS:,} timesteps")
    progress_callback = DetailedProgressCallback(
        total_timesteps=TOTAL_TIMESTEPS,
        log_every_steps=LOG_EVERY_STEPS,
    )

    # Keep training fast: use lightweight callback logging instead of rich progress bars.
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=progress_callback, progress_bar=False)
    model.save("best_sac_agent")
    print("Saved: best_sac_agent.zip")
    print("Now run: python extract_weights.py")


if __name__ == "__main__":
    train()
