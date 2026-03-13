"""Environment wrappers shared across training and analysis scripts."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

OBSERVATION_SCALE = 100.0


class NormalizedObsWrapper(gym.ObservationWrapper):
    """Scale observations from ``[0, 100]`` to ``[0, 1]``."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs / OBSERVATION_SCALE
