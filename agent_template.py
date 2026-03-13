import numpy as np


class ParticipantAgent:
    """Base participant interface used by evaluation and submission scripts."""

    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

    def act(self, observation: np.ndarray) -> np.ndarray:
        """Return an action for the current observation."""
        return self.action_space.sample()

    def reward_function(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """Default reward aligned with environment error minimization."""
        del action, truncated
        pressure, temp, target_pressure, target_temp = state
        next_pressure, next_temp, _, _ = next_state
        p_error = abs(next_pressure - target_pressure)
        t_error = abs(next_temp - target_temp)
        reward = -(p_error + t_error)
        if terminated:
            reward -= 100
        return float(reward)
