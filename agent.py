"""
agent.py — BELLATRIX Hackathon Submission
==========================================
Algorithm:  Soft Actor-Critic (SAC)
Framework:  Trained with stable-baselines3, inference with numpy only
Architecture: MLP [4] → [256, ReLU] → [256, ReLU] → [3, Tanh] → scale to [0,1]

The agent loads pre-trained weights from sac_weights.npz and performs
inference using pure numpy matrix operations. No PyTorch, TensorFlow,
or stable-baselines3 is required at evaluation time.

Allowed libraries used: numpy, gymnasium (matplotlib for optional plotting)
"""

from agent_template import ParticipantAgent
import numpy as np
import os

OBSERVATION_SCALE = 100.0
WEIGHTS_FILENAME = "sac_weights.npz"


class MySmartAgent(ParticipantAgent):
    """
    SAC-trained agent for MysteryControlEnv.

    Observation: [pressure, temp, target_pressure, target_temp]  (0–100 each)
    Action:      [inlet_valve, outlet_valve, heater_power]       (0–1 each)

    The neural network was trained via SAC to minimize tracking error
    (|pressure - target| + |temp - target|) while avoiding safety violations
    (pressure > 95 or temp > 95).
    """

    def __init__(self, action_space, observation_space):
        super().__init__(action_space, observation_space)
        self.hidden_weights = []
        self.hidden_biases = []
        self.mu_weight = None
        self.mu_bias = None
        self._load_weights()

    def _resolve_weights_path(self) -> str:
        weights_path = os.path.join(os.path.dirname(__file__), WEIGHTS_FILENAME)
        if not os.path.exists(weights_path):
            weights_path = WEIGHTS_FILENAME
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                "sac_weights.npz not found. "
                "Place it in the same directory as agent.py"
            )
        return weights_path

    def _load_weights(self) -> None:
        weights_path = self._resolve_weights_path()
        data = np.load(weights_path)

        layer_index = 0
        while f"latent_{layer_index}_w" in data:
            self.hidden_weights.append(data[f"latent_{layer_index}_w"].astype(np.float32))
            self.hidden_biases.append(data[f"latent_{layer_index}_b"].astype(np.float32))
            layer_index += 2

        self.mu_weight = data["mu_w"].astype(np.float32)
        self.mu_bias = data["mu_b"].astype(np.float32)

    def _forward_policy(self, normalized_observation: np.ndarray) -> np.ndarray:
        activation = normalized_observation
        for weight, bias in zip(self.hidden_weights, self.hidden_biases):
            activation = np.maximum(0.0, activation @ weight.T + bias)

        action_mean = activation @ self.mu_weight.T + self.mu_bias
        return np.tanh(action_mean)

    # ── Inference ───────────────────────────────────────────────
    def act(self, observation):
        """
        Select an action given a raw environment observation.

        Parameters
        ----------
        observation : array-like, shape (4,)
            [pressure, temp, target_pressure, target_temp], values in [0, 100]

        Returns
        -------
        action : np.ndarray, shape (3,)
            [inlet_valve, outlet_valve, heater_power], values in [0, 1]
        """
        normalized_observation = np.asarray(observation, dtype=np.float32) / OBSERVATION_SCALE
        squashed_action = self._forward_policy(normalized_observation)
        action = (squashed_action + 1.0) / 2.0

        return np.clip(action, 0.0, 1.0).astype(np.float32)

    # ── Custom Reward Function ──────────────────────────────────
    def reward_function(self, state, action, next_state, terminated, truncated):
        """Shaped reward used during SAC training."""
        del action, truncated
        pressure, temp, target_pressure, target_temp = state
        next_pressure, next_temp, _, _ = next_state

        p_error = abs(next_pressure - target_pressure)
        t_error = abs(next_temp - target_temp)
        prev_p_error = abs(pressure - target_pressure)
        prev_t_error = abs(temp - target_temp)

        reward = -(p_error + t_error)
        reward += 2.0 * ((prev_p_error - p_error) + (prev_t_error - t_error))

        if p_error < 2.0:
            reward += 3.0
        if t_error < 2.0:
            reward += 3.0
        if p_error < 0.5 and t_error < 0.5:
            reward += 10.0

        if next_pressure > 85:
            reward -= 5.0 * (next_pressure - 85)
        if next_temp > 85:
            reward -= 5.0 * (next_temp - 85)

        if terminated:
            reward -= 200.0

        return reward
