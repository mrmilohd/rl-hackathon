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
import time

# Force numpy to use all CPU cores for matrix operations
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['OPENBLAS_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())
os.environ['VECLIB_MAXIMUM_THREADS'] = str(os.cpu_count())

# Also set numpy to use optimized BLAS
np.set_printoptions(precision=4, suppress=True)

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

        # ── Load trained weights ────────────────────────────────
        weights_path = os.path.join(os.path.dirname(__file__), "sac_weights.npz")
        if not os.path.exists(weights_path):
            # Fallback: look in current working directory
            weights_path = "sac_weights.npz"
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                "sac_weights.npz not found. "
                "Place it in the same directory as agent.py"
            )

        # Profile single inference
        test_obs = np.random.randn(4).astype(np.float32)
        times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = self.act(test_obs)
            times.append((time.perf_counter() - start) * 1000)
        
        avg_time = np.mean(times[10:])  # Skip first 10 (warmup)
        self.inference_time_ms = avg_time
        
        print(f"✓ Inference time: {avg_time:.3f}ms per step")
        print(f"  → Expected 50 episodes: {avg_time * 200 * 50 / 1000:.1f}s")

        data = np.load(weights_path)

        # Hidden layers (Linear + ReLU pairs)
        self.hidden_w = []
        self.hidden_b = []
        i = 0
        while f"latent_{i}_w" in data:
            self.hidden_w.append(data[f"latent_{i}_w"].astype(np.float32))
            self.hidden_b.append(data[f"latent_{i}_b"].astype(np.float32))
            i += 2  # odd indices are ReLU (no parameters)

        # Output layer (mean action)
        self.mu_w = data["mu_w"].astype(np.float32)
        self.mu_b = data["mu_b"].astype(np.float32)

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
        # Normalize to [0, 1] — same preprocessing used during training
        x = np.asarray(observation, dtype=np.float32) / 100.0

        # Forward pass through hidden layers (using pre-transposed weights)
        for w_T, b in zip(self.hidden_w_T, self.hidden_b):
            x = np.maximum(0.0, x @ w_T + b)  # ReLU activation
    
        # Output layer with tanh squashing
        x = x @ self.mu_w_T + self.mu_b
        x = np.tanh(x)
    
    # Rescale from [-1, 1] → [0, 1]
        action = (x + 1.0) / 2.0
    
        return np.clip(action, 0.0, 1.0).astype(np.float32)


    def __init__(self, action_space, observation_space):
        """Initialize agent with CPU-optimized weight loading."""
        super().__init__(action_space, observation_space)  
        # ── Load trained weights ────────────────────────────────
        weights_path = os.path.join(os.path.dirname(__file__), "sac_weights.npz")
        if not os.path.exists(weights_path):
            weights_path = "sac_weights.npz"
        if not os.path.exists(weights_path):
            raise FileNotFoundError("sac_weights.npz not found.") 
        data = np.load(weights_path)   
    
        # Hidden layers (Linear + ReLU pairs) 
        self.hidden_w = []
        self.hidden_b = []
        i = 0
        while f"latent_{i}_w" in data:
            self.hidden_w.append(data[f"latent_{i}_w"].astype(np.float32))
            self.hidden_b.append(data[f"latent_{i}_b"].astype(np.float32))
            i += 2 
        # Output layer
        self.mu_w = data["mu_w"].astype(np.float32)
        self.mu_b = data["mu_b"].astype(np.float32)
      
        # ✨ CPU OPTIMIZATION: Pre-transpose weights for faster matmul
        self.hidden_w_T = [w.T.copy() for w in self.hidden_w]  # Pre-compute
        self.mu_w_T = self.mu_w.T.copy()
    
        # Profile inference time
        import time
        test_obs = np.random.randn(4).astype(np.float32)
        times = []
        for _ in range(100):
            start = time.perf_counter()
            _ = self.act(test_obs)
            times.append((time.perf_counter() - start) * 1000)
    
        avg_time = np.mean(times[10:])
        print(f"✓ CPU Inference: {avg_time:.3f}ms/step | Expected (50 ep): {avg_time*200*50/1000:.1f}s")

    # ── Custom Reward Function ──────────────────────────────────
    def reward_function(self, state, action, next_state, terminated, truncated):
        """
        Shaped reward used during SAC training.

        Design rationale
        ----------------
        1. Primary signal: negative absolute error (aligned with eval metric).
        2. Potential-based shaping: bonus for *reducing* error step-over-step.
           This doesn't change the optimal policy but accelerates learning.
        3. Closeness bonus: extra reward when both errors are small,
           encouraging precision once the agent is near the target.
        4. Safety margin: soft penalty starting at 85 (well before the
           hard limit at 95) so the agent learns to stay away from walls.
        5. Termination penalty: large negative to strongly discourage
           any trajectory that triggers a safety violation.
        """
        pressure, temp, target_pressure, target_temp = state
        next_pressure, next_temp, _, _ = next_state

        # Current errors
        p_error = abs(next_pressure - target_pressure)
        t_error = abs(next_temp - target_temp)

        # Previous errors (for shaping)
        prev_p_error = abs(pressure - target_pressure)
        prev_t_error = abs(temp - target_temp)

        # ─ Primary: negative absolute error ─
        reward = -(p_error + t_error)

        # ─ Potential-based shaping: reward for reducing error ─
        reward += 2.0 * ((prev_p_error - p_error) + (prev_t_error - t_error))

        # ─ Closeness bonus ─
        if p_error < 2.0:
            reward += 3.0
        if t_error < 2.0:
            reward += 3.0
        if p_error < 0.5 and t_error < 0.5:
            reward += 10.0

        # ─ Safety margin penalty (soft wall at 85) ─
        if next_pressure > 85:
            reward -= 5.0 * (next_pressure - 85)
        if next_temp > 85:
            reward -= 5.0 * (next_temp - 85)

        # ─ Termination penalty ─
        if terminated:
            reward -= 200.0

        return reward

