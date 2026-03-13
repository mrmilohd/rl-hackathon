# BELLATRIX Hackathon Submission

## Algorithm: Soft Actor-Critic (SAC)

### Why SAC?
SAC is an off-policy, entropy-regularized actor-critic method well-suited for this
continuous control problem. Key advantages for this environment:

- **Continuous action space**: SAC natively handles the 3D continuous action
  (inlet valve, outlet valve, heater power) without discretization.
- **Sample efficiency**: Off-policy learning with experience replay means fewer
  environment interactions needed — critical for CPU-based training.
- **Entropy regularization**: Automatic temperature tuning encourages exploration
  early in training, then shifts to exploitation as the policy improves. This
  helps avoid local minima where the agent might find a safe but suboptimal control strategy.
- **Stability**: The dual Q-network and soft updates (τ=0.005) prevent the
  catastrophic forgetting common in other deep RL methods.

### Network Architecture
```
Input (4) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(3) → Tanh → scale [0,1]
```
Two hidden layers of 256 units. The Tanh output is rescaled from [-1,1] to [0,1]
to match the action space bounds.

---

## Reward Design

The hackathon metric is `Σ -(|p_error| + |t_error|)` over 200 steps. Our shaped
reward preserves the optimal policy while accelerating learning:

| Component | Formula | Purpose |
|-----------|---------|---------|
| Primary signal | `-(p_error + t_error)` | Aligned with eval metric |
| Potential shaping | `+2.0 × (Δp_error + Δt_error)` | Rewards error reduction |
| Closeness bonus | `+3.0` if error < 2.0 per channel | Precision incentive |
| Precision bonus | `+10.0` if both errors < 0.5 | Near-perfect control |
| Safety margin | `-5.0 × (value - 85)` if > 85 | Soft wall before hard limit |
| Termination | `-200.0` | Strong violation penalty |

The potential-based shaping term is key — it provides dense reward signal during
the ramp-up phase (steps 0–15) where the raw metric gives large constant negatives.

---

## Training Strategy

- **Observation normalization**: All values divided by 100 → [0, 1] range for
  stable gradient computation.
- **Parallel environments**: 4 DummyVecEnv instances for faster data collection.
- **Hyperparameters**: lr=3e-4, buffer=200K, batch=256, γ=0.99, τ=0.005.
- **Training duration**: 500K timesteps on GPU (~30 min) / 300K on CPU (~45 min).
- **Checkpointing**: Best model saved based on hackathon score evaluated every 5K steps.

### Weight Extraction for Submission
After training with stable-baselines3, weights are extracted to a plain numpy
file (`sac_weights.npz`). The submission agent performs inference using only
numpy matrix multiplications — no PyTorch or SB3 dependency at test time.

---

## Observations and Results

### Environment Dynamics (analyzed)
- **Pressure**: `Δp = inlet × 10 - outlet × 8`. Fast actuator (up to +10/step).
  Equilibrium: `outlet = 1.25 × inlet`.
- **Temperature**: `Δt = heater × 5 - 2 × (t/100)`. Self-cooling proportional
  to current temp. At t=70, equilibrium heater ≈ 0.28.
- **Challenge**: Pressure ramps fast and overshoots easily. Temperature is slower
  but more stable. The agent must coordinate all three actuators simultaneously.

### Performance
| Metric | Value |
|--------|-------|
| Mean hackathon score | ~-250 (20-ep eval during training) |
| Completion rate | 100% (no safety violations) |
| Settling time | ~12-15 steps (both errors < 2.0) |
| Training time | ~30 min (GPU) |

### Key Insight
The biggest score penalty comes from the first ~15 steps while the agent ramps
pressure and temperature from initial values (20-40, 20-30) to targets (50-70,
60-80). Once at target, the agent maintains sub-1.0 errors on both channels.
Further optimization should focus on faster ramp-up strategies.

---

## File Structure
```
submission/
├── agent.py              ← Main submission (judges import this)
├── sac_weights.npz       ← Trained model weights (numpy)
├── agent_template.py     ← Base class (provided by organizers)
├── environment.py        ← Environment (provided by organizers)
├── evaluate.py           ← Evaluation script
├── train.py              ← Training script (requires stable-baselines3)
├── extract_weights.py    ← Converts .zip → .npz (one-time)
└── README.md             ← This file
```

## How to Run
```bash
# Evaluate (only needs numpy + gymnasium)
python evaluate.py --episodes 100 --verbose

# Re-train from scratch (needs stable-baselines3)
python train.py
python extract_weights.py
python evaluate.py
```
