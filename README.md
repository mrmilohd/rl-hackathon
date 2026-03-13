# BELLATRIX Hackathon Submission

This repository contains a SAC-based controller for the BELLATRIX control task.

## What Is In This Repo

Core implementation files:

- `agent.py`: submission agent (numpy-only inference).
- `agent_template.py`: organizer base class.
- `environment.py`: environment implementation.
- `wrappers.py`: reusable Gym wrappers.
- `train.py`: SAC training script.
- `extract_weights.py`: exports SB3 model weights to numpy format.
- `evaluate.py`: multi-episode evaluation script.
- `sac_weights.npz`: exported inference weights used by `agent.py`.
- `best_sac_agent.zip`: saved SB3 model checkpoint.
- `config_and_results.txt`: consolidated final metrics/results summary.

Artifacts tracked in this repo:

- `report_figures/`: generated figure set and `report_metrics.txt`.

Project metadata:

- `README.md`
- `requirements.txt`
- `.gitignore`

## Algorithm

The control policy is trained with Soft Actor-Critic (SAC) using a 2-layer MLP:

`[4] -> [256, ReLU] -> [256, ReLU] -> [3, tanh] -> rescale to [0, 1]`

Inference in `agent.py` is pure numpy and does not require stable-baselines3.

## Reward Used For Training

The official task metric is:

`score = sum_{t=0}^{199} -(|p_err| + |t_err|)`

Training uses shaped reward with:

- primary error term `-(p_err + t_err)`
- potential-based shaping `+2.0 * (delta p_err + delta t_err)`
- closeness bonuses (`<2.0`, `<0.5` thresholds)
- soft safety margin penalties beyond 85
- termination penalty `-200`

## Results Source

All reported final metrics should be taken from:

- `config_and_results.txt`

## Usage

Evaluate current agent:

```bash
python evaluate.py --episodes 100 --verbose
```

Retrain and export weights:

```bash
python train.py
python extract_weights.py
python evaluate.py --episodes 100
```
