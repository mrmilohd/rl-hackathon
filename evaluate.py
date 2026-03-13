"""
evaluate.py — BELLATRIX Hackathon Evaluation Script
=====================================================
Tests a submitted agent against the MysteryControlEnv across
multiple randomized seeds and reports the hackathon score.

Scoring metric:
    score = Σ_{step=0}^{199} -(|pressure_error| + |temp_error|)
    Higher (less negative) is better.  Hardcoded best ≈ -300.

Usage:
    python evaluate.py                        # default: 50 episodes
    python evaluate.py --episodes 100         # more episodes
    python evaluate.py --verbose              # per-episode printout
    python evaluate.py --seed_start 1000      # custom seed range
"""

import numpy as np
import argparse
import time
from environment import MysteryControlEnv
from agent import MySmartAgent


def evaluate(n_episodes=50, seed_start=0, verbose=False):
    """
    Run the agent for n_episodes, each with a different seed.
    Returns per-episode scores and summary statistics.
    """
    env = MysteryControlEnv()
    agent = MySmartAgent(env.action_space, env.observation_space)

    scores = []
    completions = 0
    total_steps_survived = 0

    print(f"\nEvaluating over {n_episodes} episodes "
          f"(seeds {seed_start}–{seed_start + n_episodes - 1})...\n")

    if verbose:
        print(f"{'Ep':>4} | {'Score':>10} | {'Steps':>5} | {'Status':>8} | "
              f"{'Final P_err':>10} | {'Final T_err':>10}")
        print("-" * 65)

    start_time = time.time()

    for ep in range(n_episodes):
        seed = seed_start + ep
        obs, _ = env.reset(seed=seed)
        episode_score = 0.0
        terminated_early = False
        steps_survived = 0

        for step in range(200):
            action = agent.act(obs)

            # Validate action bounds
            action = np.clip(action, 0.0, 1.0).astype(np.float32)

            next_obs, reward, terminated, truncated, info = env.step(action)

            # ── Hackathon scoring (raw metric) ──
            p, t, tp, tt = next_obs
            p_err = abs(p - tp)
            t_err = abs(t - tt)
            episode_score += -(p_err + t_err)
            steps_survived = step + 1

            obs = next_obs

            if terminated:
                # Penalize remaining steps as if error stayed constant
                remaining = 200 - step - 1
                episode_score += remaining * -(p_err + t_err)
                terminated_early = True
                break

        if not terminated_early:
            completions += 1
        total_steps_survived += steps_survived
        scores.append(episode_score)

        if verbose:
            status = "DIED" if terminated_early else "OK"
            print(f"{ep:>4} | {episode_score:>10.1f} | {steps_survived:>5} | "
                  f"{status:>8} | {p_err:>10.2f} | {t_err:>10.2f}")

    elapsed = time.time() - start_time
    scores = np.array(scores)

    # ── Summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Episodes:          {n_episodes}")
    print(f"  Mean Score:        {scores.mean():.2f} ± {scores.std():.2f}")
    print(f"  Median Score:      {np.median(scores):.2f}")
    print(f"  Best Episode:      {scores.max():.2f}")
    print(f"  Worst Episode:     {scores.min():.2f}")
    print(f"  25th Percentile:   {np.percentile(scores, 25):.2f}")
    print(f"  75th Percentile:   {np.percentile(scores, 75):.2f}")
    print(f"  Completion Rate:   {completions}/{n_episodes} "
          f"({completions / n_episodes * 100:.1f}%)")
    print(f"  Avg Steps Survived:{total_steps_survived / n_episodes:.1f} / 200")
    print(f"  Eval Time:         {elapsed:.1f}s "
          f"({elapsed / n_episodes * 1000:.1f}ms per episode)")
    print(f"{'='*60}")

    env.close()
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BELLATRIX Evaluation")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed_start", type=int, default=0,
                        help="Starting seed for reproducibility")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-episode scores")
    args = parser.parse_args()

    evaluate(
        n_episodes=args.episodes,
        seed_start=args.seed_start,
        verbose=args.verbose,
    )
