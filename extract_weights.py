"""
extract_weights.py — Run this ONCE on a machine with stable-baselines3 installed.
Converts best_sac_agent.zip → sac_weights.npz (pure numpy, no SB3 needed at inference).

Usage:
    python extract_weights.py                              # default paths
    python extract_weights.py --model best_sac_agent.zip   # custom model path
"""

import numpy as np
import argparse
import os


def extract(model_path="best_sac_agent.zip", output_path="sac_weights.npz"):
    try:
        from stable_baselines3 import SAC
    except ImportError:
        print("ERROR: stable-baselines3 is required to extract weights.")
        print("  pip install stable-baselines3")
        return False

    if not os.path.exists(model_path):
        # Try without .zip extension
        alt = model_path.replace(".zip", "")
        if os.path.exists(alt):
            model_path = alt
        elif os.path.exists(alt + ".zip"):
            model_path = alt + ".zip"
        else:
            print(f"ERROR: Model file not found: {model_path}")
            return False

    print(f"Loading model from: {model_path}")
    model = SAC.load(model_path)
    actor = model.actor

    weights = {}

    # Extract latent_pi (hidden layers: Linear + ReLU pairs)
    layer_count = 0
    for i, layer in enumerate(actor.latent_pi):
        if hasattr(layer, 'weight'):
            weights[f'latent_{i}_w'] = layer.weight.detach().cpu().numpy()
            weights[f'latent_{i}_b'] = layer.bias.detach().cpu().numpy()
            layer_count += 1

    # Extract mu (output mean layer)
    weights['mu_w'] = actor.mu.weight.detach().cpu().numpy()
    weights['mu_b'] = actor.mu.bias.detach().cpu().numpy()

    np.savez(output_path, **weights)

    print(f"\nExtracted weights → {output_path}")
    print(f"  Hidden layers: {layer_count}")
    for k, v in weights.items():
        print(f"    {k}: {v.shape} (dtype={v.dtype})")
    print(f"\n  File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"  Ready for submission. No SB3/torch needed at inference.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="best_sac_agent.zip")
    parser.add_argument("--output", default="sac_weights.npz")
    args = parser.parse_args()
    extract(args.model, args.output)
