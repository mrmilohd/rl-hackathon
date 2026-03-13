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

DEFAULT_MODEL_PATH = "best_sac_agent.zip"
DEFAULT_OUTPUT_PATH = "sac_weights.npz"


def resolve_model_path(model_path: str) -> str | None:
    if os.path.exists(model_path):
        return model_path

    alt = model_path.replace(".zip", "")
    if os.path.exists(alt):
        return alt
    if os.path.exists(f"{alt}.zip"):
        return f"{alt}.zip"

    return None


def extract(model_path: str = DEFAULT_MODEL_PATH, output_path: str = DEFAULT_OUTPUT_PATH) -> bool:
    try:
        from stable_baselines3 import SAC
    except ImportError:
        print("ERROR: stable-baselines3 is required to extract weights.")
        print("  pip install stable-baselines3")
        return False

    resolved_model_path = resolve_model_path(model_path)
    if resolved_model_path is None:
        print(f"ERROR: Model file not found: {model_path}")
        return False

    print(f"Loading model from: {resolved_model_path}")
    model = SAC.load(resolved_model_path)
    actor = model.actor

    weights = {}

    # Extract latent_pi (hidden layers: Linear + ReLU pairs)
    layer_count = 0
    for index, layer in enumerate(actor.latent_pi):
        if hasattr(layer, "weight"):
            weights[f"latent_{index}_w"] = layer.weight.detach().cpu().numpy()
            weights[f"latent_{index}_b"] = layer.bias.detach().cpu().numpy()
            layer_count += 1

    # Extract mu (output mean layer)
    weights["mu_w"] = actor.mu.weight.detach().cpu().numpy()
    weights["mu_b"] = actor.mu.bias.detach().cpu().numpy()

    np.savez(output_path, **weights)

    print(f"\nExtracted weights → {output_path}")
    print(f"  Hidden layers: {layer_count}")
    for k, v in weights.items():
        print(f"    {k}: {v.shape} (dtype={v.dtype})")
    print(f"\n  File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"  Ready for submission. No SB3/torch needed at inference.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract numpy weights from a trained SAC model")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    extract(args.model, args.output)


if __name__ == "__main__":
    main()
