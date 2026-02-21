"""
Temperature Scaling Calibration Script

Loads a trained model and the validation set, fits a temperature scalar T
to calibrate confidence, prints NLL and ECE metrics before/after calibration,
and saves the result to a temperature.json file.

Usage:
    python calibrate_temperature.py
    python calibrate_temperature.py --model-path models/best_model.pth \\
                                    --dataset-dir dataset \\
                                    --output models/temperature.json

What temperature scaling does:
    Instead of softmax(logits), the model uses softmax(logits / T).
    T > 1  → softer (less confident) probabilities
    T < 1  → sharper (more confident) probabilities
    T = 1  → no change (original model behaviour)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import json
import argparse
from datetime import datetime
from pathlib import Path

from temperature_scaling import (
    collect_logits_and_labels,
    fit_temperature,
    compute_calibration_metrics,
)
from train_classifier import BirdPlaneSupermanDataset


def load_model(model_path: str, device: str):
    """Load trained ResNet50 model from a checkpoint file.

    Returns:
        model: Model in eval mode on the requested device
        classes: List of class name strings
        config: Training config dict (may be empty)
    """
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint.get("classes", ["bird", "plane", "superman", "other"])
    config = checkpoint.get("config", {})

    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, classes, config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate model confidence using temperature scaling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_model.pth",
        help="Path to trained model checkpoint (.pth)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="dataset",
        help="Root directory of the dataset (must contain a 'val' split)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for the validation forward pass",
    )
    parser.add_argument(
        "--init-temperature",
        type=float,
        default=1.5,
        help="Initial temperature value for L-BFGS optimisation",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/temperature.json",
        help="Output path for the temperature JSON file",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\nLoading model from {args.model_path} ...")
    model, classes, config = load_model(args.model_path, device)
    print(f"  Classes : {classes}")
    if config.get("model"):
        print(f"  Architecture : {config['model']}")

    # ── Load validation set ─────────────────────────────────────────────────
    print(f"\nLoading validation set from {args.dataset_dir} ...")
    val_dataset = BirdPlaneSupermanDataset(args.dataset_dir, split="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ── Collect logits ──────────────────────────────────────────────────────
    print("\nCollecting logits on validation set ...")
    logits, labels = collect_logits_and_labels(model, val_loader, device)
    print(f"  Samples  : {len(labels)}")
    print(f"  Classes  : {logits.shape[1]}")

    # ── Fit temperature ─────────────────────────────────────────────────────
    print(f"\nFitting temperature with L-BFGS (init T={args.init_temperature}) ...")
    temperature = fit_temperature(logits, labels, init_temperature=args.init_temperature)

    print(f"\n  ╔══════════════════════════════╗")
    print(f"  ║  Optimal Temperature  T = {temperature:>6.4f} ║")
    print(f"  ╚══════════════════════════════╝")

    if temperature > 1.0:
        print(f"  → T > 1: model was overconfident; probabilities softened")
    elif temperature < 1.0:
        print(f"  → T < 1: model was underconfident; probabilities sharpened")
    else:
        print(f"  → T = 1: model is already well-calibrated")

    # ── Calibration metrics ─────────────────────────────────────────────────
    metrics = compute_calibration_metrics(logits, labels, temperature)

    print()
    print("=" * 52)
    print("  CALIBRATION METRICS")
    print("=" * 52)
    print(f"  {'Metric':<22} {'Before':>12} {'After':>12}")
    print(f"  {'-'*48}")
    print(
        f"  {'NLL (↓ = better)':<22} "
        f"{metrics['nll_before']:>12.6f} "
        f"{metrics['nll_after']:>12.6f}"
    )
    print(
        f"  {'ECE (↓ = better)':<22} "
        f"{metrics['ece_before']:>12.6f} "
        f"{metrics['ece_after']:>12.6f}"
    )
    print("=" * 52)

    delta_nll = metrics["nll_after"] - metrics["nll_before"]
    delta_ece = metrics["ece_after"] - metrics["ece_before"]
    nll_tag = "improved" if delta_nll < 0 else "worse"
    ece_tag = "improved" if delta_ece < 0 else "worse"
    print(f"\n  NLL change : {delta_nll:+.6f}  ({nll_tag})")
    print(f"  ECE change : {delta_ece:+.6f}  ({ece_tag})")

    # ── Save temperature.json ───────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "temperature": temperature,
        "calibration_date": datetime.now().isoformat(),
        "model_path": str(args.model_path),
        "dataset_dir": str(args.dataset_dir),
        "num_val_samples": int(len(labels)),
        "classes": classes,
        "metrics": {
            "nll_before": float(metrics["nll_before"]),
            "nll_after": float(metrics["nll_after"]),
            "ece_before": float(metrics["ece_before"]),
            "ece_after": float(metrics["ece_after"]),
            "nll_delta": float(delta_nll),
            "ece_delta": float(delta_ece),
        },
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved temperature calibration → {output_path}")
    print(
        "\nNext step: inference will automatically load this file and apply "
        "softmax(logits / T) for calibrated probabilities."
    )


if __name__ == "__main__":
    main()
