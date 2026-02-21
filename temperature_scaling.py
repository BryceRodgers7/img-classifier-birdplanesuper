"""
Temperature Scaling Calibration Module

Post-hoc calibration of neural network confidence using temperature scaling.
Temperature scaling divides logits by a learned scalar T before softmax,
which calibrates the model's confidence without affecting accuracy.

Reference:
    Guo et al., "On Calibration of Modern Neural Networks", ICML 2017
    https://arxiv.org/abs/1706.04599
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def collect_logits_and_labels(
    model: nn.Module,
    val_loader,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect raw logits and ground-truth labels on the validation set.

    Args:
        model: Trained PyTorch model (in eval mode or will be set to eval)
        val_loader: Validation DataLoader (yields (images, labels) or
                    (images, labels, paths) batches)
        device: Device string ('cuda' or 'cpu')

    Returns:
        logits: (N, C) float32 tensor of raw model logits
        labels: (N,) int64 tensor of ground-truth class indices
    """
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch[0], batch[1]
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_logits), torch.cat(all_labels)


class _TemperatureScaler(nn.Module):
    """Single learnable log-temperature parameter, constrained positive via exp."""

    def __init__(self, init_temperature: float = 1.5) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.tensor(float(init_temperature)).log()
        )

    @property
    def temperature(self) -> float:
        return self.log_temperature.exp().item()


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    init_temperature: float = 1.5,
    max_iter: int = 50,
) -> float:
    """
    Fit a single temperature scalar T by minimising NLL (CrossEntropyLoss)
    on logits / T using L-BFGS.

    Args:
        logits: (N, C) float32 tensor of raw model logits (CPU or GPU)
        labels: (N,) int64 tensor of ground-truth class indices
        init_temperature: Starting value for T (default 1.5)
        max_iter: Maximum L-BFGS iterations (default 50)

    Returns:
        Optimal temperature T > 0
    """
    scaler = _TemperatureScaler(init_temperature)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.LBFGS(
        [scaler.log_temperature],
        lr=0.01,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        scaled_logits = logits / scaler.log_temperature.exp()
        loss = criterion(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler.temperature


def compute_ece(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Bins predictions by confidence and measures the mean absolute gap between
    average confidence and observed accuracy within each bin, weighted by the
    fraction of samples in the bin.

    Args:
        probs: (N, C) tensor of calibrated probabilities (rows must sum to 1)
        labels: (N,) tensor of integer ground-truth class indices
        n_bins: Number of equal-width confidence bins (default 15)

    Returns:
        ECE as a float in [0, 1]
    """
    confidences, predictions = probs.max(dim=1)
    correct = predictions.eq(labels).float()

    ece = torch.zeros(1)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = confidences.gt(lower) & confidences.le(upper)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_conf_in_bin = confidences[in_bin].mean()
            ece += (avg_conf_in_bin - accuracy_in_bin).abs() * prop_in_bin

    return ece.item()


def compute_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> float:
    """
    Compute mean Negative Log-Likelihood (cross-entropy) under temperature T.

    Args:
        logits: (N, C) tensor of raw model logits
        labels: (N,) tensor of integer ground-truth class indices
        temperature: Temperature scalar (default 1.0 = uncalibrated)

    Returns:
        Mean NLL as a float
    """
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        loss = criterion(logits / temperature, labels)
    return loss.item()


def compute_calibration_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> dict:
    """
    Compute NLL and ECE before (T=1) and after (T=temperature) calibration.

    Args:
        logits: (N, C) tensor of raw model logits
        labels: (N,) tensor of integer ground-truth class indices
        temperature: Fitted temperature scalar

    Returns:
        Dictionary with keys: nll_before, nll_after, ece_before, ece_after
    """
    with torch.no_grad():
        probs_before = torch.softmax(logits, dim=1)
        probs_after = torch.softmax(logits / temperature, dim=1)

    return {
        "nll_before": compute_nll(logits, labels, temperature=1.0),
        "nll_after": compute_nll(logits, labels, temperature=temperature),
        "ece_before": compute_ece(probs_before, labels),
        "ece_after": compute_ece(probs_after, labels),
    }
