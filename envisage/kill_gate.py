"""Conservative Stage 2 acceptance gate for pipeline v3.

This module provides a pair-wise classifier that looks at the output of
Stage 1 and Stage 2 and predicts whether Stage 2 is a genuine
improvement (higher fidelity to the expected surgical outcome).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class PairResNet18Gate(nn.Module):
    """ResNet-18 binary head over concatenated Stage 1 and Stage 2 RGB images."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.net = resnet18(weights=weights)

        # Replace first conv to accept 6 channels (3 from Stage 1 + 3 from Stage 2)
        old = self.net.conv1
        self.net.conv1 = nn.Conv2d(
            6,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )

        # Initialize new weights by duplicating old ones and halving
        with torch.no_grad():
            self.net.conv1.weight[:, :3].copy_(old.weight)
            self.net.conv1.weight[:, 3:].copy_(old.weight)
            self.net.conv1.weight.mul_(0.5)

        # Replace FC with a single logit output
        self.net.fc = nn.Linear(self.net.fc.in_features, 1)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """Forward pass. Expects (B, 6, H, W) tensor."""
        return self.net(pair).flatten(1).squeeze(1)


def load_kill_gate(
    path: str | Path,
    device: str | torch.device | None = None,
) -> PairResNet18Gate:
    """Load a trained kill gate from disk."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    checkpoint = torch.load(Path(path), map_location=device)
    model = PairResNet18Gate(pretrained=False).to(device)
    # Handle both full dicts and state_dicts
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()
    return model


def pair_to_tensor(
    stage1_bgr: np.ndarray,
    stage2_bgr: np.ndarray,
    size: int = 224,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Prepare a pair of BGR images for the gate."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    def _prep(img: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        # ResNet ImageNet normalization
        arr = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (arr - mean) / std

    pair = np.concatenate([_prep(stage1_bgr), _prep(stage2_bgr)], axis=2)
    tensor = torch.from_numpy(pair).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


@torch.no_grad()
def should_keep_stage2(
    stage1_bgr: np.ndarray,
    stage2_bgr: np.ndarray,
    gate: PairResNet18Gate,
    threshold: float = 0.55,
    device: str | torch.device | None = None,
) -> tuple[bool, float, float]:
    """Inference helper for the kill gate.

    Returns:
        (keep_bool, logit, probability)
    """
    tensor = pair_to_tensor(stage1_bgr, stage2_bgr, device=device)
    logit = float(gate(tensor).item())
    probability = float(torch.sigmoid(torch.tensor(logit)).item())
    return probability >= threshold, logit, probability


def load_gate_if_available(
    path: str | Path | None,
    device: str | torch.device | None = None,
) -> PairResNet18Gate | None:
    """Lazy-load helper that returns None if path doesn't exist."""
    if path is None:
        return None
    gate_path = Path(path)
    if not gate_path.exists():
        log.warning("Kill gate model not found at %s; will default to Stage 1 fallback", gate_path)
        return None
    try:
        return load_kill_gate(gate_path, device=device)
    except Exception as e:
        log.error("Failed to load kill gate: %s", e)
        return None
