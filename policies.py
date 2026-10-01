"""
Custom CNN feature extractors for the (C, H, W) occupancy / prediction grid.

SB3 NatureCNN assumes uint8 image obs and divides by 255 — wrong for our float
grid (categorical occupancy + normalized velocities). Use these extractors with
``CnnPolicy`` via ``policy_kwargs``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GridCnnExtractor(BaseFeaturesExtractor):
    """
    Small CNN for ``(C, 30, 30)`` float grid observations.

    Default ``features_dim=128`` keeps the policy head light for CPU training.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)
        n_input_channels = int(observation_space.shape[0])

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def cnn_policy_kwargs(features_dim: int = 128) -> dict:
    """Policy kwargs for SB3 / sb3-contrib ``CnnPolicy``."""
    return {
        "features_extractor_class": GridCnnExtractor,
        "features_extractor_kwargs": {"features_dim": int(features_dim)},
    }
