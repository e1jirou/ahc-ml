from __future__ import annotations

import torch
from torch import nn


class MnistCnn(nn.Module):
    def __init__(self, channels: tuple[int, int] = (32, 64), hidden_size: int = 128) -> None:
        super().__init__()
        first_channels, second_channels = channels
        self.features = nn.Sequential(
            nn.Conv2d(1, first_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(first_channels, second_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(second_channels * 7 * 7, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))
