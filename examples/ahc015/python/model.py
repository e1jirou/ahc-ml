from __future__ import annotations

import torch
from torch import nn

CHANNELS = 144
RESIDUAL_BLOCKS = 9
PARAMETER_COUNT = 368_209


class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        values = torch.relu(self.depthwise(inputs))
        values = self.pointwise(values)
        return torch.relu(values + residual)


class Ahc015ValueNet(nn.Module):
    """Predicts the residual G(W) from an 18-channel afterstate."""

    def __init__(self) -> None:
        super().__init__()
        self.board_stem = nn.Conv2d(15, CHANNELS, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            ResidualDepthwiseBlock(CHANNELS) for _ in range(RESIDUAL_BLOCKS)
        )
        self.future_fc1 = nn.Linear(3 * 10 * 10, CHANNELS)
        self.future_fc2 = nn.Linear(CHANNELS, CHANNELS)
        self.fusion_fc = nn.Linear(CHANNELS * 2, CHANNELS * 2)
        self.output = nn.Linear(CHANNELS * 2, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Exact phi-greedy behavior before the first update.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        board = torch.relu(self.board_stem(inputs[:, :15]))
        for block in self.blocks:
            board = block(board)
        board = board.mean(dim=(2, 3))

        future = inputs[:, 15:].flatten(start_dim=1)
        future = torch.relu(self.future_fc1(future))
        future = torch.relu(self.future_fc2(future))

        fused = torch.cat((board, future), dim=1)
        fused = torch.relu(self.fusion_fc(fused))
        return self.output(fused).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
