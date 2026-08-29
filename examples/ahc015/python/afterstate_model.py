from __future__ import annotations

import torch
from torch import nn

from .features import BOARD_CHANNELS
from .model import ResidualDepthwiseBlock


class AfterstateValueNet(nn.Module):
    """Predict one residual from one action-normalized afterstate."""

    def __init__(self, channels: int = 64, residual_blocks: int = 10) -> None:
        super().__init__()
        if channels <= 0 or residual_blocks <= 0:
            raise ValueError("channels and residual_blocks must be positive")
        self.channels = channels
        self.residual_blocks = residual_blocks
        self.board_stem = nn.Conv2d(BOARD_CHANNELS, channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            ResidualDepthwiseBlock(channels) for _ in range(residual_blocks)
        )
        self.output = nn.Linear(channels, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, board_inputs: torch.Tensor) -> torch.Tensor:
        board = torch.relu(self.board_stem(board_inputs))
        for block in self.blocks:
            board = block(board)
        return self.output(board.mean(dim=(2, 3))).squeeze(1)


class AfterstatePpoNet(nn.Module):
    def __init__(self, channels: int = 64, residual_blocks: int = 10) -> None:
        super().__init__()
        self.actor = AfterstateValueNet(channels, residual_blocks)
        self.critic = AfterstateValueNet(channels, residual_blocks)

    def forward(
        self,
        candidate_boards: torch.Tensor,
        candidate_potentials: torch.Tensor,
        logit_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, action_count = candidate_boards.shape[:2]
        flat_boards = candidate_boards.flatten(0, 1)
        residuals = self.actor(flat_boards).reshape(batch_size, action_count)
        action_values = self.critic(flat_boards).reshape(batch_size, action_count)
        return (
            logit_scale * (candidate_potentials + residuals),
            action_values.mean(dim=1),
        )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
