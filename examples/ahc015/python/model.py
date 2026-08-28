from __future__ import annotations

import torch
from torch import nn

from .features import BOARD_CHANNELS
from .game import ACTION_COUNT

STUDENT_CHANNELS = 64
STUDENT_RESIDUAL_BLOCKS = 10
TEACHER_CHANNELS = 64
TEACHER_RESIDUAL_BLOCKS = 10


class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = torch.relu(self.depthwise(inputs))
        values = self.pointwise(values)
        return torch.relu(values + inputs)


class Ahc015ValueNet(nn.Module):
    """Predict four action residuals from one normalized pre-tilt board."""

    def __init__(
        self,
        channels: int = STUDENT_CHANNELS,
        residual_blocks: int = STUDENT_RESIDUAL_BLOCKS,
        output_size: int = ACTION_COUNT,
    ) -> None:
        super().__init__()
        if channels <= 0 or residual_blocks <= 0 or output_size <= 0:
            raise ValueError("channels, residual_blocks, and output_size must be positive")
        self.channels = channels
        self.residual_blocks = residual_blocks
        self.board_stem = nn.Conv2d(BOARD_CHANNELS, channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            ResidualDepthwiseBlock(channels) for _ in range(residual_blocks)
        )
        self.output = nn.Linear(channels, output_size)
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
        return self.output(board.mean(dim=(2, 3)))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def dimensions_from_state_dict(
    state_dict: dict[str, torch.Tensor], *, prefix: str = ""
) -> tuple[int, int]:
    channels = int(state_dict[f"{prefix}board_stem.weight"].shape[0])
    block_indices = {
        int(name[len(f"{prefix}blocks.") :].split(".", 1)[0])
        for name in state_dict
        if name.startswith(f"{prefix}blocks.")
    }
    if not block_indices:
        raise ValueError("checkpoint does not contain residual blocks")
    residual_blocks = max(block_indices) + 1
    if block_indices != set(range(residual_blocks)):
        raise ValueError("checkpoint residual block indices are not contiguous")
    return channels, residual_blocks


STUDENT_PARAMETER_COUNT = parameter_count(Ahc015ValueNet())
TEACHER_PARAMETER_COUNT = STUDENT_PARAMETER_COUNT


class Ahc015PpoNet(nn.Module):
    """Actor and critic sharing the compact pre-tilt input format."""

    def __init__(
        self,
        channels: int = STUDENT_CHANNELS,
        residual_blocks: int = STUDENT_RESIDUAL_BLOCKS,
    ) -> None:
        super().__init__()
        self.actor = Ahc015ValueNet(channels, residual_blocks, ACTION_COUNT)
        self.critic = Ahc015ValueNet(channels, residual_blocks, 1)

    def forward(
        self,
        board_inputs: torch.Tensor,
        candidate_potentials: torch.Tensor,
        logit_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = logit_scale * (candidate_potentials + self.actor(board_inputs))
        values = self.critic(board_inputs).squeeze(1)
        return logits, values
