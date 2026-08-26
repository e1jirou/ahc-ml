from __future__ import annotations

import torch
from torch import nn

from .features import BOARD_CHANNELS, FUTURE_CHANNELS, FUTURE_LENGTH

STUDENT_CHANNELS = 128
STUDENT_RESIDUAL_BLOCKS = 8
TEACHER_CHANNELS = 256
TEACHER_RESIDUAL_BLOCKS = 10


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
    """Predicts the residual G(W) from board and known-future inputs."""

    def __init__(
        self,
        channels: int = STUDENT_CHANNELS,
        residual_blocks: int = STUDENT_RESIDUAL_BLOCKS,
    ) -> None:
        super().__init__()
        if channels <= 0 or residual_blocks <= 0:
            raise ValueError("channels and residual_blocks must be positive")
        self.channels = channels
        self.residual_blocks = residual_blocks
        self.board_stem = nn.Conv2d(BOARD_CHANNELS, channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            ResidualDepthwiseBlock(channels) for _ in range(residual_blocks)
        )
        self.future_fc1 = nn.Linear(FUTURE_CHANNELS * FUTURE_LENGTH, channels)
        self.future_fc2 = nn.Linear(channels, channels)
        self.film = nn.Linear(channels, channels * 2)
        self.output = nn.Linear(channels, 1)
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

    def forward(
        self,
        board_inputs: torch.Tensor,
        future_inputs: torch.Tensor,
    ) -> torch.Tensor:
        future = future_inputs.flatten(start_dim=1)
        future = torch.relu(self.future_fc1(future))
        future = torch.relu(self.future_fc2(future))

        gamma, beta = self.film(future).chunk(2, dim=1)
        board = torch.relu(self.board_stem(board_inputs))
        board = board * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        for block in self.blocks:
            board = block(board)
        board = board.mean(dim=(2, 3))
        return self.output(board).squeeze(1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def dimensions_from_state_dict(
    state_dict: dict[str, torch.Tensor], *, prefix: str = ""
) -> tuple[int, int]:
    channels = int(state_dict[f"{prefix}board_stem.weight"].shape[0])
    block_indices = {
        int(name.split(".")[len(prefix.split(".")) if prefix else 1])
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
TEACHER_PARAMETER_COUNT = parameter_count(Ahc015ValueNet(TEACHER_CHANNELS, TEACHER_RESIDUAL_BLOCKS))


class Ahc015PpoNet(nn.Module):
    """Actor residual and a permutation-invariant critic over four afterstates."""

    def __init__(
        self,
        channels: int = STUDENT_CHANNELS,
        residual_blocks: int = STUDENT_RESIDUAL_BLOCKS,
    ) -> None:
        super().__init__()
        self.actor = Ahc015ValueNet(channels, residual_blocks)
        self.critic = Ahc015ValueNet(channels, residual_blocks)

    def policy_logits(
        self,
        candidate_boards: torch.Tensor,
        candidate_futures: torch.Tensor,
        candidate_potentials: torch.Tensor,
        logit_scale: float,
    ) -> torch.Tensor:
        batch_size, action_count = candidate_boards.shape[:2]
        residuals = self.actor(
            candidate_boards.flatten(0, 1), candidate_futures.flatten(0, 1)
        ).reshape(batch_size, action_count)
        return logit_scale * (candidate_potentials + residuals)

    def state_values(
        self,
        candidate_boards: torch.Tensor,
        candidate_futures: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, action_count = candidate_boards.shape[:2]
        action_values = self.critic(
            candidate_boards.flatten(0, 1), candidate_futures.flatten(0, 1)
        ).reshape(batch_size, action_count)
        return action_values.mean(dim=1)

    def forward(
        self,
        candidate_boards: torch.Tensor,
        candidate_futures: torch.Tensor,
        candidate_potentials: torch.Tensor,
        logit_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.policy_logits(
                candidate_boards,
                candidate_futures,
                candidate_potentials,
                logit_scale,
            ),
            self.state_values(candidate_boards, candidate_futures),
        )
