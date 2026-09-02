from __future__ import annotations

import torch
from torch import nn

from .afterstate_features import FUTURE_CHANNELS, FUTURE_LENGTH
from .features import BOARD_CHANNELS
from .model import ResidualDepthwiseBlock


class AfterstateValueNet(nn.Module):
    """Predict one residual from one action-normalized afterstate."""

    def __init__(
        self,
        channels: int = 64,
        residual_blocks: int = 10,
        future_mode: str = "none",
    ) -> None:
        super().__init__()
        if channels <= 0 or residual_blocks <= 0:
            raise ValueError("channels and residual_blocks must be positive")
        if future_mode not in {"none", "full_add", "full_late"}:
            raise ValueError("future_mode must be none, full_add, or full_late")
        self.channels = channels
        self.residual_blocks = residual_blocks
        self.future_mode = future_mode
        self.board_stem = nn.Conv2d(BOARD_CHANNELS, channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            ResidualDepthwiseBlock(channels) for _ in range(residual_blocks)
        )
        self.future_add = (
            nn.Linear(FUTURE_CHANNELS * FUTURE_LENGTH, channels)
            if future_mode == "full_add"
            else None
        )
        self.future_encoder = (
            nn.Linear(FUTURE_CHANNELS * FUTURE_LENGTH, channels)
            if future_mode == "full_late"
            else None
        )
        self.fusion = (
            nn.Linear(channels * 2, channels) if future_mode == "full_late" else None
        )
        self.correction = (
            nn.Linear(channels, 1) if future_mode == "full_late" else None
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
        if self.future_add is not None:
            nn.init.zeros_(self.future_add.weight)
            nn.init.zeros_(self.future_add.bias)
        if self.correction is not None:
            nn.init.zeros_(self.correction.weight)
            nn.init.zeros_(self.correction.bias)

    def forward(
        self,
        board_inputs: torch.Tensor,
        future_inputs: torch.Tensor | None = None,
        *,
        use_future_correction: bool = True,
    ) -> torch.Tensor:
        board = torch.relu(self.board_stem(board_inputs))
        if self.future_add is not None and use_future_correction:
            if future_inputs is None:
                raise ValueError("future_inputs are required in full_add mode")
            future = self.future_add(future_inputs.flatten(start_dim=1))
            board = board + future[:, :, None, None]
        for block in self.blocks:
            board = block(board)
        board = board.mean(dim=(2, 3))
        result = self.output(board)
        if self.future_encoder is not None and use_future_correction:
            if future_inputs is None:
                raise ValueError("future_inputs are required in full_late mode")
            assert self.fusion is not None and self.correction is not None
            future = torch.relu(
                self.future_encoder(future_inputs.flatten(start_dim=1))
            )
            fused = torch.relu(self.fusion(torch.cat((board, future), dim=1)))
            result = result + self.correction(fused)
        return result.squeeze(1)


class AfterstatePpoNet(nn.Module):
    def __init__(
        self,
        channels: int = 64,
        residual_blocks: int = 10,
        future_mode: str = "none",
    ) -> None:
        super().__init__()
        self.future_mode = future_mode
        self.actor = AfterstateValueNet(channels, residual_blocks, future_mode)
        self.critic = AfterstateValueNet(channels, residual_blocks, future_mode)

    def forward(
        self,
        candidate_boards: torch.Tensor,
        candidate_potentials: torch.Tensor | None,
        logit_scale: float,
        policy_phi_coefficient: float = 1.0,
        future_inputs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, action_count = candidate_boards.shape[:2]
        flat_boards = candidate_boards.flatten(0, 1)
        flat_futures = None
        if self.future_mode != "none":
            if future_inputs is None:
                raise ValueError("future_inputs are required in full_add mode")
            flat_futures = future_inputs[:, None].expand(-1, action_count, -1, -1).flatten(0, 1)
        residuals = self.actor(flat_boards, flat_futures).reshape(batch_size, action_count)
        action_values = self.critic(flat_boards, flat_futures).reshape(batch_size, action_count)
        if policy_phi_coefficient == 0.0:
            policy_values = residuals
        else:
            if candidate_potentials is None:
                raise ValueError("candidate_potentials are required when policy Phi is enabled")
            policy_values = policy_phi_coefficient * candidate_potentials + residuals
        return logit_scale * policy_values, action_values.mean(dim=1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
