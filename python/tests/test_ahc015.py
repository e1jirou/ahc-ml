from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from examples.ahc015.python.backup import bellman_residual_targets
from examples.ahc015.python.config import load_config
from examples.ahc015.python.features import (
    FEATURE_CHANNELS,
    dynamic_flavor_mapping,
    encode_afterstates,
    normalized_board,
)
from examples.ahc015.python.game import (
    ACTION_COUNT,
    BACK,
    CELL_COUNT,
    FRONT,
    LEFT,
    RIGHT,
    SIDE,
    afterstates,
    connectivity_numerator,
    empty_board,
    place_at_rank,
    tilt,
)
from examples.ahc015.python.model import PARAMETER_COUNT, Ahc015ValueNet, parameter_count
from examples.ahc015.python.replay import ReplayBatch, ReplayBuffer
from examples.ahc015.python.simulation import collect_rollouts


def test_tilts_compact_without_reordering() -> None:
    board = empty_board()
    board[2, 0] = 1
    board[7, 0] = 2
    board[1, 3] = 3
    assert np.array_equal(tilt(board, FRONT)[:2, 0], [1, 2])
    assert np.array_equal(tilt(board, BACK)[-2:, 0], [1, 2])
    assert tilt(board, LEFT)[1, 0] == 3
    assert tilt(board, RIGHT)[1, -1] == 3


def test_rank_is_row_major_and_score_is_component_squared() -> None:
    board = place_at_rank(empty_board(), 1, 1)
    board = tilt(board, RIGHT)
    board = place_at_rank(board, 1, 2)
    assert board[0, 0] == 2
    assert board[0, 9] == 1

    score_board = empty_board()
    score_board[0, :2] = 1
    score_board[2, 2] = 1
    score_board[4:6, 4] = 2
    assert connectivity_numerator(score_board) == 4 + 1 + 4


def test_dynamic_mapping_and_feature_shape() -> None:
    flavors = np.resize(np.array([3, 2, 1], dtype=np.uint8), CELL_COUNT)
    for placed in range(CELL_COUNT):
        mapping = dynamic_flavor_mapping(flavors, placed)
        assert mapping[flavors[placed]] == 1

    board = empty_board()
    board[4, 4] = flavors[0]
    candidates = afterstates(board)
    features = encode_afterstates(candidates, np.arange(ACTION_COUNT), 1, flavors)
    assert features.shape == (ACTION_COUNT, FEATURE_CHANNELS, SIDE, SIDE)
    assert features.dtype == np.float32


def test_reflection_canonicalization() -> None:
    flavors = np.resize(np.array([1, 2, 3], dtype=np.uint8), CELL_COUNT)
    board = empty_board()
    board[0, 8] = 1
    board[1, 7] = 2
    assert np.array_equal(
        normalized_board(board, FRONT, flavors, 3),
        normalized_board(board[:, ::-1], FRONT, flavors, 3),
    )


def test_action_rotation_matches_front_orientation() -> None:
    flavors = np.resize(np.array([1, 2, 3], dtype=np.uint8), CELL_COUNT)
    board = empty_board()
    board[2, 5] = 1
    board[7, 5] = 2
    candidates = afterstates(board)
    canonical = [
        normalized_board(candidates[action], action, flavors, 2)
        for action in (FRONT, BACK, LEFT, RIGHT)
    ]
    assert all(candidate.shape == (SIDE, SIDE) for candidate in canonical)


def test_model_shape_parameter_count_and_zero_residual() -> None:
    torch.manual_seed(0)
    model = Ahc015ValueNet().eval()
    assert parameter_count(model) == PARAMETER_COUNT
    inputs = torch.randn(2, FEATURE_CHANNELS, SIDE, SIDE)
    with torch.inference_mode():
        outputs = model(inputs)
    assert outputs.shape == (2,)
    assert torch.equal(outputs, torch.zeros(2))


class ZeroModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(inputs), device=inputs.device)


def test_rollout_stores_all_actions_and_mc_only_on_selected_actions() -> None:
    replay, result = collect_rollouts(
        ZeroModel(),
        torch.device("cpu"),
        episodes=2,
        rng=np.random.default_rng(7),
        epsilon=0.0,
        inference_batch_size=128,
    )
    assert replay.boards.shape == (2 * 99 * ACTION_COUNT, CELL_COUNT)
    assert np.isfinite(replay.mc_targets).sum() == 2 * 99
    assert np.all((result.potentials >= 0) & (result.potentials <= 1))


def test_terminal_bellman_target_does_not_bootstrap() -> None:
    flavors = np.ones((1, CELL_COUNT), dtype=np.uint8)
    board = np.ones((1, CELL_COUNT), dtype=np.uint8)
    board[0, -1] = 0
    batch = ReplayBatch(
        boards=board,
        actions=np.zeros(1, dtype=np.uint8),
        placed=np.asarray([99], dtype=np.uint8),
        flavors=flavors,
        mc_targets=np.asarray([np.nan], dtype=np.float32),
    )
    targets = bellman_residual_targets(
        batch,
        ZeroModel(),
        ZeroModel(),
        torch.device("cpu"),
        np.random.default_rng(8),
        placement_samples=8,
        enumerate_threshold=16,
        inference_batch_size=128,
    )
    assert np.isclose(targets[0], 1.0 - 99**2 / 100**2)


def test_replay_ring_and_default_config() -> None:
    buffer = ReplayBuffer(capacity=3)
    for placed in (1, 2, 3, 4):
        buffer.add(
            ReplayBatch(
                boards=np.full((1, CELL_COUNT), placed, dtype=np.uint8),
                actions=np.zeros(1, dtype=np.uint8),
                placed=np.asarray([placed], dtype=np.uint8),
                flavors=np.ones((1, CELL_COUNT), dtype=np.uint8),
                mc_targets=np.asarray([np.nan], dtype=np.float32),
            )
        )
    assert len(buffer) == 3
    sampled = buffer.sample(8, np.random.default_rng(9), minimum_placed=3)
    assert np.all(sampled.placed >= 3)

    config = load_config(Path(__file__).parents[2] / "examples" / "ahc015" / "config.toml")
    assert config.backup.placement_samples == 8
    assert config.training.batch_size == 128
    assert config.training.target_update_interval == 1000
    assert config.training.iterations == 300
    assert config.training.max_hours == 10.0
    assert config.wandb.mode == "online"
