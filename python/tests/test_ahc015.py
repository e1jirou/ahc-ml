from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from examples.ahc015.python.config import load_config
from examples.ahc015.python.features import (
    BOARD_CHANNELS,
    FUTURE_CHANNELS,
    FUTURE_LENGTH,
    POTENTIAL_CHANNEL,
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
from examples.ahc015.python.model import (
    STUDENT_CHANNELS,
    STUDENT_PARAMETER_COUNT,
    STUDENT_RESIDUAL_BLOCKS,
    TEACHER_CHANNELS,
    TEACHER_PARAMETER_COUNT,
    TEACHER_RESIDUAL_BLOCKS,
    Ahc015PpoNet,
    Ahc015ValueNet,
    parameter_count,
)
from examples.ahc015.python.ppo import (
    PpoRollout,
    collect_ppo_rollout,
    generalized_advantages,
    ppo_update,
)


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
    boards, futures = encode_afterstates(candidates, np.arange(ACTION_COUNT), 1, flavors)
    assert boards.shape == (ACTION_COUNT, BOARD_CHANNELS, SIDE, SIDE)
    assert futures.shape == (ACTION_COUNT, FUTURE_CHANNELS, FUTURE_LENGTH)
    assert boards.dtype == np.float32
    assert futures.dtype == np.float32
    assert np.count_nonzero(futures[:, :, 0]) == 0
    assert np.all(futures[:, :, 1:].sum(axis=1) == 1)


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
    assert parameter_count(model) == STUDENT_PARAMETER_COUNT
    board_inputs = torch.randn(2, BOARD_CHANNELS, SIDE, SIDE)
    future_inputs = torch.randn(2, FUTURE_CHANNELS, FUTURE_LENGTH)
    with torch.inference_mode():
        outputs = model(board_inputs, future_inputs)
    assert outputs.shape == (2,)
    assert torch.equal(outputs, torch.zeros(2))


def test_future_information_merges_only_through_film() -> None:
    torch.manual_seed(1)
    model = Ahc015ValueNet().eval()
    torch.nn.init.normal_(model.output.weight)
    board_inputs = torch.randn(2, BOARD_CHANNELS, SIDE, SIDE)
    future_inputs = torch.randn(2, FUTURE_CHANNELS, FUTURE_LENGTH)
    with torch.inference_mode():
        actual = model(board_inputs, future_inputs)
        future = torch.relu(model.future_fc1(future_inputs.flatten(start_dim=1)))
        future = torch.relu(model.future_fc2(future))
        gamma, beta = model.film(future).chunk(2, dim=1)
        board = torch.relu(model.board_stem(board_inputs))
        board = board * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        for block in model.blocks:
            board = block(board)
        board = board.mean(dim=(2, 3))
        expected = model.output(board).squeeze(1)
    assert torch.equal(actual, expected)


def test_teacher_is_about_four_times_the_student() -> None:
    assert TEACHER_PARAMETER_COUNT / STUDENT_PARAMETER_COUNT == pytest.approx(4.0, rel=0.1)


def test_config_and_ppo_model_shapes() -> None:
    config_directory = Path(__file__).parents[2] / "examples" / "ahc015"
    config = load_config(config_directory / "config.toml")
    assert config.training.max_hours == 10.0
    assert config.model.channels == TEACHER_CHANNELS
    assert config.model.residual_blocks == TEACHER_RESIDUAL_BLOCKS
    assert config.training.rollout_episodes == 4096
    assert config.training.batch_size == 1024
    assert config.training.learning_rate == 3e-4
    assert config.ppo.gamma == 1.0
    fine_tune_config = load_config(config_directory / "config_finetune.toml")
    assert fine_tune_config.training.max_hours == 2.0
    assert fine_tune_config.training.learning_rate == 2.5e-4
    continue_config = load_config(config_directory / "config_continue.toml")
    assert continue_config.run.seed == 15020
    assert continue_config.training.max_hours == 10.0
    assert continue_config.training.learning_rate == 2.5e-4
    rollout_config = load_config(config_directory / "config_rollout1024.toml")
    assert rollout_config.run.seed == 15021
    assert rollout_config.training.max_hours == 8.0
    assert rollout_config.training.rollout_episodes == 1024
    assert rollout_config.training.batch_size == 1024
    assert rollout_config.training.epochs == 2
    higher_lr_config = load_config(config_directory / "config_rollout1024_lr3e4.toml")
    assert higher_lr_config.run.seed == 15022
    assert higher_lr_config.training.max_hours == 10.0
    assert higher_lr_config.training.rollout_episodes == 1024
    assert higher_lr_config.training.batch_size == 1024
    assert higher_lr_config.training.epochs == 2
    assert higher_lr_config.training.learning_rate == 3e-4
    continued_config = load_config(config_directory / "config_rollout1024_continue_lr3e4.toml")
    assert continued_config.run.seed == 15023
    assert continued_config.training.max_hours == 10.0
    assert continued_config.training.rollout_episodes == 1024
    assert continued_config.training.batch_size == 1024
    assert continued_config.training.epochs == 2
    assert continued_config.training.learning_rate == 3e-4
    continued_again_config = load_config(
        config_directory / "config_rollout1024_continue2_lr3e4.toml"
    )
    assert continued_again_config.run.seed == 15024
    assert continued_again_config.training.max_hours == 10.0
    assert continued_again_config.training.rollout_episodes == 1024
    assert continued_again_config.training.batch_size == 1024
    assert continued_again_config.training.epochs == 2
    assert continued_again_config.training.learning_rate == 3e-4
    film_config = load_config(config_directory / "config_film.toml")
    assert film_config.run.seed == 15025
    assert film_config.training.max_hours == 10.0
    assert film_config.training.rollout_episodes == 1024
    assert film_config.training.batch_size == 1024
    assert film_config.training.epochs == 2
    assert film_config.training.learning_rate == 3e-4
    rollout4096_config = load_config(config_directory / "config_rollout4096.toml")
    assert rollout4096_config.run.seed == 15026
    assert rollout4096_config.training.rollout_episodes == 4096
    assert rollout4096_config.training.batch_size == 1024
    assert rollout4096_config.training.epochs == 1
    assert rollout4096_config.evaluation.interval == 1

    model = Ahc015PpoNet(STUDENT_CHANNELS, STUDENT_RESIDUAL_BLOCKS).eval()
    boards = torch.randn(2, ACTION_COUNT, BOARD_CHANNELS, SIDE, SIDE)
    futures = torch.randn(2, ACTION_COUNT, FUTURE_CHANNELS, FUTURE_LENGTH)
    potentials = torch.rand(2, ACTION_COUNT)
    with torch.inference_mode():
        logits, values = model(boards, futures, potentials, config.ppo.logit_scale)
    assert logits.shape == (2, ACTION_COUNT)
    assert values.shape == (2,)
    assert torch.allclose(logits, config.ppo.logit_scale * potentials)
    assert torch.equal(values, torch.zeros(2))


def test_generalized_advantages_terminal_and_shape() -> None:
    rewards = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)
    values = np.zeros_like(rewards)
    advantages, returns = generalized_advantages(rewards, values, gamma=1.0, gae_lambda=1.0)
    assert np.allclose(advantages, [[0.6, 0.5, 0.3]])
    assert np.array_equal(advantages, returns)


def test_ppo_rollout_and_update_smoke() -> None:
    torch.manual_seed(3)
    model = Ahc015PpoNet()
    torch.nn.init.normal_(model.actor.output.weight, std=0.01)
    torch.nn.init.normal_(model.critic.output.weight, std=0.01)
    rng = np.random.default_rng(4)
    rollout, result, metrics = collect_ppo_rollout(
        model,
        torch.device("cpu"),
        episodes=2,
        rng=rng,
        gamma=1.0,
        gae_lambda=0.95,
        logit_scale=12.0,
        inference_batch_size=4,
    )
    assert len(rollout) == 2 * 99
    assert rollout.board_features.shape == (
        2 * 99,
        ACTION_COUNT,
        BOARD_CHANNELS,
        SIDE,
        SIDE,
    )
    assert rollout.future_features.shape == (
        2 * 99,
        ACTION_COUNT,
        FUTURE_CHANNELS,
        FUTURE_LENGTH,
    )
    assert rollout.board_features.dtype == np.uint8
    assert rollout.future_features.dtype == np.uint8
    assert np.count_nonzero(rollout.board_features[:, :, POTENTIAL_CHANNEL]) == 0
    assert np.all(np.isfinite(rollout.advantages))
    assert np.all((result.potentials >= 0) & (result.potentials <= 1))
    assert metrics["rollout/mean_score"] > 0

    restored_boards = torch.from_numpy(rollout.board_features[:8]).float() / CELL_COUNT
    restored_boards[:, :, POTENTIAL_CHANNEL] = torch.from_numpy(rollout.candidate_potentials[:8])[
        :, :, None, None
    ]
    restored_futures = torch.from_numpy(rollout.future_features[:8]).float()
    with torch.inference_mode():
        logits, old_values = model(
            restored_boards,
            restored_futures,
            torch.from_numpy(rollout.candidate_potentials[:8]),
            12.0,
        )
        old_log_probs = torch.log_softmax(logits, dim=1).gather(
            1, torch.from_numpy(rollout.actions[:8, None])
        )
    assert np.allclose(old_log_probs[:, 0], rollout.old_log_probs[:8], atol=1e-6)
    assert np.allclose(old_values, rollout.old_values[:8], atol=1e-6)

    update_rollout = PpoRollout(
        board_features=rollout.board_features[:4],
        future_features=rollout.future_features[:4],
        candidate_potentials=rollout.candidate_potentials[:4],
        actions=rollout.actions[:4],
        old_log_probs=rollout.old_log_probs[:4],
        old_values=rollout.old_values[:4],
        advantages=rollout.advantages[:4],
        returns=rollout.returns[:4],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = model.actor.output.weight.detach().clone()
    update_metrics = ppo_update(
        model,
        optimizer,
        update_rollout,
        torch.device("cpu"),
        rng,
        epochs=1,
        batch_size=len(update_rollout),
        clip_ratio=0.2,
        value_clip=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        gradient_clip_norm=1.0,
        logit_scale=12.0,
        target_kl=0.03,
    )
    assert update_metrics["training/updates_this_iteration"] == 1
    assert not torch.equal(model.actor.output.weight, before)
