from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.ahc015.python import afterstate_ppo as afterstate_ppo_module
from examples.ahc015.python.afterstate_features import (
    FUTURE_CHANNELS,
    FUTURE_LENGTH,
    encode_afterstates,
    encode_future_sequences,
)
from examples.ahc015.python.afterstate_model import (
    AfterstatePpoNet,
    AfterstateValueNet,
    initialize_widened_afterstate_ppo,
)
from examples.ahc015.python.afterstate_parallel_runtime import _create_shared_rollout
from examples.ahc015.python.afterstate_ppo import (
    AfterstatePpoRollout,
    AfterstatePpoRolloutStorage,
    collect_afterstate_ppo_rollout,
    update_afterstate_ppo,
)
from examples.ahc015.python.afterstate_simulation import evaluate_afterstate_policy
from examples.ahc015.python.config import load_config
from examples.ahc015.python.features import (
    BOARD_CHANNELS,
    dynamic_flavor_mapping,
    encode_states,
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
    afterstates_batch,
    connectivity_numerator,
    denominator,
    empty_board,
    place_at_rank,
    place_at_ranks,
    tilt,
    tilt_batch,
)
from examples.ahc015.python.legacy_future import (
    LEGACY_BOARD_CHANNELS,
    LEGACY_FUTURE_CHANNELS,
    LEGACY_FUTURE_LENGTH,
    LegacyFutureValueNet,
    ablate_legacy_futures,
    encode_legacy_afterstates,
)
from examples.ahc015.python.model import (
    STUDENT_CHANNELS,
    STUDENT_PARAMETER_COUNT,
    STUDENT_RESIDUAL_BLOCKS,
    TEACHER_PARAMETER_COUNT,
    TEACHER_RESIDUAL_BLOCKS,
    Ahc015PpoNet,
    Ahc015ValueNet,
    parameter_count,
)
from examples.ahc015.python.ppo import (
    PpoRollout,
    PpoRolloutStorage,
    collect_ppo_rollout,
    generalized_advantages,
    ppo_update,
)
from examples.ahc015.python.simulation import evaluate_policy, generate_cases
from examples.ahc015.python.train import scheduled_coefficient, scheduled_policy_phi_coefficient


def test_tilts_compact_without_reordering() -> None:
    board = empty_board()
    board[2, 0] = 1
    board[7, 0] = 2
    board[1, 3] = 3
    assert np.array_equal(tilt(board, FRONT)[:2, 0], [1, 2])
    assert np.array_equal(tilt(board, BACK)[-2:, 0], [1, 2])
    assert tilt(board, LEFT)[1, 0] == 3
    assert tilt(board, RIGHT)[1, -1] == 3


def test_batched_placement_and_tilts_match_scalar_operations() -> None:
    rng = np.random.default_rng(15028)
    boards = np.zeros((8, SIDE, SIDE), dtype=np.uint8)
    for episode in range(len(boards)):
        cells = rng.choice(CELL_COUNT, size=40, replace=False)
        boards[episode].reshape(-1)[cells] = rng.integers(1, 4, size=len(cells))
    ranks = rng.integers(1, CELL_COUNT - 40 + 1, size=len(boards))
    flavors = rng.integers(1, 4, size=len(boards))

    expected_placements = np.stack(
        [
            place_at_rank(board, int(rank), int(flavor))
            for board, rank, flavor in zip(boards, ranks, flavors, strict=True)
        ]
    )
    assert np.array_equal(place_at_ranks(boards, ranks, flavors), expected_placements)
    for action in range(ACTION_COUNT):
        expected_tilts = np.stack([tilt(board, action) for board in boards])
        assert np.array_equal(tilt_batch(boards, action), expected_tilts)
    assert np.array_equal(
        afterstates_batch(boards),
        np.stack([afterstates(board) for board in boards]),
    )


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
    features, action_permutations = encode_states([board], 1, flavors)
    assert features.shape == (1, BOARD_CHANNELS, SIDE, SIDE)
    assert features.dtype == np.float32
    assert action_permutations.shape == (1, ACTION_COUNT)
    assert sorted(action_permutations[0].tolist()) == list(range(ACTION_COUNT))
    assert np.all(features.sum(axis=1) == 1)


def test_reflection_canonicalization() -> None:
    flavors = np.resize(np.array([1, 2, 3], dtype=np.uint8), CELL_COUNT)
    board = empty_board()
    board[0, 8] = 1
    board[1, 7] = 2
    normalized, _ = normalized_board(board, flavors, 3)
    mirrored, _ = normalized_board(board[:, ::-1], flavors, 3)
    assert np.array_equal(normalized, mirrored)


def test_board_rotation_preserves_action_semantics() -> None:
    flavors = np.resize(np.array([1, 2, 3], dtype=np.uint8), CELL_COUNT)
    board = empty_board()
    board[2, 5] = 1
    board[7, 5] = 2
    normalized, normalized_to_original = normalized_board(board, flavors, 2)
    rotated, rotated_to_original = normalized_board(np.rot90(board), flavors, 2)
    assert np.array_equal(normalized, rotated)
    original_candidates = afterstates(board)
    rotated_candidates = afterstates(np.rot90(board))
    for normalized_action in range(ACTION_COUNT):
        left, _ = normalized_board(
            original_candidates[normalized_to_original[normalized_action]], flavors, 2
        )
        right, _ = normalized_board(
            rotated_candidates[rotated_to_original[normalized_action]], flavors, 2
        )
        assert np.array_equal(left, right)


def test_model_shape_parameter_count_and_zero_residual() -> None:
    torch.manual_seed(0)
    model = Ahc015ValueNet().eval()
    assert parameter_count(model) == STUDENT_PARAMETER_COUNT
    board_inputs = torch.randn(2, BOARD_CHANNELS, SIDE, SIDE)
    with torch.inference_mode():
        outputs = model(board_inputs)
    assert outputs.shape == (2, ACTION_COUNT)
    assert torch.equal(outputs, torch.zeros(2, ACTION_COUNT))


def test_model_is_board_only() -> None:
    torch.manual_seed(1)
    model = Ahc015ValueNet().eval()
    torch.nn.init.normal_(model.output.weight)
    board_inputs = torch.randn(2, BOARD_CHANNELS, SIDE, SIDE)
    with torch.inference_mode():
        actual = model(board_inputs)
        board = torch.relu(model.board_stem(board_inputs))
        for block in model.blocks:
            board = block(board)
        board = board.mean(dim=(2, 3))
        expected = model.output(board)
    assert torch.equal(actual, expected)


def test_teacher_is_about_four_times_the_student() -> None:
    assert TEACHER_PARAMETER_COUNT == STUDENT_PARAMETER_COUNT


def test_config_and_ppo_model_shapes() -> None:
    config_directory = Path(__file__).parents[2] / "examples" / "ahc015"
    config = load_config(config_directory / "config.toml")
    assert config.training.max_hours == 10.0
    assert config.run.device == "mps"
    assert config.model.channels == 64
    assert config.model.residual_blocks == TEACHER_RESIDUAL_BLOCKS
    assert config.training.rollout_episodes == 4096
    assert config.training.batch_size == 1024
    assert config.training.micro_batch_size == 128
    assert config.training.rollout_processes == 1
    assert config.training.learning_rate == 3e-4
    assert config.ppo.gamma == 1.0
    afterstate_config = load_config(config_directory / "config_afterstate.toml")
    assert afterstate_config.model.input_mode == "afterstate"
    assert afterstate_config.model.channels == 64
    assert afterstate_config.model.residual_blocks == 10
    assert afterstate_config.training.max_hours == 5.0
    assert afterstate_config.evaluation.interval == 2
    distill_config = load_config(config_directory / "config_afterstate_128_distill.toml")
    assert distill_config.model.future_mode == "none"
    assert distill_config.model.channels == 128
    assert distill_config.training.learning_rate == 1e-4
    assert distill_config.distillation.coefficient_start == 1.0
    assert distill_config.distillation.coefficient_end == 0.0
    assert distill_config.distillation.anneal_hours == 3.0
    assert distill_config.training.data_parallel
    assert distill_config.training.rollout_processes == 2
    assert distill_config.run.name_prefix == "distill"
    assert afterstate_config.evaluation.episodes == 2048
    no_phi_config = load_config(config_directory / "config_afterstate_no_phi.toml")
    assert no_phi_config.run.seed == 15031
    assert no_phi_config.model.input_mode == "afterstate"
    assert no_phi_config.training.max_hours == 10.0
    assert no_phi_config.ppo.policy_phi_coefficient_start == 1.0
    assert no_phi_config.ppo.policy_phi_coefficient_end == 0.0
    assert no_phi_config.ppo.policy_phi_anneal_hours == 3.0
    assert no_phi_config.evaluation.interval == 2
    assert no_phi_config.evaluation.episodes == 2048
    alpha0_config = load_config(config_directory / "config_afterstate_alpha0.toml")
    assert alpha0_config.run.seed == 15034
    assert alpha0_config.model.channels == 64
    assert alpha0_config.model.residual_blocks == 10
    assert alpha0_config.ppo.policy_phi_coefficient_start == 0.0
    assert alpha0_config.ppo.policy_phi_coefficient_end == 0.0
    assert alpha0_config.ppo.reward_mode == "potential_shaping"
    assert alpha0_config.ppo.gae_lambda == 0.95
    assert not alpha0_config.evaluation.phi_greedy_baseline
    submission_config = load_config(config_directory / "config_afterstate_128.toml")
    assert submission_config.run.seed == 15040
    assert submission_config.model.input_mode == "afterstate"
    assert submission_config.model.future_mode == "none"
    assert submission_config.model.channels == 128
    assert submission_config.model.residual_blocks == 10
    assert submission_config.training.max_hours == 10.0
    assert submission_config.ppo.policy_phi_coefficient_start == 0.0
    assert submission_config.ppo.reward_mode == "potential_shaping"
    assert submission_config.wandb.mode == "online"
    future_add_config = load_config(config_directory / "config_afterstate_future_add.toml")
    assert future_add_config.run.seed == 15036
    assert future_add_config.model.input_mode == "afterstate"
    assert future_add_config.model.future_mode == "full_add"
    assert future_add_config.model.channels == 64
    assert future_add_config.training.max_hours == 10.0
    assert future_add_config.wandb.mode == "online"
    future_late_config = load_config(config_directory / "config_afterstate_future_late.toml")
    assert future_late_config.run.seed == 15037
    assert future_late_config.model.input_mode == "afterstate"
    assert future_late_config.model.future_mode == "full_late"
    assert future_late_config.model.channels == 64
    assert future_late_config.training.max_hours == 10.0
    assert future_late_config.wandb.mode == "online"
    terminal_config = load_config(config_directory / "config_afterstate_terminal.toml")
    assert terminal_config.run.seed == 15033
    assert terminal_config.model.channels == 64
    assert terminal_config.model.residual_blocks == 10
    assert terminal_config.ppo.policy_phi_coefficient_start == 0.0
    assert terminal_config.ppo.policy_phi_coefficient_end == 0.0
    assert terminal_config.ppo.reward_mode == "terminal"
    assert terminal_config.ppo.gae_lambda == 1.0
    assert not terminal_config.evaluation.phi_greedy_baseline
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
    boards = torch.randn(2, BOARD_CHANNELS, SIDE, SIDE)
    potentials = torch.rand(2, ACTION_COUNT)
    with torch.inference_mode():
        logits, values = model(boards, potentials, config.ppo.logit_scale)
    assert logits.shape == (2, ACTION_COUNT)
    assert values.shape == (2,)
    assert torch.allclose(logits, config.ppo.logit_scale * potentials)
    assert torch.equal(values, torch.zeros(2))


def test_distillation_schedule_and_actor_kl() -> None:
    assert scheduled_coefficient(1.0, 0.0, 3.0, 0.0) == 1.0
    assert scheduled_coefficient(1.0, 0.0, 3.0, 1.5) == 0.5
    assert scheduled_coefficient(1.0, 0.0, 3.0, 4.0) == 0.0

    torch.manual_seed(15041)
    student = AfterstatePpoNet(channels=8, residual_blocks=1)
    teacher = AfterstateValueNet(channels=4, residual_blocks=1)
    torch.nn.init.normal_(teacher.output.weight, std=0.1)
    teacher.requires_grad_(False)
    rng = np.random.default_rng(15041)
    size = 2
    rollout = AfterstatePpoRollout(
        board_features=rng.integers(
            0, 2, size=(size, ACTION_COUNT, BOARD_CHANNELS, SIDE, SIDE), dtype=np.uint8
        ),
        future_features=None,
        candidate_potentials=None,
        actions=np.asarray([0, 1], dtype=np.int64),
        old_log_probs=np.full(size, -np.log(ACTION_COUNT), dtype=np.float32),
        old_values=np.zeros(size, dtype=np.float32),
        advantages=np.asarray([1.0, -1.0], dtype=np.float32),
        returns=np.asarray([0.2, 0.4], dtype=np.float32),
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
    metrics = update_afterstate_ppo(
        student,
        optimizer,
        rollout,
        torch.device("cpu"),
        rng,
        epochs=1,
        batch_size=size,
        micro_batch_size=1,
        clip_ratio=0.2,
        value_clip=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.003,
        gradient_clip_norm=1.0,
        logit_scale=12.0,
        target_kl=0.03,
        policy_phi_coefficient=0.0,
        teacher_actor=teacher,
        distillation_coefficient=1.0,
    )
    assert metrics["training/distillation_kl"] > 0
    assert metrics["training/distillation_coefficient"] == 1.0
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_zero_actor_is_exact_phi_greedy_after_normalization() -> None:
    flavors, ranks = generate_cases(4, 15029)
    device = torch.device("cpu")
    expected = evaluate_policy(None, device, flavors, ranks, inference_batch_size=4)
    actual = evaluate_policy(
        Ahc015ValueNet().eval(), device, flavors, ranks, inference_batch_size=4
    )
    assert np.array_equal(actual.scores, expected.scores)


def test_afterstate_features_model_and_phi_greedy() -> None:
    flavors = np.resize(np.array([1, 2, 3], dtype=np.uint8), CELL_COUNT)
    board = empty_board()
    board[3, 4] = 1
    candidates = afterstates(board)
    features = encode_afterstates(candidates, np.arange(ACTION_COUNT), 1, flavors)
    assert features.shape == (ACTION_COUNT, BOARD_CHANNELS, SIDE, SIDE)
    assert np.all(features.sum(axis=1) == 1)

    model = AfterstatePpoNet().eval()
    candidate_inputs = torch.from_numpy(features[None])
    potentials = torch.rand(1, ACTION_COUNT)
    with torch.inference_mode():
        logits, values = model(candidate_inputs, potentials, 12.0)
    assert torch.allclose(logits, 12.0 * potentials)
    assert torch.equal(values, torch.zeros(1))

    evaluation_flavors, ranks = generate_cases(4, 15030)
    device = torch.device("cpu")
    expected = evaluate_afterstate_policy(
        None, device, evaluation_flavors, ranks, inference_batch_size=16
    )
    actual = evaluate_afterstate_policy(
        AfterstateValueNet().eval(),
        device,
        evaluation_flavors,
        ranks,
        inference_batch_size=16,
    )
    assert np.array_equal(actual.scores, expected.scores)


def test_afterstate_policy_phi_coefficient_and_schedule() -> None:
    torch.manual_seed(15031)
    model = AfterstatePpoNet().eval()
    torch.nn.init.normal_(model.actor.output.weight, std=0.01)
    boards = torch.randn(2, ACTION_COUNT, BOARD_CHANNELS, SIDE, SIDE)
    potentials = torch.rand(2, ACTION_COUNT)
    with torch.inference_mode():
        logits_without_phi, _ = model(boards, None, 12.0, 0.0)
        residuals = model.actor(boards.flatten(0, 1)).reshape(2, ACTION_COUNT)
        logits_with_phi, _ = model(boards, potentials, 12.0, 0.5)
    assert torch.allclose(logits_without_phi, 12.0 * residuals)
    assert torch.allclose(logits_with_phi, 12.0 * (0.5 * potentials + residuals))

    config = load_config(
        Path(__file__).parents[2] / "examples" / "ahc015" / "config_afterstate_no_phi.toml"
    )
    assert scheduled_policy_phi_coefficient(config.ppo, 0.0) == 1.0
    assert scheduled_policy_phi_coefficient(config.ppo, 1.5) == 0.5
    assert scheduled_policy_phi_coefficient(config.ppo, 3.0) == 0.0
    assert scheduled_policy_phi_coefficient(config.ppo, 8.0) == 0.0


def test_full_future_encoding_and_zero_initialized_addition() -> None:
    flavors = np.resize(np.array([3, 1, 2], dtype=np.uint8), CELL_COUNT)
    placed = 7
    future = encode_future_sequences(flavors, placed)
    assert future.shape == (1, FUTURE_CHANNELS, FUTURE_LENGTH)
    assert future.dtype == np.uint8
    assert np.all(future[:, :, :placed] == 0)
    assert np.all(future[:, :, placed:].sum(axis=1) == 1)
    assert int(future.sum()) == CELL_COUNT - placed

    torch.manual_seed(15035)
    baseline = AfterstateValueNet().eval()
    future_model = AfterstateValueNet(future_mode="full_add").eval()
    incompatible = future_model.load_state_dict(baseline.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {"future_add.weight", "future_add.bias"}
    assert not incompatible.unexpected_keys
    boards = torch.randn(2, BOARD_CHANNELS, SIDE, SIDE)
    futures = torch.from_numpy(np.repeat(future, 2, axis=0)).float()
    torch.nn.init.normal_(baseline.output.weight)
    future_model.output.load_state_dict(baseline.output.state_dict())
    with torch.inference_mode():
        expected = baseline(boards)
        actual = future_model(boards, futures)
    assert torch.equal(actual, expected)

    ppo_model = AfterstatePpoNet(future_mode="full_add").eval()
    torch.nn.init.normal_(ppo_model.actor.future_add.weight, std=0.1)
    torch.nn.init.normal_(ppo_model.actor.output.weight, std=0.1)
    candidates = torch.randn(2, ACTION_COUNT, BOARD_CHANNELS, SIDE, SIDE)
    future_a = torch.zeros(2, FUTURE_CHANNELS, FUTURE_LENGTH)
    future_b = torch.randn(2, FUTURE_CHANNELS, FUTURE_LENGTH)
    with torch.inference_mode():
        logits_a, _ = ppo_model(candidates, None, 12.0, 0.0, future_inputs=future_a)
        logits_b, _ = ppo_model(candidates, None, 12.0, 0.0, future_inputs=future_b)
    centered_a = logits_a - logits_a[:, :1]
    centered_b = logits_b - logits_b[:, :1]
    assert not torch.allclose(centered_a, centered_b)


def test_full_future_late_fusion_is_residual_and_action_dependent() -> None:
    torch.manual_seed(15037)
    baseline = AfterstateValueNet().eval()
    late_model = AfterstateValueNet(future_mode="full_late").eval()
    incompatible = late_model.load_state_dict(baseline.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {
        "future_encoder.weight",
        "future_encoder.bias",
        "fusion.weight",
        "fusion.bias",
        "correction.weight",
        "correction.bias",
    }
    assert not incompatible.unexpected_keys
    boards = torch.randn(8, BOARD_CHANNELS, SIDE, SIDE)
    futures = torch.randn(8, FUTURE_CHANNELS, FUTURE_LENGTH)
    torch.nn.init.normal_(baseline.output.weight)
    late_model.output.load_state_dict(baseline.output.state_dict())
    with torch.inference_mode():
        expected = baseline(boards)
        actual = late_model(boards, futures)
    assert torch.equal(actual, expected)

    torch.nn.init.normal_(late_model.correction.weight, std=0.1)
    with torch.inference_mode():
        future_a = torch.zeros_like(futures)
        future_b = torch.randn_like(futures)
        scores_a = late_model(boards, future_a)
        scores_b = late_model(boards, future_b)
        correction_off = late_model(
            boards,
            use_future_correction=False,
        )
    assert not torch.allclose(scores_a, scores_b)
    assert torch.equal(correction_off, expected)


def test_legacy_future_encoding_and_ablation_modes() -> None:
    flavors = np.resize(np.array([3, 1, 2], dtype=np.uint8), CELL_COUNT)
    boards = np.repeat(empty_board()[None], ACTION_COUNT, axis=0)
    boards[:, 3, 4] = 3
    actions = np.arange(ACTION_COUNT)
    placed = 7
    denominators = np.repeat(denominator(flavors), ACTION_COUNT)
    board_features, future_features, potentials = encode_legacy_afterstates(
        boards,
        actions,
        placed,
        np.repeat(flavors[None], ACTION_COUNT, axis=0),
        denominators,
    )
    assert board_features.shape == (
        ACTION_COUNT,
        LEGACY_BOARD_CHANNELS,
        SIDE,
        SIDE,
    )
    assert future_features.shape == (
        ACTION_COUNT,
        LEGACY_FUTURE_CHANNELS,
        LEGACY_FUTURE_LENGTH,
    )
    assert np.all(future_features[:, :, : CELL_COUNT - placed].sum(axis=1) == 1)
    assert np.all(future_features[:, :, CELL_COUNT - placed :] == 0)
    assert np.allclose(board_features[:, 14, 0, 0], potentials)

    two_episodes = np.concatenate((future_features, future_features[:, [1, 2, 0]]))
    permutation = np.array([1, 0])
    shifted = ablate_legacy_futures(
        two_episodes,
        "episode_shuffle",
        episode_permutation=permutation,
        remaining_length=CELL_COUNT - placed,
    )
    assert np.array_equal(shifted[:ACTION_COUNT], two_episodes[ACTION_COUNT:])
    zero = ablate_legacy_futures(
        two_episodes,
        "zero",
        remaining_length=CELL_COUNT - placed,
    )
    assert not np.any(zero)
    shuffled = ablate_legacy_futures(
        two_episodes,
        "order_shuffle",
        order_priorities=np.random.default_rng(15041).random((2, CELL_COUNT)),
        remaining_length=CELL_COUNT - placed,
    )
    assert np.array_equal(
        shuffled.reshape(2, ACTION_COUNT, 3, CELL_COUNT)[:, 0].sum(axis=2),
        two_episodes.reshape(2, ACTION_COUNT, 3, CELL_COUNT)[:, 0].sum(axis=2),
    )


def test_legacy_film_model_can_change_actions_with_future_input() -> None:
    torch.manual_seed(15041)
    model = LegacyFutureValueNet(channels=8, residual_blocks=2).eval()
    torch.nn.init.normal_(model.output.weight, std=0.1)
    boards = torch.randn(8, LEGACY_BOARD_CHANNELS, SIDE, SIDE)
    future_a = torch.zeros(8, LEGACY_FUTURE_CHANNELS, LEGACY_FUTURE_LENGTH)
    future_b = torch.randn(2, LEGACY_FUTURE_CHANNELS, LEGACY_FUTURE_LENGTH)
    future_b = future_b.repeat_interleave(ACTION_COUNT, dim=0)
    with torch.inference_mode():
        scores_a = model(boards, future_a).reshape(2, ACTION_COUNT)
        scores_b = model(boards, future_b).reshape(2, ACTION_COUNT)
    centered_a = scores_a - scores_a[:, :1]
    centered_b = scores_b - scores_b[:, :1]
    assert not torch.allclose(centered_a, centered_b)


def test_function_preserving_afterstate_widening_ignores_future_path() -> None:
    torch.manual_seed(15040)
    source = AfterstatePpoNet(channels=4, residual_blocks=2, future_mode="full_late").eval()
    for network in (source.actor, source.critic):
        torch.nn.init.normal_(network.output.weight, std=0.05)
        torch.nn.init.normal_(network.output.bias, std=0.05)
        assert network.correction is not None
        torch.nn.init.normal_(network.correction.weight, std=0.05)

    target = AfterstatePpoNet(channels=8, residual_blocks=2, future_mode="none").eval()
    dimensions = initialize_widened_afterstate_ppo(target, source.state_dict())
    assert dimensions == (4, 2)

    boards = torch.randn(7, BOARD_CHANNELS, SIDE, SIDE)
    with torch.inference_mode():
        expected_actor = source.actor(boards, use_future_correction=False)
        expected_critic = source.critic(boards, use_future_correction=False)
        actual_actor = target.actor(boards)
        actual_critic = target.critic(boards)
    assert torch.allclose(actual_actor, expected_actor, atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual_critic, expected_critic, atol=1e-6, rtol=1e-5)

    first_copy = target.actor.output.weight[:, :4]
    second_copy = target.actor.output.weight[:, 4:]
    assert not torch.equal(first_copy, second_copy)
    assert torch.allclose(first_copy + second_copy, source.actor.output.weight)


def test_afterstate_rollout_smoke() -> None:
    rollout, result, metrics = collect_afterstate_ppo_rollout(
        AfterstatePpoNet(),
        torch.device("cpu"),
        episodes=2,
        rng=np.random.default_rng(15031),
        gamma=1.0,
        gae_lambda=0.95,
        logit_scale=12.0,
        inference_batch_size=8,
    )
    assert len(rollout) == 2 * 99
    assert rollout.board_features.shape == (
        2 * 99,
        ACTION_COUNT,
        BOARD_CHANNELS,
        SIDE,
        SIDE,
    )
    assert np.all(rollout.board_features.sum(axis=2) == 1)
    assert rollout.candidate_potentials is not None
    assert np.all(np.isfinite(rollout.advantages))
    assert np.all((result.potentials >= 0) & (result.potentials <= 1))
    assert metrics["rollout/mean_score"] > 0

    explicit_storage = AfterstatePpoRolloutStorage.empty(
        2,
        future_mode="none",
        policy_phi_coefficient=1.0,
    )
    stored_rollout, stored_result, stored_metrics = collect_afterstate_ppo_rollout(
        AfterstatePpoNet(),
        torch.device("cpu"),
        episodes=2,
        rng=np.random.default_rng(15031),
        gamma=1.0,
        gae_lambda=0.95,
        logit_scale=12.0,
        inference_batch_size=8,
        storage=explicit_storage,
    )
    for field in AfterstatePpoRollout.__dataclass_fields__:
        assert np.array_equal(getattr(stored_rollout, field), getattr(rollout, field))
    assert np.array_equal(stored_result.scores, result.scores)
    assert stored_metrics == metrics

    future_rollout, _, _ = collect_afterstate_ppo_rollout(
        AfterstatePpoNet(future_mode="full_add"),
        torch.device("cpu"),
        episodes=1,
        rng=np.random.default_rng(15035),
        gamma=1.0,
        gae_lambda=0.95,
        logit_scale=12.0,
        inference_batch_size=4,
        future_mode="full_add",
    )
    assert future_rollout.future_features is not None
    assert future_rollout.future_features.shape == (99, FUTURE_CHANNELS, FUTURE_LENGTH)


def test_shared_afterstate_rollout_has_disjoint_worker_storage() -> None:
    shared = _create_shared_rollout(mp.get_context("spawn"), workers=2, episodes_per_worker=1)
    first = shared.worker_storage(0)
    second = shared.worker_storage(1)
    first.actions.fill(1)
    second.actions.fill(2)
    first.board_features.fill(0)
    second.board_features.fill(1)

    rollout = shared.as_rollout()
    assert len(rollout) == 2 * (CELL_COUNT - 1)
    assert np.all(rollout.actions[: CELL_COUNT - 1] == 1)
    assert np.all(rollout.actions[CELL_COUNT - 1 :] == 2)
    assert np.all(rollout.board_features[: CELL_COUNT - 1] == 0)
    assert np.all(rollout.board_features[CELL_COUNT - 1 :] == 1)
    assert rollout.future_features is None
    assert rollout.candidate_potentials is None


def test_afterstate_rollout_and_update_with_only_final_phi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(15032)
    model = AfterstatePpoNet()
    torch.nn.init.normal_(model.actor.output.weight, std=0.01)
    rng = np.random.default_rng(15032)
    original_potential = afterstate_ppo_module.potential
    phi_evaluations = 0

    def counted_potential(board: np.ndarray, score_denominator: int) -> float:
        nonlocal phi_evaluations
        phi_evaluations += 1
        return original_potential(board, score_denominator)

    monkeypatch.setattr(afterstate_ppo_module, "potential", counted_potential)
    rollout, _, metrics = collect_afterstate_ppo_rollout(
        model,
        torch.device("cpu"),
        episodes=1,
        rng=rng,
        gamma=1.0,
        gae_lambda=1.0,
        logit_scale=12.0,
        inference_batch_size=4,
        policy_phi_coefficient=0.0,
        reward_mode="terminal",
    )
    assert rollout.candidate_potentials is None
    assert phi_evaluations == 1
    assert metrics["rollout/state_phi_evaluations"] == 0
    assert metrics["rollout/candidate_phi_evaluations"] == 0
    assert metrics["rollout/final_phi_evaluations"] == 1
    assert np.allclose(rollout.returns, rollout.returns[-1])

    size = 2
    short_rollout = AfterstatePpoRollout(
        board_features=rollout.board_features[:size],
        future_features=None,
        candidate_potentials=None,
        actions=rollout.actions[:size],
        old_log_probs=rollout.old_log_probs[:size],
        old_values=rollout.old_values[:size],
        advantages=rollout.advantages[:size],
        returns=rollout.returns[:size],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    update_metrics = update_afterstate_ppo(
        model,
        optimizer,
        short_rollout,
        torch.device("cpu"),
        rng,
        epochs=1,
        batch_size=size,
        micro_batch_size=1,
        clip_ratio=0.2,
        value_clip=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        gradient_clip_norm=1.0,
        logit_scale=12.0,
        target_kl=0.03,
        policy_phi_coefficient=0.0,
    )
    assert update_metrics["training/updates_this_iteration"] == 1


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
        BOARD_CHANNELS,
        SIDE,
        SIDE,
    )
    assert rollout.board_features.dtype == np.uint8
    assert np.all(rollout.board_features.sum(axis=1) == 1)
    assert np.all(np.isfinite(rollout.advantages))
    assert np.all((result.potentials >= 0) & (result.potentials <= 1))
    assert metrics["rollout/mean_score"] > 0

    explicit_storage = PpoRolloutStorage.empty(2)
    stored_rollout, stored_result, stored_metrics = collect_ppo_rollout(
        model,
        torch.device("cpu"),
        episodes=2,
        rng=np.random.default_rng(4),
        gamma=1.0,
        gae_lambda=0.95,
        logit_scale=12.0,
        inference_batch_size=4,
        storage=explicit_storage,
    )
    for field in PpoRollout.__dataclass_fields__:
        assert np.array_equal(getattr(stored_rollout, field), getattr(rollout, field))
    assert np.array_equal(stored_result.scores, result.scores)
    assert stored_metrics == metrics

    restored_boards = torch.from_numpy(rollout.board_features[:8]).float()
    with torch.inference_mode():
        logits, old_values = model(
            restored_boards,
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
        micro_batch_size=2,
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
