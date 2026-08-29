from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from examples.ahc015.python.afterstate_features import encode_afterstates
from examples.ahc015.python.afterstate_model import AfterstatePpoNet, AfterstateValueNet
from examples.ahc015.python.afterstate_ppo import collect_afterstate_ppo_rollout
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
    empty_board,
    place_at_rank,
    place_at_ranks,
    tilt,
    tilt_batch,
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
    assert np.all(np.isfinite(rollout.advantages))
    assert np.all((result.potentials >= 0) & (result.potentials <= 1))
    assert metrics["rollout/mean_score"] > 0


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
