mod features;
mod game;
mod generated_mcts_prior;
mod generated_model;

use std::collections::{HashMap, VecDeque};
use std::env;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{self, BufRead, Write};
use std::str::FromStr;
use std::time::{Duration, Instant};

use ahc_ml::{Ahc015ValueNet, decode_base93, read_model, read_model_file};

use crate::features::encode_candidates;
use crate::game::{
    ACTION_COUNT, BACK, Board, CANDY_COUNT, FRONT, Input, LEFT, RIGHT, State, action_char,
    connectivity_numerator, place_on_board_at_rank, tilt,
};

const DEFAULT_EXACT_TURNS: usize = 7;

#[derive(Default)]
struct FastHasher(u64);

impl Hasher for FastHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        const MULTIPLIER: u64 = 0x517c_c1b7_2722_0a95;
        let (chunks, remainder) = bytes.as_chunks::<8>();
        for chunk in chunks {
            self.0 = (self.0.rotate_left(5) ^ u64::from_ne_bytes(*chunk)).wrapping_mul(MULTIPLIER);
        }
        if !remainder.is_empty() {
            let mut tail = [0; 8];
            tail[..remainder.len()].copy_from_slice(remainder);
            self.0 = (self.0.rotate_left(5) ^ u64::from_ne_bytes(tail)).wrapping_mul(MULTIPLIER);
        }
        self.0 ^= bytes.len() as u64;
    }
}

type ExactCache = HashMap<Board, usize, BuildHasherDefault<FastHasher>>;

#[derive(Clone, Copy, PartialEq)]
enum McStrategy {
    Equal,
    Halving,
}

#[derive(Clone, Copy, PartialEq)]
enum EndgameSearch {
    MonteCarlo,
    Mcts,
}

#[derive(Clone, Copy, PartialEq)]
enum MctsPrior {
    Uniform,
    Connectivity,
    TinyNn,
}

struct SearchSettings {
    exact_turns: usize,
    mc_turns: usize,
    mc_actions: usize,
    mc_samples: u64,
    mc_min_gain: f64,
    mc_strategy: McStrategy,
    mc_stratified_turns: usize,
    mc_exact_last_action: bool,
    endgame_search: EndgameSearch,
    mcts_simulations: u64,
    mcts_exploration: f64,
    mcts_prior: MctsPrior,
    mcts_early_prior: MctsPrior,
    mcts_rollout_depth: usize,
    mcts_rollout_cutoff_until: usize,
    mcts_tail_repair_turns: usize,
    mcts_tail_repair_passes: usize,
    mcts_early_simulations: u64,
    mcts_early_min_gain: f64,
    time_limit: Duration,
    time_reserve: Duration,
}

struct TokenReader<R> {
    reader: R,
    pending: VecDeque<String>,
}

impl<R: BufRead> TokenReader<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            pending: VecDeque::new(),
        }
    }

    fn read<T: FromStr>(&mut self) -> Result<T, String> {
        loop {
            if let Some(token) = self.pending.pop_front() {
                return token
                    .parse()
                    .map_err(|_| format!("could not parse input token: {token}"));
            }
            let mut line = String::new();
            if self
                .reader
                .read_line(&mut line)
                .map_err(|error| format!("failed to read input: {error}"))?
                == 0
            {
                return Err("unexpected end of input".to_string());
            }
            self.pending
                .extend(line.split_whitespace().map(str::to_owned));
        }
    }
}

fn load_model() -> Result<(Option<Ahc015ValueNet>, SearchSettings), String> {
    let mut arguments = env::args().skip(1);
    let mut model_path = None;
    let mut exact_turns = DEFAULT_EXACT_TURNS;
    let mut mc_turns = 20;
    let mut mc_actions = 4;
    let mut mc_samples = 96;
    let mut mc_min_gain = 20.0;
    let mut mc_strategy = McStrategy::Equal;
    let mut mc_stratified_turns = 2;
    let mut mc_exact_last_action = true;
    let mut endgame_search = EndgameSearch::Mcts;
    let mut mcts_simulations = 2560;
    let mut mcts_exploration = 700.0;
    let mut mcts_prior = MctsPrior::Connectivity;
    let mut mcts_early_prior = MctsPrior::Connectivity;
    let mut mcts_rollout_depth = 0;
    let mut mcts_rollout_cutoff_until = CANDY_COUNT;
    let mut mcts_tail_repair_turns = 2;
    let mut mcts_tail_repair_passes = 1;
    let mut mcts_early_simulations = 0;
    let mut mcts_early_min_gain = 20.0;
    let mut time_limit_ms = 1400;
    let mut time_reserve_ms = 300;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--model" => {
                model_path = Some(
                    arguments
                        .next()
                        .ok_or_else(|| "--model requires a path".to_string())?,
                );
            }
            "--exact-turns" => {
                exact_turns = arguments
                    .next()
                    .ok_or_else(|| "--exact-turns requires an integer".to_string())?
                    .parse()
                    .map_err(|_| "--exact-turns requires an integer".to_string())?;
                if exact_turns > 10 {
                    return Err("--exact-turns must be at most 10".to_string());
                }
            }
            "--mc-turns" => {
                mc_turns = parse_next(&mut arguments, "--mc-turns")?;
                if mc_turns > CANDY_COUNT {
                    return Err("--mc-turns must be at most 100".to_string());
                }
            }
            "--mc-actions" => {
                mc_actions = parse_next(&mut arguments, "--mc-actions")?;
                if !(1..=ACTION_COUNT).contains(&mc_actions) {
                    return Err("--mc-actions must be in 1..=4".to_string());
                }
            }
            "--mc-samples" => {
                mc_samples = parse_next(&mut arguments, "--mc-samples")?;
            }
            "--mc-min-gain" => {
                mc_min_gain = parse_next(&mut arguments, "--mc-min-gain")?;
                if mc_min_gain < 0.0 {
                    return Err("--mc-min-gain must be nonnegative".to_string());
                }
            }
            "--mc-strategy" => {
                mc_strategy = match arguments.next().as_deref() {
                    Some("equal") => McStrategy::Equal,
                    Some("halving") => McStrategy::Halving,
                    _ => return Err("--mc-strategy must be equal or halving".to_string()),
                };
            }
            "--mc-stratified-turns" => {
                mc_stratified_turns = parse_next(&mut arguments, "--mc-stratified-turns")?;
                if mc_stratified_turns > 3 {
                    return Err("--mc-stratified-turns must be in 0..=3".to_string());
                }
            }
            "--mc-exact-last-action" => {
                mc_exact_last_action = match arguments.next().as_deref() {
                    Some("0") => false,
                    Some("1") => true,
                    _ => return Err("--mc-exact-last-action must be 0 or 1".to_string()),
                };
            }
            "--endgame-search" => {
                endgame_search = match arguments.next().as_deref() {
                    Some("mc") => EndgameSearch::MonteCarlo,
                    Some("mcts") => EndgameSearch::Mcts,
                    _ => return Err("--endgame-search must be mc or mcts".to_string()),
                };
            }
            "--mcts-simulations" => {
                mcts_simulations = parse_next(&mut arguments, "--mcts-simulations")?;
            }
            "--mcts-exploration" => {
                mcts_exploration = parse_next(&mut arguments, "--mcts-exploration")?;
                if mcts_exploration < 0.0 {
                    return Err("--mcts-exploration must be nonnegative".to_string());
                }
            }
            "--mcts-prior" => {
                mcts_prior = match arguments.next().as_deref() {
                    Some("uniform") => MctsPrior::Uniform,
                    Some("connectivity") => MctsPrior::Connectivity,
                    Some("tiny-nn") => MctsPrior::TinyNn,
                    _ => {
                        return Err(
                            "--mcts-prior must be uniform, connectivity, or tiny-nn".to_string()
                        );
                    }
                };
            }
            "--mcts-early-prior" => {
                mcts_early_prior = match arguments.next().as_deref() {
                    Some("uniform") => MctsPrior::Uniform,
                    Some("connectivity") => MctsPrior::Connectivity,
                    Some("tiny-nn") => MctsPrior::TinyNn,
                    _ => {
                        return Err(
                            "--mcts-early-prior must be uniform, connectivity, or tiny-nn"
                                .to_string(),
                        );
                    }
                };
            }
            "--mcts-rollout-depth" => {
                mcts_rollout_depth = parse_next(&mut arguments, "--mcts-rollout-depth")?;
                if mcts_rollout_depth > CANDY_COUNT {
                    return Err("--mcts-rollout-depth must be at most 100".to_string());
                }
            }
            "--mcts-rollout-cutoff-until" => {
                mcts_rollout_cutoff_until =
                    parse_next(&mut arguments, "--mcts-rollout-cutoff-until")?;
                if mcts_rollout_cutoff_until > CANDY_COUNT {
                    return Err("--mcts-rollout-cutoff-until must be at most 100".to_string());
                }
            }
            "--mcts-tail-repair-turns" => {
                mcts_tail_repair_turns = parse_next(&mut arguments, "--mcts-tail-repair-turns")?;
                if mcts_tail_repair_turns > CANDY_COUNT {
                    return Err("--mcts-tail-repair-turns must be at most 100".to_string());
                }
            }
            "--mcts-tail-repair-passes" => {
                mcts_tail_repair_passes = parse_next(&mut arguments, "--mcts-tail-repair-passes")?;
                if !(1..=10).contains(&mcts_tail_repair_passes) {
                    return Err("--mcts-tail-repair-passes must be in 1..=10".to_string());
                }
            }
            "--mcts-early-simulations" => {
                mcts_early_simulations = parse_next(&mut arguments, "--mcts-early-simulations")?;
            }
            "--mcts-early-min-gain" => {
                mcts_early_min_gain = parse_next(&mut arguments, "--mcts-early-min-gain")?;
                if mcts_early_min_gain < 0.0 {
                    return Err("--mcts-early-min-gain must be nonnegative".to_string());
                }
            }
            "--time-limit-ms" => {
                time_limit_ms = parse_next(&mut arguments, "--time-limit-ms")?;
            }
            "--time-reserve-ms" => {
                time_reserve_ms = parse_next(&mut arguments, "--time-reserve-ms")?;
            }
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }
    let model = if let Some(path) = model_path {
        Ahc015ValueNet::from_tensors(read_model_file(path)?).map(Some)?
    } else if generated_model::MODEL_DATA_BASE93.is_empty() {
        None
    } else {
        let data = decode_base93(generated_model::MODEL_DATA_BASE93)?;
        Some(Ahc015ValueNet::from_tensors(read_model(&data)?)?)
    };
    if time_reserve_ms >= time_limit_ms {
        return Err("--time-reserve-ms must be smaller than --time-limit-ms".to_string());
    }
    Ok((
        model,
        SearchSettings {
            exact_turns,
            mc_turns,
            mc_actions,
            mc_samples,
            mc_min_gain,
            mc_strategy,
            mc_stratified_turns,
            mc_exact_last_action,
            endgame_search,
            mcts_simulations,
            mcts_exploration,
            mcts_prior,
            mcts_early_prior,
            mcts_rollout_depth,
            mcts_rollout_cutoff_until,
            mcts_tail_repair_turns,
            mcts_tail_repair_passes,
            mcts_early_simulations,
            mcts_early_min_gain,
            time_limit: Duration::from_millis(time_limit_ms),
            time_reserve: Duration::from_millis(time_reserve_ms),
        },
    ))
}

fn parse_next<T: FromStr>(
    arguments: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<T, String> {
    arguments
        .next()
        .ok_or_else(|| format!("{option} requires a value"))?
        .parse()
        .map_err(|_| format!("invalid value for {option}"))
}

fn choose_action(
    model: Option<&Ahc015ValueNet>,
    state: &State,
    input: &Input,
    rule_actions: &[[u8; CANDY_COUNT]; 24],
    settings: &SearchSettings,
    started: Instant,
    exact_cache: &mut ExactCache,
) -> Result<usize, String> {
    if state.is_terminal() {
        return Ok(FRONT);
    }
    let candidates = state.afterstates();
    if settings.exact_turns > 0 && state.placed() >= CANDY_COUNT - settings.exact_turns {
        let mut best_action = FRONT;
        let mut best_sum = 0;
        for action in 0..ACTION_COUNT {
            let sum = exact_future_sum(&candidates[action], state.placed(), input, exact_cache);
            if sum > best_sum {
                best_sum = sum;
                best_action = action;
            }
        }
        return Ok(best_action);
    }
    let residuals = if let Some(model) = model {
        model.predict(&encode_candidates(&candidates, state.placed(), input))?
    } else {
        vec![0.0; ACTION_COUNT]
    };
    if settings.mc_turns > 0
        && state.placed() >= CANDY_COUNT - settings.mc_turns
        && state.placed() < CANDY_COUNT - settings.exact_turns
    {
        let exact_start = CANDY_COUNT - settings.exact_turns;
        let remaining_mc_turns = exact_start - state.placed();
        let search_deadline = started + settings.time_limit - settings.time_reserve;
        let now = Instant::now();
        if now < search_deadline {
            let turn_budget = search_deadline.duration_since(now) / remaining_mc_turns as u32;
            let deadline = now + turn_budget;
            return Ok(match settings.endgame_search {
                EndgameSearch::MonteCarlo => monte_carlo_action(
                    &candidates,
                    &residuals,
                    state.placed(),
                    input,
                    rule_actions,
                    settings,
                    deadline,
                ),
                EndgameSearch::Mcts => mcts_action(
                    &candidates,
                    &residuals,
                    state.placed(),
                    input,
                    rule_actions,
                    settings,
                    deadline,
                ),
            });
        }
    }
    let mut best_action = FRONT;
    let mut best_value = f32::NEG_INFINITY;
    for action in 0..ACTION_COUNT {
        let value = residuals[action];
        if value > best_value {
            best_value = value;
            best_action = action;
        }
    }
    Ok(best_action)
}

#[derive(Clone)]
struct MctsNode {
    visits: u32,
    action_visits: [u32; ACTION_COUNT],
    action_sums: [u64; ACTION_COUNT],
    priorities: [u8; ACTION_COUNT],
}

type MctsTable = HashMap<Board, MctsNode, BuildHasherDefault<FastHasher>>;

fn mcts_action(
    candidates: &[Board; ACTION_COUNT],
    model_values: &[f32],
    placed: usize,
    input: &Input,
    rule_actions: &[[u8; CANDY_COUNT]; 24],
    settings: &SearchSettings,
    deadline: Instant,
) -> usize {
    let rollout_depth = if placed < settings.mcts_rollout_cutoff_until {
        settings.mcts_rollout_depth
    } else {
        0
    };
    let simulation_limit =
        if placed < settings.mcts_rollout_cutoff_until && settings.mcts_early_simulations > 0 {
            settings.mcts_early_simulations
        } else {
            settings.mcts_simulations
        };
    let min_gain = if placed < settings.mcts_rollout_cutoff_until {
        settings.mcts_early_min_gain
    } else {
        settings.mc_min_gain
    };
    let prior = if placed < settings.mcts_rollout_cutoff_until {
        settings.mcts_early_prior
    } else {
        settings.mcts_prior
    };
    let model_action = (0..ACTION_COUNT)
        .max_by(|&left, &right| {
            model_values[left]
                .total_cmp(&model_values[right])
                .then_with(|| right.cmp(&left))
        })
        .unwrap();
    let mut roots = (0..ACTION_COUNT).collect::<Vec<_>>();
    roots.sort_by(|&left, &right| {
        model_values[right]
            .total_cmp(&model_values[left])
            .then_with(|| left.cmp(&right))
    });
    roots.truncate(settings.mc_actions);

    let mut table = MctsTable::default();
    let mut sums = [0u64; ACTION_COUNT];
    let mut visits = [0u32; ACTION_COUNT];
    let mut simulation = 0u64;
    'search: loop {
        // One common random stream and rollout rule is applied to every root action.
        let ranks = playout_ranks(placed, simulation, settings.mc_stratified_turns);
        let rule = simulation as usize % rule_actions.len();
        for &root in &roots {
            let score = mcts_simulation(
                &candidates[root],
                placed,
                input,
                &rule_actions[rule],
                &ranks,
                settings.mcts_exploration,
                prior,
                rollout_depth,
                if settings.mc_exact_last_action {
                    settings.mcts_tail_repair_turns
                } else {
                    0
                },
                settings.mcts_tail_repair_passes,
                &mut table,
            );
            sums[root] += score as u64;
            visits[root] += 1;
            if Instant::now() >= deadline {
                break 'search;
            }
        }
        simulation += 1;
        if simulation_limit > 0 && simulation >= simulation_limit {
            break;
        }
    }
    let best_action = roots
        .iter()
        .copied()
        .filter(|&action| visits[action] > 0)
        .max_by(|&left, &right| {
            let left_scaled = sums[left] as u128 * visits[right] as u128;
            let right_scaled = sums[right] as u128 * visits[left] as u128;
            left_scaled
                .cmp(&right_scaled)
                .then_with(|| right.cmp(&left))
        })
        .unwrap_or(model_action);
    if visits[best_action] == 0 || visits[model_action] == 0 {
        return model_action;
    }
    let best_mean = sums[best_action] as f64 / visits[best_action] as f64;
    let model_mean = sums[model_action] as f64 / visits[model_action] as f64;
    if best_mean >= model_mean + min_gain {
        best_action
    } else {
        model_action
    }
}

#[allow(clippy::too_many_arguments)]
fn mcts_simulation(
    root: &Board,
    placed: usize,
    input: &Input,
    rollout_actions: &[u8; CANDY_COUNT],
    ranks: &[u8; CANDY_COUNT],
    exploration: f64,
    prior: MctsPrior,
    rollout_depth: usize,
    tail_repair_turns: usize,
    tail_repair_passes: usize,
    table: &mut MctsTable,
) -> usize {
    let mut board = *root;
    let mut path = [([0; CANDY_COUNT], 0); CANDY_COUNT];
    for (depth, position) in (placed..CANDY_COUNT).enumerate() {
        board = place_on_board_at_rank(&board, ranks[position] as usize, input.flavors()[position]);
        if position + 1 == CANDY_COUNT {
            let score = connectivity_numerator(&board);
            mcts_backpropagate(table, &path[..depth], score);
            return score;
        }

        let is_new = !table.contains_key(&board);
        if is_new {
            table.insert(board, new_mcts_node(&board, prior, input, position + 1));
        }
        let action = select_mcts_action(table.get(&board).unwrap(), exploration);
        let edge_unvisited = table.get(&board).unwrap().action_visits[action] == 0;
        path[depth] = (board, action);
        board = tilt(&board, action);
        if is_new || edge_unvisited {
            let score = rule_playout_after_tilt(
                &board,
                position + 1,
                input,
                rollout_actions,
                ranks,
                rollout_depth,
                tail_repair_turns,
                tail_repair_passes,
            );
            mcts_backpropagate(table, &path[..=depth], score);
            return score;
        }
    }
    unreachable!()
}

fn new_mcts_node(board: &Board, prior: MctsPrior, input: &Input, placed: usize) -> MctsNode {
    if prior == MctsPrior::Uniform {
        return MctsNode {
            visits: 0,
            action_visits: [0; ACTION_COUNT],
            action_sums: [0; ACTION_COUNT],
            priorities: [1; ACTION_COUNT],
        };
    }
    let mut actions = [FRONT, BACK, LEFT, RIGHT];
    if prior == MctsPrior::Connectivity {
        actions
            .sort_by_key(|&action| std::cmp::Reverse(connectivity_numerator(&tilt(board, action))));
    } else {
        let scores: [f32; ACTION_COUNT] =
            std::array::from_fn(|action| tiny_prior_score(&tilt(board, action), input, placed));
        actions.sort_by(|&left, &right| {
            scores[right]
                .total_cmp(&scores[left])
                .then_with(|| left.cmp(&right))
        });
    }
    let mut priorities = [0; ACTION_COUNT];
    for (rank, action) in actions.into_iter().enumerate() {
        priorities[action] = (ACTION_COUNT - rank) as u8;
    }
    MctsNode {
        visits: 0,
        action_visits: [0; ACTION_COUNT],
        action_sums: [0; ACTION_COUNT],
        priorities,
    }
}

fn tiny_prior_score(board: &Board, input: &Input, placed: usize) -> f32 {
    const FEATURE_COUNT: usize = 16;
    const HIDDEN: usize = 16;
    let mapping = crate::features::dynamic_flavor_mapping(input, placed);
    let mut features = [0.0f32; FEATURE_COUNT];
    let mut same_edges = 0usize;
    for canonical in 1..=3 {
        let mut visited = [false; CANDY_COUNT];
        let mut component_square = 0usize;
        let mut largest = 0usize;
        let mut components = 0usize;
        let mut empty_contacts = 0usize;
        let mut exposure_edges = 0usize;
        for start in 0..CANDY_COUNT {
            if visited[start] || mapping[board[start] as usize] as usize != canonical {
                continue;
            }
            components += 1;
            visited[start] = true;
            let mut stack = [0usize; CANDY_COUNT];
            stack[0] = start;
            let mut stack_len = 1;
            let mut size = 0usize;
            while stack_len > 0 {
                stack_len -= 1;
                let cell = stack[stack_len];
                size += 1;
                let row = cell / 10;
                let column = cell % 10;
                for neighbor in [
                    row.checked_sub(1).map(|next| next * 10 + column),
                    (row + 1 < 10).then_some((row + 1) * 10 + column),
                    column.checked_sub(1).map(|next| row * 10 + next),
                    (column + 1 < 10).then_some(row * 10 + column + 1),
                ]
                .into_iter()
                .flatten()
                {
                    if !visited[neighbor] && mapping[board[neighbor] as usize] as usize == canonical
                    {
                        visited[neighbor] = true;
                        stack[stack_len] = neighbor;
                        stack_len += 1;
                    }
                }
            }
            component_square += size * size;
            largest = largest.max(size);
        }
        for cell in 0..CANDY_COUNT {
            if mapping[board[cell] as usize] as usize != canonical {
                continue;
            }
            let row = cell / 10;
            let column = cell % 10;
            for neighbor in [
                row.checked_sub(1).map(|next| next * 10 + column),
                (row + 1 < 10).then_some((row + 1) * 10 + column),
                column.checked_sub(1).map(|next| row * 10 + next),
                (column + 1 < 10).then_some(row * 10 + column + 1),
            ] {
                if let Some(neighbor) = neighbor {
                    let neighbor_flavor = mapping[board[neighbor] as usize] as usize;
                    if neighbor_flavor == 0 {
                        empty_contacts += 1;
                        exposure_edges += 1;
                    } else if neighbor_flavor == canonical {
                        same_edges += 1;
                    }
                } else {
                    exposure_edges += 1;
                }
            }
        }
        let base = (canonical - 1) * 5;
        features[base] = component_square as f32 / 10_000.0;
        features[base + 1] = largest as f32 / 100.0;
        features[base + 2] = components as f32 / 100.0;
        features[base + 3] = empty_contacts as f32 / 180.0;
        features[base + 4] = exposure_edges as f32 / 220.0;
    }
    features[15] = same_edges as f32 / 360.0;
    let mut hidden = [0.0f32; HIDDEN];
    for (output, value) in hidden.iter_mut().enumerate() {
        *value = generated_mcts_prior::HIDDEN_BIAS[output];
        for (input_index, &feature) in features.iter().enumerate() {
            *value +=
                generated_mcts_prior::HIDDEN_WEIGHT[output * FEATURE_COUNT + input_index] * feature;
        }
        *value = value.max(0.0);
    }
    generated_mcts_prior::OUTPUT_BIAS[0]
        + hidden
            .iter()
            .zip(generated_mcts_prior::OUTPUT_WEIGHT)
            .map(|(&value, weight)| value * weight)
            .sum::<f32>()
}

fn select_mcts_action(node: &MctsNode, exploration: f64) -> usize {
    if let Some(action) = (0..ACTION_COUNT)
        .filter(|&action| node.action_visits[action] == 0)
        .max_by_key(|&action| (node.priorities[action], std::cmp::Reverse(action)))
    {
        return action;
    }
    let root = (node.visits as f64).sqrt();
    (0..ACTION_COUNT)
        .max_by(|&left, &right| {
            let value = |action: usize| {
                node.action_sums[action] as f64 / node.action_visits[action] as f64
                    + exploration * node.priorities[action] as f64 * root
                        / (10.0 * (node.action_visits[action] + 1) as f64)
            };
            value(left)
                .total_cmp(&value(right))
                .then_with(|| right.cmp(&left))
        })
        .unwrap()
}

fn mcts_backpropagate(table: &mut MctsTable, path: &[(Board, usize)], score: usize) {
    for &(board, action) in path {
        let node = table.get_mut(&board).unwrap();
        node.visits += 1;
        node.action_visits[action] += 1;
        node.action_sums[action] += score as u64;
    }
}

#[derive(Clone)]
struct McArm {
    action: usize,
    rule: usize,
    sum: u64,
    samples: u32,
}

fn monte_carlo_action(
    candidates: &[Board; ACTION_COUNT],
    model_values: &[f32],
    placed: usize,
    input: &Input,
    rule_actions: &[[u8; CANDY_COUNT]; 24],
    settings: &SearchSettings,
    deadline: Instant,
) -> usize {
    let mut actions = (0..ACTION_COUNT).collect::<Vec<_>>();
    actions.sort_by(|&left, &right| {
        model_values[right]
            .total_cmp(&model_values[left])
            .then_with(|| left.cmp(&right))
    });
    actions.truncate(settings.mc_actions);
    let model_action = actions[0];
    let mut rules: Vec<usize> = Vec::with_capacity(24);
    for rule in 0..24 {
        if !rules
            .iter()
            .any(|&existing| rule_actions[existing][placed..] == rule_actions[rule][placed..])
        {
            rules.push(rule);
        }
    }
    let mut arms = actions
        .into_iter()
        .flat_map(|action| {
            rules.iter().copied().map(move |rule| McArm {
                action,
                rule,
                sum: 0,
                samples: 0,
            })
        })
        .collect::<Vec<_>>();
    let mut sample = 0u64;
    let mut next_halving = 4u32;
    loop {
        let ranks = playout_ranks(placed, sample, settings.mc_stratified_turns);
        let first_boards: [Board; ACTION_COUNT] = std::array::from_fn(|action| {
            place_on_board_at_rank(
                &candidates[action],
                ranks[placed] as usize,
                input.flavors()[placed],
            )
        });
        for arm in &mut arms {
            arm.sum += rule_playout(
                &first_boards[arm.action],
                placed,
                input,
                &rule_actions[arm.rule],
                &ranks,
                settings.mc_exact_last_action,
            ) as u64;
            arm.samples += 1;
        }
        sample += 1;
        if settings.mc_strategy == McStrategy::Halving
            && arms.len() > 6
            && arms[0].samples >= next_halving
        {
            arms.sort_by(compare_mc_arms);
            let keep = arms.len().div_ceil(2);
            let model_best = arms
                .iter()
                .filter(|arm| arm.action == model_action)
                .min_by(|left, right| compare_mc_arms(left, right))
                .unwrap()
                .clone();
            arms.truncate(keep);
            if !arms.iter().any(|arm| arm.action == model_action) {
                arms[keep - 1] = model_best;
            }
            next_halving *= 2;
        }
        if (settings.mc_samples > 0 && sample >= settings.mc_samples) || Instant::now() >= deadline
        {
            break;
        }
    }
    arms.sort_by(compare_mc_arms);
    let best = &arms[0];
    let model_best = arms
        .iter()
        .filter(|arm| arm.action == model_action)
        .min_by(|left, right| compare_mc_arms(left, right))
        .unwrap();
    let best_mean = best.sum as f64 / best.samples as f64;
    let model_mean = model_best.sum as f64 / model_best.samples as f64;
    if best_mean >= model_mean + settings.mc_min_gain {
        best.action
    } else {
        model_action
    }
}

fn compare_mc_arms(left: &McArm, right: &McArm) -> std::cmp::Ordering {
    let left_scaled = left.sum as u128 * right.samples as u128;
    let right_scaled = right.sum as u128 * left.samples as u128;
    right_scaled
        .cmp(&left_scaled)
        .then_with(|| left.action.cmp(&right.action))
        .then_with(|| left.rule.cmp(&right.rule))
}

fn build_rule_actions(input: &Input) -> [[u8; CANDY_COUNT]; 24] {
    const PERMUTATIONS: [[u8; 3]; 6] = [
        [1, 2, 3],
        [1, 3, 2],
        [2, 1, 3],
        [2, 3, 1],
        [3, 1, 2],
        [3, 2, 1],
    ];
    std::array::from_fn(|rule| {
        let rotation = rule / 6;
        let targets = PERMUTATIONS[rule % 6];
        let mut abstract_flavor = [0usize; 4];
        for (abstract_id, &flavor) in targets.iter().enumerate() {
            abstract_flavor[flavor as usize] = abstract_id;
        }
        std::array::from_fn(|position| {
            if position + 1 == CANDY_COUNT {
                return FRONT as u8;
            }
            let current = abstract_flavor[input.flavors()[position] as usize];
            let immediate_next = abstract_flavor[input.flavors()[position + 1] as usize];
            let later_different = input.flavors()[position + 1..]
                .iter()
                .map(|&flavor| abstract_flavor[flavor as usize])
                .find(|&flavor| flavor != current);
            rotate_action(
                rule_action(current, immediate_next, later_different),
                rotation,
            ) as u8
        })
    })
}

fn rule_playout(
    board_after_first_placement: &Board,
    placed: usize,
    input: &Input,
    actions: &[u8; CANDY_COUNT],
    ranks: &[u8; CANDY_COUNT],
    exact_last_action: bool,
) -> usize {
    let mut result = *board_after_first_placement;
    if placed + 1 == CANDY_COUNT {
        return connectivity_numerator(&result);
    }
    if exact_last_action && placed + 2 == CANDY_COUNT {
        return optimal_last_action_score(&result, input.flavors()[placed + 1]);
    }
    result = tilt(&result, actions[placed] as usize);
    for position in placed + 1..CANDY_COUNT {
        let empty_count = CANDY_COUNT - position;
        result =
            place_on_board_at_rank(&result, ranks[position] as usize, input.flavors()[position]);
        if exact_last_action && empty_count == 2 {
            return optimal_last_action_score(&result, input.flavors()[position + 1]);
        }
        if empty_count == 1 {
            break;
        }
        result = tilt(&result, actions[position] as usize);
    }
    connectivity_numerator(&result)
}

fn rule_playout_after_tilt(
    board: &Board,
    placed: usize,
    input: &Input,
    actions: &[u8; CANDY_COUNT],
    ranks: &[u8; CANDY_COUNT],
    rollout_depth: usize,
    tail_repair_turns: usize,
    tail_repair_passes: usize,
) -> usize {
    if tail_repair_turns <= 1 {
        return rule_playout_after_tilt_legacy(
            board,
            placed,
            input,
            actions,
            ranks,
            rollout_depth,
            tail_repair_turns == 1,
        );
    }
    let last_action = if rollout_depth == 0 {
        CANDY_COUNT - 2
    } else {
        (placed + rollout_depth - 1).min(CANDY_COUNT - 2)
    };
    let terminal = last_action == CANDY_COUNT - 2;
    let mut repaired_actions = *actions;
    let repair_start = placed.max((last_action + 1).saturating_sub(tail_repair_turns));
    let mut boards_before_action = Vec::with_capacity(last_action + 1 - repair_start);
    let mut best_score = 0;

    for pass in 0..tail_repair_passes {
        boards_before_action.clear();
        let mut result = *board;
        for position in placed..=last_action {
            result = place_on_board_at_rank(
                &result,
                ranks[position] as usize,
                input.flavors()[position],
            );
            if position >= repair_start {
                boards_before_action.push(result);
            }
            result = tilt(&result, repaired_actions[position] as usize);
        }
        if terminal {
            result = place_on_board_at_rank(&result, 1, input.flavors()[CANDY_COUNT - 1]);
        }
        if pass == 0 {
            best_score = scaled_playout_score(&result, input, last_action, terminal);
        }

        for position in (repair_start..=last_action).rev() {
            let original = repaired_actions[position];
            let mut best_action = original;
            for action in 0..ACTION_COUNT {
                repaired_actions[position] = action as u8;
                let score = replay_playout_suffix(
                    &boards_before_action[position - repair_start],
                    position,
                    last_action,
                    terminal,
                    input,
                    &repaired_actions,
                    ranks,
                );
                if score > best_score {
                    best_score = score;
                    best_action = action as u8;
                }
            }
            repaired_actions[position] = best_action;
        }
    }
    best_score
}

fn rule_playout_after_tilt_legacy(
    board: &Board,
    placed: usize,
    input: &Input,
    actions: &[u8; CANDY_COUNT],
    ranks: &[u8; CANDY_COUNT],
    rollout_depth: usize,
    exact_last_action: bool,
) -> usize {
    let mut result = *board;
    let mut actions_played = 0;
    for position in placed..CANDY_COUNT {
        let empty_count = CANDY_COUNT - position;
        result =
            place_on_board_at_rank(&result, ranks[position] as usize, input.flavors()[position]);
        if exact_last_action && empty_count == 2 {
            return optimal_last_action_score(&result, input.flavors()[position + 1]);
        }
        if empty_count == 1 {
            return connectivity_numerator(&result);
        }
        result = tilt(&result, actions[position] as usize);
        actions_played += 1;
        if rollout_depth > 0 && actions_played >= rollout_depth {
            let partial_denominator = input.prefix_denominator(position + 1);
            let final_denominator: usize = input.totals().iter().map(|count| count * count).sum();
            return connectivity_numerator(&result) * final_denominator / partial_denominator;
        }
    }
    unreachable!()
}

#[allow(clippy::too_many_arguments)]
fn replay_playout_suffix(
    board_before_action: &Board,
    first_action: usize,
    last_action: usize,
    terminal: bool,
    input: &Input,
    actions: &[u8; CANDY_COUNT],
    ranks: &[u8; CANDY_COUNT],
) -> usize {
    let mut result = tilt(board_before_action, actions[first_action] as usize);
    for position in first_action + 1..=last_action {
        result =
            place_on_board_at_rank(&result, ranks[position] as usize, input.flavors()[position]);
        result = tilt(&result, actions[position] as usize);
    }
    if terminal {
        result = place_on_board_at_rank(&result, 1, input.flavors()[CANDY_COUNT - 1]);
    }
    scaled_playout_score(&result, input, last_action, terminal)
}

fn scaled_playout_score(board: &Board, input: &Input, last_action: usize, terminal: bool) -> usize {
    let score = connectivity_numerator(board);
    if terminal {
        score
    } else {
        let partial_denominator = input.prefix_denominator(last_action + 1);
        let final_denominator: usize = input.totals().iter().map(|count| count * count).sum();
        score * final_denominator / partial_denominator
    }
}

fn optimal_last_action_score(board: &Board, final_flavor: u8) -> usize {
    (0..ACTION_COUNT)
        .map(|action| {
            let tilted = tilt(board, action);
            let terminal = place_on_board_at_rank(&tilted, 1, final_flavor);
            connectivity_numerator(&terminal)
        })
        .max()
        .unwrap()
}

fn playout_ranks(placed: usize, sample: u64, stratified_turns: usize) -> [u8; CANDY_COUNT] {
    let seed = splitmix64(0x9e37_79b9_7f4a_7c15 ^ (placed as u64) << 32 ^ sample);
    let mut random = seed;
    let depth = stratified_turns.min(CANDY_COUNT - placed);
    let combination_count = (0..depth)
        .map(|offset| (CANDY_COUNT - placed - offset) as u64)
        .product::<u64>();
    let mut combination = if depth == 0 {
        0
    } else {
        let scramble = splitmix64(0xd1b5_4a32_d192_ed03 ^ (placed as u64) << 32);
        let mut step = splitmix64(scramble) % combination_count;
        while greatest_common_divisor(step, combination_count) != 1 {
            step = (step + 1) % combination_count;
        }
        (scramble % combination_count + sample % combination_count * step) % combination_count
    };
    let mut ranks = [0; CANDY_COUNT];
    for position in placed..CANDY_COUNT {
        random = splitmix64(random);
        let empty_count = CANDY_COUNT - position;
        ranks[position] = if position - placed < depth {
            let rank = combination % empty_count as u64;
            combination /= empty_count as u64;
            (rank + 1) as u8
        } else {
            (random as usize % empty_count + 1) as u8
        };
    }
    ranks
}

fn greatest_common_divisor(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

fn rule_action(current: usize, immediate_next: usize, later_different: Option<usize>) -> usize {
    if current != immediate_next {
        return base_rule_action(current, Some(immediate_next));
    }
    match (current, later_different) {
        (0, Some(1)) => RIGHT,
        (0, Some(2)) => LEFT,
        _ => base_rule_action(current, later_different),
    }
}

fn base_rule_action(current: usize, next_different: Option<usize>) -> usize {
    match (current, next_different) {
        (1 | 2, Some(0)) => FRONT,
        (0, Some(1 | 2)) => BACK,
        (1, Some(2)) => RIGHT,
        (2, Some(1)) => LEFT,
        (0, None) => BACK,
        (1, None) => LEFT,
        (2, None) => RIGHT,
        _ => unreachable!(),
    }
}

fn rotate_action(action: usize, rotation: usize) -> usize {
    const CLOCKWISE: [usize; ACTION_COUNT] = [RIGHT, LEFT, FRONT, BACK];
    let mut result = action;
    for _ in 0..rotation {
        result = CLOCKWISE[result];
    }
    result
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Sum of terminal connectivity numerators over every equally likely future
/// placement sequence, assuming optimal actions after each revealed placement.
fn exact_future_sum(board: &Board, placed: usize, input: &Input, cache: &mut ExactCache) -> usize {
    if placed == CANDY_COUNT {
        return connectivity_numerator(board);
    }
    if let Some(&value) = cache.get(board) {
        return value;
    }
    let empty_count = CANDY_COUNT - placed;
    let flavor = input.flavors()[placed];
    let mut sum = 0;
    for rank in 1..=empty_count {
        let placed_board = place_on_board_at_rank(board, rank, flavor);
        if empty_count == 1 {
            sum += connectivity_numerator(&placed_board);
        } else {
            sum += (0..ACTION_COUNT)
                .map(|action| {
                    exact_future_sum(&tilt(&placed_board, action), placed + 1, input, cache)
                })
                .max()
                .unwrap();
        }
    }
    // The number of non-empty cells uniquely determines `placed`, so the board
    // itself is sufficient as a cache key.
    cache.insert(*board, sum);
    sum
}

fn run() -> Result<(), String> {
    let started = Instant::now();
    let (model, settings) = load_model()?;
    let stdin = io::stdin();
    let mut reader = TokenReader::new(stdin.lock());
    let mut flavors = [0; CANDY_COUNT];
    for flavor in &mut flavors {
        *flavor = reader.read()?;
        if !(1..=3).contains(flavor) {
            return Err(format!("flavor must be in 1..=3, got {flavor}"));
        }
    }
    let input = Input::new(flavors);
    let rule_actions = build_rule_actions(&input);
    let mut state = State::new();
    let mut exact_cache = ExactCache::default();
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());

    for turn in 0..CANDY_COUNT {
        let rank: usize = reader.read()?;
        state.place_at_rank(rank, input.flavors()[turn]);
        let action = choose_action(
            model.as_ref(),
            &state,
            &input,
            &rule_actions,
            &settings,
            started,
            &mut exact_cache,
        )?;
        state.apply_action(action);
        writeln!(output, "{}", action_char(action))
            .map_err(|error| format!("failed to write action: {error}"))?;
        output
            .flush()
            .map_err(|error| format!("failed to flush action: {error}"))?;
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_model_chooses_first_tied_action() {
        let input = Input::new(std::array::from_fn(|index| (index % 3 + 1) as u8));
        let mut state = State::new();
        state.place_at_rank(1, 1);
        let rule_actions = build_rule_actions(&input);
        let settings = SearchSettings {
            exact_turns: 0,
            mc_turns: 0,
            mc_actions: 4,
            mc_samples: 0,
            mc_min_gain: 0.0,
            mc_strategy: McStrategy::Equal,
            mc_stratified_turns: 0,
            mc_exact_last_action: false,
            endgame_search: EndgameSearch::MonteCarlo,
            mcts_simulations: 0,
            mcts_exploration: 0.0,
            mcts_prior: MctsPrior::Connectivity,
            mcts_early_prior: MctsPrior::Connectivity,
            mcts_rollout_depth: 0,
            mcts_rollout_cutoff_until: CANDY_COUNT,
            mcts_tail_repair_turns: 2,
            mcts_tail_repair_passes: 1,
            mcts_early_simulations: 0,
            mcts_early_min_gain: 20.0,
            time_limit: Duration::from_secs(2),
            time_reserve: Duration::from_millis(100),
        };
        assert_eq!(
            choose_action(
                None,
                &state,
                &input,
                &rule_actions,
                &settings,
                Instant::now(),
                &mut ExactCache::default(),
            )
            .unwrap(),
            FRONT
        );
    }

    #[test]
    fn exact_sum_counts_each_future_placement_sequence() {
        let input = Input::new([1; CANDY_COUNT]);
        let mut board = [1; CANDY_COUNT];
        board[..4].fill(0);
        let mut cache = ExactCache::default();
        assert_eq!(
            exact_future_sum(&board, 96, &input, &mut cache),
            24 * 10_000
        );
    }

    #[test]
    fn mcts_backpropagation_updates_a_shared_dag_node() {
        let input = Input::new([1; CANDY_COUNT]);
        let board = [0; CANDY_COUNT];
        let mut table = MctsTable::default();
        table.insert(board, new_mcts_node(&board, MctsPrior::Uniform, &input, 1));
        mcts_backpropagate(&mut table, &[(board, LEFT)], 123);
        mcts_backpropagate(&mut table, &[(board, LEFT)], 321);
        assert_eq!(table.len(), 1);
        assert_eq!(table[&board].visits, 2);
        assert_eq!(table[&board].action_visits[LEFT], 2);
        assert_eq!(table[&board].action_sums[LEFT], 444);
    }

    #[test]
    fn cutoff_projection_preserves_a_perfect_single_flavor_board() {
        let input = Input::new([1; CANDY_COUNT]);
        let mut board = [0; CANDY_COUNT];
        board[..80].fill(1);
        let actions = [FRONT as u8; CANDY_COUNT];
        let ranks = [1; CANDY_COUNT];
        assert_eq!(
            rule_playout_after_tilt(&board, 80, &input, &actions, &ranks, 4, 0, 1),
            10_000
        );
    }

    #[test]
    fn repairing_more_tail_actions_does_not_reduce_playout_score() {
        let input = Input::new(std::array::from_fn(|index| (index % 3 + 1) as u8));
        let mut board = [0; CANDY_COUNT];
        for (index, cell) in board.iter_mut().take(94).enumerate() {
            *cell = (index % 3 + 1) as u8;
        }
        let actions = std::array::from_fn(|index| (index % ACTION_COUNT) as u8);
        let ranks = [1; CANDY_COUNT];
        let one = rule_playout_after_tilt(&board, 94, &input, &actions, &ranks, 0, 1, 1);
        let four = rule_playout_after_tilt(&board, 94, &input, &actions, &ranks, 0, 4, 1);
        assert!(four >= one);
    }

    #[test]
    fn stratified_pairs_do_not_repeat_before_exhaustion() {
        let mut seen = [[false; 11]; 12];
        for sample in 0..132 {
            let ranks = playout_ranks(88, sample, 2);
            let first = ranks[88] as usize - 1;
            let second = ranks[89] as usize - 1;
            assert!(!seen[first][second]);
            seen[first][second] = true;
        }
    }

    #[test]
    fn one_turn_stratification_is_balanced() {
        let mut counts = [0; 8];
        for sample in 0..128 {
            counts[playout_ranks(92, sample, 1)[92] as usize - 1] += 1;
        }
        assert_eq!(counts, [16; 8]);
    }

    #[test]
    fn playout_optimizes_the_last_nontrivial_action() {
        let input = Input::new(std::array::from_fn(|index| (index % 3 + 1) as u8));
        let mut board = std::array::from_fn(|index| (index % 3 + 1) as u8);
        board[44] = 0;
        let actions = [FRONT as u8; CANDY_COUNT];
        let ranks = [1; CANDY_COUNT];
        let score = rule_playout(&board, 98, &input, &actions, &ranks, true);
        assert_eq!(
            score,
            optimal_last_action_score(&board, input.flavors()[99])
        );
        assert!(score >= rule_playout(&board, 98, &input, &actions, &ranks, false));
    }

    #[test]
    fn base_rule_matches_the_six_directed_flavor_transitions() {
        assert_eq!(base_rule_action(1, Some(0)), FRONT);
        assert_eq!(base_rule_action(2, Some(0)), FRONT);
        assert_eq!(base_rule_action(0, Some(1)), BACK);
        assert_eq!(base_rule_action(0, Some(2)), BACK);
        assert_eq!(base_rule_action(1, Some(2)), RIGHT);
        assert_eq!(base_rule_action(2, Some(1)), LEFT);
    }

    #[test]
    fn repeated_bottom_flavor_anticipates_the_next_side() {
        assert_eq!(rule_action(0, 0, Some(1)), RIGHT);
        assert_eq!(rule_action(0, 0, Some(2)), LEFT);
        assert_eq!(rule_action(0, 1, Some(1)), BACK);
        assert_eq!(rule_action(0, 2, Some(2)), BACK);
    }

    #[test]
    fn rotating_a_rule_four_times_is_identity() {
        for action in 0..ACTION_COUNT {
            assert_eq!(rotate_action(action, 4), action);
        }
    }
}
