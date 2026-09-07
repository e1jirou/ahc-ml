mod features;
mod game;
mod generated_model;

use std::collections::{HashMap, VecDeque};
use std::env;
use std::io::{self, BufRead, Write};
use std::str::FromStr;
use std::time::{Duration, Instant};

use ahc_ml::{Ahc015ValueNet, decode_base93, read_model, read_model_file};

use crate::features::encode_candidates;
use crate::game::{
    ACTION_COUNT, BACK, Board, CANDY_COUNT, FRONT, Input, LEFT, RIGHT, State, action_char,
    connectivity_numerator, place_on_board_at_rank, tilt,
};

const DEFAULT_EXACT_TURNS: usize = 6;

#[derive(Clone, Copy, PartialEq)]
enum McStrategy {
    Equal,
    Halving,
}

struct SearchSettings {
    exact_turns: usize,
    mc_turns: usize,
    mc_actions: usize,
    mc_samples: u64,
    mc_min_gain: f64,
    mc_strategy: McStrategy,
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
    let mut mc_turns = 12;
    let mut mc_actions = 4;
    let mut mc_samples = 128;
    let mut mc_min_gain = 20.0;
    let mut mc_strategy = McStrategy::Equal;
    let mut time_limit_ms = 1900;
    let mut time_reserve_ms = 200;
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
    settings: &SearchSettings,
    started: Instant,
) -> Result<usize, String> {
    if state.is_terminal() {
        return Ok(FRONT);
    }
    let candidates = state.afterstates();
    if settings.exact_turns > 0 && state.placed() >= CANDY_COUNT - settings.exact_turns {
        let mut best_action = FRONT;
        let mut best_sum = 0;
        let mut cache = HashMap::new();
        for action in 0..ACTION_COUNT {
            let sum = exact_future_sum(&candidates[action], state.placed(), input, &mut cache);
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
            return Ok(monte_carlo_action(
                &candidates,
                &residuals,
                state.placed(),
                input,
                settings,
                now + turn_budget,
            ));
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
    let mut arms = actions
        .into_iter()
        .flat_map(|action| {
            (0..24).map(move |rule| McArm {
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
        let seed = splitmix64(0x9e37_79b9_7f4a_7c15 ^ (placed as u64) << 32 ^ sample);
        for arm in &mut arms {
            arm.sum += rule_playout(&candidates[arm.action], placed, input, arm.rule, seed) as u64;
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

fn rule_playout(board: &Board, placed: usize, input: &Input, rule: usize, seed: u64) -> usize {
    const PERMUTATIONS: [[u8; 3]; 6] = [
        [1, 2, 3],
        [1, 3, 2],
        [2, 1, 3],
        [2, 3, 1],
        [3, 1, 2],
        [3, 2, 1],
    ];
    let rotation = rule / 6;
    let targets = PERMUTATIONS[rule % 6];
    let mut abstract_flavor = [0usize; 4];
    for (abstract_id, &flavor) in targets.iter().enumerate() {
        abstract_flavor[flavor as usize] = abstract_id;
    }
    let mut result = *board;
    let mut random = seed;
    for position in placed..CANDY_COUNT {
        random = splitmix64(random);
        let empty_count = CANDY_COUNT - position;
        let rank = random as usize % empty_count + 1;
        result = place_on_board_at_rank(&result, rank, input.flavors()[position]);
        if empty_count == 1 {
            break;
        }
        let current = abstract_flavor[input.flavors()[position] as usize];
        let immediate_next = abstract_flavor[input.flavors()[position + 1] as usize];
        let later_different = input.flavors()[position + 1..]
            .iter()
            .map(|&flavor| abstract_flavor[flavor as usize])
            .find(|&flavor| flavor != current);
        let base_action = rule_action(current, immediate_next, later_different);
        result = tilt(&result, rotate_action(base_action, rotation));
    }
    connectivity_numerator(&result)
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
fn exact_future_sum(
    board: &Board,
    placed: usize,
    input: &Input,
    cache: &mut HashMap<(Board, usize), usize>,
) -> usize {
    if placed == CANDY_COUNT {
        return connectivity_numerator(board);
    }
    if let Some(&value) = cache.get(&(*board, placed)) {
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
    cache.insert((*board, placed), sum);
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
    let mut state = State::new();
    let stdout = io::stdout();
    let mut output = io::BufWriter::new(stdout.lock());

    for turn in 0..CANDY_COUNT {
        let rank: usize = reader.read()?;
        state.place_at_rank(rank, input.flavors()[turn]);
        let action = choose_action(model.as_ref(), &state, &input, &settings, started)?;
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
        let settings = SearchSettings {
            exact_turns: 0,
            mc_turns: 0,
            mc_actions: 4,
            mc_samples: 0,
            mc_min_gain: 0.0,
            mc_strategy: McStrategy::Equal,
            time_limit: Duration::from_secs(2),
            time_reserve: Duration::from_millis(100),
        };
        assert_eq!(
            choose_action(None, &state, &input, &settings, Instant::now()).unwrap(),
            FRONT
        );
    }

    #[test]
    fn exact_sum_counts_each_future_placement_sequence() {
        let input = Input::new([1; CANDY_COUNT]);
        let mut board = [1; CANDY_COUNT];
        board[..4].fill(0);
        let mut cache = HashMap::new();
        assert_eq!(
            exact_future_sum(&board, 96, &input, &mut cache),
            24 * 10_000
        );
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
