mod features;
mod game;
mod generated_model;

use std::collections::{HashMap, VecDeque};
use std::env;
use std::io::{self, BufRead, Write};
use std::str::FromStr;

use ahc_ml::{Ahc015ValueNet, decode_base93, read_model, read_model_file};

use crate::features::encode_candidates;
use crate::game::{
    ACTION_COUNT, Board, CANDY_COUNT, FRONT, Input, State, action_char, connectivity_numerator,
    place_on_board_at_rank, tilt,
};

const DEFAULT_EXACT_TURNS: usize = 6;

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

fn load_model() -> Result<(Option<Ahc015ValueNet>, usize), String> {
    let mut arguments = env::args().skip(1);
    let mut model_path = None;
    let mut exact_turns = DEFAULT_EXACT_TURNS;
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
    Ok((model, exact_turns))
}

fn choose_action(
    model: Option<&Ahc015ValueNet>,
    state: &State,
    input: &Input,
    exact_turns: usize,
) -> Result<usize, String> {
    if state.is_terminal() {
        return Ok(FRONT);
    }
    let candidates = state.afterstates();
    if exact_turns > 0 && state.placed() >= CANDY_COUNT - exact_turns {
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
    let (model, exact_turns) = load_model()?;
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
        let action = choose_action(model.as_ref(), &state, &input, exact_turns)?;
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
        assert_eq!(choose_action(None, &state, &input, 0).unwrap(), FRONT);
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
}
