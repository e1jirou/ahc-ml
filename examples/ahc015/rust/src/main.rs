mod features;
mod game;
mod generated_model;

use std::collections::VecDeque;
use std::env;
use std::io::{self, BufRead, Write};
use std::str::FromStr;

use ahc_ml::{Ahc015ValueNet, decode_base93, read_model, read_model_file};

use crate::features::encode_candidates;
use crate::game::{ACTION_COUNT, CANDY_COUNT, FRONT, Input, State, action_char, potential};

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

fn load_model() -> Result<Option<Ahc015ValueNet>, String> {
    let mut arguments = env::args().skip(1);
    if let Some(argument) = arguments.next() {
        if argument != "--model" {
            return Err(format!("unknown argument: {argument}"));
        }
        let path = arguments
            .next()
            .ok_or_else(|| "--model requires a path".to_string())?;
        if arguments.next().is_some() {
            return Err("unexpected extra argument".to_string());
        }
        return Ahc015ValueNet::from_tensors(read_model_file(path)?).map(Some);
    }
    if generated_model::MODEL_DATA_BASE93.is_empty() {
        Ok(None)
    } else {
        let data = decode_base93(generated_model::MODEL_DATA_BASE93)?;
        Ahc015ValueNet::from_tensors(read_model(&data)?).map(Some)
    }
}

fn choose_action(
    model: Option<&Ahc015ValueNet>,
    state: &State,
    input: &Input,
) -> Result<usize, String> {
    if state.is_terminal() {
        return Ok(FRONT);
    }
    let candidates = state.afterstates();
    let residuals = if let Some(model) = model {
        model.predict(&encode_candidates(&candidates, state.placed(), input))?
    } else {
        vec![0.0; ACTION_COUNT]
    };
    let mut best_action = FRONT;
    let mut best_value = f32::NEG_INFINITY;
    for action in 0..ACTION_COUNT {
        let value = potential(&candidates[action], input.denominator()) + residuals[action];
        if value > best_value {
            best_value = value;
            best_action = action;
        }
    }
    Ok(best_action)
}

fn run() -> Result<(), String> {
    let model = load_model()?;
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
        let action = choose_action(model.as_ref(), &state, &input)?;
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
    fn absent_model_is_exact_phi_greedy() {
        let input = Input::new(std::array::from_fn(|index| (index % 3 + 1) as u8));
        let mut state = State::new();
        state.place_at_rank(1, 1);
        let candidates = state.afterstates();
        let expected = (0..ACTION_COUNT)
            .max_by(|&left, &right| {
                potential(&candidates[left], input.denominator())
                    .total_cmp(&potential(&candidates[right], input.denominator()))
                    .then_with(|| right.cmp(&left))
            })
            .unwrap();
        assert_eq!(choose_action(None, &state, &input).unwrap(), expected);
    }
}
