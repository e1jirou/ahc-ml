use crate::game::{
    ACTION_COUNT, Action, BACK, Board, CANDY_COUNT, FRONT, Input, LEFT, RIGHT, SIDE, cell,
};

pub const FEATURE_CHANNELS: usize = 4;

pub fn encode_candidates(boards: &[Board; ACTION_COUNT], placed: usize, input: &Input) -> Vec<f32> {
    let mut output = vec![0.0; ACTION_COUNT * FEATURE_CHANNELS * CANDY_COUNT];
    for action in 0..ACTION_COUNT {
        encode_one(&mut output, action, &boards[action], action, placed, input);
    }
    output
}

fn encode_one(
    output: &mut [f32],
    sample: usize,
    board: &Board,
    action: Action,
    placed: usize,
    input: &Input,
) {
    let mapping = dynamic_flavor_mapping(input, placed);
    let board = normalized_board(board, action, &mapping);
    let sample_base = sample * FEATURE_CHANNELS * CANDY_COUNT;

    for index in 0..CANDY_COUNT {
        let flavor = board[index];
        let channel = if flavor == 0 { 3 } else { flavor as usize - 1 };
        output[sample_base + channel * CANDY_COUNT + index] = 1.0;
    }
}

pub fn dynamic_flavor_mapping(input: &Input, placed: usize) -> [u8; 4] {
    let totals = input.totals();
    let first = input.first_positions();
    let mut original = [0usize, 1, 2];
    original.sort_by_key(|&flavor| {
        let next = input.flavors()[placed..]
            .iter()
            .position(|&value| value as usize == flavor + 1)
            .map(|offset| placed + offset)
            .unwrap_or(usize::MAX);
        (next, totals[flavor], first[flavor])
    });
    let mut mapping = [0; 4];
    for (canonical, flavor) in original.into_iter().enumerate() {
        mapping[flavor + 1] = canonical as u8 + 1;
    }
    mapping
}

fn normalized_board(board: &Board, action: Action, mapping: &[u8; 4]) -> Board {
    let oriented = rotate_action_to_front(board, action);
    let mut normalized = [0; CANDY_COUNT];
    for index in 0..CANDY_COUNT {
        normalized[index] = mapping[oriented[index] as usize];
    }
    if mirror_is_smaller(&normalized) {
        mirror_left_right(&normalized)
    } else {
        normalized
    }
}

fn rotate_action_to_front(board: &Board, action: Action) -> Board {
    let mut result = [0; CANDY_COUNT];
    for row in 0..SIDE {
        for column in 0..SIDE {
            let (destination_row, destination_column) = match action {
                FRONT => (row, column),
                BACK => (SIDE - 1 - row, SIDE - 1 - column),
                LEFT => (column, SIDE - 1 - row),
                RIGHT => (SIDE - 1 - column, row),
                _ => panic!("invalid action: {action}"),
            };
            result[cell(destination_row, destination_column)] = board[cell(row, column)];
        }
    }
    result
}

fn mirror_left_right(board: &Board) -> Board {
    let mut result = [0; CANDY_COUNT];
    for row in 0..SIDE {
        for column in 0..SIDE {
            result[cell(row, SIDE - 1 - column)] = board[cell(row, column)];
        }
    }
    result
}

fn mirror_is_smaller(board: &Board) -> bool {
    for row in 0..SIDE {
        for column in 0..SIDE {
            let left = board[cell(row, column)];
            let right = board[cell(row, SIDE - 1 - column)];
            if left != right {
                return right < left;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> Input {
        Input::new(std::array::from_fn(|index| (index % 3 + 1) as u8))
    }

    #[test]
    fn next_flavor_is_canonical_one() {
        let input = input();
        for placed in 0..CANDY_COUNT {
            let mapping = dynamic_flavor_mapping(&input, placed);
            assert_eq!(mapping[input.flavors()[placed] as usize], 1);
        }
    }

    #[test]
    fn candidate_batch_has_documented_shape() {
        let boards = [[0; CANDY_COUNT]; ACTION_COUNT];
        let features = encode_candidates(&boards, 0, &input());
        assert_eq!(
            features.len(),
            ACTION_COUNT * FEATURE_CHANNELS * CANDY_COUNT
        );
    }

    #[test]
    fn reflection_has_same_normalized_board() {
        let mut board = [0; CANDY_COUNT];
        board[cell(0, 8)] = 1;
        board[cell(1, 7)] = 2;
        let mirrored = mirror_left_right(&board);
        let mapping = dynamic_flavor_mapping(&input(), 3);
        assert_eq!(
            normalized_board(&board, FRONT, &mapping),
            normalized_board(&mirrored, FRONT, &mapping)
        );
    }
}
