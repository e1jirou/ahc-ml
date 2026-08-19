use crate::game::{
    ACTION_COUNT, Action, BACK, Board, CANDY_COUNT, FRONT, Input, LEFT, RIGHT, SIDE, cell,
    potential,
};

pub const FEATURE_CHANNELS: usize = 18;
const COMPONENT_CHANNEL_START: usize = 4;
const TURN_CHANNEL: usize = 7;
const TOTAL_CHANNEL_START: usize = 8;
const REMAINING_CHANNEL_START: usize = 11;
const POTENTIAL_CHANNEL: usize = 14;
const SEQUENCE_CHANNEL_START: usize = 15;

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

    let component_planes = component_size_planes(&board);
    for flavor in 0..3 {
        let channel_base = sample_base + (COMPONENT_CHANNEL_START + flavor) * CANDY_COUNT;
        output[channel_base..channel_base + CANDY_COUNT].copy_from_slice(&component_planes[flavor]);
    }

    fill_plane(
        output,
        sample_base,
        TURN_CHANNEL,
        placed as f32 / CANDY_COUNT as f32,
    );
    let totals = input.totals();
    let mut remaining = [0; 3];
    for &flavor in &input.flavors()[placed..] {
        remaining[flavor as usize - 1] += 1;
    }
    for original in 0..3 {
        let canonical = mapping[original + 1] as usize - 1;
        fill_plane(
            output,
            sample_base,
            TOTAL_CHANNEL_START + canonical,
            totals[original] as f32 / CANDY_COUNT as f32,
        );
        fill_plane(
            output,
            sample_base,
            REMAINING_CHANNEL_START + canonical,
            remaining[original] as f32 / CANDY_COUNT as f32,
        );
    }
    fill_plane(
        output,
        sample_base,
        POTENTIAL_CHANNEL,
        potential(&board, input.denominator()),
    );

    for offset in 0..CANDY_COUNT - placed {
        let original = input.flavors()[placed + offset] as usize;
        let canonical = mapping[original] as usize - 1;
        output[sample_base + (SEQUENCE_CHANNEL_START + canonical) * CANDY_COUNT + offset] = 1.0;
    }
}

fn fill_plane(output: &mut [f32], sample_base: usize, channel: usize, value: f32) {
    let start = sample_base + channel * CANDY_COUNT;
    output[start..start + CANDY_COUNT].fill(value);
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

fn component_size_planes(board: &Board) -> [[f32; CANDY_COUNT]; 3] {
    let mut planes = [[0.0; CANDY_COUNT]; 3];
    let mut visited = [false; CANDY_COUNT];
    let mut stack = [0usize; CANDY_COUNT];
    let mut component = [0usize; CANDY_COUNT];
    for start in 0..CANDY_COUNT {
        let flavor = board[start];
        if flavor == 0 || visited[start] {
            continue;
        }
        let mut stack_size = 1;
        let mut component_size = 0;
        stack[0] = start;
        visited[start] = true;
        while stack_size > 0 {
            stack_size -= 1;
            let current = stack[stack_size];
            component[component_size] = current;
            component_size += 1;
            let row = current / SIDE;
            let column = current % SIDE;
            let mut neighbors = [usize::MAX; 4];
            if row > 0 {
                neighbors[0] = cell(row - 1, column);
            }
            if row + 1 < SIDE {
                neighbors[1] = cell(row + 1, column);
            }
            if column > 0 {
                neighbors[2] = cell(row, column - 1);
            }
            if column + 1 < SIDE {
                neighbors[3] = cell(row, column + 1);
            }
            for next in neighbors {
                if next != usize::MAX && !visited[next] && board[next] == flavor {
                    visited[next] = true;
                    stack[stack_size] = next;
                    stack_size += 1;
                }
            }
        }
        let value = component_size as f32 / CANDY_COUNT as f32;
        for &index in &component[..component_size] {
            planes[flavor as usize - 1][index] = value;
        }
    }
    planes
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
