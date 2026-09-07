pub const SIDE: usize = 10;
pub const CANDY_COUNT: usize = SIDE * SIDE;
pub const ACTION_COUNT: usize = 4;

pub type Board = [u8; CANDY_COUNT];
pub type Action = usize;

pub const FRONT: Action = 0;
pub const BACK: Action = 1;
pub const LEFT: Action = 2;
pub const RIGHT: Action = 3;

pub fn action_char(action: Action) -> char {
    match action {
        FRONT => 'F',
        BACK => 'B',
        LEFT => 'L',
        RIGHT => 'R',
        _ => panic!("invalid action: {action}"),
    }
}

#[derive(Clone, Debug)]
pub struct Input {
    flavors: [u8; CANDY_COUNT],
    totals: [usize; 3],
    first_positions: [usize; 3],
}

impl Input {
    pub fn new(flavors: [u8; CANDY_COUNT]) -> Self {
        let mut totals = [0; 3];
        let mut first_positions = [usize::MAX; 3];
        for (index, &flavor) in flavors.iter().enumerate() {
            assert!((1..=3).contains(&flavor));
            let flavor = flavor as usize - 1;
            totals[flavor] += 1;
            first_positions[flavor] = first_positions[flavor].min(index);
        }
        Self {
            flavors,
            totals,
            first_positions,
        }
    }

    pub fn flavors(&self) -> &[u8; CANDY_COUNT] {
        &self.flavors
    }

    pub fn totals(&self) -> [usize; 3] {
        self.totals
    }

    pub fn first_positions(&self) -> [usize; 3] {
        self.first_positions
    }
}

#[derive(Clone, Debug)]
pub struct State {
    board: Board,
    placed: usize,
}

impl State {
    pub fn new() -> Self {
        Self {
            board: [0; CANDY_COUNT],
            placed: 0,
        }
    }

    pub fn placed(&self) -> usize {
        self.placed
    }

    pub fn is_terminal(&self) -> bool {
        self.placed == CANDY_COUNT
    }

    pub fn place_at_rank(&mut self, rank: usize, flavor: u8) {
        assert!((1..=CANDY_COUNT - self.placed).contains(&rank));
        let mut empty_rank = 0;
        for cell in 0..CANDY_COUNT {
            if self.board[cell] == 0 {
                empty_rank += 1;
                if empty_rank == rank {
                    self.board[cell] = flavor;
                    self.placed += 1;
                    return;
                }
            }
        }
        unreachable!("validated rank must identify an empty cell");
    }

    pub fn afterstates(&self) -> [Board; ACTION_COUNT] {
        std::array::from_fn(|action| tilt(&self.board, action))
    }

    pub fn apply_action(&mut self, action: Action) {
        self.board = tilt(&self.board, action);
    }
}

pub const fn cell(row: usize, column: usize) -> usize {
    row * SIDE + column
}

pub fn tilt(board: &Board, action: Action) -> Board {
    let mut result = [0; CANDY_COUNT];
    match action {
        FRONT => {
            for column in 0..SIDE {
                let mut destination = 0;
                for row in 0..SIDE {
                    let flavor = board[cell(row, column)];
                    if flavor != 0 {
                        result[cell(destination, column)] = flavor;
                        destination += 1;
                    }
                }
            }
        }
        BACK => {
            for column in 0..SIDE {
                let mut destination = SIDE;
                for row in (0..SIDE).rev() {
                    let flavor = board[cell(row, column)];
                    if flavor != 0 {
                        destination -= 1;
                        result[cell(destination, column)] = flavor;
                    }
                }
            }
        }
        LEFT => {
            for row in 0..SIDE {
                let mut destination = 0;
                for column in 0..SIDE {
                    let flavor = board[cell(row, column)];
                    if flavor != 0 {
                        result[cell(row, destination)] = flavor;
                        destination += 1;
                    }
                }
            }
        }
        RIGHT => {
            for row in 0..SIDE {
                let mut destination = SIDE;
                for column in (0..SIDE).rev() {
                    let flavor = board[cell(row, column)];
                    if flavor != 0 {
                        destination -= 1;
                        result[cell(row, destination)] = flavor;
                    }
                }
            }
        }
        _ => panic!("invalid action: {action}"),
    }
    result
}

pub fn place_on_board_at_rank(board: &Board, rank: usize, flavor: u8) -> Board {
    let mut result = *board;
    let mut empty_rank = 0;
    for cell in &mut result {
        if *cell == 0 {
            empty_rank += 1;
            if empty_rank == rank {
                *cell = flavor;
                return result;
            }
        }
    }
    panic!("rank {rank} does not identify an empty cell");
}

pub fn connectivity_numerator(board: &Board) -> usize {
    let mut visited = [false; CANDY_COUNT];
    let mut stack = [0usize; CANDY_COUNT];
    let mut numerator = 0;
    for start in 0..CANDY_COUNT {
        let flavor = board[start];
        if flavor == 0 || visited[start] {
            continue;
        }
        let mut stack_size = 1;
        stack[0] = start;
        visited[start] = true;
        let mut component_size = 0;
        while stack_size > 0 {
            stack_size -= 1;
            let current = stack[stack_size];
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
        numerator += component_size * component_size;
    }
    numerator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tilt_preserves_order_and_compacts() {
        let mut board = [0; CANDY_COUNT];
        board[cell(2, 0)] = 1;
        board[cell(7, 0)] = 2;
        board[cell(1, 3)] = 3;
        let front = tilt(&board, FRONT);
        assert_eq!(front[cell(0, 0)], 1);
        assert_eq!(front[cell(1, 0)], 2);
        assert_eq!(front[cell(0, 3)], 3);
        let right = tilt(&board, RIGHT);
        assert_eq!(right[cell(2, 9)], 1);
        assert_eq!(right[cell(7, 9)], 2);
    }

    #[test]
    fn rank_is_row_major_after_a_tilt() {
        let mut state = State::new();
        state.place_at_rank(1, 1);
        state.apply_action(RIGHT);
        state.place_at_rank(1, 2);
        assert_eq!(state.board[0], 2);
        assert_eq!(state.board[9], 1);
    }

    #[test]
    fn component_score_matches_hand_calculation() {
        let mut board = [0; CANDY_COUNT];
        board[cell(0, 0)] = 1;
        board[cell(0, 1)] = 1;
        board[cell(2, 2)] = 1;
        board[cell(4, 4)] = 2;
        board[cell(5, 4)] = 2;
        assert_eq!(connectivity_numerator(&board), 4 + 1 + 4);
    }
}
