//! Azul Game Engine
//!
//! A Markov game state engine for the board game Azul, designed for RL training.
//! Core object is a single `GameState` (plain data). No logic baked into methods;
//! pure functions operate on it.

use rand::Rng;

// =============================================================================
// Section 3.1: Basic types and constants
// =============================================================================

/// Index into players array: 0..num_players-1
pub type PlayerIdx = u8;

/// Row index (0..=4)
pub type Row = u8;

/// Column index (0..=4)
pub type Col = u8;

pub const BOARD_SIZE: usize = 5;
pub const MAX_PLAYERS: usize = 4;
pub const MAX_FACTORIES: usize = 9;
pub const FACTORY_CAPACITY: usize = 4;
pub const FLOOR_CAPACITY: usize = 7;
pub const TILE_COLORS: usize = 5;
pub const TILES_PER_COLOR: u8 = 20;

/// Tile colors (order fixed for serialization)
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Color {
    Blue = 0,
    Yellow = 1,
    Red = 2,
    Black = 3,
    Teal = 4, // "light blue"
}

impl Color {
    /// Convert from u8 index to Color
    pub fn from_index(idx: u8) -> Option<Color> {
        match idx {
            0 => Some(Color::Blue),
            1 => Some(Color::Yellow),
            2 => Some(Color::Red),
            3 => Some(Color::Black),
            4 => Some(Color::Teal),
            _ => None,
        }
    }
}

pub const ALL_COLORS: [Color; TILE_COLORS] = [
    Color::Blue,
    Color::Yellow,
    Color::Red,
    Color::Black,
    Color::Teal,
];

/// Floor penalties (fixed table)
pub const FLOOR_PENALTY: [i8; FLOOR_CAPACITY] = [-1, -1, -2, -2, -2, -3, -3];

/// Game phase / status
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Phase {
    FactoryOffer, // Players drafting
    GameOver,     // Terminal; no more actions
}

// =============================================================================
// Section 3.2: Wall layout constants (colored side)
// =============================================================================

/// Wall pattern: wall_pattern[row][col] = Color at that position
/// This is the fixed Latin-square pattern on the colored side of the player board.
pub const WALL_PATTERN: [[Color; BOARD_SIZE]; BOARD_SIZE] = [
    // row 0
    [
        Color::Blue,
        Color::Yellow,
        Color::Red,
        Color::Black,
        Color::Teal,
    ],
    // row 1
    [
        Color::Teal,
        Color::Blue,
        Color::Yellow,
        Color::Red,
        Color::Black,
    ],
    // row 2
    [
        Color::Black,
        Color::Teal,
        Color::Blue,
        Color::Yellow,
        Color::Red,
    ],
    // row 3
    [
        Color::Red,
        Color::Black,
        Color::Teal,
        Color::Blue,
        Color::Yellow,
    ],
    // row 4
    [
        Color::Yellow,
        Color::Red,
        Color::Black,
        Color::Teal,
        Color::Blue,
    ],
];

/// Destination column lookup: WALL_DEST_COL[row][color_index] => col
/// Precomputed from WALL_PATTERN for O(1) lookup.
pub const WALL_DEST_COL: [[u8; TILE_COLORS]; BOARD_SIZE] = [
    // row 0: Blue=0, Yellow=1, Red=2, Black=3, Teal=4
    [0, 1, 2, 3, 4],
    // row 1: Teal=0, Blue=1, Yellow=2, Red=3, Black=4
    [1, 2, 3, 4, 0],
    // row 2: Black=0, Teal=1, Blue=2, Yellow=3, Red=4
    [2, 3, 4, 0, 1],
    // row 3: Red=0, Black=1, Teal=2, Blue=3, Yellow=4
    [3, 4, 0, 1, 2],
    // row 4: Yellow=0, Red=1, Black=2, Teal=3, Blue=4
    [4, 0, 1, 2, 3],
];

// =============================================================================
// Section 3.3: Factory and Center data structures
// =============================================================================

/// A single factory display (holds up to 4 tiles)
#[derive(Clone, Debug)]
pub struct Factory {
    pub len: u8,
    pub tiles: [Color; FACTORY_CAPACITY],
}

impl Default for Factory {
    fn default() -> Self {
        Factory {
            len: 0,
            tiles: [Color::Blue; FACTORY_CAPACITY], // placeholder values
        }
    }
}

/// All factories (max 9; only first `num_factories` valid)
#[derive(Clone, Debug)]
pub struct Factories {
    pub num_factories: u8, // 5, 7, or 9
    pub factories: [Factory; MAX_FACTORIES],
}

impl Default for Factories {
    fn default() -> Self {
        Factories {
            num_factories: 0,
            factories: std::array::from_fn(|_| Factory::default()),
        }
    }
}

/// Token in center pool or floor line
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Token {
    Tile(Color),
    FirstPlayerMarker,
}

/// Center pool: contains colored tiles plus possibly the first player marker
#[derive(Clone, Debug)]
pub struct CenterPool {
    pub len: u8,
    pub items: [Token; 100], // upper bound: all tiles could theoretically be here
}

impl Default for CenterPool {
    fn default() -> Self {
        CenterPool {
            len: 0,
            items: [Token::Tile(Color::Blue); 100], // placeholder
        }
    }
}

// =============================================================================
// Section 3.4: Bag and discard (tile supply)
// =============================================================================

/// Tile supply: counts of tiles in bag and discard
#[derive(Clone, Debug)]
pub struct TileSupply {
    /// Count of tiles of each color in the bag
    pub bag: [u8; TILE_COLORS],
    /// Count of tiles of each color in the discard ("lid")
    pub discard: [u8; TILE_COLORS],
}

impl Default for TileSupply {
    fn default() -> Self {
        TileSupply {
            bag: [0; TILE_COLORS],
            discard: [0; TILE_COLORS],
        }
    }
}

// =============================================================================
// Section 3.5: Player board state structures
// =============================================================================

/// A single pattern line (one of 5 rows, capacities 1-5)
#[derive(Copy, Clone, Debug, Default)]
pub struct PatternLine {
    pub color: Option<Color>, // None => empty; Some(c) => all tiles are c
    pub count: u8,            // 0..=capacity(row_index)
}

/// Wall: 5x5 grid, each cell either empty (None) or occupied by a color
pub type Wall = [[Option<Color>; BOARD_SIZE]; BOARD_SIZE];

/// Floor line: ordered sequence of tokens (tiles + maybe FirstPlayerMarker)
#[derive(Clone, Debug)]
pub struct FloorLine {
    pub len: u8,
    pub slots: [Token; FLOOR_CAPACITY],
}

impl Default for FloorLine {
    fn default() -> Self {
        FloorLine {
            len: 0,
            slots: [Token::Tile(Color::Blue); FLOOR_CAPACITY], // placeholder
        }
    }
}

/// Complete state for one player
#[derive(Clone, Debug)]
pub struct PlayerState {
    pub wall: Wall,
    pub pattern_lines: [PatternLine; BOARD_SIZE],
    pub floor: FloorLine,
    pub score: i16, // can go negative from floor penalties
}

impl Default for PlayerState {
    fn default() -> Self {
        PlayerState {
            wall: [[None; BOARD_SIZE]; BOARD_SIZE],
            pattern_lines: [PatternLine::default(); BOARD_SIZE],
            floor: FloorLine::default(),
            score: 0,
        }
    }
}

// =============================================================================
// Section 3.6: Top-level GameState
// =============================================================================

/// Complete game state - fully Markov (no history needed)
#[derive(Clone, Debug)]
pub struct GameState {
    pub num_players: u8,                     // 2..=4
    pub players: [PlayerState; MAX_PLAYERS], // only 0..num_players used

    pub factories: Factories,
    pub center: CenterPool,
    pub supply: TileSupply,

    /// Whose turn it is in FactoryOffer phase
    pub current_player: PlayerIdx,

    /// Who will start the *next* round (determined by who took FP marker this round)
    pub starting_player_next_round: PlayerIdx,

    /// Is the game in drafting or over?
    pub phase: Phase,

    /// Number of rounds completed so far (0 at game start)
    pub round: u16,

    /// Whether this is the *final* round (at least one player has a full horizontal row)
    pub final_round_triggered: bool,
}

impl Default for GameState {
    fn default() -> Self {
        GameState {
            num_players: 2,
            players: std::array::from_fn(|_| PlayerState::default()),
            factories: Factories::default(),
            center: CenterPool::default(),
            supply: TileSupply::default(),
            current_player: 0,
            starting_player_next_round: 0,
            phase: Phase::FactoryOffer,
            round: 0,
            final_round_triggered: false,
        }
    }
}

// =============================================================================
// Section 4: Action representation
// =============================================================================

/// Source of tiles for drafting
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DraftSource {
    Factory(u8), // index 0..num_factories-1
    Center,
}

/// Destination for drafted tiles
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DraftDestination {
    PatternLine(Row), // 0..4
    Floor,
}

/// A player action: draft tiles of a color from a source to a destination
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Action {
    pub source: DraftSource,
    pub color: Color,
    pub dest: DraftDestination,
}

// =============================================================================
// Section 5: State transitions and rules as pure functions
// =============================================================================

/// Error types for apply_action
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ApplyError {
    NotPlayersTurn,
    WrongPhase,
    IllegalAction,
}

/// Result of applying an action
#[derive(Clone, Debug)]
pub struct StepResult {
    pub state: GameState,
    /// If Some, the game became terminal and this is final per-player score
    pub final_scores: Option<[i16; MAX_PLAYERS]>,
}

/// Get the number of factories for a given player count
fn num_factories_for_players(num_players: u8) -> u8 {
    match num_players {
        2 => 5,
        3 => 7,
        4 => 9,
        _ => panic!("Invalid number of players: {num_players}"),
    }
}

/// Draw a single random tile from the bag (refilling from discard if needed)
/// Returns None if both bag and discard are empty
fn draw_tile(supply: &mut TileSupply, rng: &mut impl Rng) -> Option<Color> {
    let bag_total: u8 = supply.bag.iter().sum();

    if bag_total == 0 {
        // Refill bag from discard
        let discard_total: u8 = supply.discard.iter().sum();
        if discard_total == 0 {
            return None; // Both empty
        }
        supply.bag = supply.discard;
        supply.discard = [0; TILE_COLORS];
    }

    let bag_total: u8 = supply.bag.iter().sum();
    if bag_total == 0 {
        return None;
    }

    // Pick a random tile from the bag
    let mut pick = rng.random_range(0..bag_total as u32) as u8;
    for (i, &count) in supply.bag.iter().enumerate() {
        if pick < count {
            supply.bag[i] -= 1;
            return Color::from_index(i as u8);
        }
        pick -= count;
    }

    unreachable!("Should have picked a tile")
}

/// Initialize a new game with 2-4 players
pub fn new_game(num_players: u8, starting_player: PlayerIdx, rng: &mut impl Rng) -> GameState {
    assert!((2..=4).contains(&num_players), "Must have 2-4 players");
    assert!(starting_player < num_players, "Invalid starting player");

    let mut state = GameState {
        num_players,
        players: std::array::from_fn(|_| PlayerState::default()),
        factories: Factories::default(),
        center: CenterPool::default(),
        supply: TileSupply {
            bag: [TILES_PER_COLOR; TILE_COLORS], // 20 of each color
            discard: [0; TILE_COLORS],
        },
        current_player: starting_player,
        starting_player_next_round: starting_player,
        phase: Phase::FactoryOffer,
        round: 0,
        final_round_triggered: false,
    };

    // Set up factories
    let num_fac = num_factories_for_players(num_players);
    state.factories.num_factories = num_fac;

    // Fill each factory with 4 tiles
    for f in 0..num_fac as usize {
        state.factories.factories[f].len = 0;
        for slot in 0..FACTORY_CAPACITY {
            if let Some(color) = draw_tile(&mut state.supply, rng) {
                state.factories.factories[f].tiles[slot] = color;
                state.factories.factories[f].len += 1;
            }
        }
    }

    // Place first player marker in center
    state.center.len = 1;
    state.center.items[0] = Token::FirstPlayerMarker;

    state
}

/// Enumerate all legal actions for the current player
pub fn legal_actions(state: &GameState) -> Vec<Action> {
    if state.phase != Phase::FactoryOffer {
        return Vec::new();
    }

    let mut actions = Vec::new();
    let p = state.current_player as usize;
    let player = &state.players[p];

    // Helper: check if a (source, color) -> dest is legal
    let check_dest = |color: Color, dest: DraftDestination| -> bool {
        match dest {
            DraftDestination::Floor => true, // Always allowed
            DraftDestination::PatternLine(row) => {
                let r = row as usize;
                let cap = (r + 1) as u8;
                let line = &player.pattern_lines[r];

                // Wall constraint: can't place if color already in wall row
                let dest_col = WALL_DEST_COL[r][color as usize] as usize;
                if player.wall[r][dest_col].is_some() {
                    return false;
                }

                // Pattern line homogeneity
                if let Some(existing_color) = line.color {
                    if existing_color != color {
                        return false;
                    }
                }

                // Line not already full
                if line.count >= cap {
                    return false;
                }

                true
            }
        }
    };

    // Collect (source, color) pairs from factories
    for f in 0..state.factories.num_factories as usize {
        let factory = &state.factories.factories[f];
        if factory.len == 0 {
            continue;
        }

        // Get unique colors in this factory
        let mut seen_colors = [false; TILE_COLORS];
        for i in 0..factory.len as usize {
            let color = factory.tiles[i];
            if !seen_colors[color as usize] {
                seen_colors[color as usize] = true;

                // Check each destination
                for r in 0..BOARD_SIZE {
                    if check_dest(color, DraftDestination::PatternLine(r as u8)) {
                        actions.push(Action {
                            source: DraftSource::Factory(f as u8),
                            color,
                            dest: DraftDestination::PatternLine(r as u8),
                        });
                    }
                }
                // Floor always allowed
                actions.push(Action {
                    source: DraftSource::Factory(f as u8),
                    color,
                    dest: DraftDestination::Floor,
                });
            }
        }
    }

    // Collect colors from center (ignoring FirstPlayerMarker)
    let mut center_colors = [false; TILE_COLORS];
    for i in 0..state.center.len as usize {
        if let Token::Tile(color) = state.center.items[i] {
            center_colors[color as usize] = true;
        }
    }

    for (ci, &present) in center_colors.iter().enumerate() {
        if present {
            let color = Color::from_index(ci as u8).unwrap();

            // Check each destination
            for r in 0..BOARD_SIZE {
                if check_dest(color, DraftDestination::PatternLine(r as u8)) {
                    actions.push(Action {
                        source: DraftSource::Center,
                        color,
                        dest: DraftDestination::PatternLine(r as u8),
                    });
                }
            }
            // Floor always allowed
            actions.push(Action {
                source: DraftSource::Center,
                color,
                dest: DraftDestination::Floor,
            });
        }
    }

    actions
}

/// Score a tile placement on the wall
fn score_placement(wall: &Wall, row: usize, col: usize) -> i16 {
    // Count horizontal adjacency (including placed tile)
    let mut horiz = 1;
    // Left
    let mut c = col;
    while c > 0 {
        c -= 1;
        if wall[row][c].is_some() {
            horiz += 1;
        } else {
            break;
        }
    }
    // Right
    c = col;
    while c < BOARD_SIZE - 1 {
        c += 1;
        if wall[row][c].is_some() {
            horiz += 1;
        } else {
            break;
        }
    }

    // Count vertical adjacency (including placed tile)
    let mut vert = 1;
    // Up
    let mut r = row;
    while r > 0 {
        r -= 1;
        if wall[r][col].is_some() {
            vert += 1;
        } else {
            break;
        }
    }
    // Down
    r = row;
    while r < BOARD_SIZE - 1 {
        r += 1;
        if wall[r][col].is_some() {
            vert += 1;
        } else {
            break;
        }
    }

    // Scoring logic
    if horiz == 1 && vert == 1 {
        // Isolated tile
        1
    } else {
        let h_score = if horiz > 1 { horiz } else { 0 };
        let v_score = if vert > 1 { vert } else { 0 };
        h_score + v_score
    }
}

/// Apply final scoring bonuses
fn apply_final_scoring(state: &mut GameState) -> [i16; MAX_PLAYERS] {
    let mut scores = [0i16; MAX_PLAYERS];

    for (player, score) in state.players[..state.num_players as usize]
        .iter_mut()
        .zip(scores[..state.num_players as usize].iter_mut())
    {
        // +2 per complete horizontal row
        for row in 0..BOARD_SIZE {
            let complete = (0..BOARD_SIZE).all(|col| player.wall[row][col].is_some());
            if complete {
                player.score += 2;
            }
        }

        // +7 per complete vertical column
        for col in 0..BOARD_SIZE {
            let complete = (0..BOARD_SIZE).all(|row| player.wall[row][col].is_some());
            if complete {
                player.score += 7;
            }
        }

        // +10 per complete color set (all 5 tiles of one color on wall)
        for color in ALL_COLORS {
            let mut count = 0;
            for row in 0..BOARD_SIZE {
                for col in 0..BOARD_SIZE {
                    if player.wall[row][col] == Some(color) {
                        count += 1;
                    }
                }
            }
            if count == 5 {
                player.score += 10;
            }
        }

        *score = player.score;
    }

    scores
}

/// Refill factories from the bag
fn refill_factories(supply: &mut TileSupply, factories: &mut Factories, rng: &mut impl Rng) {
    for f in 0..factories.num_factories as usize {
        factories.factories[f].len = 0;
        for slot in 0..FACTORY_CAPACITY {
            if let Some(color) = draw_tile(supply, rng) {
                factories.factories[f].tiles[slot] = color;
                factories.factories[f].len += 1;
            }
        }
    }
}

/// Resolve end of round (wall tiling, scoring, prepare next round or end game)
fn resolve_end_of_round(state: &mut GameState, rng: &mut impl Rng) -> Option<[i16; MAX_PLAYERS]> {
    // Step 1: Wall tiling and placement scoring for each player
    for p in 0..state.num_players as usize {
        let player = &mut state.players[p];

        for (r, dest_cols) in WALL_DEST_COL.iter().enumerate() {
            let cap = (r + 1) as u8;
            let line = &player.pattern_lines[r];

            if line.count == cap {
                let color = line.color.expect("complete line must have color");

                // Find destination column
                let col = dest_cols[color as usize] as usize;

                // Place tile on wall
                debug_assert!(player.wall[r][col].is_none());
                player.wall[r][col] = Some(color);

                // Score placement
                let delta = score_placement(&player.wall, r, col);
                player.score += delta;

                // Discard remaining tiles (cap - 1)
                let tiles_to_discard = cap - 1;
                state.supply.discard[color as usize] += tiles_to_discard;

                // Reset pattern line
                // Need to use index since we borrowed player mutably
            }
        }

        // Reset completed pattern lines (separate loop to avoid borrow issues)
        for r in 0..BOARD_SIZE {
            let cap = (r + 1) as u8;
            if player.pattern_lines[r].count == cap {
                player.pattern_lines[r].color = None;
                player.pattern_lines[r].count = 0;
            }
        }
    }

    // Step 2: Floor cleanup (penalties already applied immediately when tiles were placed)
    // We only need to discard tiles and detect first player marker here.
    for p in 0..state.num_players as usize {
        let player = &mut state.players[p];
        let mut fp_marker_here = false;

        for slot in player.floor.slots[..player.floor.len as usize].iter() {
            match *slot {
                Token::Tile(color) => {
                    // Discard tile (penalty was already applied when placed)
                    state.supply.discard[color as usize] += 1;
                }
                Token::FirstPlayerMarker => {
                    // Penalty was already applied when placed
                    fp_marker_here = true;
                }
            }
        }
        player.floor.len = 0;

        if fp_marker_here {
            state.starting_player_next_round = p as u8;
        }
    }

    // Step 3: Check for game end (any player has complete horizontal row)
    for p in 0..state.num_players as usize {
        for row in 0..BOARD_SIZE {
            let complete = (0..BOARD_SIZE).all(|col| state.players[p].wall[row][col].is_some());
            if complete {
                state.final_round_triggered = true;
                break;
            }
        }
        if state.final_round_triggered {
            break;
        }
    }

    if state.final_round_triggered {
        let scores = apply_final_scoring(state);
        state.phase = Phase::GameOver;
        return Some(scores);
    }

    // Step 4: Prepare next round
    state.round += 1;
    refill_factories(&mut state.supply, &mut state.factories, rng);
    state.center.len = 1;
    state.center.items[0] = Token::FirstPlayerMarker;
    state.current_player = state.starting_player_next_round;

    None
}

/// Apply a player action to the game state
pub fn apply_action(
    mut state: GameState,
    action: Action,
    rng: &mut impl Rng,
) -> Result<StepResult, ApplyError> {
    // Validity checks
    if state.phase != Phase::FactoryOffer {
        return Err(ApplyError::WrongPhase);
    }

    // Verify action is legal
    let legal = legal_actions(&state);
    if !legal.contains(&action) {
        return Err(ApplyError::IllegalAction);
    }

    let p = state.current_player as usize;

    // Step 1: Extract tiles from source
    let mut taken: Vec<Color> = Vec::new();
    let mut took_first_player_marker = false;

    match action.source {
        DraftSource::Factory(f) => {
            let factory = &mut state.factories.factories[f as usize];

            // Collect tiles of the chosen color, move others to center
            for i in 0..factory.len as usize {
                let tile_color = factory.tiles[i];
                if tile_color == action.color {
                    taken.push(tile_color);
                } else {
                    // Move to center
                    state.center.items[state.center.len as usize] = Token::Tile(tile_color);
                    state.center.len += 1;
                }
            }
            factory.len = 0;
        }
        DraftSource::Center => {
            // Collect tiles of chosen color and check for first player marker
            let mut new_center_items = Vec::new();

            for i in 0..state.center.len as usize {
                match state.center.items[i] {
                    Token::Tile(c) if c == action.color => {
                        taken.push(c);
                    }
                    Token::FirstPlayerMarker => {
                        took_first_player_marker = true;
                        // Don't keep in center
                    }
                    other => {
                        new_center_items.push(other);
                    }
                }
            }

            // Rebuild center
            for (i, item) in new_center_items.iter().enumerate() {
                state.center.items[i] = *item;
            }
            state.center.len = new_center_items.len() as u8;
        }
    }

    if took_first_player_marker {
        state.starting_player_next_round = p as u8;
    }

    // Step 2: Place tiles into destination
    let player = &mut state.players[p];

    // Helper to add tile to floor (with overflow to discard)
    // IMPORTANT: Floor penalties are applied IMMEDIATELY when tiles are placed,
    // not at end of round. This gives MCTS/RL agents immediate negative signal.
    let add_to_floor = |player: &mut PlayerState, supply: &mut TileSupply, token: Token| {
        if player.floor.len < FLOOR_CAPACITY as u8 {
            let slot_idx = player.floor.len as usize;
            player.floor.slots[slot_idx] = token;
            // Apply penalty immediately for this slot
            player.score += FLOOR_PENALTY[slot_idx] as i16;
            player.floor.len += 1;
        } else {
            // Overflow: discard tile (marker never discarded but also doesn't overflow in practice)
            if let Token::Tile(c) = token {
                supply.discard[c as usize] += 1;
            }
        }
    };

    match action.dest {
        DraftDestination::Floor => {
            // All tiles go to floor
            for &tile in &taken {
                add_to_floor(player, &mut state.supply, Token::Tile(tile));
            }
            if took_first_player_marker {
                add_to_floor(player, &mut state.supply, Token::FirstPlayerMarker);
            }
        }
        DraftDestination::PatternLine(row) => {
            let r = row as usize;
            let line = &mut player.pattern_lines[r];
            let cap = (r + 1) as u8;

            // Set color if empty
            if line.color.is_none() {
                line.color = Some(action.color);
            }

            // Fill pattern line
            let space = cap - line.count;
            let n_to_line = std::cmp::min(space as usize, taken.len());
            line.count += n_to_line as u8;

            // Overflow to floor
            for &tile in &taken[n_to_line..] {
                add_to_floor(player, &mut state.supply, Token::Tile(tile));
            }

            // First player marker to floor
            if took_first_player_marker {
                add_to_floor(player, &mut state.supply, Token::FirstPlayerMarker);
            }
        }
    }

    // Step 3: Check if round ended (all factories empty and center has no tiles)
    let all_factories_empty =
        (0..state.factories.num_factories as usize).all(|f| state.factories.factories[f].len == 0);

    let center_has_tiles =
        (0..state.center.len as usize).any(|i| matches!(state.center.items[i], Token::Tile(_)));

    if all_factories_empty && !center_has_tiles {
        // End of round
        let final_scores = resolve_end_of_round(&mut state, rng);
        Ok(StepResult {
            state,
            final_scores,
        })
    } else {
        // Advance to next player
        state.current_player =
            ((state.current_player as usize + 1) % state.num_players as usize) as u8;
        Ok(StepResult {
            state,
            final_scores: None,
        })
    }
}

// =============================================================================
// Debug assertions for invariants (Section 8.2)
// =============================================================================

#[cfg(debug_assertions)]
pub fn assert_tile_invariants(state: &GameState) {
    for color in ALL_COLORS {
        let ci = color as usize;
        let mut total = 0u8;

        // Bag
        total += state.supply.bag[ci];

        // Discard
        total += state.supply.discard[ci];

        // Factories
        for f in 0..state.factories.num_factories as usize {
            let factory = &state.factories.factories[f];
            for i in 0..factory.len as usize {
                if factory.tiles[i] == color {
                    total += 1;
                }
            }
        }

        // Center
        for i in 0..state.center.len as usize {
            if let Token::Tile(c) = state.center.items[i] {
                if c == color {
                    total += 1;
                }
            }
        }

        // Players
        for p in 0..state.num_players as usize {
            let player = &state.players[p];

            // Pattern lines
            for r in 0..BOARD_SIZE {
                if player.pattern_lines[r].color == Some(color) {
                    total += player.pattern_lines[r].count;
                }
            }

            // Wall
            for row in 0..BOARD_SIZE {
                for col in 0..BOARD_SIZE {
                    if player.wall[row][col] == Some(color) {
                        total += 1;
                    }
                }
            }

            // Floor
            for i in 0..player.floor.len as usize {
                if let Token::Tile(c) = player.floor.slots[i] {
                    if c == color {
                        total += 1;
                    }
                }
            }
        }

        assert_eq!(
            total, TILES_PER_COLOR,
            "Tile count invariant violated for {color:?}: expected {TILES_PER_COLOR}, got {total}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    // =========================================================================
    // Basic game state tests
    // =========================================================================

    #[test]
    fn test_new_game_creates_valid_state() {
        let mut rng = StdRng::seed_from_u64(42);
        let state = new_game(2, 0, &mut rng);

        assert_eq!(state.num_players, 2);
        assert_eq!(state.factories.num_factories, 5);
        assert_eq!(state.current_player, 0);
        assert_eq!(state.phase, Phase::FactoryOffer);

        // Check first player marker in center
        assert!(state.center.len >= 1);
        assert!(matches!(state.center.items[0], Token::FirstPlayerMarker));

        // Verify tile invariants
        assert_tile_invariants(&state);
    }

    #[test]
    fn test_new_game_3_players() {
        let mut rng = StdRng::seed_from_u64(42);
        let state = new_game(3, 0, &mut rng);

        assert_eq!(state.num_players, 3);
        assert_eq!(state.factories.num_factories, 7);
        assert_tile_invariants(&state);
    }

    #[test]
    fn test_new_game_4_players() {
        let mut rng = StdRng::seed_from_u64(42);
        let state = new_game(4, 0, &mut rng);

        assert_eq!(state.num_players, 4);
        assert_eq!(state.factories.num_factories, 9);
        assert_tile_invariants(&state);
    }

    // =========================================================================
    // Tile count invariant tests (azul-e9k.22)
    // =========================================================================

    #[test]
    fn test_tile_invariants_throughout_game() {
        let mut rng = StdRng::seed_from_u64(99999);
        let mut state = new_game(2, 0, &mut rng);

        // Check invariants after every action
        for _ in 0..100 {
            if state.phase == Phase::GameOver {
                break;
            }
            assert_tile_invariants(&state);

            let actions = legal_actions(&state);
            if actions.is_empty() {
                break;
            }

            let action_idx = rng.random_range(0..actions.len() as u32) as usize;
            let result = apply_action(state, actions[action_idx], &mut rng).unwrap();
            state = result.state;
        }
        assert_tile_invariants(&state);
    }

    // =========================================================================
    // Pattern line and floor invariant tests (azul-e9k.23)
    // =========================================================================

    #[test]
    fn test_pattern_line_invariants() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut state = new_game(2, 0, &mut rng);

        for _ in 0..50 {
            if state.phase == Phase::GameOver {
                break;
            }

            // Check pattern line invariants for all players
            for p in 0..state.num_players as usize {
                for r in 0..BOARD_SIZE {
                    let line = &state.players[p].pattern_lines[r];
                    let cap = (r + 1) as u8;

                    // count <= capacity
                    assert!(line.count <= cap, "Pattern line count exceeds capacity");

                    // count == 0 iff color == None
                    if line.count == 0 {
                        assert!(line.color.is_none(), "Empty line should have no color");
                    } else {
                        assert!(line.color.is_some(), "Non-empty line should have a color");
                    }
                }

                // Floor length <= capacity
                assert!(
                    state.players[p].floor.len <= FLOOR_CAPACITY as u8,
                    "Floor length exceeds capacity"
                );

                // Player indices valid
                assert!(state.current_player < state.num_players);
                assert!(state.starting_player_next_round < state.num_players);
            }

            let actions = legal_actions(&state);
            if actions.is_empty() {
                break;
            }
            let action_idx = rng.random_range(0..actions.len() as u32) as usize;
            let result = apply_action(state, actions[action_idx], &mut rng).unwrap();
            state = result.state;
        }
    }

    #[test]
    fn test_wall_positions_match_pattern() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut state = new_game(2, 0, &mut rng);

        // Play a few rounds to get some tiles on walls
        for _ in 0..100 {
            if state.phase == Phase::GameOver {
                break;
            }
            let actions = legal_actions(&state);
            if actions.is_empty() {
                break;
            }
            let action_idx = rng.random_range(0..actions.len() as u32) as usize;
            let result = apply_action(state, actions[action_idx], &mut rng).unwrap();
            state = result.state;
        }

        // Verify wall positions match WALL_PATTERN
        for p in 0..state.num_players as usize {
            for row in 0..BOARD_SIZE {
                for col in 0..BOARD_SIZE {
                    if let Some(color) = state.players[p].wall[row][col] {
                        assert_eq!(
                            color, WALL_PATTERN[row][col],
                            "Wall color at ({},{}) doesn't match pattern",
                            row, col
                        );
                    }
                }
            }
        }
    }

    // =========================================================================
    // Placement scoring tests (azul-e9k.24)
    // =========================================================================

    #[test]
    fn test_score_placement_isolated() {
        let mut wall: Wall = [[None; BOARD_SIZE]; BOARD_SIZE];
        wall[2][2] = Some(Color::Blue);

        let score = score_placement(&wall, 2, 2);
        assert_eq!(score, 1, "Isolated tile should score 1");
    }

    #[test]
    fn test_score_placement_horizontal() {
        let mut wall: Wall = [[None; BOARD_SIZE]; BOARD_SIZE];
        wall[2][1] = Some(Color::Blue);
        wall[2][2] = Some(Color::Yellow);
        wall[2][3] = Some(Color::Red);

        let score = score_placement(&wall, 2, 2);
        assert_eq!(score, 3, "Horizontal line of 3 should score 3");
    }

    #[test]
    fn test_score_placement_vertical() {
        let mut wall: Wall = [[None; BOARD_SIZE]; BOARD_SIZE];
        wall[1][2] = Some(Color::Blue);
        wall[2][2] = Some(Color::Yellow);
        wall[3][2] = Some(Color::Red);

        let score = score_placement(&wall, 2, 2);
        assert_eq!(score, 3, "Vertical line of 3 should score 3");
    }

    #[test]
    fn test_score_placement_cross() {
        let mut wall: Wall = [[None; BOARD_SIZE]; BOARD_SIZE];
        wall[2][1] = Some(Color::Blue);
        wall[2][2] = Some(Color::Yellow);
        wall[2][3] = Some(Color::Red);
        wall[1][2] = Some(Color::Black);
        wall[3][2] = Some(Color::Teal);

        let score = score_placement(&wall, 2, 2);
        assert_eq!(score, 6, "Cross shape (3 horiz + 3 vert) should score 6");
    }

    #[test]
    fn test_score_placement_corner() {
        let mut wall: Wall = [[None; BOARD_SIZE]; BOARD_SIZE];
        wall[0][0] = Some(Color::Blue);

        let score = score_placement(&wall, 0, 0);
        assert_eq!(score, 1, "Corner isolated tile should score 1");
    }

    #[test]
    fn test_score_placement_edge_horizontal() {
        let mut wall: Wall = [[None; BOARD_SIZE]; BOARD_SIZE];
        wall[0][0] = Some(Color::Blue);
        wall[0][1] = Some(Color::Yellow);

        let score = score_placement(&wall, 0, 0);
        assert_eq!(score, 2, "Edge tile with one neighbor should score 2");
    }

    // =========================================================================
    // Floor penalty tests (azul-e9k.25)
    // =========================================================================

    #[test]
    fn test_floor_penalty_calculation() {
        let total_penalty: i16 = FLOOR_PENALTY.iter().map(|&p| p as i16).sum();
        assert_eq!(total_penalty, -14, "Full floor penalty should be -14");
    }

    #[test]
    fn test_floor_partial_penalties() {
        // Verify individual penalties
        assert_eq!(FLOOR_PENALTY[0], -1);
        assert_eq!(FLOOR_PENALTY[1], -1);
        assert_eq!(FLOOR_PENALTY[2], -2);
        assert_eq!(FLOOR_PENALTY[3], -2);
        assert_eq!(FLOOR_PENALTY[4], -2);
        assert_eq!(FLOOR_PENALTY[5], -3);
        assert_eq!(FLOOR_PENALTY[6], -3);
    }

    #[test]
    fn test_floor_penalty_allows_negative_scores() {
        // Verify that floor penalties can cause negative scores.
        // With immediate penalties, we test this via apply_action.
        let mut rng = StdRng::seed_from_u64(42);
        let mut state = new_game(2, 0, &mut rng);

        // Set player to low score
        state.players[0].score = 3;

        // Add a tile to center so we can pick from it
        state.center.items[state.center.len as usize] = Token::Tile(Color::Blue);
        state.center.len += 1;
        state.center.items[state.center.len as usize] = Token::Tile(Color::Blue);
        state.center.len += 1;
        state.center.items[state.center.len as usize] = Token::Tile(Color::Blue);
        state.center.len += 1;
        state.center.items[state.center.len as usize] = Token::Tile(Color::Blue);
        state.center.len += 1;
        state.center.items[state.center.len as usize] = Token::Tile(Color::Blue);
        state.center.len += 1;

        // Pick tiles to floor (will get 5 tiles + first player marker = 6 items)
        // Penalties: -1 -1 -2 -2 -2 -3 = -11 for first 6 slots
        let action = Action {
            source: DraftSource::Center,
            color: Color::Blue,
            dest: DraftDestination::Floor,
        };

        let result = apply_action(state, action, &mut rng).unwrap();

        // Score should now be negative: 3 + (-1 -1 -2 -2 -2 -3) = 3 - 11 = -8
        assert!(
            result.state.players[0].score < 0,
            "Score should be allowed to go negative; got {}",
            result.state.players[0].score
        );
        assert_eq!(
            result.state.players[0].score, -8,
            "Score should be 3 - 11 = -8, got {}",
            result.state.players[0].score
        );
    }

    /// Test A: Immediate floor penalty - verify score decreases immediately when tiles placed
    #[test]
    fn test_immediate_floor_penalty() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut state = new_game(2, 0, &mut rng);

        // Record initial score
        let initial_score = state.players[0].score;
        assert_eq!(initial_score, 0, "Initial score should be 0");

        // Add exactly 1 tile to center
        state.center.items[state.center.len as usize] = Token::Tile(Color::Red);
        state.center.len += 1;

        // Pick that single tile and send it to floor
        let action = Action {
            source: DraftSource::Center,
            color: Color::Red,
            dest: DraftDestination::Floor,
        };

        let result = apply_action(state, action, &mut rng).unwrap();

        // The single tile goes to floor slot 0, plus first player marker goes to slot 1
        // Penalties: slot 0 = -1, slot 1 = -1 (for FP marker)
        // Total immediate penalty = -2
        assert_eq!(
            result.state.players[0].score, -2,
            "Score should be -2 after placing 1 tile + FP marker on floor (slots 0 and 1), got {}",
            result.state.players[0].score
        );

        // Verify floor has 2 items (tile + FP marker)
        assert_eq!(
            result.state.players[0].floor.len, 2,
            "Floor should have 2 items"
        );
    }

    /// Test B: No double-penalty - verify resolve_end_of_round doesn't change score
    #[test]
    fn test_no_double_floor_penalty() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut state = new_game(2, 0, &mut rng);

        // Manually place tiles on floor WITH the penalty already applied
        // (simulating what would have happened during apply_action)
        state.players[0].floor.len = 3;
        state.players[0].floor.slots[0] = Token::Tile(Color::Blue);
        state.players[0].floor.slots[1] = Token::Tile(Color::Red);
        state.players[0].floor.slots[2] = Token::Tile(Color::Yellow);

        // Set score as if penalties were already applied: -1 -1 -2 = -4
        state.players[0].score = -4;

        // Record score before resolve_end_of_round
        let score_before = state.players[0].score;

        // Call resolve_end_of_round
        let _final = resolve_end_of_round(&mut state, &mut rng);

        // Score should NOT have changed (penalties already applied)
        assert_eq!(
            state.players[0].score, score_before,
            "resolve_end_of_round should NOT apply floor penalties again. \
             Score was {} before, {} after",
            score_before, state.players[0].score
        );

        // Floor should be cleared
        assert_eq!(state.players[0].floor.len, 0, "Floor should be cleared");
    }

    // =========================================================================
    // Endgame bonus tests (azul-e9k.26)
    // =========================================================================

    #[test]
    fn test_endgame_bonus_horizontal_row() {
        let mut state = GameState::default();
        state.num_players = 2;
        state.players[0].score = 0;

        // Fill one horizontal row
        for col in 0..BOARD_SIZE {
            state.players[0].wall[0][col] = Some(WALL_PATTERN[0][col]);
        }

        let scores = apply_final_scoring(&mut state);
        assert_eq!(scores[0], 2, "One complete row should give +2 bonus");
    }

    #[test]
    fn test_endgame_bonus_vertical_column() {
        let mut state = GameState::default();
        state.num_players = 2;
        state.players[0].score = 0;

        // Fill one vertical column
        for row in 0..BOARD_SIZE {
            state.players[0].wall[row][0] = Some(WALL_PATTERN[row][0]);
        }

        let scores = apply_final_scoring(&mut state);
        assert_eq!(scores[0], 7, "One complete column should give +7 bonus");
    }

    #[test]
    fn test_endgame_bonus_complete_color() {
        let mut state = GameState::default();
        state.num_players = 2;
        state.players[0].score = 0;

        // Place all 5 Blue tiles on wall (at their designated positions)
        for row in 0..BOARD_SIZE {
            let col = WALL_DEST_COL[row][Color::Blue as usize] as usize;
            state.players[0].wall[row][col] = Some(Color::Blue);
        }

        let scores = apply_final_scoring(&mut state);
        assert_eq!(scores[0], 10, "Complete color set should give +10 bonus");
    }

    #[test]
    fn test_endgame_bonus_combined() {
        // Test: 1 row + 1 column + 1 color = +2 +7 +10 = +19
        let mut state = GameState::default();
        state.num_players = 2;
        state.players[0].score = 0;

        // This is tricky - we need a configuration where we have:
        // - A complete row
        // - A complete column
        // - A complete color set
        // Fill entire wall with appropriate tiles
        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                state.players[0].wall[row][col] = Some(WALL_PATTERN[row][col]);
            }
        }

        let scores = apply_final_scoring(&mut state);
        // Full wall: 5 rows * 2 = 10, 5 cols * 7 = 35, 5 colors * 10 = 50
        // Total = 95
        assert_eq!(scores[0], 95, "Full wall should give +95 bonus");
    }

    // =========================================================================
    // Factory refill tests (azul-e9k.27)
    // =========================================================================

    #[test]
    fn test_factory_refill_from_discard() {
        let mut rng = StdRng::seed_from_u64(42);

        // Create supply with few tiles in bag, rest in discard
        let mut supply = TileSupply {
            bag: [2, 2, 2, 2, 2],          // 10 tiles total in bag
            discard: [18, 18, 18, 18, 18], // 90 tiles in discard
        };

        let mut factories = Factories::default();
        factories.num_factories = 5; // Need 20 tiles

        refill_factories(&mut supply, &mut factories, &mut rng);

        // Should have refilled from discard
        let total_in_factories: u8 = (0..5).map(|f| factories.factories[f].len).sum();
        assert_eq!(
            total_in_factories, 20,
            "Should have filled 5 factories with 4 tiles each"
        );

        // Verify tile count invariant
        let bag_total: u8 = supply.bag.iter().sum();
        let discard_total: u8 = supply.discard.iter().sum();
        assert_eq!(bag_total + discard_total + total_in_factories, 100);
    }

    #[test]
    fn test_factory_refill_partial_when_empty() {
        let mut rng = StdRng::seed_from_u64(42);

        // Very few tiles available
        let mut supply = TileSupply {
            bag: [1, 1, 1, 0, 0], // Only 3 tiles total
            discard: [0, 0, 0, 0, 0],
        };

        let mut factories = Factories::default();
        factories.num_factories = 5;

        refill_factories(&mut supply, &mut factories, &mut rng);

        let total_in_factories: u8 = (0..5).map(|f| factories.factories[f].len).sum();
        assert_eq!(
            total_in_factories, 3,
            "Should only have 3 tiles in factories"
        );
    }

    // =========================================================================
    // Legal actions tests (azul-e9k.28)
    // =========================================================================

    #[test]
    fn test_legal_actions_non_empty() {
        let mut rng = StdRng::seed_from_u64(42);
        let state = new_game(2, 0, &mut rng);

        let actions = legal_actions(&state);
        assert!(
            !actions.is_empty(),
            "Should have legal actions at game start"
        );
    }

    #[test]
    fn test_legal_actions_wall_constraint() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut state = new_game(2, 0, &mut rng);

        // Place Blue in row 0 of wall (col 0 based on WALL_PATTERN)
        state.players[0].wall[0][0] = Some(Color::Blue);

        let actions = legal_actions(&state);

        // No action should allow placing Blue in pattern line 0
        for action in &actions {
            if action.color == Color::Blue {
                if let DraftDestination::PatternLine(row) = action.dest {
                    assert_ne!(
                        row, 0,
                        "Should not allow Blue in pattern line 0 when already in wall row 0"
                    );
                }
            }
        }
    }

    #[test]
    fn test_legal_actions_pattern_line_homogeneity() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut state = new_game(2, 0, &mut rng);

        // Put a Red tile in pattern line 2
        state.players[0].pattern_lines[2].color = Some(Color::Red);
        state.players[0].pattern_lines[2].count = 1;

        let actions = legal_actions(&state);

        // Only Red should be allowed in pattern line 2
        for action in &actions {
            if let DraftDestination::PatternLine(2) = action.dest {
                assert_eq!(
                    action.color,
                    Color::Red,
                    "Only Red should be allowed in pattern line with Red"
                );
            }
        }
    }

    #[test]
    fn test_legal_actions_full_pattern_line() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut state = new_game(2, 0, &mut rng);

        // Fill pattern line 0 (capacity 1)
        state.players[0].pattern_lines[0].color = Some(Color::Blue);
        state.players[0].pattern_lines[0].count = 1;

        let actions = legal_actions(&state);

        // No action should target pattern line 0
        for action in &actions {
            if let DraftDestination::PatternLine(row) = action.dest {
                assert_ne!(row, 0, "Should not allow placing in full pattern line");
            }
        }
    }

    #[test]
    fn test_legal_actions_floor_always_available() {
        let mut rng = StdRng::seed_from_u64(42);
        let state = new_game(2, 0, &mut rng);

        let actions = legal_actions(&state);

        // There should be at least one action with Floor destination
        let floor_actions: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a.dest, DraftDestination::Floor))
            .collect();
        assert!(
            !floor_actions.is_empty(),
            "Floor should always be an available destination"
        );
    }

    // =========================================================================
    // First player marker tests (azul-e9k.29)
    // =========================================================================

    #[test]
    fn test_first_player_marker_starts_in_center() {
        let mut rng = StdRng::seed_from_u64(42);
        let state = new_game(2, 0, &mut rng);

        assert!(state.center.len >= 1);
        assert!(matches!(state.center.items[0], Token::FirstPlayerMarker));
    }

    #[test]
    fn test_first_player_marker_taken_on_center_pick() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut state = new_game(2, 0, &mut rng);

        // Add some tiles to center to enable center pick
        state.center.items[state.center.len as usize] = Token::Tile(Color::Blue);
        state.center.len += 1;

        // Find an action that picks from center
        let center_action = Action {
            source: DraftSource::Center,
            color: Color::Blue,
            dest: DraftDestination::Floor,
        };

        let result = apply_action(state, center_action, &mut rng).unwrap();

        // First player marker should no longer be in center
        let marker_in_center = (0..result.state.center.len as usize)
            .any(|i| matches!(result.state.center.items[i], Token::FirstPlayerMarker));
        assert!(
            !marker_in_center,
            "First player marker should be taken from center"
        );

        // Should be on player 0's floor
        let marker_on_floor = (0..result.state.players[0].floor.len as usize).any(|i| {
            matches!(
                result.state.players[0].floor.slots[i],
                Token::FirstPlayerMarker
            )
        });
        assert!(
            marker_on_floor,
            "First player marker should be on player's floor"
        );
    }

    #[test]
    fn test_first_player_marker_determines_next_starter() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut state = new_game(2, 0, &mut rng);

        // Player 1 takes from center (will get marker)
        state.current_player = 1;
        state.center.items[state.center.len as usize] = Token::Tile(Color::Red);
        state.center.len += 1;

        let center_action = Action {
            source: DraftSource::Center,
            color: Color::Red,
            dest: DraftDestination::Floor,
        };

        let result = apply_action(state, center_action, &mut rng).unwrap();

        // Player 1 should be the starting player for next round
        assert_eq!(result.state.starting_player_next_round, 1);
    }

    // =========================================================================
    // Full game simulation tests (azul-e9k.30)
    // =========================================================================

    #[test]
    fn test_full_game_simulation() {
        let mut rng = StdRng::seed_from_u64(12345);
        let mut state = new_game(2, 0, &mut rng);

        let mut moves = 0;
        while state.phase != Phase::GameOver && moves < 1000 {
            assert_tile_invariants(&state);

            let actions = legal_actions(&state);
            if actions.is_empty() {
                panic!("No legal actions but game not over");
            }

            let action_idx = rng.random_range(0..actions.len() as u32) as usize;
            let action = actions[action_idx];

            let result = apply_action(state, action, &mut rng).expect("Action should succeed");
            state = result.state;

            if result.final_scores.is_some() {
                break;
            }

            moves += 1;
        }

        assert_eq!(state.phase, Phase::GameOver, "Game should end");
        assert_tile_invariants(&state);
    }

    #[test]
    fn test_full_game_3_players() {
        let mut rng = StdRng::seed_from_u64(54321);
        let mut state = new_game(3, 0, &mut rng);

        let mut moves = 0;
        while state.phase != Phase::GameOver && moves < 1000 {
            assert_tile_invariants(&state);

            let actions = legal_actions(&state);
            if actions.is_empty() {
                panic!("No legal actions but game not over");
            }

            let action_idx = rng.random_range(0..actions.len() as u32) as usize;
            let result = apply_action(state, actions[action_idx], &mut rng).unwrap();
            state = result.state;
            moves += 1;
        }

        assert_eq!(state.phase, Phase::GameOver);
        assert_tile_invariants(&state);
    }

    #[test]
    fn test_full_game_4_players() {
        let mut rng = StdRng::seed_from_u64(11111);
        let mut state = new_game(4, 0, &mut rng);

        let mut moves = 0;
        while state.phase != Phase::GameOver && moves < 1000 {
            assert_tile_invariants(&state);

            let actions = legal_actions(&state);
            if actions.is_empty() {
                panic!("No legal actions but game not over");
            }

            let action_idx = rng.random_range(0..actions.len() as u32) as usize;
            let result = apply_action(state, actions[action_idx], &mut rng).unwrap();
            state = result.state;
            moves += 1;
        }

        assert_eq!(state.phase, Phase::GameOver);
        assert_tile_invariants(&state);
    }

    // =========================================================================
    // Deterministic replay tests (azul-e9k.31)
    // =========================================================================

    #[test]
    fn test_deterministic_replay() {
        // Use separate RNGs for action selection vs game state
        // This ensures the "game RNG" stays synchronized between runs
        let mut game_rng1 = StdRng::seed_from_u64(42);
        let mut selection_rng = StdRng::seed_from_u64(999); // Separate for action selection

        let mut state1 = new_game(2, 0, &mut game_rng1);
        let mut actions_taken = Vec::new();
        let mut scores_at_each_step = Vec::new();

        for _ in 0..50 {
            if state1.phase == Phase::GameOver {
                break;
            }
            let actions = legal_actions(&state1);
            if actions.is_empty() {
                break;
            }
            // Use separate RNG for action selection
            let action_idx = selection_rng.random_range(0..actions.len() as u32) as usize;
            let action = actions[action_idx];
            actions_taken.push(action);

            // Game RNG is only used for apply_action
            let result = apply_action(state1, action, &mut game_rng1).unwrap();
            state1 = result.state;
            scores_at_each_step.push(state1.players[0].score);
        }

        let final_state1 = state1.clone();

        // Replay with same game seed (action sequence is fixed)
        let mut game_rng2 = StdRng::seed_from_u64(42);
        let mut state2 = new_game(2, 0, &mut game_rng2);

        for (i, &action) in actions_taken.iter().enumerate() {
            if state2.phase == Phase::GameOver {
                break;
            }
            let result = apply_action(state2, action, &mut game_rng2).unwrap();
            state2 = result.state;

            // Verify states match at each step
            assert_eq!(
                scores_at_each_step[i], state2.players[0].score,
                "Scores diverged at move {}",
                i
            );
        }

        // Final states should be identical
        assert_eq!(final_state1.phase, state2.phase);
        for p in 0..2 {
            assert_eq!(final_state1.players[p].score, state2.players[p].score);
            assert_eq!(final_state1.players[p].wall, state2.players[p].wall);
        }
    }

    #[test]
    fn test_deterministic_new_game() {
        // Same seed should produce identical initial states
        let mut rng1 = StdRng::seed_from_u64(12345);
        let state1 = new_game(2, 0, &mut rng1);

        let mut rng2 = StdRng::seed_from_u64(12345);
        let state2 = new_game(2, 0, &mut rng2);

        // Check factories are identical
        for f in 0..state1.factories.num_factories as usize {
            assert_eq!(
                state1.factories.factories[f].len,
                state2.factories.factories[f].len
            );
            for i in 0..state1.factories.factories[f].len as usize {
                assert_eq!(
                    state1.factories.factories[f].tiles[i],
                    state2.factories.factories[f].tiles[i]
                );
            }
        }
    }

    // =========================================================================
    // Basic action tests
    // =========================================================================

    #[test]
    fn test_apply_action_basic() {
        let mut rng = StdRng::seed_from_u64(42);
        let state = new_game(2, 0, &mut rng);

        let actions = legal_actions(&state);
        assert!(!actions.is_empty());

        let action = actions[0];
        let result = apply_action(state, action, &mut rng);
        assert!(result.is_ok());

        let step = result.unwrap();
        assert_tile_invariants(&step.state);
    }

    #[test]
    fn test_apply_action_wrong_phase() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut state = new_game(2, 0, &mut rng);
        state.phase = Phase::GameOver;

        let action = Action {
            source: DraftSource::Factory(0),
            color: Color::Blue,
            dest: DraftDestination::Floor,
        };

        let result = apply_action(state, action, &mut rng);
        assert!(matches!(result, Err(ApplyError::WrongPhase)));
    }

    #[test]
    fn test_apply_action_illegal() {
        let mut rng = StdRng::seed_from_u64(42);
        let state = new_game(2, 0, &mut rng);

        // Try an action from an empty factory
        let action = Action {
            source: DraftSource::Factory(99), // Invalid factory
            color: Color::Blue,
            dest: DraftDestination::Floor,
        };

        let result = apply_action(state, action, &mut rng);
        assert!(matches!(result, Err(ApplyError::IllegalAction)));
    }
}
