use std::fmt::Write as _;
use std::sync::LazyLock;

use rand::rngs::StdRng;
use rand::SeedableRng;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use azul_engine::{
    apply_action, legal_actions, new_game, Action, Color, DraftDestination, DraftSource, GameState,
    Phase, PlayerIdx, Token, ALL_COLORS, BOARD_SIZE, FLOOR_CAPACITY, MAX_FACTORIES, MAX_PLAYERS,
    TILES_PER_COLOR, TILE_COLORS, WALL_DEST_COL,
};

pub type ActionId = u16;

/// Total size of the discrete action space.
/// (9 factories + center) * 5 colors * 2 destination types * 5 rows = 500
pub const ACTION_SPACE_SIZE: usize = 500;

/// Precomputed lookup table for O(1) action decoding.
static ACTION_LUT: LazyLock<[Action; ACTION_SPACE_SIZE]> = LazyLock::new(|| {
    let mut lut = [Action {
        source: DraftSource::Center,
        color: Color::Blue,
        dest: DraftDestination::Floor,
    }; ACTION_SPACE_SIZE];

    for id in 0..ACTION_SPACE_SIZE as u16 {
        lut[id as usize] = decode_action_impl(id);
    }
    lut
});

fn decode_action_impl(id: ActionId) -> Action {
    let mut x = id;

    let d_idx = x % BOARD_SIZE as u16;
    x /= BOARD_SIZE as u16;

    let d_type = x % 2;
    x /= 2;

    let c_idx = x % TILE_COLORS as u16;
    x /= TILE_COLORS as u16;

    let s_idx = x % MAX_FACTORIES as u16;
    x /= MAX_FACTORIES as u16;

    let s_type = x; // 0 or 1

    let source = if s_type == 0 {
        DraftSource::Factory(s_idx as u8)
    } else {
        DraftSource::Center
    };

    let color =
        Color::from_index(c_idx as u8).expect("color index should be valid within action space");

    let dest = if d_type == 0 {
        DraftDestination::PatternLine(d_idx as u8)
    } else {
        DraftDestination::Floor
    };

    Action {
        source,
        color,
        dest,
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ActionEncoder;

impl ActionEncoder {
    pub const fn action_space_size() -> usize {
        ACTION_SPACE_SIZE
    }

    pub fn encode(action: &Action) -> ActionId {
        let (s_type, s_idx) = match action.source {
            DraftSource::Factory(f) => (0u16, f as u16),
            DraftSource::Center => (1u16, 0),
        };

        let c_idx = action.color as u16;

        let (d_type, d_idx) = match action.dest {
            DraftDestination::PatternLine(r) => (0u16, r as u16),
            DraftDestination::Floor => (1u16, 0),
        };

        let id = (((s_type * MAX_FACTORIES as u16 + s_idx) * TILE_COLORS as u16 + c_idx) * 2
            + d_type)
            * BOARD_SIZE as u16
            + d_idx;

        debug_assert!(
            (id as usize) < ACTION_SPACE_SIZE,
            "ActionId {id} out of range"
        );

        id
    }

    #[inline]
    pub fn decode(id: ActionId) -> Action {
        ACTION_LUT[id as usize]
    }
}

fn color_char(color: Color) -> char {
    match color {
        Color::Blue => 'B',
        Color::Yellow => 'Y',
        Color::Red => 'R',
        Color::Black => 'K',
        Color::Teal => 'T',
    }
}

fn color_name(color: Color) -> &'static str {
    match color {
        Color::Blue => "Blue",
        Color::Yellow => "Yellow",
        Color::Red => "Red",
        Color::Black => "Black",
        Color::Teal => "Teal",
    }
}

fn color_slug(color: Color) -> &'static str {
    match color {
        Color::Blue => "blue",
        Color::Yellow => "amber",
        Color::Red => "rose",
        Color::Black => "zinc",
        Color::Teal => "emerald",
    }
}

fn token_char(token: Token) -> char {
    match token {
        Token::Tile(color) => color_char(color),
        Token::FirstPlayerMarker => '1',
    }
}

fn format_action_compact(action: &Action) -> String {
    let source = match action.source {
        DraftSource::Factory(f) => format!("F{f}"),
        DraftSource::Center => "C".to_string(),
    };

    let dest = match action.dest {
        DraftDestination::PatternLine(r) => format!("L{}", r + 1),
        DraftDestination::Floor => "Floor".to_string(),
    };

    format!("{} {} {}", source, color_name(action.color), dest)
}

fn obs_size() -> usize {
    // Factories: MAX_FACTORIES * TILE_COLORS (count of each color)
    let factory_features = MAX_FACTORIES * TILE_COLORS;

    // Center: TILE_COLORS (counts) + 1 (first player marker present)
    let center_features = TILE_COLORS + 1;

    // Per player features
    let pattern_line_features = BOARD_SIZE * (TILE_COLORS + 1);
    let wall_features = BOARD_SIZE * BOARD_SIZE;
    let wall_derived_features = (BOARD_SIZE * 2) + TILE_COLORS + (TILE_COLORS * BOARD_SIZE);
    let floor_features = TILE_COLORS + 1 + 1;
    let score_feature = 1;
    let per_player_features = pattern_line_features
        + wall_features
        + wall_derived_features
        + floor_features
        + score_feature;

    let all_players_features = per_player_features * MAX_PLAYERS;

    let current_player_features = MAX_PLAYERS;
    let starting_player_features = MAX_PLAYERS;
    let round_feature = 1;
    let final_round_triggered_feature = 1;
    let supply_features = TILE_COLORS * 2;

    factory_features
        + center_features
        + all_players_features
        + current_player_features
        + starting_player_features
        + round_feature
        + final_round_triggered_feature
        + supply_features
}

fn encode_features(state: &GameState, player: PlayerIdx) -> Vec<f32> {
    let obs_size = obs_size();
    let mut features = vec![0.0f32; obs_size];
    let mut idx = 0;

    macro_rules! write_feature {
        ($val:expr) => {
            features[idx] = $val;
            idx += 1;
        };
    }

    // 1. Factories
    for f in 0..MAX_FACTORIES {
        if f < state.factories.num_factories as usize {
            let factory = &state.factories.factories[f];
            let mut color_counts = [0u8; TILE_COLORS];
            for i in 0..factory.len as usize {
                color_counts[factory.tiles[i] as usize] += 1;
            }
            for &count in &color_counts {
                write_feature!(count as f32 / 4.0);
            }
        } else {
            for _ in 0..TILE_COLORS {
                write_feature!(0.0);
            }
        }
    }

    // 2. Center pool
    let mut center_color_counts = [0u8; TILE_COLORS];
    let mut fp_marker_in_center = false;
    for i in 0..state.center.len as usize {
        match state.center.items[i] {
            Token::Tile(color) => center_color_counts[color as usize] += 1,
            Token::FirstPlayerMarker => fp_marker_in_center = true,
        }
    }
    for &count in &center_color_counts {
        write_feature!(count as f32 / 20.0);
    }
    write_feature!(if fp_marker_in_center { 1.0 } else { 0.0 });

    // 3. Players (rotated)
    for p_offset in 0..MAX_PLAYERS {
        let p = ((player as usize) + p_offset) % (state.num_players as usize);
        let is_active_player = p_offset < state.num_players as usize;

        if is_active_player {
            let player_state = &state.players[p];

            for r in 0..BOARD_SIZE {
                let line = &player_state.pattern_lines[r];
                let capacity = (r + 1) as f32;

                for color in ALL_COLORS {
                    write_feature!(if line.color == Some(color) { 1.0 } else { 0.0 });
                }
                write_feature!(line.count as f32 / capacity);
            }

            for row in 0..BOARD_SIZE {
                for col in 0..BOARD_SIZE {
                    write_feature!(if player_state.wall[row][col].is_some() {
                        1.0
                    } else {
                        0.0
                    });
                }
            }

            let mut row_counts = [0u8; BOARD_SIZE];
            let mut col_counts = [0u8; BOARD_SIZE];
            let mut color_counts = [0u8; TILE_COLORS];
            for (row, wall_row) in player_state.wall.iter().enumerate() {
                for (col, cell) in wall_row.iter().enumerate() {
                    if let Some(color) = cell {
                        row_counts[row] += 1;
                        col_counts[col] += 1;
                        color_counts[*color as usize] += 1;
                    }
                }
            }
            for &count in &row_counts {
                write_feature!(count as f32 / BOARD_SIZE as f32);
            }
            for &count in &col_counts {
                write_feature!(count as f32 / BOARD_SIZE as f32);
            }
            for color in ALL_COLORS {
                write_feature!(color_counts[color as usize] as f32 / BOARD_SIZE as f32);
            }
            for color in ALL_COLORS {
                let mut missing_row: Option<usize> = None;
                for (row, (&dest_cols, wall_row)) in WALL_DEST_COL
                    .iter()
                    .zip(player_state.wall.iter())
                    .enumerate()
                {
                    let col = dest_cols[color as usize] as usize;
                    if wall_row[col].is_none() {
                        missing_row = Some(row);
                        break;
                    }
                }
                for row in 0..BOARD_SIZE {
                    write_feature!(if missing_row == Some(row) { 1.0 } else { 0.0 });
                }
            }

            let mut floor_color_counts = [0u8; TILE_COLORS];
            let mut fp_on_floor = false;
            for i in 0..player_state.floor.len as usize {
                match player_state.floor.slots[i] {
                    Token::Tile(color) => floor_color_counts[color as usize] += 1,
                    Token::FirstPlayerMarker => fp_on_floor = true,
                }
            }
            for &count in &floor_color_counts {
                write_feature!(count as f32 / FLOOR_CAPACITY as f32);
            }
            write_feature!(if fp_on_floor { 1.0 } else { 0.0 });
            write_feature!(player_state.floor.len as f32 / FLOOR_CAPACITY as f32);

            write_feature!(player_state.score as f32 / 100.0);
        } else {
            let pattern_line_features = BOARD_SIZE * (TILE_COLORS + 1);
            let wall_features = BOARD_SIZE * BOARD_SIZE;
            let wall_derived_features =
                (BOARD_SIZE * 2) + TILE_COLORS + (TILE_COLORS * BOARD_SIZE);
            let floor_features = TILE_COLORS + 1 + 1;
            let score_feature = 1;
            let per_player_features = pattern_line_features
                + wall_features
                + wall_derived_features
                + floor_features
                + score_feature;

            for _ in 0..per_player_features {
                write_feature!(0.0);
            }
        }
    }

    // 4. Current player one-hot
    for p in 0..MAX_PLAYERS {
        write_feature!(if p == state.current_player as usize { 1.0 } else { 0.0 });
    }

    // 5. Starting player next round one-hot
    for p in 0..MAX_PLAYERS {
        write_feature!(if p == state.starting_player_next_round as usize {
            1.0
        } else {
            0.0
        });
    }

    // 6. Round number (normalized)
    write_feature!(state.round as f32 / 10.0);

    // 7. Final-round triggered flag
    write_feature!(if state.final_round_triggered { 1.0 } else { 0.0 });

    // 8. Supply: bag + discard counts per color
    for color in ALL_COLORS {
        write_feature!(state.supply.bag[color as usize] as f32 / TILES_PER_COLOR as f32);
    }
    for color in ALL_COLORS {
        write_feature!(state.supply.discard[color as usize] as f32 / TILES_PER_COLOR as f32);
    }

    debug_assert_eq!(idx, obs_size, "Feature count mismatch");

    features
}

fn render_state_text(state: &GameState, highlight_player: Option<u8>) -> String {
    let mut out = String::new();
    let _ = writeln!(
        &mut out,
        "Round {} | Current Player: {}",
        state.round + 1,
        state.current_player
    );
    let _ = writeln!(&mut out, "");

    let _ = writeln!(&mut out, "Factories:");
    for f in 0..state.factories.num_factories as usize {
        let factory = &state.factories.factories[f];
        let _ = write!(&mut out, "  F{f}: ");
        if factory.len == 0 {
            let _ = writeln!(&mut out, "(empty)");
        } else {
            for i in 0..factory.len as usize {
                let _ = write!(&mut out, "{} ", color_char(factory.tiles[i]));
            }
            let _ = writeln!(&mut out);
        }
    }

    let _ = write!(&mut out, "\nCenter: ");
    if state.center.len == 0 {
        let _ = writeln!(&mut out, "(empty)");
    } else {
        for i in 0..state.center.len as usize {
            let _ = write!(&mut out, "{} ", token_char(state.center.items[i]));
        }
        let _ = writeln!(&mut out);
    }

    let _ = writeln!(&mut out, "");

    for p in 0..state.num_players as usize {
        let player = &state.players[p];
        let header_prefix = if highlight_player == Some(p as u8) { ">" } else { " " };
        let _ = writeln!(&mut out, "{} Player {} (Score: {})", header_prefix, p, player.score);
        let _ = writeln!(&mut out, "  Pattern Lines          Wall");

        for (row, (line, (wall_row, pattern_row))) in player
            .pattern_lines
            .iter()
            .zip(player.wall.iter().zip(azul_engine::WALL_PATTERN.iter()))
            .enumerate()
        {
            let cap = row + 1;
            let empty = cap - line.count as usize;

            let _ = write!(&mut out, "  ");
            for _ in 0..(BOARD_SIZE - cap) {
                let _ = write!(&mut out, "  ");
            }
            for _ in 0..empty {
                let _ = write!(&mut out, ". ");
            }
            if let Some(color) = line.color {
                for _ in 0..line.count {
                    let _ = write!(&mut out, "{} ", color_char(color));
                }
            }

            let _ = write!(&mut out, " -> ");

            for (cell, &expected) in wall_row.iter().zip(pattern_row.iter()) {
                if let Some(color) = cell {
                    let _ = write!(&mut out, "{} ", color_char(*color));
                } else {
                    let _ = write!(&mut out, "{} ", color_char(expected));
                }
            }
            let _ = writeln!(&mut out);
        }

        let _ = write!(&mut out, "  Floor: ");
        if player.floor.len == 0 {
            let _ = writeln!(&mut out, "(empty)");
        } else {
            for i in 0..player.floor.len as usize {
                let _ = write!(&mut out, "{} ", token_char(player.floor.slots[i]));
            }
            let _ = writeln!(&mut out);
        }
        let _ = writeln!(&mut out, "");
    }

    out
}

#[derive(Serialize)]
struct ApplyResult {
    reward: f32,
    game_over: bool,
    scores: Option<Vec<i16>>,
    current_player: u8,
    round: u16,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ActionSourceView {
    Factory { index: u8 },
    Center,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ActionDestView {
    Pattern { row: u8 },
    Floor,
}

#[derive(Serialize)]
struct ActionDetail {
    id: u16,
    source: ActionSourceView,
    color: String,
    dest: ActionDestView,
}

#[derive(Serialize)]
struct PatternLineView {
    color: Option<String>,
    count: u8,
    capacity: u8,
}

#[derive(Serialize)]
struct PlayerView {
    pattern_lines: Vec<PatternLineView>,
    wall: Vec<Vec<Option<String>>>,
    floor: Vec<String>,
    score: i16,
}

#[derive(Serialize)]
struct GameStateView {
    num_players: u8,
    current_player: u8,
    round: u16,
    final_round_triggered: bool,
    factories: Vec<Vec<String>>,
    center: Vec<String>,
    has_origin: bool,
    players: Vec<PlayerView>,
}

#[wasm_bindgen]
pub struct GameStateHandle {
    state: GameState,
    rng: StdRng,
}

#[wasm_bindgen]
pub fn action_space_size() -> usize {
    ACTION_SPACE_SIZE
}

#[wasm_bindgen]
pub fn observation_size() -> usize {
    obs_size()
}

#[wasm_bindgen]
pub fn new_game_state(seed: u64) -> GameStateHandle {
    let mut rng = StdRng::seed_from_u64(seed);
    let state = new_game(2, 0, &mut rng);
    GameStateHandle { state, rng }
}

#[wasm_bindgen]
impl GameStateHandle {
    #[wasm_bindgen]
    pub fn clone_handle(&self) -> GameStateHandle {
        GameStateHandle {
            state: self.state.clone(),
            rng: self.rng.clone(),
        }
    }

    #[wasm_bindgen]
    pub fn current_player(&self) -> u8 {
        self.state.current_player
    }

    #[wasm_bindgen]
    pub fn is_game_over(&self) -> bool {
        self.state.phase == Phase::GameOver
    }

    #[wasm_bindgen]
    pub fn round(&self) -> u16 {
        self.state.round
    }

    #[wasm_bindgen]
    pub fn scores(&self) -> Vec<i16> {
        let n = self.state.num_players as usize;
        (0..n).map(|i| self.state.players[i].score).collect()
    }

    #[wasm_bindgen]
    pub fn legal_action_ids(&self) -> Vec<u16> {
        legal_actions(&self.state)
            .iter()
            .map(ActionEncoder::encode)
            .collect()
    }

    #[wasm_bindgen]
    pub fn legal_action_strings(&self) -> Vec<String> {
        legal_actions(&self.state)
            .iter()
            .map(|action| format_action_compact(action))
            .collect()
    }

    #[wasm_bindgen]
    pub fn action_id_to_string(&self, action_id: u16) -> String {
        let action = ActionEncoder::decode(action_id);
        format_action_compact(&action)
    }

    #[wasm_bindgen]
    pub fn encode_observation(&self, player: u8) -> Vec<f32> {
        encode_features(&self.state, player)
    }

    #[wasm_bindgen]
    pub fn render_text(&self, highlight_player: Option<u8>) -> String {
        render_state_text(&self.state, highlight_player)
    }

    #[wasm_bindgen]
    pub fn state_view(&self) -> Result<JsValue, JsValue> {
        let mut factories = Vec::with_capacity(self.state.factories.num_factories as usize);
        for f in 0..self.state.factories.num_factories as usize {
            let factory = &self.state.factories.factories[f];
            let mut tiles = Vec::with_capacity(factory.len as usize);
            for i in 0..factory.len as usize {
                tiles.push(color_slug(factory.tiles[i]).to_string());
            }
            factories.push(tiles);
        }

        let mut center = Vec::new();
        let mut has_origin = false;
        for i in 0..self.state.center.len as usize {
            match self.state.center.items[i] {
                Token::Tile(color) => center.push(color_slug(color).to_string()),
                Token::FirstPlayerMarker => has_origin = true,
            }
        }

        let mut players = Vec::with_capacity(self.state.num_players as usize);
        for p in 0..self.state.num_players as usize {
            let player = &self.state.players[p];
            let mut pattern_lines = Vec::with_capacity(BOARD_SIZE);
            for r in 0..BOARD_SIZE {
                let line = &player.pattern_lines[r];
                pattern_lines.push(PatternLineView {
                    color: line.color.map(|c| color_slug(c).to_string()),
                    count: line.count,
                    capacity: (r + 1) as u8,
                });
            }

            let mut wall = vec![vec![None; BOARD_SIZE]; BOARD_SIZE];
            for row in 0..BOARD_SIZE {
                for col in 0..BOARD_SIZE {
                    wall[row][col] = player.wall[row][col].map(|c| color_slug(c).to_string());
                }
            }

            let mut floor = Vec::with_capacity(player.floor.len as usize);
            for i in 0..player.floor.len as usize {
                match player.floor.slots[i] {
                    Token::Tile(color) => floor.push(color_slug(color).to_string()),
                    Token::FirstPlayerMarker => floor.push("origin".to_string()),
                }
            }

            players.push(PlayerView {
                pattern_lines,
                wall,
                floor,
                score: player.score,
            });
        }

        let view = GameStateView {
            num_players: self.state.num_players,
            current_player: self.state.current_player,
            round: self.state.round,
            final_round_triggered: self.state.final_round_triggered,
            factories,
            center,
            has_origin,
            players,
        };

        serde_wasm_bindgen::to_value(&view)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize state: {e}")))
    }

    #[wasm_bindgen]
    pub fn legal_action_details(&self) -> Result<JsValue, JsValue> {
        let actions = legal_actions(&self.state);
        let mut details = Vec::with_capacity(actions.len());
        for action in actions {
            let id = ActionEncoder::encode(&action);
            let source = match action.source {
                DraftSource::Factory(idx) => ActionSourceView::Factory { index: idx },
                DraftSource::Center => ActionSourceView::Center,
            };
            let dest = match action.dest {
                DraftDestination::PatternLine(row) => ActionDestView::Pattern { row },
                DraftDestination::Floor => ActionDestView::Floor,
            };
            details.push(ActionDetail {
                id,
                source,
                color: color_slug(action.color).to_string(),
                dest,
            });
        }

        serde_wasm_bindgen::to_value(&details)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize actions: {e}")))
    }

    #[wasm_bindgen]
    pub fn apply_action_id(&mut self, action_id: u16) -> Result<JsValue, JsValue> {
        let action = ActionEncoder::decode(action_id);
        let current_state = std::mem::take(&mut self.state);
        let acting_player = current_state.current_player as usize;

        let my_before = current_state.players[acting_player].score as f32;
        let opp_before = if current_state.num_players == 2 {
            current_state.players[1 - acting_player].score as f32
        } else {
            0.0
        };

        let result = apply_action(current_state, action, &mut self.rng)
            .map_err(|e| JsValue::from_str(&format!("Apply error: {e:?}")))?;

        self.state = result.state;

        let my_after = self.state.players[acting_player].score as f32;
        let opp_after = if self.state.num_players == 2 {
            self.state.players[1 - acting_player].score as f32
        } else {
            0.0
        };

        let reward = ((my_after - my_before) - (opp_after - opp_before)) / 20.0;

        let scores = result
            .final_scores
            .map(|arr| arr[..self.state.num_players as usize].to_vec());

        let payload = ApplyResult {
            reward,
            game_over: scores.is_some(),
            scores,
            current_player: self.state.current_player,
            round: self.state.round,
        };

        serde_wasm_bindgen::to_value(&payload).map_err(|e| {
            JsValue::from_str(&format!("Failed to serialize ApplyResult: {e}"))
        })
    }
}
