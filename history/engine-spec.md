Here’s a spec you can hand to Future-You and say “implement this Azul engine in Rust” without needing to think about rules again.

I’ll assume **base Azul (2017)**, 2–4 players, colored side of the player board, no expansions or gray-side variant (I’ll note how to extend to gray side at the end).

Rules are taken from the official rulebook and some secondary sources for details like floor penalties. ([cdn.1j1ju.com][1])

---

## 1. Goals & constraints

**Design goals**

- Engine for **Azul** suitable for an **RL playground**.
- Core object is a **single `GameState` (plain data)**. No logic baked into methods; pure functions operate on it.
- The state must be **Markov**: given this one object, you can:
  - determine all legal actions,
  - compute all transition probabilities (given a RNG),
  - compute rewards and game termination.

- Support **2–4 players**.
- Model actions as **purely functional reducers**:
  - `next_state = reduce(state, action, rng)` (or in-place version).
  - No dependence on past states.

**Non-goals (for the initial version)**

- Gray-side variant where wall placement is free-form.
- Expansions (Crystal Mosaic, etc.).
- Fancy performance tuning beyond “reasonable arrays and small enums”.

---

## 2. Game rules distilled (for the engine)

You don’t need to re-read the rulebook while coding; this is what matters for the engine.

### 2.1 Components

- **Players**: 2–4.
- **Tiles**: 5 colors × 20 each = 100 tiles, all identical within a color. ([cdn.1j1ju.com][1])
- **Factories**:
  - 2 players → 5 factories
  - 3 players → 7 factories
  - 4 players → 9 factories ([cdn.1j1ju.com][1])

- **First player marker**: 1 token.
- Each player has:
  - **Wall**: 5×5 grid (color layout fixed by player board pattern).
  - **Pattern lines**: 5 rows of capacities 1..=5.
  - **Floor line**: 7 slots with penalties `[-1, -1, -2, -2, -2, -3, -3]`. ([dspace.cuni.cz][2])
  - **Score** (integer, never below 0).

### 2.2 Round structure

Each round has three phases. ([cdn.1j1ju.com][1])

1. **Factory Offer (Drafting)**
   Players alternate turns in clockwise order. On your turn:
   1. Pick a **source**:
      - either one **factory** with tiles,
      - or the **center**.

   2. Choose a **color** present in that source.
   3. Take **all tiles of that color** from the source.
      - If it’s a factory: remaining tiles from that factory go to the center.
      - If it’s the center and this is the **first pick from center this round**, you also take the **first player marker**; it goes to your floor line as a tile.

   4. Choose a **destination**:
      - either one of your 5 **pattern lines**, or
      - your **floor line**.

   5. Tile placement rules:
      - Pattern line `r` (0..4) has capacity `r+1`.
      - If pattern line is non-empty, it must already contain the same color (homogeneous).
      - You **cannot** place color `c` into pattern line `r` if that row of your wall already contains color `c`. ([cdn.1j1ju.com][1])
      - Place tiles from **right to left** in the chosen pattern line until it’s full.
      - Any tiles that don’t fit go to the **floor line**, filling it **left to right**.
      - You may also choose to put **all of them directly into the floor** (legal but usually bad).
      - If the floor line overflows beyond 7 slots, extra tiles go straight to **discard** (no extra penalty).
   - The **round ends** when **all factories and center** are empty.

2. **Wall Tiling (End-of-round scoring)** ([cdn.1j1ju.com][1])
   For each player:
   1. For each pattern line `r` from top to bottom:
      - If the line is **complete** (capacity reached):
        - Move the **rightmost tile** to the wall, in that row, in the column corresponding to that color.
        - Score the placement (see below).
        - All other tiles in that pattern line go to **discard**.

      - If the line is incomplete, it stays as-is.

   2. **Placement scoring** (each placement is scored immediately):
      - Let `(r, c)` be the cell.
      - Check horizontal adjacency: count contiguous tiles including `(r,c)` left and right.
      - Check vertical adjacency: count contiguous tiles including `(r,c)` up and down.
      - If both horizontal and vertical counts are 1 (i.e. isolated tile), score **1 point**.
      - Otherwise, score `(horizontal_count>1 ? horizontal_count : 0) + (vertical_count>1 ? vertical_count : 0)`. ([cdn.1j1ju.com][1])

   3. **Floor line penalties**:
      - For each tile in floor position `i` (0-based):
        - Subtract penalty `[-1, -1, -2, -2, -2, -3, -3][i]` from score, but clamp score at ≥ 0. ([dspace.cuni.cz][2])

      - All tiles on floor go to **discard**.
      - **Exception**: the **first player marker** is not discarded; it moves to “in front” of that player and determines the starting player next round. It still counted as one floor tile for penalty. ([cdn.1j1ju.com][1])

3. **Prepare next round** ([cdn.1j1ju.com][1])
   - If **no one** has completed a full horizontal row (5 tiles) on their wall, then:
     - The holder of the first player marker is the **next starting player**.
     - Refill each factory with 4 tiles drawn randomly from the bag. If the bag empties, refill the bag from discard and continue. If everything empties, leave some factories partially full.
     - Place the **first player marker** into the **center**.
     - Set `current_player = starting_player`, and begin a new Factory Offer phase.

   - If **someone has completed at least one horizontal row**, the game will end after this round; see below.

### 2.3 End of game & final scoring

- Game ends after the Wall Tiling phase **of the round where at least one player completed a horizontal row** (5 tiles in a row on the wall). ([cdn.1j1ju.com][1])
- Then apply final bonuses for each player: ([cdn.1j1ju.com][1])
  - +2 points for each complete **horizontal** row (5 tiles).
  - +7 points for each complete **vertical** column.
  - +10 points for each **color** where the player has all 5 tiles on their wall.

- Highest score wins. Ties broken by who has more completed horizontal rows; remaining ties shared.

---

## 3. Core data model (Rust-ish)

### 3.1 Basic types

```rust
/// Index into players array: 0..num_players-1
pub type PlayerIdx = u8;  // 0..=3

/// Rows and columns are always 0..4
pub type Row = u8;  // 0..=4
pub type Col = u8;  // 0..=4

pub const BOARD_SIZE: usize = 5;
pub const MAX_PLAYERS: usize = 4;
pub const MAX_FACTORIES: usize = 9;
pub const FACTORY_CAPACITY: usize = 4;
pub const FLOOR_CAPACITY: usize = 7;
pub const TILE_COLORS: usize = 5;
pub const TILES_PER_COLOR: u8 = 20;
```

Tile colors (order doesn’t really matter but fix it for serialization):

```rust
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Color {
    Blue = 0,
    Yellow = 1,
    Red = 2,
    Black = 3,
    Teal = 4, // “light blue”
}
pub const ALL_COLORS: [Color; TILE_COLORS] = [
    Color::Blue,
    Color::Yellow,
    Color::Red,
    Color::Black,
    Color::Teal,
];
```

Floor penalties (fixed table):

```rust
pub const FLOOR_PENALTY: [i8; FLOOR_CAPACITY] = [-1, -1, -2, -2, -2, -3, -3];
```

Game phase / status:

```rust
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Phase {
    FactoryOffer,   // Players drafting
    GameOver,       // Terminal; no more actions
}
```

### 3.2 Wall layout constants (colored side)

On the colored side, the positions of colors on the wall are fixed and form a Latin-square–like pattern. Hard-code the pattern matrix and a reverse lookup from `(row, color)` to `col`. ([cdn.1j1ju.com][1])

```rust
// wall_pattern[row][col] = Color at that position.
pub const WALL_PATTERN: [[Color; BOARD_SIZE]; BOARD_SIZE] = [
    // row 0
    [Color::Blue, Color::Yellow, Color::Red,   Color::Black, Color::Teal],
    // row 1
    [Color::Teal, Color::Blue,   Color::Yellow,Color::Red,   Color::Black],
    // row 2
    [Color::Black,Color::Teal,   Color::Blue,  Color::Yellow,Color::Red],
    // row 3
    [Color::Red,  Color::Black,  Color::Teal,  Color::Blue,  Color::Yellow],
    // row 4
    [Color::Yellow,Color::Red,   Color::Black, Color::Teal,  Color::Blue],
];

// dest_col[row][color_index] => col
pub const WALL_DEST_COL: [[u8; TILE_COLORS]; BOARD_SIZE] = /* precomputed */;
```

You can precompute `WALL_DEST_COL` at compile time (macro/const fn) or hand-code.

### 3.3 Factory and center

Factories: fixed and small, so prefer arrays.

```rust
#[derive(Clone)]
pub struct Factory {
    pub len: u8,                   // 0..=4
    pub tiles: [Color; FACTORY_CAPACITY], // only first len used
}

/// Max 9 factories; only first `num_factories` valid.
#[derive(Clone)]
pub struct Factories {
    pub num_factories: u8,                       // 5,7,9
    pub factories: [Factory; MAX_FACTORIES],
}
```

Center: contains colored tiles plus possibly the first player marker.

```rust
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Token {
    Tile(Color),
    FirstPlayerMarker,
}

#[derive(Clone)]
pub struct CenterPool {
    pub len: u8,                   // 0..=max
    pub items: [Token; 100],       // upper bound: all tiles could theoretically be here.
    // (You can make this tighter, but 100 is trivial.)
}
```

### 3.4 Bag and discard (tile supply)

To keep the state Markov but simple, store **color counts**, not exact order. Random draws depend only on these counts.

```rust
#[derive(Clone)]
pub struct TileSupply {
    /// Count of tiles of each color in the bag.
    pub bag: [u8; TILE_COLORS],
    /// Count of tiles of each color in the discard (“lid”).
    pub discard: [u8; TILE_COLORS],
}
```

Invariants:

- For each color `c`,
  `bag[c] + discard[c] + on_factories[c] + in_center[c] + on_player_boards[c] + on_floors[c] = TILES_PER_COLOR`.

You don’t strictly need to store this invariant; but tests should assert it.

### 3.5 Player board state

Pattern lines: each row is homogeneous color or empty.

```rust
#[derive(Copy, Clone)]
pub struct PatternLine {
    pub color: Option<Color>,  // None => empty; Some(c) => all tiles are c
    pub count: u8,             // 0..=capacity(row_index)
}
```

Wall: we only need occupancy; color is implied by `WALL_PATTERN`. For clarity and future gray-side support, store color explicitly.

```rust
pub type Wall = [[Option<Color>; BOARD_SIZE]; BOARD_SIZE];
```

Floor line: ordered sequence of tokens (tiles + maybe FirstPlayerMarker).

```rust
#[derive(Clone)]
pub struct FloorLine {
    pub len: u8,                        // 0..=FLOOR_CAPACITY
    pub slots: [Token; FLOOR_CAPACITY], // positions 0..len-1 used, left->right
}
```

Player state:

```rust
#[derive(Clone)]
pub struct PlayerState {
    pub wall: Wall,
    pub pattern_lines: [PatternLine; BOARD_SIZE],
    pub floor: FloorLine,
    pub score: i16, // clamp at >=0 on updates
}
```

### 3.6 Top-level GameState

```rust
#[derive(Clone)]
pub struct GameState {
    pub num_players: u8,                     // 2..=4
    pub players: [PlayerState; MAX_PLAYERS], // only 0..num_players used

    pub factories: Factories,
    pub center: CenterPool,
    pub supply: TileSupply,

    /// Whose turn it is in FactoryOffer phase.
    pub current_player: PlayerIdx,

    /// Who will start the *next* round (determined by who took FP marker this round).
    /// At the start of game/round, this equals the current starting player.
    pub starting_player_next_round: PlayerIdx,

    /// Is the game in drafting or over? (Wall tiling + prepare next round are done
    /// automatically as a consequence of the last drafting action.)
    pub phase: Phase,

    /// Number of rounds completed so far (0 at game start; increments after each full round).
    pub round: u16,

    /// Whether this is the *final* round (at least one player has a full horizontal row),
    /// used mainly to decide when to run endgame scoring.
    pub final_round_triggered: bool,
}
```

Notes:

- There is **no explicit “WallTiling” phase** in state:
  - After each drafting action, if the factories and center are empty, the engine automatically:
    - runs Wall Tiling and floor scoring for all players,
    - checks for game end,
    - either sets `phase = GameOver` or prepares the next round and sets `phase = FactoryOffer`.

- `starting_player_next_round` is updated when the **first player marker** leaves the center (i.e. first time someone drafts from center this round).

---

## 4. Action representation

In this engine, **only drafting choices are actions**. Wall tiling and round setup are automatic deterministic transitions.

### 4.1 Action enum

There is exactly one kind of action: “draft tiles of color C from some source to some destination”.

```rust
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DraftSource {
    Factory(u8), // index 0..num_factories-1
    Center,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DraftDestination {
    PatternLine(Row), // 0..4
    Floor,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Action {
    pub source: DraftSource,
    pub color: Color,
    pub dest: DraftDestination,
}
```

Notes:

- The **number of tiles taken** is implied by the current board:
  - all tiles of `color` from source.

- The mapping of tiles between pattern line and floor is deterministic:
  - you always fill pattern line to capacity (subject to rules) and then overflow to floor.

- Direct “all to floor” is represented by `DraftDestination::Floor`.

---

## 5. State transitions and rules as pure functions

We’ll define the core functional API and the semantics each function must satisfy.

### 5.1 Game construction

```rust
/// Initialize a new game with 2–4 players.
/// `starting_player` is initial player index (you can randomize externally).
/// `rng` is only used to randomly fill factories from the bag.
pub fn new_game(
    num_players: u8,
    starting_player: PlayerIdx,
    rng: &mut impl Rng,
) -> GameState;
```

**new_game semantics**:

1. Validate `2 <= num_players <= 4`.
2. `GameState.num_players = num_players`.
3. Create empty `PlayerState`:
   - `wall` all `None`.
   - `pattern_lines` all `{ color: None, count: 0 }`.
   - `floor.len = 0`.
   - `score = 0`.

4. Supply:
   - `bag` = `[TILES_PER_COLOR; TILE_COLORS]` (20 of each color).
   - `discard` = `[0; TILE_COLORS]`.

5. Factories:
   - `num_factories = {2→5, 3→7, 4→9}`.
   - Draw 4 tiles for each factory from `supply.bag` using random draws:
     - If `bag` empty at any point, move all `discard` into `bag` and continue.

6. Center:
   - empty; then **place first player marker** (a `Token::FirstPlayerMarker`) into center.

7. `current_player = starting_player`.
8. `starting_player_next_round = starting_player`.
9. `phase = Phase::FactoryOffer`.
10. `round = 0`.
11. `final_round_triggered = false`.

### 5.2 Legal actions

```rust
pub fn legal_actions(state: &GameState) -> Vec<Action>;
```

**Preconditions**:

- Use only when `state.phase == Phase::FactoryOffer`.
- If `Phase::GameOver`, return empty vector.

**Algorithm**:

1. `p = state.current_player`.

2. Build all possible `(source, color)` pairs:
   - For each **factory index `f`** in `0..state.factories.num_factories`:
     - If `factory[f].len > 0`, compute the set of unique colors `{c}` present there.

   - From **center**:
     - Consider only `Token::Tile(c)` entries; ignore `FirstPlayerMarker` when enumerating colors.
     - For each unique color present, `source = Center`.

3. For each `(source, color)`:
   - Compute **allowed destinations** for player `p`:
     - Always allow `DraftDestination::Floor` (even if floor is full; overflow is just discarded).
     - For each row `r in 0..5`:

       Let:
       - `cap = r + 1` (pattern line capacity).
       - `line = state.players[p].pattern_lines[r]`.
       - `wall = state.players[p].wall`.

       Validate:
       1. **Wall constraint**: It’s illegal if the wall row `r` already has color `color`
          (i.e. there exists `c` s.t. `wall[r][c] == Some(color)`).
       2. **Pattern color homogeneity**:
          - If `line.color == Some(other)` and `other != color` → illegal.

       3. **Line fullness**:
          - If `line.count == cap` → illegal (you can’t choose an already-full pattern line as target).

       If all constraints pass, then `DraftDestination::PatternLine(r)` is allowed.

4. For each `(source, color)` and each allowed `dest`, generate `Action { source, color, dest }`.

This gives a complete enumeration of legal moves for the current player.

### 5.3 Applying a player action

```rust
pub enum ApplyError {
    NotPlayersTurn,
    WrongPhase,
    IllegalAction, // not included in legal_actions(state)
}

pub struct StepResult {
    pub state: GameState,
    /// If Some, the game became terminal and this is final per-player score.
    pub final_scores: Option<[i16; MAX_PLAYERS]>,
}

pub fn apply_action(
    mut state: GameState,
    action: Action,
    rng: &mut impl Rng,
) -> Result<StepResult, ApplyError>;
```

**High-level semantics**:

- Only valid if `phase == FactoryOffer`.
- Only the `current_player` may act.
- `apply_action`:
  - performs the drafting move,
  - checks if the round ended; if yes, runs Wall Tiling, floor scoring, and prepare-next-round (including endgame),
  - advances `current_player` appropriately,
  - returns the new `GameState` and possibly final scores in `final_scores`.

We’ll break this into substeps.

#### 5.3.1 Validity checks

1. If `state.phase != Phase::FactoryOffer` → `Err(WrongPhase)`.
2. If `state` has no legal actions and `phase != GameOver`, you’ve got a bug; but under correct rules, this cannot happen.
3. Optionally (for safety), recompute `legal_actions(&state)` and check that `action` is in the list:
   - If not, return `Err(IllegalAction)`.

#### 5.3.2 Execute drafting

Let `p = state.current_player`, `player = &mut state.players[p]`.

1. **Extract tiles from source**:

   ```rust
   let mut taken: Vec<Color> = Vec::new();
   let mut took_first_player_marker = false;
   ```

   - If `action.source == Factory(f)`:
     - Iterate `factory.tiles[0..factory.len)`:
       - If `tile == action.color`, push into `taken`.
       - Else, push into **center** (`Token::Tile(tile)`).

     - Set `factory.len = 0`.

   - If `action.source == Center`:
     - Iterate over `center.items[0..center.len)`:
       - If `Token::Tile(c)` where `c == action.color`:
         - push `c` into `taken`.

       - Else if `Token::FirstPlayerMarker` and **we haven’t yet assigned next starting player this round**:
         - `took_first_player_marker = true`.
         - Remove marker.

       - Otherwise, keep token in center.

     - Compact `center.items` to remove taken tiles and (maybe) marker, updating `center.len`.

   - If `took_first_player_marker`:
     - Set `state.starting_player_next_round = p`.

2. **Place tiles into destination for player `p`**:

   Case `DraftDestination::Floor`:
   - For each tile in `taken`:
     - If `player.floor.len < FLOOR_CAPACITY`:
       - write `Token::Tile(color)` to `floor.slots[floor.len]`, `floor.len += 1`.

     - Else:
       - Increment `supply.discard[color]` (overflow).

   - If `took_first_player_marker`:
     - If `floor.len < FLOOR_CAPACITY`:
       - `floor.slots[floor.len] = Token::FirstPlayerMarker; floor.len += 1;`

     - Else:
       - The marker can _never_ be discarded; if floor is full when someone takes from center, the marker should logically occupy an “invisible” slot with penalty?
         The rules say extra fallen tiles go to the box lid once floor is full, but they don’t explicitly say floor might be too full for the marker. We can resolve this by:
         - still applying **one additional penalty** (as if on 7th slot), and
         - tracking the marker separately on the player as “held but not in floor”. That edge case is practically impossible in real play, but you can define it precisely if you care.

       - For simplicity, you can assume we always have capacity for marker because you can’t get more than 7 other tiles before taking from center; but strict reinforcement should handle overflow explicitly.

   Case `DraftDestination::PatternLine(r)`:
   - Let `line = &mut player.pattern_lines[r]`.
   - Precondition: verified by legality checks.
   - If `line.color.is_none()`:
     - `line.color = Some(action.color)`.

   - Let `cap = r + 1`.
   - Let `space = cap - line.count`.
   - Let `n_to_line = min(space, taken.len() as u8)`.
   - Add `n_to_line` tiles to this pattern line:
     - `line.count += n_to_line`.

   - For each of the remaining tiles (overflow), add to floor line as in floor case above (subject to capacity / discard).
   - If `took_first_player_marker`:
     - Add marker to floor exactly as in floor case.

3. **Round-end check**:
   - After this action, if **all factories are empty and center has no tiles** (ignore marker):
     - Call `resolve_end_of_round(state, rng)` (below).

   - Else:
     - Advance `current_player` to `(p + 1) % num_players`.
     - Return `StepResult { state, final_scores: None }`.

#### 5.3.3 End-of-round resolution

```rust
fn resolve_end_of_round(
    state: &mut GameState,
    rng: &mut impl Rng,
) -> Option<[i16; MAX_PLAYERS]>; // returns Some(scores) if game ends
```

Steps:

1. **Wall tiling and placement scoring for each player**:

   For each `p in 0..state.num_players`:

   ```rust
   let player = &mut state.players[p];

   for r in 0..BOARD_SIZE {
       let cap = (r + 1) as u8;
       let line = &mut player.pattern_lines[r];

       if line.count == cap {
           let color = line.color.expect("complete line must have color");

           // Find dest column
           let col = WALL_DEST_COL[r][color as usize] as usize;

           // Place tile on wall if not already occupied (should not be)
           debug_assert!(player.wall[r][col].is_none());
           player.wall[r][col] = Some(color);

           // Score placement
           let delta = score_placement(&player.wall, r as usize, col);
           player.score = (player.score + delta).max(0);

           // Remove the placed tile: this is conceptually from pattern, but we’ve
           // already moved it to wall. The remaining (cap-1) tiles go to discard.
           let tiles_to_discard = cap - 1;
           state.supply.discard[color as usize] += tiles_to_discard;

           // Reset pattern line
           line.color = None;
           line.count = 0;
       }
       // else: incomplete line remains, no changes.
   }
   ```

   `score_placement` implements the adjacency scoring described earlier.

2. **Floor penalties and cleanup**:

   For each `p` again:

   ```rust
   let player = &mut state.players[p];
   let mut fp_marker_here = false;

   for i in 0..player.floor.len as usize {
       match player.floor.slots[i] {
           Token::Tile(color) => {
               let penalty = FLOOR_PENALTY[i] as i16;
               player.score = (player.score + penalty).max(0);
               state.supply.discard[color as usize] += 1;
           }
           Token::FirstPlayerMarker => {
               let penalty = FLOOR_PENALTY[i] as i16;
               player.score = (player.score + penalty).max(0);
               fp_marker_here = true;
           }
       }
   }
   player.floor.len = 0; // all floor positions cleared

   if fp_marker_here {
       state.starting_player_next_round = p;
   }
   ```

3. **Check for end-of-game condition**:
   - For each `p`, check if `player.wall[row]` has all 5 positions filled for any `row`.
   - If at least one such row exists:
     - Set `state.final_round_triggered = true`.

   According to rules, the game ends **after this round**, i.e. we now do final scoring and set `phase=GameOver`. ([cdn.1j1ju.com][1])

   So:

   ```rust
   if state.final_round_triggered {
       let scores = apply_final_scoring(state);
       state.phase = Phase::GameOver;
       return Some(scores);
   }
   ```

4. **Prepare next round (if not final)**:

   ```rust
   state.round += 1;
   refill_factories(&mut state.supply, &mut state.factories, rng);
   state.center.len = 0;
   state.center.items[0] = Token::FirstPlayerMarker;
   state.center.len = 1;

   state.current_player = state.starting_player_next_round;
   state.phase = Phase::FactoryOffer;
   state.final_round_triggered = false;
   ```

5. Return `None` for `final_scores` if the game continues.

#### 5.3.4 Final scoring

```rust
fn apply_final_scoring(state: &mut GameState) -> [i16; MAX_PLAYERS];
```

For each player:

1. **Horizontal rows**: for each `row`:
   - If `wall[row][col].is_some()` for all `col`, add `+2`.

2. **Vertical columns**: for each `col`:
   - If `wall[row][col].is_some()` for all `row`, add `+7`.

3. **Color sets**: for each `color`:
   - Count occurrences of that `color` on wall; if count == 5, add `+10`. ([cdn.1j1ju.com][1])

Return the resulting scores array (pad unused players with 0 or actual scores; all indices 0..num_players are meaningful).

---

## 7. Markov property checklist

To verify that `GameState` is truly Markov (you never need history):

- **All public info** is contained:
  - Each player’s wall, pattern lines, floor, and score.
  - Factories and center contents (colors + FP marker).

- **Tile supply** is represented as counts in bag and discard, enough to determine the distribution of future draws.
- **Turn order & next round starter**:
  - `current_player` and `starting_player_next_round`.

- **Game phase**:
  - `phase` tells if further actions possible.

- **Game progression**:
  - `round` (not strictly needed for rules, but useful for debugging).

- **First-player marker location** is logically encoded:
  - If the marker is in the center, it’s represented as `Token::FirstPlayerMarker` in `center.items`.
  - If it was taken this round, that’s reflected by `starting_player_next_round` and maybe in a floor slot until end-of-round.

From this state, you can:

- Reconstruct all legal drafting actions.
- Compute exact state transition distribution (given RNG for draws).
- Compute final outcome.

No reference to previous states is needed.

---

## 8. Implementation notes & tests

### 8.1 Efficiency choices

- Use fixed-size arrays (`[T; N]`) as above to make vectorization for RL straightforward.
- Avoid heap allocations on hot paths:
  - `legal_actions` will allocate at most `num_sources * colors_per_source * destinations_per_color` (bounded by a small constant ~ 5×5×6).
  - You can reuse a preallocated buffer or slab if needed.

### 8.2 Invariants to assert in debug builds

- For each color `c`:

  ```text
  bag[c] + discard[c] +
  Σ factories tiles of color c +
  center tiles of color c +
  Σ players (pattern + wall + floor tiles of color c)
  == 20
  ```

- `floor.len <= FLOOR_CAPACITY`.

- For each pattern line row r:
  - `line.count <= r+1`.
  - `line.count == 0` ↔ `line.color == None`.
  - If `line.count > 0`, then no other color appears in that line (by construction).

- For each wall position (r,c):
  - `wall[r][c] == Some(color)` implies `color == WALL_PATTERN[r][c]` (for colored side).

- `current_player < num_players`.

- `starting_player_next_round < num_players`.

### 8.3 Rule-based tests

Write deterministic tests reproducing examples from the rulebook and online rules:

- Scoring adjacency: single tile; horizontal group; vertical group; cross shape. ([cdn.1j1ju.com][1])
- Floor penalties: fill floor with 7 tiles, verify total penalty `-1-1-2-2-2-3-3 = -14` but clamped to ≥0 from current score. ([dspace.cuni.cz][2])
- Endgame bonus examples:
  - Player with 1 row, 1 column, full color set: compute +2 +7 +10.

- Factory refill from depleted bag:
  - Fill `bag` with small known counts, `discard` with others; ensure `refill_factories` pulls from discard when bag empties. ([cdn.1j1ju.com][1])

---

If you follow this spec, you’ll end up with:

- A **single plain `GameState` struct** that fully describes an Azul game.
- A **small `Action` enum** describing all meaningful player choices.
- A set of **pure functions** (`new_game`, `legal_actions`, `apply_action`, etc.) that you can plug directly into your RL harness.

[1]: https://cdn.1j1ju.com/medias/03/14/fd-azul-rulebook.pdf "Azul Rulebook - 1jour-1jeu.com"
[2]: https://dspace.cuni.cz/bitstream/handle/20.500.11956/127953/130308802.pdf?isAllowed=y&sequence=1&utm_source=chatgpt.com "Michal Pocatko AI for the Board Game Azul"
