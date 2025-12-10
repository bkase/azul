## Part 2 — Updated engineering spec (RL environment + agent API, MLX-based)

### 0. Summary

- We assume the **Azul engine** from the first spec is implemented:
  - `GameState`, `Action`, `Phase`, `new_game`, `legal_actions`, `apply_action`, etc.

- We now specify:
  - RL **environment** for Azul.
  - **Action encoding** into a fixed discrete space.
  - **Feature extraction** into MLX arrays.
  - **Agent API**.
  - **Testing strategy** for this layer.

---

### 1. Dependencies and assumptions

- Crates:
  - `azul_engine`: your engine crate with `GameState`, `Action`, etc.
  - `mlx-rs`: MLX bindings for Apple Silicon. ([GitHub][2])

- RNG:
  - `rand` or `rand_core` for `Rng` trait.

- This spec assumes:
  - `GameState` is Markov and fully describes the Azul game.
  - `apply_action` handles round‑end tiling, scoring, and game termination.

---

### 2. Core RL types

#### 2.1 Action IDs and rewards

```rust
pub type ActionId = u16;   // 0..ACTION_SPACE_SIZE-1
pub type Reward   = f32;
```

#### 2.2 Observations as MLX arrays

```rust
use mlx_rs::Array;

/// All observations are MLX Arrays of dtype f32 and shape [obs_size].
pub type Observation = Array;
```

Invariant:

- For any `FeatureExtractor` implementation used with this environment:
  - `encode(state, player)` must produce an `Array` with:
    - rank 1 (1D),
    - shape `[obs_size]`,
    - element type `f32`.

Batching (outside the environment):

- You can build batched arrays of shape `[batch_size, obs_size]` by stacking multiple `Observation`s using MLX ops (e.g., some `stack`/concatenate function).

---

### 3. Environment configuration

```rust
/// Reward schemes supported by the environment.
#[derive(Copy, Clone, Debug)]
pub enum RewardScheme {
    /// Dense incremental reward:
    /// reward[player] = (score_after[player] - score_before[player])
    /// at every step, including all final bonuses at the terminal step.
    DenseScoreDelta,

    /// Terminal-only reward:
    /// rewards are 0 until game over; at game over:
    /// reward[player] = final_score[player] - mean(final_scores).
    TerminalOnly,

    // (Optional future extension: add Shaped reward scheme here.)
}

/// Environment configuration parameters.
#[derive(Clone, Debug)]
pub struct EnvConfig {
    /// Number of players in the game.
    /// Must match GameState.num_players.
    pub num_players: u8,              // 2..=4

    /// Reward computation strategy (see RewardScheme).
    pub reward_scheme: RewardScheme,

    /// If true, EnvStep.state contains a full GameState clone
    /// for debugging and analysis.
    pub include_full_state_in_step: bool,
}
```

---

### 4. Feature extraction: `GameState` → `Observation`

#### 4.1 Trait

```rust
use crate::azul_engine::{GameState, PlayerIdx};

/// Converts a full GameState into a fixed-length MLX Array observation
/// from the perspective of a given player.
pub trait FeatureExtractor {
    /// Returns the length of the flattened observation vector.
    ///
    /// For all states and players, encode(...) must return an Array
    /// with shape [obs_size].
    fn obs_size(&self) -> usize;

    /// Encode state from the perspective of `player` into a 1D MLX Array.
    ///
    /// Requirements:
    /// - Deterministic given (state, player).
    /// - Shape is [obs_size()], rank-1, dtype f32.
    /// - No side effects or internal randomness.
    fn encode(&self, state: &GameState, player: PlayerIdx) -> Observation;
}
```

Notes:

- You are free to implement `encode` using multi‑channel binary planes internally and then flatten to 1D.
- Remember to encode at least:
  - all board structures (walls, pattern lines, floors),
  - whose turn it is (`current_player`),
  - perhaps who will start next round (`starting_player_next_round`).

---

### 5. Action encoding: `Action` ↔ `ActionId`

We define a **fixed global action space** of size 300, covering all syntactically possible Azul draft moves regardless of current legality.

#### 5.1 Constants

```rust
use crate::azul_engine::{Action, DraftSource, DraftDestination, Color};
use crate::azul_engine::{MAX_FACTORIES, TILE_COLORS, BOARD_SIZE};

pub const ACTION_SPACE_SIZE: usize = 300;
```

We’ll use:

- `source_type` ∈ {Factory, Center} encoded as 0 or 1.
- `source_index` ∈ 0..MAX_FACTORIES-1 (0..8), only used if `source_type == Factory`.
- `color_index` ∈ 0..4 (Color as u8).
- `dest_type` ∈ {PatternLine, Floor} encoded as 0 or 1.
- `dest_index` ∈ 0..BOARD_SIZE-1 (pattern line row), only used if `dest_type == PatternLine`.

Packing scheme:

```text
let s_type ∈ {0,1}
let s_idx  ∈ 0..MAX_FACTORIES
let c_idx  ∈ 0..TILE_COLORS        // 0..4
let d_type ∈ {0,1}
let d_idx  ∈ 0..BOARD_SIZE         // 0..4

id = ((((s_type * MAX_FACTORIES + s_idx) * TILE_COLORS + c_idx)
       * 2
       + d_type)
      * BOARD_SIZE
      + d_idx)

ACTION_SPACE_SIZE = (MAX_FACTORIES + 1) * TILE_COLORS * 2 * BOARD_SIZE
                  = (9 + 1) * 5 * 2 * 5
                  = 300
```

#### 5.2 Encoder

```rust
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ActionEncoder;

impl ActionEncoder {
    pub const fn action_space_size() -> usize {
        ACTION_SPACE_SIZE
    }

    /// Encode a concrete Action into a discrete ActionId.
    ///
    /// Panics in debug builds if the Action has indices out of range.
    pub fn encode(action: &Action) -> ActionId {
        let (s_type, s_idx) = match action.source {
            DraftSource::Factory(f) => (0_u16, f as u16),
            DraftSource::Center     => (1_u16, 0),
        };

        let c_idx = action.color as u16;

        let (d_type, d_idx) = match action.dest {
            DraftDestination::PatternLine(r) => (0_u16, r as u16),
            DraftDestination::Floor          => (1_u16, 0),
        };

        let id = ((((s_type * MAX_FACTORIES as u16 + s_idx) * TILE_COLORS as u16 + c_idx)
                   * 2
                   + d_type)
                  * BOARD_SIZE as u16
                  + d_idx);

        debug_assert!((id as usize) < ACTION_SPACE_SIZE);

        id
    }

    /// Decode an ActionId back into an Action.
    ///
    /// This may result in syntactically valid but *illegal* Actions for
    /// the current state. The environment must mask illegal IDs.
    pub fn decode(id: ActionId) -> Action {
        assert!((id as usize) < ACTION_SPACE_SIZE);

        let mut x = id as u16;

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

        let color = Color::from_index(c_idx as u8); // you implement from_index()

        let dest = if d_type == 0 {
            DraftDestination::PatternLine(d_idx as u8)
        } else {
            DraftDestination::Floor
        };

        Action { source, color, dest }
    }
}
```

You’ll need to implement `Color::from_index(u8) -> Color`.

**Testing requirement**: For all syntactically valid `Action`s in range, `decode(encode(a)) == a`.

---

### 6. Generic environment trait

We define a generic trait with fully spelled‑out type parameters:

```rust
/// Generic environment interface for RL.
pub trait Environment {
    /// Type used to represent observations (for us: MLX Array).
    type ObservationType;

    /// Type used to represent actions (for us: ActionId).
    type ActionType;

    /// Type used to represent rewards (for us: f32).
    type RewardType;

    /// Reset the environment to a fresh episode (new game).
    ///
    /// Returns the first EnvStep, representing the initial state prior
    /// to any actions.
    fn reset(
        &mut self,
        rng: &mut impl rand::Rng,
    ) -> EnvStep<Self::ObservationType, Self::RewardType>;

    /// Apply an action for the current player, advance the environment
    /// by one step, and return the resulting EnvStep.
    fn step(
        &mut self,
        action: Self::ActionType,
        rng: &mut impl rand::Rng,
    ) -> Result<EnvStep<Self::ObservationType, Self::RewardType>, StepError>;
}
```

---

### 7. RL-facing step struct and errors

#### 7.1 EnvStep

```rust
use crate::azul_engine::{GameState, PlayerIdx, MAX_PLAYERS};

/// The result of either reset() or step() in an environment.
pub struct EnvStep<O, R> {
    /// Observation per player, from that player's perspective.
    ///
    /// Only indices 0..num_players-1 are meaningful.
    pub observations: [O; MAX_PLAYERS],

    /// Reward per player for the most recent transition.
    ///
    /// For reset(), this is all zeros.
    pub rewards: [R; MAX_PLAYERS],

    /// True if the episode has terminated (game over).
    pub done: bool;

    /// Index of the player whose turn it is *after* this step.
    pub current_player: PlayerIdx,

    /// Mask over the discrete action space:
    ///
    /// legal_action_mask[id] == true if that ActionId is legal
    /// for `current_player` in this state.
    pub legal_action_mask: Vec<bool>,

    /// The last action taken, if any.
    ///
    /// For reset(), this is None.
    pub last_action: Option<ActionId>,

    /// Optional full GameState for debugging.
    ///
    /// Populated only if EnvConfig.include_full_state_in_step is true.
    pub state: Option<GameState>,
}
```

#### 7.2 StepError

```rust
#[derive(Debug)]
pub enum StepError {
    /// step() called after the episode has already terminated.
    EpisodeDone,

    /// ActionId could not be decoded into a syntactically valid Action
    /// (e.g., indices out of range).
    InvalidActionId(ActionId),

    /// ActionId decodes to an Action which is illegal in the current state
    /// (e.g., placing tiles on an invalid pattern line).
    IllegalAction(ActionId),
}
```

---

### 8. Azul environment struct (`AzulEnv`)

#### 8.1 Definition

```rust
pub struct AzulEnv<F: FeatureExtractor> {
    /// Underlying Azul engine state.
    pub game_state: GameState,

    /// Environment configuration.
    pub config: EnvConfig,

    /// Feature extractor for building observations.
    pub features: F,

    /// Last action applied (ActionId), if any.
    pub last_action: Option<ActionId>,

    /// Whether this episode has ended.
    pub done: bool,
}
```

**Invariants:**

- `game_state.num_players == config.num_players`.
- `done == (game_state.phase == Phase::GameOver)`.
- If `done == false`, then `game_state.phase == Phase::FactoryOffer`.

#### 8.2 Helper: legal action mask

```rust
impl<F: FeatureExtractor> AzulEnv<F> {
    fn build_legal_action_mask(&self) -> Vec<bool> {
        let mut mask = vec![false; ActionEncoder::action_space_size()];
        let actions = crate::azul_engine::legal_actions(&self.game_state);

        for action in actions {
            let id = ActionEncoder::encode(&action) as usize;
            mask[id] = true;
        }

        mask
    }
}
```

---

### 9. `Environment` implementation for `AzulEnv`

```rust
impl<F: FeatureExtractor> Environment for AzulEnv<F> {
    type ObservationType = Observation;
    type ActionType      = ActionId;
    type RewardType      = Reward;

    fn reset(
        &mut self,
        rng: &mut impl rand::Rng,
    ) -> EnvStep<Self::ObservationType, Self::RewardType> {
        // 1. Build a fresh GameState using the engine's new_game()
        self.game_state = crate::azul_engine::new_game(
            self.config.num_players,
            /* starting_player = */ 0, // or randomize with rng if desired
            rng,
        );
        self.last_action = None;
        self.done = false;

        // 2. Build per-player observations
        let mut observations: [Observation; MAX_PLAYERS] =
            std::array::from_fn(|idx| {
                if (idx as u8) < self.game_state.num_players {
                    self.features.encode(&self.game_state, idx as u8)
                } else {
                    // For unused slots, create a zero vector of length obs_size
                    // You can implement a helper to create such an Array.
                    create_zero_observation(self.features.obs_size())
                }
            });

        // 3. Rewards are all zero on reset
        let rewards = [0.0_f32; MAX_PLAYERS];

        // 4. Build legal action mask for the current player
        let legal_action_mask = self.build_legal_action_mask();

        // 5. Optional full state
        let state_clone = if self.config.include_full_state_in_step {
            Some(self.game_state.clone())
        } else {
            None
        };

        EnvStep {
            observations,
            rewards,
            done: false,
            current_player: self.game_state.current_player,
            legal_action_mask,
            last_action: None,
            state: state_clone,
        }
    }

    fn step(
        &mut self,
        action_id: Self::ActionType,
        rng: &mut impl rand::Rng,
    ) -> Result<EnvStep<Self::ObservationType, Self::RewardType>, StepError> {
        if self.done {
            return Err(StepError::EpisodeDone);
        }

        // 1. Decode action
        let action = ActionEncoder::decode(action_id);

        // 2. Check legality
        let legal_mask = self.build_legal_action_mask();
        if (action_id as usize) >= legal_mask.len() {
            return Err(StepError::InvalidActionId(action_id));
        }
        if !legal_mask[action_id as usize] {
            return Err(StepError::IllegalAction(action_id));
        }

        // 3. Record previous scores
        let mut prev_scores = [0_i16; MAX_PLAYERS];
        for p in 0..self.game_state.num_players {
            prev_scores[p as usize] = self.game_state.players[p as usize].score;
        }

        // 4. Apply action via engine
        let apply_result =
            crate::azul_engine::apply_action(self.game_state.clone(), action, rng)
            .expect("Engine-level errors should be impossible if masking is correct");

        self.game_state = apply_result.state;

        // 5. Compute rewards based on RewardScheme
        let mut rewards = [0.0_f32; MAX_PLAYERS];

        match self.config.reward_scheme {
            RewardScheme::DenseScoreDelta => {
                for p in 0..self.game_state.num_players {
                    let new_score = self.game_state.players[p as usize].score;
                    let delta = (new_score - prev_scores[p as usize]) as f32;
                    rewards[p as usize] = delta;
                }
            }

            RewardScheme::TerminalOnly => {
                if self.game_state.phase == crate::azul_engine::Phase::GameOver {
                    let mut final_scores = [0.0_f32; MAX_PLAYERS];
                    for p in 0..self.game_state.num_players {
                        final_scores[p as usize] =
                            self.game_state.players[p as usize].score as f32;
                    }
                    let n = self.game_state.num_players as usize;
                    let mean = final_scores[0..n].iter().sum::<f32>() / (n as f32);
                    for p in 0..n {
                        rewards[p] = final_scores[p] - mean;
                    }
                } else {
                    // Non-terminal steps have zero reward.
                }
            }
        }

        // 6. Update done flag
        self.done = self.game_state.phase == crate::azul_engine::Phase::GameOver;

        // 7. Build new observations
        let observations: [Observation; MAX_PLAYERS] =
            std::array::from_fn(|idx| {
                if (idx as u8) < self.game_state.num_players {
                    self.features.encode(&self.game_state, idx as u8)
                } else {
                    create_zero_observation(self.features.obs_size())
                }
            });

        // 8. Build new legal action mask (or all-false if done)
        let legal_action_mask = if self.done {
            vec![false; ActionEncoder::action_space_size()]
        } else {
            self.build_legal_action_mask()
        };

        // 9. Update last_action
        self.last_action = Some(action_id);

        // 10. Optional state clone
        let state_clone = if self.config.include_full_state_in_step {
            Some(self.game_state.clone())
        } else {
            None
        };

        Ok(EnvStep {
            observations,
            rewards,
            done: self.done,
            current_player: self.game_state.current_player,
            legal_action_mask,
            last_action: self.last_action,
            state: state_clone,
        })
    }
}
```

`create_zero_observation(obs_size)` is a small helper that constructs an `Array` of zeros with shape `[obs_size]` and dtype `f32` using MLX.

---

### 10. Agent API

#### 10.1 Agent input

```rust
/// Inputs provided to an agent when selecting an action.
pub struct AgentInput<'a> {
    /// Observation vector for the player whose turn it is.
    pub observation: &'a Observation,

    /// Mask over action IDs:
    /// legal_action_mask[id] == true if the action is legal.
    pub legal_action_mask: &'a [bool],

    /// Index of the player whose turn it is.
    pub current_player: PlayerIdx,
}
```

#### 10.2 Agent trait

```rust
/// Trait for anything that can choose actions in the environment:
/// random policy, neural-net-based policy, or human input.
pub trait Agent {
    /// Choose a legal action given an observation and legal-action mask.
    ///
    /// Requirement:
    /// - Must only return ActionIds for which legal_action_mask[id as usize] == true.
    /// - May use rng for exploration.
    fn select_action(
        &mut self,
        input: &AgentInput,
        rng: &mut impl rand::Rng,
    ) -> ActionId;
}
```

---

### 11. Example self‑play loop (shared policy)

This is not API, just “how you use it”:

```rust
pub fn self_play_episode<F: FeatureExtractor, A: Agent>(
    env: &mut AzulEnv<F>,
    agent: &mut A,
    rng: &mut impl rand::Rng,
    replay_buffer: &mut Vec<Transition>, // your replay buffer struct
) {
    // 1. Reset environment
    let mut step = env.reset(rng);

    while !step.done {
        let p = step.current_player as usize;

        // 2. Build input for current player
        let agent_input = AgentInput {
            observation: &step.observations[p],
            legal_action_mask: &step.legal_action_mask,
            current_player: step.current_player,
        };

        // 3. Select action
        let action_id = agent.select_action(&agent_input, rng);

        // 4. Step environment
        let next_step = env.step(action_id, rng)
                           .expect("step failed unexpectedly");

        // 5. Log transition for training (replay buffer)
        let transition = Transition {
            player: step.current_player,
            observation_before: step.observations[p].clone(),
            action_id,
            reward: next_step.rewards[p],
            observation_after: next_step.observations[p].clone(),
            done: next_step.done,
        };
        replay_buffer.push(transition);

        // 6. Move on
        step = next_step;
    }
}
```

Later, a **training loop** will sample mini‑batches from `replay_buffer`, build MLX batched arrays, and run gradient updates.

---

### 12. Testing strategy for RL layer

Here’s how we test the RL side (in addition to engine tests):

#### 12.1 Action encoding/decoding

- Property-based test:
  - For randomly generated syntactically valid `Action`s:
    - `id = ActionEncoder::encode(&action)`
    - `action2 = ActionEncoder::decode(id)`
    - assert `action2 == action`.

- For random `ActionId`s in `[0, ACTION_SPACE_SIZE)`:
  - `ActionEncoder::decode(id)` must produce an `Action` whose fields are within valid index ranges.

#### 12.2 Legal action mask correctness

- For many random `GameState`s (reachable states):
  - Compute `legal = legal_actions(&game_state)`.
  - Compute `mask = env.build_legal_action_mask()`.
  - For each `a in legal`:
    - `id = encode(&a)` → assert `mask[id as usize] == true`.

  - For each `id` with `mask[id] == true`:
    - `a = decode(id)` → assert `a` is in `legal` (set equality).

#### 12.3 Reward correctness

For `RewardScheme::DenseScoreDelta`:

- Run a full random episode:
  - Track per-player cumulative reward `cum_reward[p] = Σ_t reward_t[p]`.
  - At the end, check:
    - `cum_reward[p] == final_score[p] - initial_score[p]` (initial_score is 0).

- This ensures our step rewards match engine scoring exactly.

For `RewardScheme::TerminalOnly`:

- Check that:
  - `reward_t[p] == 0` for all non-terminal steps.
  - At terminal step:
    - `Σ_p reward_terminal[p] == 0` (because we subtract the mean).
    - `reward_terminal[p] == final_score[p] - mean(final_scores)`.

#### 12.4 Determinism

- With fixed RNG seed and fixed sequence of `ActionId`s:
  - Run `env.reset()` and repeated `env.step(action_id)` sequences twice.
  - Assert that:
    - `EnvStep.state` (if included) and observations are identical across runs.

This ensures the environment is pure given seed + actions.

#### 12.5 Observation shape and type

- For arbitrary reachable `GameState`s and players:
  - `obs = features.encode(&state, player)`:
    - check that `obs` has rank 1,
    - length `obs_size()`,
    - dtype `f32` (MLX Array must be created that way).

#### 12.6 Illegal action handling

- Pick random states and random `ActionId`s.
- If `mask[id] == false`, then:
  - `env.step(id, rng)` must return `Err(StepError::IllegalAction(..))` (or panic in debug code paths if you prefer).

- Confirm that `EpisodeDone` is returned for `step` calls after `done == true`.

