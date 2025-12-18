//! Core RL types for the Azul environment

use azul_engine::{GameState, PlayerIdx, MAX_PLAYERS};

/// Discrete action identifier (0..ACTION_SPACE_SIZE-1)
pub type ActionId = u16;

/// Reward value (float)
pub type Reward = f32;

/// Observation as MLX array of dtype f32 and shape [obs_size]
pub type Observation = mlx_rs::Array;

/// Helper trait to get f32 slice from observation arrays
pub trait ObservationExt {
    fn as_f32_slice(&self) -> &[f32];
}

impl ObservationExt for Observation {
    fn as_f32_slice(&self) -> &[f32] {
        self.as_slice::<f32>()
    }
}

/// Reward schemes supported by the environment
#[derive(Copy, Clone, Debug, Default)]
pub enum RewardScheme {
    /// Dense incremental reward:
    /// reward[player] = score_after[player] - score_before[player]
    /// at every step, including all final bonuses at the terminal step.
    #[default]
    DenseScoreDelta,
}

/// Environment configuration parameters
#[derive(Clone, Debug)]
pub struct EnvConfig {
    /// Number of players in the game (2..=4)
    /// Must match GameState.num_players
    pub num_players: u8,

    /// Reward computation strategy
    pub reward_scheme: RewardScheme,

    /// If true, EnvStep.state contains a full GameState clone for debugging
    pub include_full_state_in_step: bool,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            num_players: 2,
            reward_scheme: RewardScheme::default(),
            include_full_state_in_step: false,
        }
    }
}

/// Error types for Environment::step()
#[derive(Debug)]
pub enum StepError {
    /// step() called after the episode has already terminated
    EpisodeDone,

    /// ActionId could not be decoded into a syntactically valid Action
    InvalidActionId(ActionId),

    /// ActionId decodes to an Action which is illegal in the current state
    IllegalAction(ActionId),
}

/// The result of either reset() or step() in an environment
pub struct EnvStep<O, R> {
    /// Observation per player, from that player's perspective.
    /// Only indices 0..num_players-1 are meaningful.
    pub observations: [O; MAX_PLAYERS],

    /// Reward per player for the most recent transition.
    /// For reset(), this is all zeros.
    pub rewards: [R; MAX_PLAYERS],

    /// True if the episode has terminated (game over)
    pub done: bool,

    /// Index of the player whose turn it is *after* this step
    pub current_player: PlayerIdx,

    /// Mask over the discrete action space:
    /// legal_action_mask[id] == true if that ActionId is legal
    /// for `current_player` in this state.
    pub legal_action_mask: Vec<bool>,

    /// The last action taken, if any.
    /// For reset(), this is None.
    pub last_action: Option<ActionId>,

    /// Optional full GameState for debugging.
    /// Populated only if EnvConfig.include_full_state_in_step is true.
    pub state: Option<GameState>,
}

/// Transition struct for storing experience in replay buffer
#[derive(Clone)]
pub struct Transition {
    /// Which player made this transition
    pub player: PlayerIdx,

    /// Observation before the action
    pub observation_before: Observation,

    /// The action taken
    pub action_id: ActionId,

    /// Reward received
    pub reward: Reward,

    /// Observation after the action
    pub observation_after: Observation,

    /// Whether the episode ended after this action
    pub done: bool,
}
