//! Core RL types for the Azul environment

use crate::{GameState, PlayerIdx, MAX_PLAYERS};

/// Discrete action identifier (0..ACTION_SPACE_SIZE-1)
pub type ActionId = u16;

/// Reward value (float)
pub type Reward = f32;

/// Simple array type for observations when MLX is not available.
/// This mimics the MLX Array API for compatibility.
#[derive(Clone, Debug)]
pub struct SimpleArray {
    data: Vec<f32>,
    shape: Vec<i32>,
}

impl SimpleArray {
    /// Create a new array from a slice with the given shape
    pub fn from_slice(data: &[f32], shape: &[i32]) -> Self {
        let expected_len: i32 = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len as usize,
            "Data length {} doesn't match shape {:?}",
            data.len(),
            shape
        );
        Self {
            data: data.to_vec(),
            shape: shape.to_vec(),
        }
    }

    /// Create a zero-filled array with the given shape
    pub fn zeros(shape: &[i32]) -> Result<Self, &'static str> {
        let len: i32 = shape.iter().product();
        Ok(Self {
            data: vec![0.0; len as usize],
            shape: shape.to_vec(),
        })
    }

    /// Get the shape of the array
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }

    /// Get the data as a slice
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }
}

/// Observation as array of dtype f32 and shape [obs_size]
#[cfg(feature = "mlx")]
pub type Observation = mlx_rs::Array;

#[cfg(not(feature = "mlx"))]
pub type Observation = SimpleArray;

/// Alias for the array type being used
#[cfg(feature = "mlx")]
pub type Array = mlx_rs::Array;

#[cfg(not(feature = "mlx"))]
pub type Array = SimpleArray;

/// Reward schemes supported by the environment
#[derive(Copy, Clone, Debug, Default)]
pub enum RewardScheme {
    /// Dense incremental reward:
    /// reward[player] = score_after[player] - score_before[player]
    /// at every step, including all final bonuses at the terminal step.
    #[default]
    DenseScoreDelta,

    /// Terminal-only reward:
    /// rewards are 0 until game over; at game over:
    /// reward[player] = final_score[player] - mean(final_scores).
    TerminalOnly,
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
