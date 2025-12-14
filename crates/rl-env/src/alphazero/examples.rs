//! Training example types for AlphaZero self-play
//!
//! Defines the data structures for storing training examples during self-play.

use crate::Observation;
use azul_engine::PlayerIdx;
use crate::ActionId;

/// One training example: (s, π, z) in AlphaZero notation.
#[derive(Clone, Debug)]
pub struct TrainingExample {
    /// Observation from the acting player's perspective.
    pub observation: Observation, // shape [obs_size]

    /// MCTS-improved policy π over ACTION_SPACE_SIZE actions.
    /// Stored as a flat f32 vec; converted to Array at batch time.
    pub policy: Vec<f32>, // len == ACTION_SPACE_SIZE

    /// The action actually taken at this move (after applying temperature sampling).
    /// Useful for debugging/training diagnostics (e.g., floor-action rate).
    pub action: ActionId,

    /// Discounted return / advantage from this player's perspective.
    /// Typically normalized to [-1, 1].
    pub value: f32,
}

/// Move record before we know the final outcome.
/// Converted to TrainingExample when game ends.
#[derive(Clone)]
pub struct PendingMove {
    /// The player who made this move
    pub player: PlayerIdx,

    /// Observation at the time of the move
    pub observation: Observation,

    /// MCTS policy distribution (from visit counts)
    pub policy: Vec<f32>, // len == ACTION_SPACE_SIZE

    /// Action taken at this move (ActionId).
    pub action: ActionId,

    /// Immediate reward for this transition from the acting player's perspective.
    /// For 2-player, this is typically the normalized score-delta difference.
    pub reward: f32,
}
