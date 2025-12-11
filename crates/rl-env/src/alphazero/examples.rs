//! Training example types for AlphaZero self-play
//!
//! Defines the data structures for storing training examples during self-play.

use crate::Observation;
use azul_engine::PlayerIdx;

/// One training example: (s, π, z) in AlphaZero notation.
#[derive(Clone)]
pub struct TrainingExample {
    /// Observation from the acting player's perspective.
    pub observation: Observation, // shape [obs_size]

    /// MCTS-improved policy π over ACTION_SPACE_SIZE actions.
    /// Stored as a flat f32 vec; converted to Array at batch time.
    pub policy: Vec<f32>, // len == ACTION_SPACE_SIZE

    /// Final outcome from this player's perspective.
    /// Typically in [-1, 1] or normalized score difference.
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
}
