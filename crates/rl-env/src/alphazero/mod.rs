//! AlphaZero self-play training pipeline
//!
//! This module provides:
//! - `ReplayBuffer` for storing training examples
//! - `SelfPlayConfig` and `TrainerConfig` for configuration
//! - `Trainer` for running the training loop
//! - Supporting types: `PendingMove`, `MctsSearchResult`

pub mod examples;
pub mod replay_buffer;
pub mod training;

pub use examples::{PendingMove, TrainingExample};
pub use replay_buffer::ReplayBuffer;
pub use training::{
    build_training_batch, compute_outcomes_from_scores, self_play_game, MctsSearchResult,
    SelfPlayConfig, Trainer, TrainerConfig,
};
