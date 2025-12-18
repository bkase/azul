//! RL Environment module for Azul game
//!
//! This module provides:
//! - Action encoding/decoding (ActionId ↔ Action)
//! - Feature extraction (GameState → Observation)
//! - Environment trait and AzulEnv implementation
//! - Agent trait and RandomAgent
//! - AlphaZero-style MCTS agent and neural network
//! - AlphaZero training pipeline (self-play, replay buffer, trainer)

mod action_encoder;
mod agent;
pub mod alphazero;
mod alphazero_net;
mod environment;
mod feature_extractor;
mod mcts;
mod mlx_config;
#[cfg(feature = "profiling")]
pub mod profiling;
mod types;

pub use action_encoder::*;
pub use agent::*;
pub use alphazero::*;
pub use alphazero_net::AlphaZeroNet;
pub use environment::*;
pub use feature_extractor::*;
pub use mcts::{AlphaZeroMctsAgent, Batch, MctsConfig, PolicyValueNet};
pub(crate) use mlx_config::configure_mlx_for_current_thread;
pub use types::*;
