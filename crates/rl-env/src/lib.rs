//! RL Environment module for Azul game
//!
//! This module provides:
//! - Action encoding/decoding (ActionId ↔ Action)
//! - Feature extraction (GameState → Observation)
//! - Environment trait and AzulEnv implementation
//! - Agent trait and RandomAgent

mod action_encoder;
mod agent;
mod environment;
mod feature_extractor;
mod types;

pub use action_encoder::*;
pub use agent::*;
pub use environment::*;
pub use feature_extractor::*;
pub use types::*;
