//! Azul Game Engine and RL Environment
//!
//! A Markov game state engine for the board game Azul, designed for RL training.
//!
//! This crate re-exports the engine and rl-env crates for convenience.

pub use azul_engine::*;
pub use azul_rl_env as rl_env;
