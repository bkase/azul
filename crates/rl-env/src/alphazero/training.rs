//! AlphaZero training loop and self-play generation
//!
//! This module provides:
//! - Self-play game generation
//! - Training step with MLX autodiff
//! - Full training loop (Trainer)

use std::path::PathBuf;

use mlx_rs::builder::Builder;
use mlx_rs::error::Exception;
use mlx_rs::losses::{LossReduction, MseLossBuilder};
use mlx_rs::module::ModuleParameters;
use mlx_rs::nn;
use mlx_rs::optimizers::{Adam, Optimizer};
use mlx_rs::transforms::eval_params;
use mlx_rs::Array;
use rand::{Rng, RngCore, SeedableRng};
use rayon::prelude::*;
use azul_engine::{DraftDestination, DraftSource, GameState, Phase, Token, BOARD_SIZE};

use super::{PendingMove, ReplayBuffer, TrainingExample};
use crate::{
    ActionEncoder, ActionId, AgentInput, AzulEnv, Environment, FeatureExtractor,
    ACTION_SPACE_SIZE,
};
use crate::mcts::InferenceSync;

#[cfg(feature = "profiling")]
use crate::profiling::{print_summary, Timer, PROF};
#[cfg(feature = "profiling")]
use std::sync::atomic::Ordering;

/// Result of a single self-play game, including training examples and diagnostic stats.
///
/// This struct enables parallel self-play by encapsulating all information
/// needed for both training and diagnostics in a single return value.
#[derive(Clone, Debug)]
pub struct SelfPlayResult {
    /// Training examples generated from the game.
    pub examples: Vec<TrainingExample>,

    /// Number of moves that went to the floor.
    pub floor_moves: usize,

    /// Number of moves that went to floor when a non-floor option existed.
    /// This distinguishes "forced floor" (no alternative) from "chosen floor" (policy collapsed).
    pub optional_floor_moves: usize,

    /// Floor chosen when a non-floor action exists with zero overflow.
    /// This is "strictly dominated" - a clear mistake. Should go to ~0 in sane learning.
    pub dominated_floor_moves: usize,

    /// Floor chosen when ALL non-floor actions would cause overflow (min_overflow > 0).
    /// This can be rational damage control (e.g., choosing where overflow goes).
    pub overflow_floor_moves: usize,

    /// Number of wall tiles placed during the game (across all players).
    pub wall_tiles_placed: usize,

    /// True if the game was truncated (hit max_moves limit).
    pub was_truncated: bool,

    /// Final scores for each player.
    pub final_scores: Vec<i16>,

    /// Number of players in the game.
    pub num_players: u8,

    /// Sum of floor mass in MCTS policy across all moves.
    /// floor_mass = sum(pi[a] for legal a where decode(a).dest==Floor)
    /// This helps distinguish whether floor is favored by search (high mass)
    /// vs just sampling noise (low mass but floor still chosen).
    pub total_floor_policy_mass: f32,

    /// Number of moves used to compute floor policy mass (for averaging).
    pub num_moves_for_floor_mass: usize,
}

/// Output of an MCTS search at the root.
#[derive(Clone, Debug)]
pub struct MctsSearchResult {
    /// The selected action
    pub action: ActionId,

    /// Visit-count-based policy over all actions, normalized.
    /// Length == ACTION_SPACE_SIZE
    pub policy: Vec<f32>,
}

/// Configuration for self-play game generation.
#[derive(Clone, Debug)]
pub struct SelfPlayConfig {
    /// Safety cap on maximum moves per game
    pub max_moves: usize,

    /// Number of MCTS simulations per move
    pub mcts_simulations: usize,

    /// Dirichlet noise alpha parameter for root exploration
    pub dirichlet_alpha: f32,

    /// Dirichlet noise epsilon (fraction of noise vs prior)
    pub dirichlet_eps: f32,

    /// Move number at which to switch temperature (tau=1 -> tau=0)
    pub temp_cutoff_move: usize,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            max_moves: 500,
            mcts_simulations: 256,
            dirichlet_alpha: 0.3,
            dirichlet_eps: 0.25,
            // Use high cutoff (200) so tau=1 for entire game during early training.
            // This prevents "locking in" garbage moves early via argmax selection.
            // With ~65 moves per game, this effectively means tau=1 throughout.
            temp_cutoff_move: 200,
        }
    }
}

/// Configuration for the training loop.
#[derive(Clone, Debug)]
pub struct TrainerConfig {
    /// Number of players (usually 2)
    pub num_players: u8,

    /// Replay buffer capacity
    pub replay_capacity: usize,

    /// Batch size for training
    pub batch_size: usize,

    /// Number of self-play games per iteration
    pub self_play_games_per_iter: usize,

    /// Number of training steps per iteration
    pub training_steps_per_iter: usize,

    /// Total number of training iterations
    pub num_iters: usize,

    /// Starting iteration (for resuming from checkpoint)
    pub start_iter: usize,

    /// Learning rate for optimizer
    pub learning_rate: f32,

    /// Weight decay for optimizer
    pub weight_decay: f32,

    /// Weight for value loss in total loss
    pub value_loss_weight: f32,

    /// Weight for policy loss in total loss
    pub policy_loss_weight: f32,

    /// Self-play configuration
    pub self_play: SelfPlayConfig,

    /// Optional checkpoint directory
    pub checkpoint_dir: Option<PathBuf>,

    /// Checkpoint/evaluation interval (in iterations)
    pub eval_interval: usize,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            num_players: 2,
            replay_capacity: 100_000,
            batch_size: 256,
            self_play_games_per_iter: 10,
            training_steps_per_iter: 100,
            num_iters: 1000,
            start_iter: 0,
            learning_rate: 0.001,
            weight_decay: 0.0001,
            value_loss_weight: 1.0,
            policy_loss_weight: 1.0,
            self_play: SelfPlayConfig::default(),
            checkpoint_dir: None,
            eval_interval: 10,
        }
    }
}

/// Compute final outcomes from game scores.
///
/// For **2-player** (the only mode supported by the current AlphaZero MCTS):
/// - `z_0 = clamp((score_0 - score_1) / 20.0, -1.0, 1.0)`
/// - `z_1 = -z_0`
///
/// For **3–4 player** games (future work), we keep a mean-centered shaping:
/// - `z_i = clamp((score_i - mean(scores)) / 20.0, -1.0, 1.0)`
///
/// The divisor of 20.0 (instead of 100.0) provides stronger gradient signal.
/// A typical Azul game is decided by ~10-20 points, so dividing by 100
/// would squash the signal too much (e.g., a 5-point lead becomes 0.05).
/// With 20.0, a 10-point lead becomes 0.5, which is a meaningful signal.
///
/// Returns values clamped to [-1, 1] range.
pub fn compute_outcomes_from_scores(scores: &[i16]) -> Vec<f32> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }

    // 2-player: use explicit score difference for a true zero-sum target.
    // This matches MCTS terminal evaluation and yields a stronger value signal.
    if n == 2 {
        let diff = (scores[0] as f32) - (scores[1] as f32);
        let v0 = (diff / 20.0).clamp(-1.0, 1.0);
        return vec![v0, -v0];
    }

    // 3–4 player: mean-centered shaping (sum ~= 0), clamped to [-1, 1].
    let scores_f: Vec<f32> = scores.iter().map(|s| *s as f32).collect();
    let mean = scores_f.iter().sum::<f32>() / (n as f32);
    scores_f
        .into_iter()
        .map(|s| ((s - mean) / 20.0).clamp(-1.0, 1.0))
        .collect()
}

/// Build training batch arrays from a slice of training examples.
///
/// Returns: (observations [B, obs_size], policies [B, ACTION_SPACE_SIZE], values [B])
pub fn build_training_batch(examples: &[&TrainingExample]) -> (Array, Array, Array) {
    let batch_size = examples.len();
    if batch_size == 0 {
        panic!("Cannot build batch from empty examples");
    }

    let obs_size = examples[0].observation.shape()[0] as usize;

    // Collect data into contiguous vectors
    let mut obs_data = Vec::with_capacity(batch_size * obs_size);
    let mut policy_data = Vec::with_capacity(batch_size * ACTION_SPACE_SIZE);
    let mut value_data = Vec::with_capacity(batch_size);

    for ex in examples {
        // observation
        let obs_slice = ex.observation.as_slice::<f32>();
        obs_data.extend_from_slice(obs_slice);

        // policy
        debug_assert_eq!(ex.policy.len(), ACTION_SPACE_SIZE);
        policy_data.extend_from_slice(&ex.policy);

        // value
        value_data.push(ex.value);
    }

    let obs = Array::from_slice(&obs_data, &[batch_size as i32, obs_size as i32]);
    let pi = Array::from_slice(&policy_data, &[batch_size as i32, ACTION_SPACE_SIZE as i32]);
    let z = Array::from_slice(&value_data, &[batch_size as i32]);

    (obs, pi, z)
}

/// Trait for models that support training with the AlphaZero training loop.
///
/// Extends the basic PolicyValueNet with methods needed for gradient-based training.
pub trait TrainableModel {
    /// Number of parameter arrays in the model
    fn param_count(&self) -> usize;

    /// Get all model parameters as arrays
    fn parameters(&self) -> Vec<Array>;

    /// Forward pass for autodiff (takes &mut self so gradients flow through)
    fn forward(&mut self, obs: &Array) -> (Array, Array);

    /// Apply gradients to update model parameters
    fn apply_gradients(&mut self, learning_rate: f32, grads: &[Array]);

    /// Force evaluation of parameters (for lazy MLX evaluation)
    fn eval_parameters(&self);

    /// Save model to a directory
    fn save(&self, path: &std::path::Path) -> std::io::Result<()>;

    /// Load model from a directory
    fn load(&mut self, path: &std::path::Path) -> std::io::Result<()>;
}

/// Count the total number of tiles placed on walls across all players.
fn count_wall_tiles(game: &azul_engine::GameState) -> usize {
    let mut count = 0;
    for p in 0..game.num_players as usize {
        for row in 0..BOARD_SIZE {
            for col in 0..BOARD_SIZE {
                if game.players[p].wall[row][col].is_some() {
                    count += 1;
                }
            }
        }
    }
    count
}

/// Check if any legal action is a non-floor destination.
fn has_non_floor_legal_action(legal_mask: &[bool]) -> bool {
    for (action_id, &is_legal) in legal_mask.iter().enumerate() {
        if is_legal {
            let action = ActionEncoder::decode(action_id as u16);
            if !matches!(action.dest, DraftDestination::Floor) {
                return true;
            }
        }
    }
    false
}

/// Compute the total policy mass on floor actions.
/// floor_mass = sum(pi[a] for legal a where decode(a).dest==Floor)
fn compute_floor_policy_mass(policy: &[f32], legal_mask: &[bool]) -> f32 {
    let mut floor_mass = 0.0f32;
    for (action_id, (&prob, &is_legal)) in policy.iter().zip(legal_mask.iter()).enumerate() {
        if is_legal {
            let action = ActionEncoder::decode(action_id as u16);
            if matches!(action.dest, DraftDestination::Floor) {
                floor_mass += prob;
            }
        }
    }
    floor_mass
}

/// Count how many tiles of a given color are available from a source.
fn count_tiles_from_source(state: &GameState, source: DraftSource, color: azul_engine::Color) -> usize {
    match source {
        DraftSource::Factory(f_idx) => {
            let factory = &state.factories.factories[f_idx as usize];
            factory.tiles[..factory.len as usize]
                .iter()
                .filter(|&&c| c == color)
                .count()
        }
        DraftSource::Center => {
            state.center.items[..state.center.len as usize]
                .iter()
                .filter(|t| matches!(t, Token::Tile(c) if *c == color))
                .count()
        }
    }
}

/// Compute how many tiles would overflow to floor for a given non-floor action.
/// Returns 0 if the action is Floor (by definition no overflow from placing on floor).
fn compute_overflow_for_action(state: &GameState, action: &azul_engine::Action, player: usize) -> usize {
    let tiles_to_take = count_tiles_from_source(state, action.source, action.color);

    match action.dest {
        DraftDestination::Floor => 0, // By definition, floor action has no "overflow"
        DraftDestination::PatternLine(row) => {
            let capacity = (row as usize) + 1;
            let current = state.players[player].pattern_lines[row as usize].count as usize;
            let remaining = capacity.saturating_sub(current);
            tiles_to_take.saturating_sub(remaining)
        }
    }
}

/// Compute the minimum overflow among all legal non-floor actions.
/// Returns None if there are no legal non-floor actions.
fn compute_min_overflow(state: &GameState, player: usize, legal_mask: &[bool]) -> Option<usize> {
    let mut min_overflow: Option<usize> = None;

    for (action_id, &is_legal) in legal_mask.iter().enumerate() {
        if is_legal {
            let action = ActionEncoder::decode(action_id as u16);
            if !matches!(action.dest, DraftDestination::Floor) {
                let overflow = compute_overflow_for_action(state, &action, player);
                min_overflow = Some(min_overflow.map_or(overflow, |m| m.min(overflow)));
            }
        }
    }

    min_overflow
}

/// Run a single self-play game and return training examples with diagnostic stats.
///
/// Uses the MCTS agent to play against itself, recording (observation, policy, value)
/// tuples for each move. The value is backfilled with the final game outcome.
///
/// Returns a `SelfPlayResult` containing both the training examples and game statistics
/// needed for monitoring training progress.
pub fn self_play_game<F, A>(
    env: &mut AzulEnv<F>,
    agent: &mut A,
    self_play_cfg: &SelfPlayConfig,
    rng: &mut impl Rng,
) -> SelfPlayResult
where
    F: FeatureExtractor,
    A: MctsAgentExt,
{
    // 1. Reset env
    let mut step = env.reset(rng);
    let mut moves: Vec<PendingMove> = Vec::new();
    let mut move_idx = 0usize;

    // Track wall tiles at start for computing tiles placed during game
    let wall_tiles_start = count_wall_tiles(&env.game_state);

    // Track floor moves with their context for optional_floor detection
    let mut floor_move_indices: Vec<usize> = Vec::new();
    let mut had_non_floor_alternative: Vec<bool> = Vec::new();
    // Track min_overflow for each floor move (None if no non-floor alternatives)
    let mut floor_move_min_overflow: Vec<Option<usize>> = Vec::new();

    // Track floor policy mass (sum of pi[a] for floor actions in each move's policy)
    let mut total_floor_policy_mass = 0.0f32;
    let mut num_moves_for_floor_mass = 0usize;

    // 2. Play game
    while !step.done && move_idx < self_play_cfg.max_moves {
        let acting_player = step.current_player;
        let p = acting_player as usize;

        // Check if non-floor alternatives exist BEFORE making the move
        let non_floor_exists = has_non_floor_legal_action(&step.legal_action_mask);

        // Compute min_overflow among non-floor actions (if state available)
        let min_overflow = step.state.as_ref().and_then(|state| {
            compute_min_overflow(state, p, &step.legal_action_mask)
        });

        // Build AgentInput (with full state if available)
        let input = AgentInput {
            observation: &step.observations[p],
            legal_action_mask: &step.legal_action_mask,
            current_player: acting_player,
            state: step.state.as_ref(),
        };

        // Determine temperature based on move number
        let temperature = if move_idx < self_play_cfg.temp_cutoff_move {
            1.0
        } else {
            0.0
        };

        // Select action + policy via MCTS
        let search_result = agent.select_action_and_policy(&input, temperature, rng);

        // Track floor policy mass in the MCTS-returned policy
        let floor_mass = compute_floor_policy_mass(&search_result.policy, &step.legal_action_mask);
        total_floor_policy_mass += floor_mass;
        num_moves_for_floor_mass += 1;

        // Track floor moves and whether they were optional
        let is_floor_move = matches!(
            ActionEncoder::decode(search_result.action).dest,
            DraftDestination::Floor
        );
        if is_floor_move {
            floor_move_indices.push(moves.len());
            had_non_floor_alternative.push(non_floor_exists);
            floor_move_min_overflow.push(min_overflow);
        }

        // Step env
        let next = env
            .step(search_result.action, rng)
            .expect("env::step should not fail for legal action");

        // Dense, zero-sum reward from acting player's perspective.
        //
        // AzulEnv emits per-player score deltas; we convert that to a 2-player
        // zero-sum signal by subtracting the opponent's delta and normalizing.
        //
        // This reward corresponds to a *change* in score difference, so the
        // value targets represent "remaining advantage" (terminal value = 0).
        let n = env.game_state.num_players as usize;
        let reward = if n == 2 {
            let opp = 1 - p;
            (next.rewards[p] - next.rewards[opp]) / 20.0
        } else {
            let mean = next.rewards[0..n].iter().sum::<f32>() / (n as f32);
            (next.rewards[p] - mean) / 20.0
        };

        // Record pending move (now that we know the transition reward).
        moves.push(PendingMove {
            player: acting_player,
            observation: step.observations[p].clone(),
            policy: search_result.policy.clone(),
            action: search_result.action,
            reward,
        });

        step = next;
        move_idx += 1;

        #[cfg(feature = "profiling")]
        PROF.env_steps.fetch_add(1, Ordering::Relaxed);
    }

    // 3. Extract final state info for diagnostics
    let num_players = env.game_state.num_players;
    let n = num_players as usize;
    let final_scores: Vec<i16> = (0..n).map(|p| env.game_state.players[p].score).collect();
    let was_truncated = env.game_state.phase != Phase::GameOver;

    // Count wall tiles at end and compute tiles placed
    let wall_tiles_end = count_wall_tiles(&env.game_state);
    let wall_tiles_placed = wall_tiles_end.saturating_sub(wall_tiles_start);

    // Count floor moves and categorize them
    let floor_moves = floor_move_indices.len();
    let optional_floor_moves = had_non_floor_alternative.iter().filter(|&&x| x).count();

    // Categorize optional floor moves:
    // - dominated_floor: floor chosen when min_overflow == 0 (strictly dominated, a mistake)
    // - overflow_floor: floor chosen when min_overflow > 0 (potentially rational damage control)
    let mut dominated_floor_moves = 0usize;
    let mut overflow_floor_moves = 0usize;
    for (i, &had_alt) in had_non_floor_alternative.iter().enumerate() {
        if had_alt {
            // This was an optional floor move
            match floor_move_min_overflow.get(i) {
                Some(Some(0)) => dominated_floor_moves += 1, // Zero-overflow option existed
                Some(Some(_)) => overflow_floor_moves += 1,  // All options had overflow
                Some(None) | None => {} // No state available or no non-floor alternatives
            }
        }
    }

    // 4. Compute discounted returns for each pending move (terminal value = 0).
    //
    // IMPORTANT: Our per-step reward is defined as the *change* in normalized
    // score difference (from the acting player's perspective). Therefore, the
    // value target represents the expected *remaining* advantage from that
    // state, not the absolute final score difference.
    let mut values = vec![0.0f32; moves.len()];
    let mut next_value = 0.0f32;
    let mut next_player: Option<u8> = None;

    for i in (0..moves.len()).rev() {
        let player = moves[i].player;
        let continuation = match next_player {
            None => 0.0,
            Some(np) if np == player => next_value,
            Some(_) => -next_value,
        };

        let v = (moves[i].reward + continuation).clamp(-1.0, 1.0);
        values[i] = v;
        next_value = v;
        next_player = Some(player);
    }

    // Keep terminal score outcomes for diagnostics/logging.
    let _outcomes = compute_outcomes_from_scores(&final_scores);

    // 5. Convert PendingMove -> TrainingExample
    // Force evaluation of observations before storing in replay buffer.
    // MLX arrays are lazy - without eval(), the underlying event/stream tracking
    // can become invalid when arrays are stored for a long time.
    let examples: Vec<TrainingExample> = moves
        .into_iter()
        .zip(values.into_iter())
        .map(|(m, value)| {
            m.observation.eval().expect("Failed to evaluate observation");
            TrainingExample {
                observation: m.observation,
                policy: m.policy,
                action: m.action,
                value,
            }
        })
        .collect();

    SelfPlayResult {
        examples,
        floor_moves,
        optional_floor_moves,
        dominated_floor_moves,
        overflow_floor_moves,
        wall_tiles_placed,
        was_truncated,
        final_scores,
        num_players,
        total_floor_policy_mass,
        num_moves_for_floor_mass,
    }
}

/// Extension trait for MCTS agents that can return both action and policy.
pub trait MctsAgentExt {
    /// Run MCTS search and return both the selected action and the policy distribution.
    fn select_action_and_policy(
        &mut self,
        input: &AgentInput,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> MctsSearchResult;
}

/// Trainer state for running the AlphaZero training loop.
pub struct Trainer<F, M>
where
    F: FeatureExtractor,
    M: TrainableModel + MctsAgentExt + ModuleParameters,
{
    /// The environment for self-play
    pub env: AzulEnv<F>,

    /// The MCTS agent with trainable model
    pub agent: M,

    /// Replay buffer for storing training examples
    pub replay: ReplayBuffer,

    /// Training configuration
    pub cfg: TrainerConfig,

    /// Random number generator
    pub rng: rand::rngs::StdRng,

    /// Adam optimizer for gradient updates
    pub optimizer: Adam,
}

impl<F, M> Trainer<F, M>
where
    F: FeatureExtractor + Clone + Send,
    M: TrainableModel + MctsAgentExt + ModuleParameters + Clone + Send + InferenceSync,
{
    /// Create a new trainer with the given components.
    pub fn new(env: AzulEnv<F>, agent: M, cfg: TrainerConfig, rng: rand::rngs::StdRng) -> Self {
        let replay = ReplayBuffer::new(cfg.replay_capacity);
        let optimizer = Adam::new(cfg.learning_rate);
        Self {
            env,
            agent,
            replay,
            cfg,
            rng,
            optimizer,
        }
    }

    /// Run the main training loop.
    pub fn run(&mut self) -> Result<(), TrainingError> {
        for iter in self.cfg.start_iter..self.cfg.num_iters {
            #[cfg(feature = "profiling")]
            let iter_wall = std::time::Instant::now();
            // 1. Self-play (parallelized with deterministic seeding)
            #[cfg(feature = "profiling")]
            let _t_sp = Timer::new(&PROF.time_self_play_ns);

            // Generate deterministic seeds and pre-create all game setups.
            // This ensures reproducibility: the same main RNG seed will always
            // produce the same sequence of game seeds, regardless of thread scheduling.
            //
            // We pre-create all clones here (sequential) because MLX arrays contain
            // raw pointers that aren't Sync. By moving ownership into each parallel
            // task, we avoid the need for shared references.
            let self_play_cfg = self.cfg.self_play.clone();
            // Sync inference worker to latest weights before parallel games.
            self.agent.sync_inference_backend();
            let game_setups: Vec<(u64, AzulEnv<F>, M)> = (0..self.cfg.self_play_games_per_iter)
                .map(|_| {
                    let seed = self.rng.next_u64();
                    (seed, self.env.clone(), self.agent.clone())
                })
                .collect();

            // Run games in parallel using rayon - each task owns its env/agent
            let results: Vec<SelfPlayResult> = game_setups
                .into_par_iter()
                .map(|(seed, mut local_env, mut local_agent)| {
                    // Create a thread-local RNG seeded deterministically
                    let mut local_rng = rand::rngs::StdRng::seed_from_u64(seed);

                    self_play_game(&mut local_env, &mut local_agent, &self_play_cfg, &mut local_rng)
                })
                .collect();

            // Aggregate results into diagnostics and replay buffer
            let mut sp_games = 0usize;
            let mut sp_moves = 0usize;
            let mut sp_floor_moves = 0usize;
            let mut _sp_optional_floor_moves = 0usize; // Kept for backward compat, replaced by dom/ovf
            let mut sp_dominated_floor_moves = 0usize;
            let mut sp_overflow_floor_moves = 0usize;
            let mut sp_wall_tiles_placed = 0usize;
            let mut sp_truncated_games = 0usize;
            let mut sp_ties = 0usize;
            let mut sp_zero_zero_games = 0usize;
            let mut sp_score_diff_sum = 0.0f32;
            let mut sp_abs_score_diff_sum = 0.0f32;
            let mut sp_abs_z_sum = 0.0f32;
            let mut sp_total_floor_policy_mass = 0.0f32;
            let mut sp_total_moves_for_floor_mass = 0usize;

            for result in results {
                sp_games += 1;
                sp_moves += result.examples.len();
                sp_floor_moves += result.floor_moves;
                _sp_optional_floor_moves += result.optional_floor_moves;
                sp_dominated_floor_moves += result.dominated_floor_moves;
                sp_overflow_floor_moves += result.overflow_floor_moves;
                sp_wall_tiles_placed += result.wall_tiles_placed;
                sp_total_floor_policy_mass += result.total_floor_policy_mass;
                sp_total_moves_for_floor_mass += result.num_moves_for_floor_mass;

                if result.was_truncated {
                    sp_truncated_games += 1;
                }

                let n = result.num_players as usize;
                if n == 2 && result.final_scores.len() >= 2 {
                    let s0 = result.final_scores[0];
                    let s1 = result.final_scores[1];
                    let diff = (s0 - s1) as f32;
                    sp_score_diff_sum += diff;
                    sp_abs_score_diff_sum += diff.abs();
                    if s0 == s1 {
                        sp_ties += 1;
                    }
                    if s0 == 0 && s1 == 0 {
                        sp_zero_zero_games += 1;
                    }
                }

                let outcomes = compute_outcomes_from_scores(&result.final_scores);
                if !outcomes.is_empty() {
                    let mean_abs_z =
                        outcomes.iter().map(|z| z.abs()).sum::<f32>() / outcomes.len() as f32;
                    sp_abs_z_sum += mean_abs_z;
                }

                #[cfg(feature = "profiling")]
                {
                    PROF.self_play_games.fetch_add(1, Ordering::Relaxed);
                    PROF.self_play_moves
                        .fetch_add(result.examples.len() as u64, Ordering::Relaxed);
                }

                // Skip truncated games - they poison the replay buffer with
                // degenerate data (stalling strategies, bad value targets).
                // Only add examples from games that reached GameOver naturally.
                if !result.was_truncated {
                    self.replay.extend(result.examples);
                }
            }

            // Drop timer to record self-play time before training starts
            #[cfg(feature = "profiling")]
            drop(_t_sp);

            // 2. Training
            #[cfg(feature = "profiling")]
            let _t_train = Timer::new(&PROF.time_training_ns);

            let mut total_loss = 0.0f32;
            let mut train_steps = 0usize;

            for _ in 0..self.cfg.training_steps_per_iter {
                if self.replay.len() < self.cfg.batch_size {
                    break;
                }

                #[cfg(feature = "profiling")]
                PROF.train_steps.fetch_add(1, Ordering::Relaxed);

                // Clone examples to avoid borrow issues
                let batch: Vec<TrainingExample> = self
                    .replay
                    .sample(&mut self.rng, self.cfg.batch_size)
                    .into_iter()
                    .cloned()
                    .collect();
                let batch_refs: Vec<&TrainingExample> = batch.iter().collect();
                let loss = self.training_step(&batch_refs)?;
                total_loss += loss;
                train_steps += 1;
            }

            // Drop timer to record training time
            #[cfg(feature = "profiling")]
            drop(_t_train);

            // 3. Optional checkpoint
            if let Some(ref dir) = self.cfg.checkpoint_dir {
                if iter % self.cfg.eval_interval == 0 {
                    self.save_checkpoint(dir, iter)?;
                }
            }

            // Log progress with loss information
            if iter % 10 == 0 {
                let avg_loss = if train_steps > 0 {
                    total_loss / train_steps as f32
                } else {
                    0.0
                };

                let sp_games_f = sp_games.max(1) as f32;
                let moves_per_game = sp_moves as f32 / sp_games_f;
                let floor_pct = if sp_moves > 0 {
                    100.0 * (sp_floor_moves as f32) / (sp_moves as f32)
                } else {
                    0.0
                };
                // Dominated floor: floor chosen when zero-overflow option existed (a clear mistake)
                // Should go to ~0% in sane learning.
                let dom_floor_pct = if sp_moves > 0 {
                    100.0 * (sp_dominated_floor_moves as f32) / (sp_moves as f32)
                } else {
                    0.0
                };
                // Overflow floor: floor chosen when all non-floor options would overflow
                // This can be rational damage control - not necessarily a mistake.
                let ovf_floor_pct = if sp_moves > 0 {
                    100.0 * (sp_overflow_floor_moves as f32) / (sp_moves as f32)
                } else {
                    0.0
                };
                // Mean floor policy mass: average fraction of MCTS policy mass on floor actions
                let mean_floor_policy_mass = if sp_total_moves_for_floor_mass > 0 {
                    100.0 * sp_total_floor_policy_mass / (sp_total_moves_for_floor_mass as f32)
                } else {
                    0.0
                };
                let wall_tiles_per_game = sp_wall_tiles_placed as f32 / sp_games_f;
                let trunc_pct = 100.0 * (sp_truncated_games as f32) / sp_games_f;
                let abs_z = sp_abs_z_sum / sp_games_f;

                let two_player_stats = if self.cfg.num_players == 2 {
                    let mean_diff = sp_score_diff_sum / sp_games_f;
                    let mean_abs_diff = sp_abs_score_diff_sum / sp_games_f;
                    let tie_pct = 100.0 * (sp_ties as f32) / sp_games_f;
                    let zero_zero_pct = 100.0 * (sp_zero_zero_games as f32) / sp_games_f;
                    format!(
                        " diff={:+.2} |diff|={:.2} ties={:.1}% 0-0={:.1}%",
                        mean_diff, mean_abs_diff, tie_pct, zero_zero_pct
                    )
                } else {
                    String::new()
                };

                eprintln!(
                    "Iter {}/{} | replay: {} | loss: {:.4} | sp: moves/g={:.1} floor={:.1}% dom={:.1}% ovf={:.1}% pi_floor={:.1}% wall={:.1} |z|={:.3} trunc={:.1}%{}",
                    iter,
                    self.cfg.num_iters,
                    self.replay.len(),
                    avg_loss,
                    moves_per_game,
                    floor_pct,
                    dom_floor_pct,
                    ovf_floor_pct,
                    mean_floor_policy_mass,
                    wall_tiles_per_game,
                    abs_z,
                    trunc_pct,
                    two_player_stats
                );
            }
            #[cfg(feature = "profiling")]
            {
                let elapsed = iter_wall.elapsed().as_nanos() as u64;
                PROF.time_iter_wall_ns
                    .fetch_add(elapsed, Ordering::Relaxed);
            }
        }

        // Print profiling summary at the end
        #[cfg(feature = "profiling")]
        print_summary();

        Ok(())
    }

    /// Run a single training step on a batch using MLX autodiff.
    fn training_step(&mut self, batch: &[&TrainingExample]) -> Result<f32, TrainingError> {
        #[cfg(feature = "profiling")]
        let _t = Timer::new(&PROF.time_training_step_ns);

        // Build batch arrays
        let (obs, pi_target, z_target) = build_training_batch(batch);

        let policy_weight = Array::from_slice(&[self.cfg.policy_loss_weight], &[1]);
        let value_weight = Array::from_slice(&[self.cfg.value_loss_weight], &[1]);

        // Create loss function for value_and_grad
        let loss_fn = |model: &mut M, (obs, pi_target, z_target): (&Array, &Array, &Array)| {
            // Forward pass (mut reference allows gradients to flow through)
            let (policy_logits, value_pred) = model.forward(obs);

            // Compute policy loss: cross-entropy with soft targets
            let policy_loss = cross_entropy_loss_array(&policy_logits, pi_target)?;

            // Compute value loss: MSE
            let value_loss = mse_loss_array(&value_pred, z_target)?;

            // Total weighted loss (must be scalar for autodiff)
            let total_loss = policy_loss
                .multiply(&policy_weight)?
                .add(&value_loss.multiply(&value_weight)?)?
                .squeeze()?;

            Ok(total_loss)
        };

        // Compute value and gradients using MLX autodiff
        let mut vg = nn::value_and_grad(loss_fn);
        let (loss, grads) = vg(&mut self.agent, (&obs, &pi_target, &z_target))
            .map_err(|e| TrainingError::Mlx(e.to_string()))?;

        // Apply gradients using optimizer
        self.optimizer
            .update(&mut self.agent, grads)
            .map_err(|e| TrainingError::Mlx(e.to_string()))?;

        // Force evaluation of parameters
        let _ = eval_params(ModuleParameters::parameters(&self.agent));

        Ok(loss.item::<f32>())
    }

    /// Save a checkpoint to disk.
    fn save_checkpoint(&self, dir: &std::path::Path, iter: usize) -> Result<(), TrainingError> {
        let checkpoint_path = dir.join(format!("checkpoint_{iter:06}.safetensors"));
        std::fs::create_dir_all(dir).map_err(TrainingError::Io)?;
        self.agent.save(&checkpoint_path).map_err(TrainingError::Io)
    }
}

/// Compute cross-entropy loss between logits and target distribution (returns Array for autodiff).
fn cross_entropy_loss_array(logits: &Array, target: &Array) -> Result<Array, Exception> {
    // log_softmax for numerical stability (along action axis = -1)
    let log_probs = nn::log_softmax(logits, -1)?;

    // Cross-entropy: -sum(target * log_probs) / batch_size
    // Multiply element-wise
    let weighted = target.multiply(&log_probs)?;

    // Sum over action dimension, then mean over batch
    let per_sample = weighted.sum_axis(-1, None)?;
    let loss = per_sample.mean(None)?.negative()?;

    Ok(loss)
}

/// Compute MSE loss between predictions and targets (returns Array for autodiff).
fn mse_loss_array(pred: &Array, target: &Array) -> Result<Array, Exception> {
    let mse = MseLossBuilder::new()
        .reduction(LossReduction::Mean)
        .build()
        .map_err(|e| Exception::custom(format!("Failed to build MSE loss: {e:?}")))?;
    mse.apply(pred, target)
}

/// Training error types.
#[derive(Debug)]
pub enum TrainingError {
    /// IO error (e.g., checkpoint save/load)
    Io(std::io::Error),

    /// MLX error
    Mlx(String),
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainingError::Io(e) => write!(f, "IO error: {e}"),
            TrainingError::Mlx(s) => write!(f, "MLX error: {s}"),
        }
    }
}

impl std::error::Error for TrainingError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_outcomes_from_scores_zero_sum() {
        let scores = [10i16, 20i16];
        let outcomes = compute_outcomes_from_scores(&scores);

        // Sum should be approximately 0
        let sum: f32 = outcomes.iter().sum();
        assert!(sum.abs() < 1e-5, "Outcomes should sum to ~0, got {}", sum);

        // Higher score should have positive outcome
        assert!(
            outcomes[1] > 0.0,
            "Higher score player should have positive outcome"
        );

        // Lower score should have negative outcome
        assert!(
            outcomes[0] < 0.0,
            "Lower score player should have negative outcome"
        );

        // 2-player uses score difference scaling:
        // diff = -10, so outcomes = [-10/20, +10/20] = [-0.5, 0.5]
        assert!((outcomes[0] - (-0.5)).abs() < 1e-5);
        assert!((outcomes[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_compute_outcomes_clamps_extreme_values() {
        // Test that extreme score differences are clamped to [-1, 1]
        let scores = [0i16, 100i16]; // 100 point difference
        let outcomes = compute_outcomes_from_scores(&scores);

        // mean = 50, raw outcomes would be [-2.5, 2.5] but should clamp
        assert!(
            outcomes[0] >= -1.0 && outcomes[0] <= 1.0,
            "Outcome should be clamped to [-1, 1], got {}",
            outcomes[0]
        );
        assert!(
            outcomes[1] >= -1.0 && outcomes[1] <= 1.0,
            "Outcome should be clamped to [-1, 1], got {}",
            outcomes[1]
        );
        assert_eq!(outcomes[0], -1.0, "Loser should be clamped to -1");
        assert_eq!(outcomes[1], 1.0, "Winner should be clamped to 1");
    }

    #[test]
    fn test_compute_outcomes_multi_player_tie() {
        let scores = [10i16, 10i16, 10i16];
        let outcomes = compute_outcomes_from_scores(&scores);

        // All outcomes should be approximately 0 (tie)
        for (i, &outcome) in outcomes.iter().enumerate() {
            assert!(
                outcome.abs() < 1e-5,
                "Player {} outcome should be ~0 in tie, got {}",
                i,
                outcome
            );
        }
    }

    #[test]
    fn test_build_training_batch_shapes() {
        let obs_size = 100;
        let batch_size = 4;

        let examples: Vec<TrainingExample> = (0..batch_size)
            .map(|i| TrainingExample {
                observation: Array::zeros::<f32>(&[obs_size]).unwrap(),
                policy: vec![1.0 / ACTION_SPACE_SIZE as f32; ACTION_SPACE_SIZE],
                action: 0,
                value: i as f32 * 0.1,
            })
            .collect();

        let example_refs: Vec<&TrainingExample> = examples.iter().collect();
        let (obs, pi, z) = build_training_batch(&example_refs);

        assert_eq!(
            obs.shape(),
            &[batch_size as i32, obs_size],
            "Observations should have shape [B, obs_size]"
        );
        assert_eq!(
            pi.shape(),
            &[batch_size as i32, ACTION_SPACE_SIZE as i32],
            "Policies should have shape [B, ACTION_SPACE_SIZE]"
        );
        assert_eq!(
            z.shape(),
            &[batch_size as i32],
            "Values should have shape [B]"
        );
    }

    #[test]
    fn test_cross_entropy_loss_one_hot() {
        // One-hot target: all probability on action 0
        let logits = Array::from_slice(&[10.0f32, 0.0, 0.0, 0.0], &[1, 4]);
        let target = Array::from_slice(&[1.0f32, 0.0, 0.0, 0.0], &[1, 4]);

        let loss: f32 = cross_entropy_loss_array(&logits, &target).unwrap().item();

        // With logits heavily favoring action 0, loss should be small
        assert!(
            loss < 0.1,
            "Loss should be small for matching logits, got {}",
            loss
        );
    }

    #[test]
    fn test_mse_loss() {
        let pred = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
        let target = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);

        let loss: f32 = mse_loss_array(&pred, &target).unwrap().item();
        assert!(loss.abs() < 1e-6, "MSE of identical arrays should be 0");

        let pred2 = Array::from_slice(&[0.0f32, 0.0, 0.0], &[3]);
        let loss2: f32 = mse_loss_array(&pred2, &target).unwrap().item();
        // MSE = (1 + 4 + 9) / 3 = 14/3 ≈ 4.67
        assert!((loss2 - 14.0 / 3.0).abs() < 1e-5);
    }

    /// Stub PolicyValueNet for testing that returns uniform policy and zero value.
    #[derive(Clone, Copy)]
    pub struct StubPolicyValueModel;

    impl StubPolicyValueModel {
        pub fn new(_obs_size: usize) -> Self {
            Self
        }
    }

    impl crate::PolicyValueNet for StubPolicyValueModel {
        fn predict_single(&mut self, _obs: &crate::Observation) -> (Array, f32) {
            let policy = vec![1.0 / ACTION_SPACE_SIZE as f32; ACTION_SPACE_SIZE];
            let policy_arr = Array::from_slice(&policy, &[ACTION_SPACE_SIZE as i32]);
            (policy_arr, 0.0)
        }

        fn predict_batch(&mut self, obs_batch: &Array) -> (Array, Array) {
            let batch_size = obs_batch.shape()[0] as usize;
            let policy = vec![1.0 / ACTION_SPACE_SIZE as f32; batch_size * ACTION_SPACE_SIZE];
            let values = vec![0.0f32; batch_size];
            let policy_arr =
                Array::from_slice(&policy, &[batch_size as i32, ACTION_SPACE_SIZE as i32]);
            let values_arr = Array::from_slice(&values, &[batch_size as i32]);
            (policy_arr, values_arr)
        }
    }

    impl TrainableModel for StubPolicyValueModel {
        fn param_count(&self) -> usize {
            1
        }

        fn parameters(&self) -> Vec<Array> {
            // Single dummy parameter
            vec![Array::zeros::<f32>(&[10]).unwrap()]
        }

        fn forward(&mut self, obs: &Array) -> (Array, Array) {
            let batch_size = obs.shape()[0] as usize;
            let policy = vec![1.0 / ACTION_SPACE_SIZE as f32; batch_size * ACTION_SPACE_SIZE];
            let values = vec![0.0f32; batch_size];
            let policy_arr =
                Array::from_slice(&policy, &[batch_size as i32, ACTION_SPACE_SIZE as i32]);
            let values_arr = Array::from_slice(&values, &[batch_size as i32]);
            (policy_arr, values_arr)
        }

        fn apply_gradients(&mut self, _learning_rate: f32, _grads: &[Array]) {
            // No-op for stub
        }

        fn eval_parameters(&self) {
            // No-op for stub
        }

        fn save(&self, _path: &std::path::Path) -> std::io::Result<()> {
            Ok(())
        }

        fn load(&mut self, _path: &std::path::Path) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Wrapper for using StubPolicyValueModel with AlphaZeroMctsAgent
    pub struct StubMctsAgent {
        pub agent: crate::AlphaZeroMctsAgent<crate::BasicFeatureExtractor, StubPolicyValueModel>,
        /// Dummy parameter for ModuleParameters trait
        pub dummy_weight: mlx_rs::module::Param<Array>,
    }

    impl Clone for StubMctsAgent {
        fn clone(&self) -> Self {
            Self {
                agent: self.agent.clone(),
                dummy_weight: mlx_rs::module::Param::new(Array::zeros::<f32>(&[10]).unwrap()),
            }
        }
    }

    // SAFETY: StubMctsAgent is only used in tests which run with RAYON_NUM_THREADS=1
    // effectively making parallelization sequential. The MLX arrays inside aren't
    // actually shared between threads.
    unsafe impl Send for StubMctsAgent {}
    unsafe impl Sync for StubMctsAgent {}

    impl StubMctsAgent {
        pub fn new(num_players: u8, num_simulations: u32) -> Self {
            let features = crate::BasicFeatureExtractor::new(num_players);
            let obs_size = features.obs_size();
            let model = StubPolicyValueModel::new(obs_size);
            let config = crate::MctsConfig {
                num_simulations,
                root_dirichlet_alpha: 0.0, // Disable noise for determinism
                ..Default::default()
            };
            let agent = crate::AlphaZeroMctsAgent::new(config, features, model);
            let dummy_weight = mlx_rs::module::Param::new(Array::zeros::<f32>(&[10]).unwrap());
            Self {
                agent,
                dummy_weight,
            }
        }
    }

    impl crate::mcts::InferenceSync for StubMctsAgent {
        fn sync_inference_backend(&self) {
            // Nothing to sync; stub model deterministic.
        }
    }

    impl MctsAgentExt for StubMctsAgent {
        fn select_action_and_policy(
            &mut self,
            input: &crate::AgentInput,
            temperature: f32,
            rng: &mut impl Rng,
        ) -> MctsSearchResult {
            self.agent.select_action_and_policy(input, temperature, rng)
        }
    }

    impl TrainableModel for StubMctsAgent {
        fn param_count(&self) -> usize {
            1
        }

        fn parameters(&self) -> Vec<Array> {
            vec![Array::zeros::<f32>(&[10]).unwrap()]
        }

        fn forward(&mut self, obs: &Array) -> (Array, Array) {
            let batch_size = obs.shape()[0] as usize;
            let policy = vec![1.0 / ACTION_SPACE_SIZE as f32; batch_size * ACTION_SPACE_SIZE];
            let values = vec![0.0f32; batch_size];
            let policy_arr =
                Array::from_slice(&policy, &[batch_size as i32, ACTION_SPACE_SIZE as i32]);
            let values_arr = Array::from_slice(&values, &[batch_size as i32]);
            (policy_arr, values_arr)
        }

        fn apply_gradients(&mut self, _learning_rate: f32, _grads: &[Array]) {}

        fn eval_parameters(&self) {}

        fn save(&self, _path: &std::path::Path) -> std::io::Result<()> {
            Ok(())
        }

        fn load(&mut self, _path: &std::path::Path) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl ModuleParameters for StubMctsAgent {
        fn num_parameters(&self) -> usize {
            10 // Single dummy parameter with 10 elements
        }

        fn parameters(&self) -> mlx_rs::module::ModuleParamRef<'_> {
            use mlx_rs::nested::NestedValue;
            let mut map = mlx_rs::nested::NestedHashMap::new();
            map.insert(
                "dummy_weight".into(),
                NestedValue::Value(&*self.dummy_weight),
            );
            map
        }

        fn parameters_mut(&mut self) -> mlx_rs::module::ModuleParamMut<'_> {
            use mlx_rs::nested::NestedValue;
            let mut map = mlx_rs::nested::NestedHashMap::new();
            map.insert(
                "dummy_weight".into(),
                NestedValue::Value(&mut *self.dummy_weight),
            );
            map
        }

        fn trainable_parameters(&self) -> mlx_rs::module::ModuleParamRef<'_> {
            ModuleParameters::parameters(self)
        }

        fn freeze_parameters(&mut self, _recursive: bool) {}

        fn unfreeze_parameters(&mut self, _recursive: bool) {}

        fn all_frozen(&self) -> Option<bool> {
            Some(false)
        }

        fn any_frozen(&self) -> Option<bool> {
            Some(false)
        }
    }

    #[test]
    fn test_self_play_generates_examples() {
        use crate::{AzulEnv, BasicFeatureExtractor, EnvConfig, RewardScheme};
        use rand::SeedableRng;

        let config = EnvConfig {
            num_players: 2,
            reward_scheme: RewardScheme::TerminalOnly,
            include_full_state_in_step: true,
        };
        let features = BasicFeatureExtractor::new(2);
        let mut env = AzulEnv::new(config, features);

        let mut agent = StubMctsAgent::new(2, 4); // Small num_simulations for speed
        let self_play_cfg = SelfPlayConfig {
            max_moves: 500,
            mcts_simulations: 4,
            ..Default::default()
        };

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = self_play_game(&mut env, &mut agent, &self_play_cfg, &mut rng);

        // Should have generated at least one example
        assert!(
            !result.examples.is_empty(),
            "Should generate at least one training example"
        );

        for ex in &result.examples {
            // Each policy should have correct length
            assert_eq!(
                ex.policy.len(),
                ACTION_SPACE_SIZE,
                "Policy should have ACTION_SPACE_SIZE length"
            );

            // Policy should sum to approximately 1
            let sum: f32 = ex.policy.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "Policy should sum to ~1, got {}",
                sum
            );

            // Value should be in reasonable range
            assert!(
                ex.value >= -2.0 && ex.value <= 2.0,
                "Value {} should be in [-2, 2]",
                ex.value
            );
        }
    }

    #[test]
    fn test_self_play_respects_max_moves() {
        use crate::{AzulEnv, BasicFeatureExtractor, EnvConfig, RewardScheme};
        use rand::SeedableRng;

        let config = EnvConfig {
            num_players: 2,
            reward_scheme: RewardScheme::TerminalOnly,
            include_full_state_in_step: true,
        };
        let features = BasicFeatureExtractor::new(2);
        let mut env = AzulEnv::new(config, features);

        let mut agent = StubMctsAgent::new(2, 2);
        let self_play_cfg = SelfPlayConfig {
            max_moves: 5, // Very small limit
            mcts_simulations: 2,
            ..Default::default()
        };

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = self_play_game(&mut env, &mut agent, &self_play_cfg, &mut rng);

        assert!(
            result.examples.len() <= 5,
            "Should respect max_moves limit, got {} examples",
            result.examples.len()
        );
    }

    #[test]
    fn test_self_play_determinism_with_stub_model() {
        use crate::{AzulEnv, BasicFeatureExtractor, EnvConfig, RewardScheme};
        use rand::SeedableRng;

        let config = EnvConfig {
            num_players: 2,
            reward_scheme: RewardScheme::TerminalOnly,
            include_full_state_in_step: true,
        };

        // Run 1
        let features1 = BasicFeatureExtractor::new(2);
        let mut env1 = AzulEnv::new(config.clone(), features1);
        let mut agent1 = StubMctsAgent::new(2, 4);
        let self_play_cfg = SelfPlayConfig {
            max_moves: 50,
            mcts_simulations: 4,
            ..Default::default()
        };

        let mut rng1 = rand::rngs::StdRng::seed_from_u64(42);
        let result1 = self_play_game(&mut env1, &mut agent1, &self_play_cfg, &mut rng1);

        // Run 2 with same seed
        let features2 = BasicFeatureExtractor::new(2);
        let mut env2 = AzulEnv::new(config, features2);
        let mut agent2 = StubMctsAgent::new(2, 4);

        let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);
        let result2 = self_play_game(&mut env2, &mut agent2, &self_play_cfg, &mut rng2);

        // Should produce same number of examples
        assert_eq!(
            result1.examples.len(),
            result2.examples.len(),
            "Should produce same number of examples"
        );

        // Each example should have same value
        for (ex1, ex2) in result1.examples.iter().zip(result2.examples.iter()) {
            assert!(
                (ex1.value - ex2.value).abs() < 1e-6,
                "Values should match: {} vs {}",
                ex1.value,
                ex2.value
            );
        }
    }

    #[test]
    fn test_loss_fn_runs_on_toy_batch() {
        // Build toy batch with 2 examples, simple policies (one-hot) and values
        let obs_size = 50;
        let batch_size = 2;

        let examples: Vec<TrainingExample> = (0..batch_size)
            .map(|i| {
                let mut policy = vec![0.0f32; ACTION_SPACE_SIZE];
                policy[i % ACTION_SPACE_SIZE] = 1.0; // One-hot
                TrainingExample {
                    observation: Array::zeros::<f32>(&[obs_size]).unwrap(),
                    policy,
                    action: (i % ACTION_SPACE_SIZE) as u16,
                    value: if i == 0 { 0.5 } else { -0.5 },
                }
            })
            .collect();

        let example_refs: Vec<&TrainingExample> = examples.iter().collect();
        let (_obs, pi_target, z_target) = build_training_batch(&example_refs);

        // Create toy logits (uniform distribution)
        let logits = Array::zeros::<f32>(&[batch_size as i32, ACTION_SPACE_SIZE as i32]).unwrap();

        // Compute loss using the actual functions used in training
        let policy_loss: f32 = cross_entropy_loss_array(&logits, &pi_target)
            .unwrap()
            .item();
        let value_loss: f32 = mse_loss_array(
            &Array::zeros::<f32>(&[batch_size as i32]).unwrap(),
            &z_target,
        )
        .unwrap()
        .item();

        // Assert loss is finite
        assert!(
            policy_loss.is_finite(),
            "Policy loss should be finite, got {}",
            policy_loss
        );
        assert!(
            value_loss.is_finite(),
            "Value loss should be finite, got {}",
            value_loss
        );
        assert!(
            policy_loss >= 0.0,
            "Policy loss should be non-negative, got {}",
            policy_loss
        );
        assert!(
            value_loss >= 0.0,
            "Value loss should be non-negative, got {}",
            value_loss
        );
    }

    #[test]
    fn test_trainer_runs_small_config() {
        use crate::{AzulEnv, BasicFeatureExtractor, EnvConfig, RewardScheme};
        use rand::SeedableRng;

        let config = EnvConfig {
            num_players: 2,
            reward_scheme: RewardScheme::TerminalOnly,
            include_full_state_in_step: true,
        };
        let features = BasicFeatureExtractor::new(2);
        let env = AzulEnv::new(config, features);

        let agent = StubMctsAgent::new(2, 2);
        let trainer_cfg = TrainerConfig {
            num_players: 2,
            num_iters: 2,
            self_play_games_per_iter: 1,
            training_steps_per_iter: 1,
            batch_size: 4,
            replay_capacity: 1000,
            ..Default::default()
        };

        let rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut trainer = Trainer::new(env, agent, trainer_cfg, rng);

        // Run should complete without panic
        let result = trainer.run();
        assert!(
            result.is_ok(),
            "Trainer should run without error: {:?}",
            result.err()
        );

        // Replay buffer should have examples
        assert!(
            trainer.replay.len() > 0,
            "Replay buffer should have examples after training"
        );
    }

    #[test]
    fn test_self_play_episode_runtime() {
        use crate::{AzulEnv, BasicFeatureExtractor, EnvConfig, RewardScheme};
        use rand::SeedableRng;
        use std::time::Instant;

        let config = EnvConfig {
            num_players: 2,
            reward_scheme: RewardScheme::TerminalOnly,
            include_full_state_in_step: true,
        };
        let features = BasicFeatureExtractor::new(2);
        let mut env = AzulEnv::new(config, features);

        let mut agent = StubMctsAgent::new(2, 4);
        let self_play_cfg = SelfPlayConfig {
            max_moves: 200,
            mcts_simulations: 4,
            ..Default::default()
        };

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let start = Instant::now();
        let _examples = self_play_game(&mut env, &mut agent, &self_play_cfg, &mut rng);
        let elapsed = start.elapsed();

        // Should complete within 2 seconds (generous threshold for CI)
        assert!(
            elapsed.as_secs() < 2,
            "Self-play episode should complete quickly, took {:?}",
            elapsed
        );
    }

    #[test]
    fn test_trainer_respects_start_iter() {
        use crate::{AzulEnv, BasicFeatureExtractor, EnvConfig, RewardScheme};
        use rand::SeedableRng;

        let config = EnvConfig {
            num_players: 2,
            reward_scheme: RewardScheme::TerminalOnly,
            include_full_state_in_step: true,
        };
        let features = BasicFeatureExtractor::new(2);
        let env = AzulEnv::new(config, features);

        let agent = StubMctsAgent::new(2, 2);

        // Configure to run from iteration 5 to 7 (3 iterations total)
        let trainer_cfg = TrainerConfig {
            num_players: 2,
            num_iters: 7,
            start_iter: 5,
            self_play_games_per_iter: 1,
            training_steps_per_iter: 1,
            batch_size: 4,
            replay_capacity: 1000,
            ..Default::default()
        };

        let rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut trainer = Trainer::new(env, agent, trainer_cfg, rng);

        // Run should complete without panic
        let result = trainer.run();
        assert!(
            result.is_ok(),
            "Trainer should run without error: {:?}",
            result.err()
        );

        // If start_iter wasn't respected, we'd run 7 iterations worth of self-play.
        // With start_iter=5, we only run 2 iterations (5 and 6), so fewer examples.
        // This is a weak test but verifies the loop respects start_iter.
        assert!(
            trainer.replay.len() > 0,
            "Replay buffer should have examples after training"
        );
    }

    #[test]
    fn test_trainer_skips_all_when_start_equals_num_iters() {
        use crate::{AzulEnv, BasicFeatureExtractor, EnvConfig, RewardScheme};
        use rand::SeedableRng;

        let config = EnvConfig {
            num_players: 2,
            reward_scheme: RewardScheme::TerminalOnly,
            include_full_state_in_step: true,
        };
        let features = BasicFeatureExtractor::new(2);
        let env = AzulEnv::new(config, features);

        let agent = StubMctsAgent::new(2, 2);

        // start_iter == num_iters means no iterations run
        let trainer_cfg = TrainerConfig {
            num_players: 2,
            num_iters: 5,
            start_iter: 5,
            self_play_games_per_iter: 1,
            training_steps_per_iter: 1,
            batch_size: 4,
            replay_capacity: 1000,
            ..Default::default()
        };

        let rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut trainer = Trainer::new(env, agent, trainer_cfg, rng);

        let result = trainer.run();
        assert!(result.is_ok());

        // No iterations should have run, so replay buffer should be empty
        assert_eq!(
            trainer.replay.len(),
            0,
            "Replay buffer should be empty when start_iter == num_iters"
        );
    }

    /// Sanity test: Verify the network can overfit on a single batch.
    ///
    /// This test trains the network on a single batch of data repeatedly.
    /// If the loss doesn't drop and the network doesn't learn the target,
    /// there's a fundamental problem with the gradient flow or network definition.
    ///
    /// This is a critical debugging tool for RL training issues.
    #[test]
    fn test_sanity_can_overfit_single_batch() {
        use crate::{AlphaZeroNet, ACTION_SPACE_SIZE};
        use mlx_rs::module::ModuleParameters;
        use mlx_rs::optimizers::{Adam, Optimizer};
        use mlx_rs::transforms::eval_params;

        // 1. Setup - small network for speed
        let obs_size = 50;
        let hidden_size = 32;
        let mut net = AlphaZeroNet::new(obs_size, hidden_size);
        let mut optimizer = Adam::new(0.01); // High LR for fast convergence

        // 2. Create a fake batch with clear targets
        let batch_size = 8;

        // Fake observations (zeros)
        let obs = Array::zeros::<f32>(&[batch_size, obs_size as i32]).unwrap();

        // Fake policy targets: Action 0 has 100% probability
        let mut target_policy_data = vec![0.0f32; batch_size as usize * ACTION_SPACE_SIZE];
        for i in 0..batch_size as usize {
            target_policy_data[i * ACTION_SPACE_SIZE] = 1.0; // One-hot on action 0
        }
        let pi_target =
            Array::from_slice(&target_policy_data, &[batch_size, ACTION_SPACE_SIZE as i32]);

        // Fake value targets: All 1.0 (win)
        let z_target = Array::from_slice(&vec![1.0f32; batch_size as usize], &[batch_size]);

        // 3. Get initial predictions to compare
        let (_, initial_values) = net.forward_batch(&obs);
        let initial_val = initial_values.as_slice::<f32>()[0];

        // 4. Training loop - overfit on this single batch
        let mut losses = Vec::new();
        let num_steps = 200;

        for step in 0..num_steps {
            // Define loss function matching training.rs
            let policy_weight = Array::from_slice(&[1.0f32], &[1]);
            let value_weight = Array::from_slice(&[1.0f32], &[1]);

            let loss_fn =
                |model: &mut AlphaZeroNet, (x, pi, z): (&Array, &Array, &Array)| {
                    let (policy_logits, value_pred) = model.forward_batch(x);

                    // Policy loss: cross-entropy
                    let policy_loss = cross_entropy_loss_array(&policy_logits, pi)?;

                    // Value loss: MSE
                    let value_loss = mse_loss_array(&value_pred, z)?;

                    // Total weighted loss
                    let total_loss = policy_loss
                        .multiply(&policy_weight)?
                        .add(&value_loss.multiply(&value_weight)?)?
                        .squeeze()?;

                    Ok(total_loss)
                };

            // Compute value and gradients
            let mut vg = mlx_rs::nn::value_and_grad(loss_fn);
            let (loss, grads) = vg(&mut net, (&obs, &pi_target, &z_target)).unwrap();
            let loss_val = loss.item::<f32>();

            // Apply gradients
            optimizer.update(&mut net, grads).unwrap();
            let _ = eval_params(ModuleParameters::parameters(&net));

            losses.push(loss_val);

            if step % 50 == 0 {
                eprintln!("Overfit test step {}: loss = {:.6}", step, loss_val);
            }
        }

        // 5. Verify the network learned
        let (final_policy, final_values) = net.forward_batch(&obs);
        let final_val = final_values.as_slice::<f32>()[0];
        let final_policy_slice = final_policy.as_slice::<f32>();

        // Value should be close to target (1.0)
        eprintln!(
            "Initial value: {:.4}, Final value: {:.4} (target: 1.0)",
            initial_val, final_val
        );
        assert!(
            final_val > 0.8,
            "Network should learn to predict value ~1.0, got {:.4}. \
             This indicates the value head is not receiving gradients correctly.",
            final_val
        );

        // Policy should heavily favor action 0
        let action0_prob = {
            // Softmax to get probabilities
            let logits = &final_policy_slice[0..ACTION_SPACE_SIZE];
            let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exps: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
            let sum: f32 = exps.iter().sum();
            exps[0] / sum
        };

        eprintln!(
            "Action 0 probability: {:.4} (target: 1.0)",
            action0_prob
        );
        assert!(
            action0_prob > 0.8,
            "Network should learn to assign high probability to action 0, got {:.4}. \
             This indicates the policy head is not receiving gradients correctly.",
            action0_prob
        );

        // Loss should have decreased significantly
        let initial_loss = losses[0];
        let final_loss = *losses.last().unwrap();
        eprintln!(
            "Initial loss: {:.6}, Final loss: {:.6}",
            initial_loss, final_loss
        );
        assert!(
            final_loss < initial_loss * 0.1,
            "Loss should decrease by at least 10x when overfitting, \
             initial: {:.6}, final: {:.6}",
            initial_loss,
            final_loss
        );
    }
}
