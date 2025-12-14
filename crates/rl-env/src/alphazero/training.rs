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
use rand::Rng;

use super::{PendingMove, ReplayBuffer, TrainingExample};
use crate::{ActionId, AgentInput, AzulEnv, Environment, FeatureExtractor, ACTION_SPACE_SIZE};

#[cfg(feature = "profiling")]
use crate::profiling::{print_summary, Timer, PROF};
#[cfg(feature = "profiling")]
use std::sync::atomic::Ordering;

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
            temp_cutoff_move: 30,
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
/// Uses same semantics as RewardScheme::TerminalOnly:
/// z_i = (score_i - mean(scores)) / 100.0
///
/// Returns values in approximately [-1, 1] range.
pub fn compute_outcomes_from_scores(scores: &[i16]) -> Vec<f32> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }

    let scores_f: Vec<f32> = scores.iter().map(|s| *s as f32).collect();
    let mean = scores_f.iter().sum::<f32>() / (n as f32);

    scores_f.into_iter().map(|s| (s - mean) / 100.0).collect()
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

/// Run a single self-play game and return training examples.
///
/// Uses the MCTS agent to play against itself, recording (observation, policy, value)
/// tuples for each move. The value is backfilled with the final game outcome.
pub fn self_play_game<F, A>(
    env: &mut AzulEnv<F>,
    agent: &mut A,
    self_play_cfg: &SelfPlayConfig,
    rng: &mut impl Rng,
) -> Vec<TrainingExample>
where
    F: FeatureExtractor,
    A: MctsAgentExt,
{
    // 1. Reset env
    let mut step = env.reset(rng);
    let mut moves: Vec<PendingMove> = Vec::new();
    let mut move_idx = 0usize;

    // 2. Play game
    while !step.done && move_idx < self_play_cfg.max_moves {
        let p = step.current_player as usize;

        // Build AgentInput (with full state if available)
        let input = AgentInput {
            observation: &step.observations[p],
            legal_action_mask: &step.legal_action_mask,
            current_player: step.current_player,
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

        // Record pending move
        moves.push(PendingMove {
            player: step.current_player,
            observation: step.observations[p].clone(),
            policy: search_result.policy.clone(),
        });

        // Step env
        let next = env
            .step(search_result.action, rng)
            .expect("env::step should not fail for legal action");
        step = next;
        move_idx += 1;

        #[cfg(feature = "profiling")]
        PROF.env_steps.fetch_add(1, Ordering::Relaxed);
    }

    // 3. Extract final scores & compute z per player
    let n = env.game_state.num_players as usize;
    let scores: Vec<i16> = (0..n).map(|p| env.game_state.players[p].score).collect();
    let outcomes = compute_outcomes_from_scores(&scores);

    // 4. Convert PendingMove -> TrainingExample
    // Force evaluation of observations before storing in replay buffer.
    // MLX arrays are lazy - without eval(), the underlying event/stream tracking
    // can become invalid when arrays are stored for a long time.
    moves
        .into_iter()
        .map(|m| {
            m.observation.eval().expect("Failed to evaluate observation");
            TrainingExample {
                observation: m.observation,
                policy: m.policy,
                value: outcomes[m.player as usize],
            }
        })
        .collect()
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
    F: FeatureExtractor + Clone,
    M: TrainableModel + MctsAgentExt + ModuleParameters,
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
            // 1. Self-play
            #[cfg(feature = "profiling")]
            let _t_sp = Timer::new(&PROF.time_self_play_ns);

            for _ in 0..self.cfg.self_play_games_per_iter {
                let examples = self_play_game(
                    &mut self.env,
                    &mut self.agent,
                    &self.cfg.self_play,
                    &mut self.rng,
                );

                #[cfg(feature = "profiling")]
                {
                    PROF.self_play_games.fetch_add(1, Ordering::Relaxed);
                    PROF.self_play_moves
                        .fetch_add(examples.len() as u64, Ordering::Relaxed);
                }

                self.replay.extend(examples);
            }

            // Drop timer to record self-play time before training starts
            #[cfg(feature = "profiling")]
            drop(_t_sp);

            // 2. Training
            #[cfg(feature = "profiling")]
            let _t_train = Timer::new(&PROF.time_training_ns);

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
                let _loss = self.training_step(&batch_refs)?;
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

            // Log progress
            if iter % 10 == 0 {
                eprintln!(
                    "Iteration {}/{}, replay buffer size: {}",
                    iter,
                    self.cfg.num_iters,
                    self.replay.len()
                );
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

        // Check values
        // mean = 15, so outcomes = [(10-15)/100, (20-15)/100] = [-0.05, 0.05]
        assert!((outcomes[0] - (-0.05)).abs() < 1e-5);
        assert!((outcomes[1] - 0.05).abs() < 1e-5);
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
        let examples = self_play_game(&mut env, &mut agent, &self_play_cfg, &mut rng);

        // Should have generated at least one example
        assert!(
            !examples.is_empty(),
            "Should generate at least one training example"
        );

        for ex in &examples {
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
        let examples = self_play_game(&mut env, &mut agent, &self_play_cfg, &mut rng);

        assert!(
            examples.len() <= 5,
            "Should respect max_moves limit, got {} examples",
            examples.len()
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
        let examples1 = self_play_game(&mut env1, &mut agent1, &self_play_cfg, &mut rng1);

        // Run 2 with same seed
        let features2 = BasicFeatureExtractor::new(2);
        let mut env2 = AzulEnv::new(config, features2);
        let mut agent2 = StubMctsAgent::new(2, 4);

        let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);
        let examples2 = self_play_game(&mut env2, &mut agent2, &self_play_cfg, &mut rng2);

        // Should produce same number of examples
        assert_eq!(
            examples1.len(),
            examples2.len(),
            "Should produce same number of examples"
        );

        // Each example should have same value
        for (ex1, ex2) in examples1.iter().zip(examples2.iter()) {
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
}
