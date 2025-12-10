## 0. Scope

**In scope**

- Single-process **self-play generation** using `AlphaZeroMctsAgent` and `AzulEnv`.
- **Replay buffer** of `(observation, policy, value)` training examples.
- **Training loop** using `mlx-rs` to optimize a `PolicyValueNet`:
  - Cross-entropy policy loss vs MCTS search policy.
  - MSE value loss vs final game outcome.

- Basic **checkpointing** and **evaluation hooks**.

**Out of scope (for this spec)**

- Distributed self-play / training.
- Fancy replay strategies (prioritized replay, curriculum).
- Hyperparameter search.
- GUI / UX; just a CLI binary or library API.

---

## 1. New modules and types

### 1.1 Module layout

Target layout (within `crates/rl-env`):

```text
crates/rl-env/src/alphazero/
    mod.rs
    training.rs         // Trainer + training loop
    replay_buffer.rs    // ReplayBuffer implementation
    examples.rs         // TrainingExample / move history helpers
```

Assume `alphazero/mcts.rs` (or similar) already exists per previous spec, exporting:

- `struct AlphaZeroMctsAgent<M: PolicyValueModel> { ... }`
- `trait PolicyValueModel { ... }`
  (This is your mlx-rs-backed neural net wrapper.)
- A way to obtain the **root policy distribution** (visit counts normalized) for the last chosen action.

If that interface differs, the training module will adapt to whatever `alphazero-mcts` exposes.

### 1.2 Training example representation

```rust
// crates/rl-env/src/alphazero/examples.rs

use crate::{Observation, ActionId, ACTION_SPACE_SIZE};
use mlx_rs::Array;

/// One training example: (s, π, z) in AlphaZero notation.
pub struct TrainingExample {
    /// Observation from the acting player's perspective.
    pub observation: Observation,      // shape [obs_size]

    /// MCTS-improved policy π over ACTION_SPACE_SIZE actions.
    /// Stored as a flat f32 vec; converted to Array at batch time.
    pub policy: Vec<f32>,              // len == ACTION_SPACE_SIZE

    /// Final outcome from this player's perspective.
    /// Typically in [-1, 1] or normalized score difference.
    pub value: f32,
}
```

Internal helper while a game is still running:

```rust
/// Move record before we know the final outcome.
pub struct PendingMove {
    pub player: azul_engine::PlayerIdx,
    pub observation: Observation,
    pub policy: Vec<f32>,             // from MCTS visit counts
}
```

---

## 2. Replay buffer

### 2.1 API

```rust
// crates/rl-env/src/alphazero/replay_buffer.rs

use super::TrainingExample;
use rand::Rng;

pub struct ReplayBuffer {
    capacity: usize,
    data: Vec<TrainingExample>,
    write_index: usize,
    is_full: bool,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self { ... }

    pub fn len(&self) -> usize { ... }
    pub fn is_empty(&self) -> bool { ... }

    /// Add a single example (overwrites oldest when full).
    pub fn push(&mut self, example: TrainingExample) { ... }

    /// Add many examples.
    pub fn extend<I: IntoIterator<Item = TrainingExample>>(&mut self, it: I) { ... }

    /// Uniformly sample `batch_size` examples (with replacement).
    pub fn sample<'a>(
        &'a self,
        rng: &mut impl Rng,
        batch_size: usize,
    ) -> Vec<&'a TrainingExample> { ... }
}
```

### 2.2 Implementation details

- Storage is a **ring buffer**:
  - `data` length is `min(total_seen, capacity)`.
  - `write_index` points to where the next element will be written.

- `push`:
  - If `data.len() < capacity`, `data.push(example)` and increment write_index.
  - Else `data[write_index] = example` and move `write_index = (write_index + 1) % capacity`.

- `sample`:
  - `assert!(!self.is_empty())`.
  - For `i in 0..batch_size`: sample `idx = rng.random_range(0..len as u32) as usize;` and push `&self.data[idx]`.

---

## 3. Self-play generation

### 3.1 Requirements

- Use `AzulEnv<F>` and `AlphaZeroMctsAgent<M>` to generate **full games**.
- For each move:
  - Snapshot the **observation** for the acting player.
  - Record the **MCTS-improved policy** π (visit counts normalized).

- At game end:
  - Compute final outcomes `z[player]` for each player.
  - Turn `PendingMove`s into `TrainingExample`s with `value = z[player]`.

### 3.2 Interface to MCTS agent

We assume the MCTS module provides something like:

```rust
/// Output of a MCTS search at the root.
pub struct MctsSearchResult {
    pub action: ActionId,
    /// Visit-count-based policy over all actions, normalized.
    pub policy: Vec<f32>, // len == ACTION_SPACE_SIZE
    // optional: q_values, visit_counts, etc.
}

impl<M: PolicyValueModel> AlphaZeroMctsAgent<M> {
    pub fn select_action_and_policy(
        &mut self,
        input: &crate::AgentInput,
        rng: &mut impl rand::Rng,
    ) -> MctsSearchResult { ... }
}
```

> If `alphazero-mcts` uses a different but equivalent API, adapt self-play to that.

### 3.3 Computing final outcomes

Define a helper:

```rust
fn compute_outcomes_from_scores(
    scores: &[i16],
) -> Vec<f32> {
    // Same semantics as RewardScheme::TerminalOnly:
    // z_i = score_i - mean(score_j), normalized by 100.0.
    let n = scores.len();
    let scores_f: Vec<f32> = scores.iter().map(|s| *s as f32).collect();
    let mean = scores_f.iter().sum::<f32>() / (n as f32);

    scores_f
        .into_iter()
        .map(|s| (s - mean) / 100.0)
        .collect()
}
```

Call this at the end of the game using `env.game_state.players[p].score`.

### 3.4 Self-play for one game

```rust
// crates/rl-env/src/alphazero/training.rs

use crate::{AzulEnv, FeatureExtractor, EnvConfig, Environment, ACTION_SPACE_SIZE};
use crate::{AgentInput, ActionId};
use super::{TrainingExample, PendingMove, ReplayBuffer};
use rand::Rng;

pub struct SelfPlayConfig {
    pub max_moves: usize,                // safety cap
    pub mcts_simulations: usize,         // passed into MCTS agent
    pub dirichlet_alpha: f32,            // for root exploration
    pub dirichlet_eps: f32,
    pub temp_cutoff_move: usize,         // temperature schedule switch
}

pub fn self_play_game<F, M>(
    env: &mut AzulEnv<F>,
    mcts_agent: &mut AlphaZeroMctsAgent<M>,
    self_play_cfg: &SelfPlayConfig,
    rng: &mut impl Rng,
) -> Vec<TrainingExample>
where
    F: FeatureExtractor,
    M: PolicyValueModel,
{
    // 1. Reset env
    let mut step = env.reset(rng);
    let mut moves: Vec<PendingMove> = Vec::new();
    let mut move_idx = 0usize;

    while !step.done && move_idx < self_play_cfg.max_moves {
        let p = step.current_player as usize;

        // 2. Build AgentInput (with full state if available)
        let input = AgentInput {
            observation: &step.observations[p],
            legal_action_mask: &step.legal_action_mask,
            current_player: step.current_player,
            state: step.state.as_ref(),   // assuming AgentInput now has this
        };

        // 3. Select action + policy via MCTS
        let search_result = mcts_agent.select_action_and_policy(&input, rng);

        // 4. Record pending move
        moves.push(PendingMove {
            player: step.current_player,
            observation: step.observations[p].clone(),
            policy: search_result.policy.clone(),
        });

        // 5. Step env
        let next = env.step(search_result.action, rng)
            .expect("env::step should not fail for legal action");
        step = next;
        move_idx += 1;
    }

    // 6. Extract final scores & compute z per player
    let n = env.game_state.num_players as usize;
    let scores: Vec<i16> = (0..n)
        .map(|p| env.game_state.players[p].score)
        .collect();
    let outcomes = compute_outcomes_from_scores(&scores); // Vec<f32> len n

    // 7. Convert PendingMove → TrainingExample
    moves
        .into_iter()
        .map(|m| TrainingExample {
            observation: m.observation,
            policy: m.policy,
            value: outcomes[m.player as usize],
        })
        .collect()
}
```

---

## 4. Training loop with mlx-rs

### 4.1 Trainer configuration

```rust
pub struct TrainerConfig {
    pub num_players: u8,                 // usually 2
    pub replay_capacity: usize,
    pub batch_size: usize,
    pub self_play_games_per_iter: usize,
    pub training_steps_per_iter: usize,
    pub num_iters: usize,

    // Optimizer hyperparams
    pub learning_rate: f32,
    pub weight_decay: f32,

    // Loss weights
    pub value_loss_weight: f32,
    pub policy_loss_weight: f32,

    // MCTS/self-play settings
    pub self_play: SelfPlayConfig,

    // Optional evaluation/checkpointing
    pub checkpoint_dir: Option<std::path::PathBuf>,
    pub eval_interval: usize,            // in iterations
}
```

### 4.2 Trainer state

```rust
pub struct Trainer<F, M>
where
    F: FeatureExtractor,
    M: PolicyValueModel,
{
    pub env: AzulEnv<F>,
    pub mcts_agent: AlphaZeroMctsAgent<M>,
    pub replay: ReplayBuffer,
    pub cfg: TrainerConfig,
    pub rng: rand::rngs::StdRng,

    // mlx-rs training state
    pub optimizer: mlx_rs::optim::Adam,  // or generic optimizer
}
```

> Exact optimizer type can be whatever `mlx-rs` exposes; spec assumes something like `optim::Adam`.

### 4.3 Building mini-batches

Conversion helper:

```rust
fn build_training_batch(
    examples: &[&TrainingExample],
) -> (Array, Array, Array) {
    let batch_size = examples.len();
    let obs_size = examples[0].observation.shape()[0] as usize;

    // obs: [B, obs_size]
    let mut obs_data = Vec::with_capacity(batch_size * obs_size);
    let mut policy_data = Vec::with_capacity(batch_size * ACTION_SPACE_SIZE);
    let mut value_data = Vec::with_capacity(batch_size);

    for ex in examples {
        // observation
        obs_data.extend_from_slice(ex.observation.as_slice::<f32>());

        // policy
        debug_assert_eq!(ex.policy.len(), ACTION_SPACE_SIZE);
        policy_data.extend_from_slice(&ex.policy);

        // value
        value_data.push(ex.value);
    }

    let obs = Array::from_slice::<f32>(&obs_data, &[
        batch_size as i32,
        obs_size as i32,
    ]);
    let pi = Array::from_slice::<f32>(&policy_data, &[
        batch_size as i32,
        ACTION_SPACE_SIZE as i32,
    ]);
    let z = Array::from_slice::<f32>(&value_data, &[batch_size as i32]);

    (obs, pi, z)
}
```

### 4.4 Loss function (conceptual)

We follow the MLX recommendation: **do not close over arrays** in the loss closure. Instead, we pass parameters and data arrays explicitly and call `transforms::grad(loss_fn, argnums)` where `argnums` indicates which inputs are differentiated. ([GitHub][1])

The exact API depends on how `PolicyValueModel` exposes parameters, but conceptually:

```rust
use mlx_rs::{Array, Exception};
use mlx_rs::{ops, transforms};
use crate::alphazero::PolicyValueModel;

fn make_loss_fn<M: PolicyValueModel>(
    model: &M,
    value_loss_weight: f32,
    policy_loss_weight: f32,
) -> impl Fn(&[Array]) -> Result<Array, Exception> + '_ {
    move |inputs: &[Array]| -> Result<Array, Exception> {
        // Suppose PolicyValueModel gives us:
        // - param_count()
        // - forward_with_params(params: &[Array], obs: &Array) -> (logits, value)
        let param_count = model.param_count();

        let (params, rest) = inputs.split_at(param_count);
        let obs = &rest[0];
        let pi_target = &rest[1];   // [B, A]
        let z_target = &rest[2];    // [B]

        let (logits, value_pred) = model.forward_with_params(params, obs)?;

        // Policy loss: cross-entropy between π_target and softmax(logits)
        // logits: [B, A]
        let log_probs = ops::log_softmax(&logits, Some(1))?;  // axis=1 over actions
        // pi_target assumed to sum to 1; loss = -sum π * log p
        let policy_loss = -ops::mean(&ops::sum(&(pi_target * &log_probs)?, Some(&[1]), false)?, None, None)?;

        // Value loss: MSE(z_pred, z_target)
        let value_loss = ops::mean(&ops::square(&(value_pred - z_target)?)?, None, None)?;

        // Total loss
        let total = policy_loss_weight * policy_loss
            + value_loss_weight * value_loss;

        Ok(total)
    }
}
```

Then in the training step:

```rust
fn training_step<M: PolicyValueModel>(
    model: &mut M,
    optimizer: &mut Adam,  // or generic optimizer
    batch: &[&TrainingExample],
    cfg: &TrainerConfig,
) -> Result<f32, Exception> {
    let (obs, pi, z) = build_training_batch(batch);

    let param_arrays = model.parameters(); // Vec<Array>, maybe references

    // Build inputs: [params..., obs, pi, z]
    let mut inputs: Vec<Array> = Vec::new();
    inputs.extend(param_arrays.iter().cloned());
    inputs.push(obs);
    inputs.push(pi);
    inputs.push(z);

    let loss_fn = make_loss_fn(model, cfg.value_loss_weight, cfg.policy_loss_weight);

    // We differentiate w.r.t. args 0..param_count
    let argnums: Vec<usize> = (0..param_arrays.len()).collect();
    let grad_fn = transforms::grad(loss_fn, &argnums);

    let grads = grad_fn(&inputs)?; // Vec<Array> of gradients for each param

    // Apply optimizer update to model parameters
    model.apply_gradients(optimizer, &grads)?;

    // Because MLX is lazily evaluated, force evaluation of updated params.
    model.eval_parameters()?; // e.g. call .eval() on each Array under the hood

    // Optionally recompute loss for logging
    let loss = loss_fn(&inputs)?.as_slice::<f32>()[0];

    Ok(loss)
}
```

> Note: the exact shape and API (`Adam`, `ops::log_softmax`, etc.) may differ slightly in real mlx-rs; this spec is conceptual but follows the documented pattern: closures over **inputs slice** and **transforms::grad** with explicit `argnums`. ([GitHub][1])

### 4.5 Top-level training loop

```rust
impl<F, M> Trainer<F, M>
where
    F: FeatureExtractor,
    M: PolicyValueModel,
{
    pub fn run(&mut self) -> Result<(), Exception> {
        for iter in 0..self.cfg.num_iters {
            // 1. Self-play
            for _ in 0..self.cfg.self_play_games_per_iter {
                let examples = self_play_game(
                    &mut self.env,
                    &mut self.mcts_agent,
                    &self.cfg.self_play,
                    &mut self.rng,
                );
                self.replay.extend(examples);
            }

            // 2. Training
            for _ in 0..self.cfg.training_steps_per_iter {
                if self.replay.len() < self.cfg.batch_size {
                    break;
                }
                let batch = self.replay.sample(&mut self.rng, self.cfg.batch_size);
                let loss = training_step(
                    &mut self.mcts_agent.model,
                    &mut self.optimizer,
                    &batch,
                    &self.cfg,
                )?;
                // log loss
            }

            // 3. Optional checkpoint
            if let Some(dir) = &self.cfg.checkpoint_dir {
                if iter %  self.cfg.eval_interval == 0 {
                    self.save_checkpoint(dir, iter)?;
                    // optional: run evaluation vs RandomAgent
                }
            }
        }
        Ok(())
    }

    fn save_checkpoint(
        &self,
        dir: &std::path::Path,
        iter: usize,
    ) -> Result<(), std::io::Error> {
        // Implement via PolicyValueModel::save, etc.
        Ok(())
    }
}
```

---

## 5. Entry point / CLI

Add a simple training binary:

```text
src/bin/train_alphazero.rs
```

Sketch:

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse CLI args or config file.

    let cfg = TrainerConfig { ... };
    let env_cfg = EnvConfig {
        num_players: cfg.num_players,
        reward_scheme: RewardScheme::TerminalOnly,
        include_full_state_in_step: true,
    };
    let features = BasicFeatureExtractor::new(cfg.num_players);
    let mut env = AzulEnv::new(env_cfg, features);

    let model = PolicyValueNet::new(/* obs_size, action_space_size, etc. */);
    let mcts_agent = AlphaZeroMctsAgent::new(model, /* mcts config */);
    let replay = ReplayBuffer::new(cfg.replay_capacity);
    let rng = StdRng::seed_from_u64(42);

    let optimizer = make_adam_for_model(&mcts_agent.model, cfg.learning_rate, cfg.weight_decay)?;

    let mut trainer = Trainer {
        env,
        mcts_agent,
        replay,
        cfg,
        rng,
        optimizer,
    };

    trainer.run()?;
    Ok(())
}
```

---

## 6. Test Plan

Focus on **correctness, MLX AD usage, determinism, and integration**.

### 6.1 Unit tests

**File:** `replay_buffer.rs`

1. `test_replay_buffer_push_and_len`
   - Create buffer capacity 3.
   - Push 2 samples → `len() == 2`.
   - Push 3rd sample → `len() == 3`.

2. `test_replay_buffer_overwrite_oldest`
   - Capacity 2; push A, B; verify order.
   - Push C; buffer should contain {B, C}, not A.

3. `test_replay_buffer_sample_returns_valid_refs`
   - Fill buffer; sample 5 times; each sample is one of stored items, never out-of-bounds; handle empty case with assert panic.

**File:** `examples.rs` (or training.rs helpers)

4. `test_build_training_batch_shapes`
   - Create synthetic `TrainingExample`s with known `obs_size` and `ACTION_SPACE_SIZE`.
   - Build batch; assert shapes:
     - `obs.shape() == [B, obs_size]`
     - `pi.shape() == [B, ACTION_SPACE_SIZE]`
     - `z.shape() == [B]`.

**File:** `training.rs`

5. `test_compute_outcomes_from_scores_zero_sum`
   - Provide scores [10, 20].
   - Compute outcomes; check:
     - sum outcomes ≈ 0.
     - Higher-score outcome > 0, lower < 0.

6. `test_compute_outcomes_multi_player`
   - Scores [10, 10, 10]; outcomes all ≈ 0.

### 6.2 Self-play tests

**File:** `training.rs` or `alphazero/self_play_tests.rs`

Use a **stub model** with deterministic behavior (e.g., uniform policy, zero value).

7. `test_self_play_generates_examples`
   - Configure `EnvConfig` with `num_players = 2`.
   - Create `StubPolicyValueModel` that always returns uniform logits and zero value.
   - Create `AlphaZeroMctsAgent<Stub>` with small `mcts_simulations`.
   - Run `self_play_game` with `max_moves` large enough.
   - Assert:
     - Returned `examples.len() > 0`.
     - For each example, `policy.len() == ACTION_SPACE_SIZE`.
     - Each `policy` sums to ~1.
     - `value` within reasonable bounds (e.g., [-2, 2]).

8. `test_self_play_respects_max_moves`
   - Set `max_moves = 5`.
   - Ensure `examples.len() <= 5`.

9. `test_self_play_determinism_with_stub_model`
   - Use fixed `StdRng` seed and stub model.
   - Run `self_play_game` twice.
   - Verify that the **sequence of actions** (or `policy` vectors and `value`s) is identical between runs.

### 6.3 Training step & MLX gradient tests

These ensure we are using mlx-rs AD correctly per docs. ([GitHub][1])

10. `test_loss_fn_runs_on_toy_batch`
    - Build toy batch with 2 examples, simple policies (one-hot) and values.
    - Build a tiny `PolicyValueModel` (e.g., linear layer).
    - Call `training_step` once; assert that:
      - It returns `Ok(loss)`.
      - `loss` is finite (not NaN or inf).

11. `test_training_step_updates_model_parameters`
    - Clone model params before training.
    - Run `training_step` on toy batch.
    - Retrieve params after; assert at least one parameter array differs from before (norm difference > 0).

12. `test_training_step_gradient_shapes`
    - Internal helper: call `transforms::grad` directly on the loss fn with known param shapes; ensure gradient arrays have the same shapes as parameters.

### 6.4 End-to-end integration tests

**File:** `alphazero/integration_tests.rs`

13. `test_trainer_runs_small_config`
    - Configure `TrainerConfig` with:
      - `num_iters = 2`
      - `self_play_games_per_iter = 1`
      - `training_steps_per_iter = 1`
      - Small `replay_capacity`, `batch_size`.

    - Use stub/simple model + small MCTS.
    - Run `trainer.run()`.
    - Assert:
      - No panic.
      - At end, `replay.len() > 0`.
      - Model params changed vs initial.

14. `test_trainer_determinism_with_stub_model`
    - Fix RNG seed and use stub model & MCTS config with no randomness except RNG.
    - Run training loop for 1 iteration twice from same initial state.
    - Confirm:
      - Sequence of self-play game lengths identical.
      - Final model parameters identical (within numerical tolerance).

### 6.5 Performance / sanity checks (optional)

15. `test_self_play_episode_runtime_under_threshold`
    - Run one self-play game with some `mcts_simulations` and assert it completes within a sane time bound (e.g., 1 second) in CI; helps catch pathological recursion/loops.
