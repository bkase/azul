Below is a concrete engineering spec for adding an AlphaZero-style MCTS agent on top of your existing Azul engine + rl-env, using `mlx-rs` for the neural network.

I’ll write it as if it were a new doc, e.g. `history/alphazero-mcts.md`.

---

# AlphaZero-style MCTS + Neural Net for Azul (MLX-RS)

**Status:** Draft
**Target crates:**

* `crates/engine` (read-only; already stable)
* `crates/rl-env` (add MCTS agent + NN)
* top-level `azul` (no changes beyond depending on new API)

---

## 1. Goals and Scope

### 1.1 Goals

1. Add an AlphaZero-style MCTS agent that can play Azul by searching forward using the *exact* engine dynamics (`azul-engine`) and a neural network policy/value function built on `mlx-rs`.
2. Integrate this agent cleanly with the existing RL environment (`azul-rl-env`) and its `Agent` trait, so it can be used in `self_play_episode` and future training loops.
3. Define an MLX-based neural network architecture (`AlphaZeroNet`) that:

   * Takes the existing `Observation` (`mlx_rs::Array` of shape `[obs_size]`),
   * Outputs a policy over the discrete action space (size `ACTION_SPACE_SIZE = 500`),
   * Outputs a scalar value for the current player.
4. Provide a clear testing strategy, including deterministic tests, integration tests with `AzulEnv`, and correctness checks for masking and search behavior.

### 1.2 Non-goals / Deferrals

* Full training pipeline (replay buffer, optimizer loop, checkpoints) is not implemented here; this spec will define the interfaces and outline how training will plug in.
* Support for 3–4 player AlphaZero-style search is deferred; initial implementation focuses on **2-player** games where `RewardScheme::TerminalOnly` is zero-sum.
* Parallel / batched MCTS and network inference is out of scope for v1 (we run simulations sequentially).

---

## 2. Background & Existing Pieces

We already have:

* **Game engine:** `azul-engine`, with:

  * `GameState` (cloneable, fully Markov).
  * `legal_actions(&GameState) -> Vec<Action>`.
  * `apply_action(GameState, Action, &mut impl Rng) -> StepResult` (pure state transition).
  * Determinism & invariants heavily tested.

* **RL env:** `azul-rl-env`, with:

  * `AzulEnv<F: FeatureExtractor>` wrapping `GameState` and handling rewards (`RewardScheme`).
  * Fixed discrete action space with `ActionId` and `ActionEncoder` (size 500).
  * `EnvStep` which *already* has `state: Option<GameState>` when `EnvConfig.include_full_state_in_step == true`.
  * `FeatureExtractor` and `BasicFeatureExtractor` which map `(GameState, PlayerIdx)` → `Observation` (`mlx_rs::Array`).
  * `Agent` trait and `RandomAgent`.

* **MLX-RS usage (already in repo):**

  * We’re already using `mlx_rs::Array::zeros::<f32>(&[obs_size])` and `Array::from_slice` in `feature_extractor.rs`. This matches the MLX-RS API (Array creation functions with generic dtype and shape).

* **Neural net primitives in MLX-RS** (from docs):

  * `mlx_rs::nn` module with:

    * `Linear`, `Relu`, `Tanh`, `LogSoftmax`, `Sequential`, etc. ([Oxide AI][1])
    * `value_and_grad` to get gradients of loss w.r.t. model parameters. ([Oxide AI][1])
  * These are the building blocks we’ll use for `AlphaZeroNet`.

---

## 3. API Design

### 3.1 New Trait: PolicyValueNet

We want MCTS to depend only on a simple “policy+value” interface, not on the concrete MLX network type. Define in `crates/rl-env/src/mcts.rs`:

```rust
use crate::{Observation, ActionId, ACTION_SPACE_SIZE};
use mlx_rs::Array;

/// Policy + value function used by MCTS.
///
/// Shapes (single-example interface):
/// - obs: [obs_size]
/// - policy_logits: [ACTION_SPACE_SIZE]
/// - value: scalar (0-dim or [1])
pub trait PolicyValueNet {
    /// Forward pass for a single observation.
    ///
    /// The implementation may internally reshape to [1, obs_size] etc.,
    /// but the inputs and outputs at this boundary are 1D and scalar.
    fn predict_single(&self, obs: &Observation) -> (Array, f32);

    /// Optional batch interface for training.
    ///
    /// - obs_batch: [batch, obs_size]
    /// - policy_logits: [batch, ACTION_SPACE_SIZE]
    /// - values: [batch]
    fn predict_batch(&self, obs_batch: &Array) -> (Array, Array);
}
```

Notes:

* `predict_single` is used by MCTS for leaf evaluation (no need for batching).
* `predict_batch` is provided for training but can be a thin wrapper that stacks/slices to/from the single version.
* We deliberately leave parameter access (for optimization) to be handled separately (e.g., concrete `AlphaZeroNet` exposing methods for training).

### 3.2 MCTS Config

In `mcts.rs`:

```rust
/// Configuration for AlphaZero-style MCTS.
pub struct MctsConfig {
    /// Number of simulations per root move.
    pub num_simulations: u32,

    /// PUCT exploration constant.
    pub cpuct: f32,

    /// Root Dirichlet noise epsilon (fraction of noise vs prior).
    pub root_dirichlet_eps: f32,

    /// Dirichlet concentration parameter α for root noise.
    /// Only used if > 0.0.
    pub root_dirichlet_alpha: f32,

    /// Softmax temperature for turning visit counts into a policy.
    /// - For training self-play: τ=1.0
    /// - For evaluation: τ→0 (argmax)
    pub temperature: f32,

    /// Maximum search depth in playouts (safety bound).
    pub max_depth: u32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            num_simulations: 256,
            cpuct: 1.5,
            root_dirichlet_eps: 0.25,
            root_dirichlet_alpha: 0.3,
            temperature: 1.0,
            max_depth: 200,
        }
    }
}
```

### 3.3 Extending `AgentInput` to Carry State

**Problem:** MCTS needs full `GameState` to roll out future states, but `AgentInput` currently only has `Observation`, `legal_action_mask`, and `current_player`.

We already *have* the full state in `EnvStep.state: Option<GameState>` when `EnvConfig.include_full_state_in_step == true`, but we don’t pass it through to the `Agent`.

**Solution: Add `state` to `AgentInput`.**

In `crates/rl-env/src/agent.rs`:

```rust
use azul_engine::GameState;

/// Inputs provided to an agent when selecting an action
pub struct AgentInput<'a> {
    pub observation: &'a Observation,
    pub legal_action_mask: &'a [bool],
    pub current_player: PlayerIdx,

    /// Optional full engine state at this decision point.
    /// - For MCTS-based agents, this must be Some(&GameState).
    /// - For simple agents (RandomAgent), it may be None.
    pub state: Option<&'a GameState>,
}
```

Update all call sites:

* In `environment::self_play_episode`:

```rust
let agent_input = AgentInput {
    observation: &step.observations[p],
    legal_action_mask: &step.legal_action_mask,
    current_player: step.current_player,
    state: step.state.as_ref(), // <- new
};
```

* In tests that construct `AgentInput` manually (`agent.rs` tests), pass `state: None` since RandomAgent doesn’t need it.

**Contract for MCTS agent:**

* `AlphaZeroMctsAgent::select_action` will assert that `input.state.is_some()` and that the `GameState` is 2-player. We will document that for MCTS self-play we must construct the env with `EnvConfig { include_full_state_in_step: true, num_players: 2, reward_scheme: RewardScheme::TerminalOnly }`.

### 3.4 MCTS Node Representation

We’ll implement a stateful tree using indices.

```rust
use azul_engine::{GameState, PlayerIdx};
use crate::ActionId;

/// Index into MCTS node arena.
pub type NodeIdx = u32;

/// Edge statistics for an action from a given node.
pub struct ChildEdge {
    pub action_id: ActionId,
    pub prior: f32,         // P(s, a)
    pub visit_count: u32,   // N(s, a)
    pub value_sum: f32,     // W(s, a), sum of backed-up values
    pub child: Option<NodeIdx>, // None until first expansion along this edge
}

/// Node in the MCTS tree.
pub struct Node {
    pub state: GameState,
    pub to_play: PlayerIdx,      // state.current_player at this node
    pub is_terminal: bool,       // state.phase == GameOver

    /// Edges for all legal actions at this node.
    pub children: Vec<ChildEdge>,

    /// Cumulative visit count at this node (sum over children)
    pub visit_count: u32,
}
```

The tree itself:

```rust
pub struct MctsTree {
    pub nodes: Vec<Node>,
}
```

We always store full `GameState` in the node; cloning is cheap enough for our initial implementation, and it keeps the search logic simple and independent from `AzulEnv`.

### 3.5 AlphaZero MCTS Agent

In `mcts.rs`:

```rust
use crate::{Agent, AgentInput, ActionId, ACTION_SPACE_SIZE, FeatureExtractor, Observation};
use azul_engine::{GameState, legal_actions};
use rand::Rng;

pub struct AlphaZeroMctsAgent<F, N>
where
    F: FeatureExtractor,
    N: PolicyValueNet,
{
    pub config: MctsConfig,
    pub features: F,
    pub net: N,
}

impl<F, N> AlphaZeroMctsAgent<F, N>
where
    F: FeatureExtractor,
    N: PolicyValueNet,
{
    pub fn new(config: MctsConfig, features: F, net: N) -> Self {
        Self { config, features, net }
    }

    // main MCTS search entrypoint (called from select_action)
    fn run_search(&mut self, root_state: &GameState, rng: &mut impl Rng) -> [f32; ACTION_SPACE_SIZE] {
        // ... see algorithm section below
    }
}
```

Implement `Agent`:

```rust
impl<F, N> Agent for AlphaZeroMctsAgent<F, N>
where
    F: FeatureExtractor,
    N: PolicyValueNet,
{
    fn select_action(&mut self, input: &AgentInput, rng: &mut impl Rng) -> ActionId {
        let state = input.state.expect("AlphaZeroMctsAgent requires full GameState in AgentInput.state");
        assert_eq!(state.num_players, 2, "AlphaZeroMctsAgent v1 only supports 2 players");

        // 1. Run search to get improved policy over ACTION_SPACE_SIZE
        let pi = self.run_search(state, rng);

        // 2. Mask illegal actions using input.legal_action_mask
        let mut masked_pi = [0.0f32; ACTION_SPACE_SIZE];
        for (id, &prob) in pi.iter().enumerate() {
            if id < input.legal_action_mask.len() && input.legal_action_mask[id] {
                masked_pi[id] = prob;
            } else {
                masked_pi[id] = 0.0;
            }
        }
        // 3. Re-normalize (with epsilon if needed) and sample according to temperature
        let action_id = sample_from_policy(&masked_pi, self.config.temperature, rng);
        action_id as ActionId
    }
}
```

`sample_from_policy` is a helper that:

* For τ≈0: returns argmax (`f32::max_by`).
* For τ>0: applies exponentiation `p_i ∝ count_i^{1/τ}` and samples with `rng`.

---

## 4. MCTS Algorithm Details

### 4.1 Overview

Standard AlphaZero MCTS loop for each root decision:

1. Build root node from current `GameState` `s0`.

2. Repeat `num_simulations` times:

   * **Selection:** Traverse the tree from root by choosing actions that maximize PUCT score until reaching a leaf node (unexpanded or terminal).
   * **Expansion & Evaluation:**

     * If terminal: `leaf_value` is known from game result if we search to terminal, otherwise from a heuristic (we’ll normally stop at non-terminal).
     * Else:

       * Use `FeatureExtractor` to encode `(state_leaf, to_play_leaf)` into `Observation`.
       * Use `PolicyValueNet::predict_single` to get `policy_logits` and `value`.
       * Mask illegal actions using `legal_actions(state_leaf)` + `ActionEncoder`.
       * Convert logits to normalized priors `P(s_leaf, a)` (via softmax over legal actions).
       * Add all legal actions as `ChildEdge`s in the leaf node with corresponding priors.
       * `leaf_value = value_output` (predicted by net).
   * **Backup:** Propagate `leaf_value` back along the path, updating edge statistics `(N, W)` while flipping sign at each step for 2-player zero-sum.

3. After simulations, use root visit counts `N(s0, a)` to produce improved policy π(a|s0), and sample / argmax.

### 4.2 Selection: PUCT

For a node with visit count (N = \sum_a N(s,a)) and edges (a \in \mathcal{A}(s)), define:

* ( Q(s,a) = \begin{cases}
  \frac{W(s,a)}{N(s,a)} & N(s,a) > 0 \
  0                     & N(s,a) = 0
  \end{cases} )

* ( U(s,a) = c_{puct} \cdot P(s,a) \cdot \frac{\sqrt{N + 1}}{1 + N(s,a)} )

We choose:

```rust
score = q + u;
```

Algorithm (within `run_search`):

```rust
fn select_child(&self, node: &Node) -> usize {
    let mut best_idx = 0;
    let mut best_score = f32::NEG_INFINITY;

    let parent_n = node.visit_count.max(1) as f32; // avoid sqrt(0)

    for (i, edge) in node.children.iter().enumerate() {
        let q = if edge.visit_count > 0 {
            edge.value_sum / edge.visit_count as f32
        } else {
            0.0
        };

        let u = self.config.cpuct * edge.prior * (parent_n.sqrt() / (1.0 + edge.visit_count as f32));

        let score = q + u;
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    best_idx
}
```

We store the path as a vector of `(node_idx, chosen_child_idx)` pairs for backup.

### 4.3 Expansion & Evaluation

At the leaf node:

1. Check terminal:

   * If `node.is_terminal` (i.e., `state.phase == GameOver`), we compute `leaf_value` as:

     * For now: 0.0 for non-terminal in partial search; but we normally won’t stop at non-terminal nodes due to `max_depth`.
     * For a true terminal, we compute the **TerminalOnly** reward from engine scores, from the perspective of `node.to_play` (same convention as env).

   However, for v1 we’ll rely on **network value only**, and treat terminal detection only to avoid expanding children (i.e., still use net’s value or 0.0 if we wish; we can refine later).

2. Non-terminal leaf:

   * Compute observation:

     ```rust
     let obs = self.features.encode(&node.state, node.to_play);
     ```

   * Evaluate network:

     ```rust
     let (policy_logits_array, value) = self.net.predict_single(&obs);
     let policy_logits: &[f32] = policy_logits_array.as_slice::<f32>();
     assert_eq!(policy_logits.len(), ACTION_SPACE_SIZE);
     ```

   * Get legal actions from engine:

     ```rust
     let legal_actions = azul_engine::legal_actions(&node.state);
     ```

   * Build a local vector of `(ActionId, logit)` for legal actions:

     ```rust
     let mut legal_ids_and_logits = Vec::with_capacity(legal_actions.len());
     for action in &legal_actions {
         let id = crate::ActionEncoder::encode(action);
         let logit = policy_logits[id as usize];
         legal_ids_and_logits.push((id, logit));
     }
     ```

   * Softmax over **legal** logits to produce priors:

     ```rust
     fn softmax(logits: &[(ActionId, f32)]) -> Vec<(ActionId, f32)> {
         let max_logit = logits.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
         let mut exps = Vec::with_capacity(logits.len());
         let mut sum = 0.0;
         for (id, l) in logits {
             let e = (l - max_logit).exp(); // numerical stability
             exps.push((*id, e));
             sum += e;
         }
         let sum = sum.max(1e-8);
         exps.into_iter().map(|(id, e)| (id, e / sum)).collect()
     }
     ```

   * `node.children = priors.iter().map(|(id, p)| ChildEdge { action_id: *id, prior: *p, visit_count: 0, value_sum: 0.0, child: None }).collect();`

   * Set `node.visit_count` unchanged (visit count is updated only in backup).

   * Set `leaf_value = value` (network prediction).

### 4.4 Root Dirichlet Noise

To get proper exploration in self-play, we add Dirichlet noise to root priors **once per root**:

* Sample a Dirichlet vector ( \eta \sim \mathrm{Dir}(\alpha) ) over root edges.
* For each edge (a):

  ( P'(s_0, a) = (1 - \varepsilon) P(s_0, a) + \varepsilon \eta_a )

We will:

* Add `rand_distr = "0.4"` (or matching `rand` minor version) to `azul-rl-env` `Cargo.toml`.
* Use `rand_distr::Dirichlet` with dimension = number of legal actions at root.

If `root_dirichlet_alpha <= 0.0`, skip noise entirely.

### 4.5 Backup

We maintain a path stack:

```rust
struct PathStep {
    node_idx: NodeIdx,
    child_idx: usize,
    to_play: PlayerIdx, // player who was about to act at this node
}
```

During selection, we push each `(node_idx, chosen_child_idx, node.to_play)`.

After obtaining `leaf_value` (from perspective of the leaf node’s `to_play`), we propagate back to the root, flipping sign at each ply for 2-player zero-sum:

```rust
fn backup(&mut self, path: &[PathStep], leaf_value_from_leaf_perspective: f32) {
    // sign = +1 if leaf_to_play == root_to_play, -1 otherwise,
    // but easier to just flip per step in reverse order.
    let mut value = leaf_value_from_leaf_perspective;

    for step in path.iter().rev() {
        let node = &mut self.tree.nodes[step.node_idx as usize];
        let edge = &mut node.children[step.child_idx];

        // Update edge stats
        edge.visit_count += 1;
        edge.value_sum += value;

        // Update node visit count
        node.visit_count += 1;

        // Flip value for parent perspective (2-player zero-sum)
        value = -value;
    }
}
```

Assumptions:

* `value` is always interpreted as the expected TerminalOnly reward for the player about to move in the **child** node. The sign flip makes it consistent w.r.t. the previous node’s player.

For now we explicitly **assert** `num_players == 2` in `select_action`.

### 4.6 From Visit Counts to Policy π

After running `num_simulations`, we read root node children:

```rust
let root = &self.tree.nodes[root_idx as usize];
let mut counts = [0.0f32; ACTION_SPACE_SIZE];
for edge in &root.children {
    counts[edge.action_id as usize] = edge.visit_count as f32;
}

// temperature τ
let pi = apply_temperature(&counts, self.config.temperature);
```

Temperature logic:

* If `τ <= 1e-6`: argmax — set `pi[a*] = 1`, others 0.
* Else: `π(a) ∝ N(a)^{1/τ}` (raise counts to power `1/τ` and re-normalize).

This `pi` array is used both for:

* Sampling the actual move in `select_action`.
* Saving as a target policy for training (e.g., if we log `(obs, pi, z)` into replay).

---

## 5. Neural Network Design (MLX-RS)

### 5.1 AlphaZeroNet Architecture

We implement a simple fully-connected architecture using `mlx_rs::nn` primitives. From docs: `nn` has `Linear`, `Relu`, `Tanh`, `LogSoftmax`, and `Sequential`, as well as `value_and_grad` for gradients. ([Oxide AI][1])

**Shape convention:**

* `Observation` currently is an `Array` of shape `[obs_size]`.
* `nn::Linear` works on arrays where the last dimension is `in_features`. For a vector `[obs_size]`, it should interpret that as a single example with dimension `obs_size`. (In training we’ll often use `[batch, obs_size]`).
* Policy head outputs logits of shape `[ACTION_SPACE_SIZE]`.
* Value head outputs scalar (0-dim) or `[1]`; we’ll convert to `f32`.

**Struct:**

In `crates/rl-env/src/alphazero_net.rs` (new file):

```rust
use mlx_rs::{Array};
use mlx_rs::nn::{self, Sequential};
use crate::{Observation, ACTION_SPACE_SIZE};

pub struct AlphaZeroNet {
    pub obs_size: usize,
    pub hidden_size: usize,
    pub trunk: Sequential,
    pub policy_head: Sequential,
    pub value_head: Sequential,
}
```

**Architecture:**

* Trunk:

  * Input: `[*, obs_size]`
  * Layers: `Linear(obs_size, hidden_size) -> Relu -> Linear(hidden_size, hidden_size) -> Relu`
  * Output: `[*, hidden_size]`

* Policy head:

  * Input: `[*, hidden_size]`
  * Layers: `Linear(hidden_size, hidden_size) -> Relu -> Linear(hidden_size, ACTION_SPACE_SIZE)`
  * (We will *not* apply softmax here; we expose raw logits for MCTS. For training, we’ll use `nn::LogSoftmax` or `mlx_rs::nn::log_softmax` in loss.)

* Value head:

  * Input: `[*, hidden_size]`
  * Layers: `Linear(hidden_size, hidden_size) -> Relu -> Linear(hidden_size, 1) -> Tanh`
    (Tanh to clamp value to [-1, 1])

**MLX-RS usage:**

* Model creation will use Mlx-rs `nn` constructors, following doc pattern:

  * `let linear = nn::Linear::new(in_features, out_features);`
  * `let relu = nn::Relu {};` (or `nn::Relu::new()` depending on API)
  * Compose with `nn::Sequential` which accepts items implementing `SequentialModuleItem` (per docs). ([Oxide AI][1])

* Forward for single observation:

  ```rust
  impl AlphaZeroNet {
      pub fn new(obs_size: usize, hidden_size: usize) -> Self {
          // instantiate trunk, policy_head, value_head
          // using nn::Linear, nn::Relu, nn::Tanh, nn::Sequential
      }

      /// Forward pass for a batched input.
      /// `obs_batch` shape: [B, obs_size]
      pub fn forward_batch(&self, obs_batch: &Array) -> (Array, Array) {
          let h = self.trunk.call(obs_batch);
          let policy_logits = self.policy_head.call(&h);
          let value = self.value_head.call(&h);
          (policy_logits, value)
      }

      /// Forward pass for a single observation [obs_size].
      pub fn forward_single(&self, obs: &Observation) -> (Array, f32) {
          // reshape to [1, obs_size]
          let obs_batch = obs.reshape(&[1, self.obs_size as i32]);
          let (policy_logits_batch, value_batch) = self.forward_batch(&obs_batch);

          // squeeze [1, ACTION_SPACE_SIZE] -> [ACTION_SPACE_SIZE]
          let policy_logits = policy_logits_batch.squeeze(0); // assume API matches Array::squeeze
          let value_array = value_batch.squeeze(0);           // [1] -> []
          let value_scalar = value_array.as_slice::<f32>()[0];

          (policy_logits, value_scalar)
      }
  }

  impl PolicyValueNet for AlphaZeroNet {
      fn predict_single(&self, obs: &Observation) -> (Array, f32) {
          self.forward_single(obs)
      }

      fn predict_batch(&self, obs_batch: &Array) -> (Array, Array) {
          self.forward_batch(obs_batch)
      }
  }
  ```

We’ll need to adapt the exact method names (`call`, `forward`, `reshape`, `squeeze`) to the MLX-RS API when implementing, but the **shapes and usage of `Array`/`nn` are aligned with the docs**.

### 5.2 Training Hook (Outline Only)

We won’t implement full training, but define a hook so future code can use `mlx_rs::nn::value_and_grad`:

```rust
pub struct TrainingExample {
    pub observation: Observation,        // [obs_size]
    pub policy_target: [f32; ACTION_SPACE_SIZE],
    pub value_target: f32,
}

pub struct Batch {
    pub observations: Array,            // [B, obs_size]
    pub policy_targets: Array,          // [B, ACTION_SPACE_SIZE]
    pub value_targets: Array,           // [B]
}
```

Training will:

1. Stack examples into a `Batch` (using `Array::from_slice` with appropriate shapes).

2. Call `nn::value_and_grad` on a closure that:

   * Runs `AlphaZeroNet::forward_batch`,
   * Computes:

     * Policy loss: cross-entropy between softmax(policy_logits) and `policy_targets`.
     * Value loss: MSE between `value` and `value_targets`.
     * L2 regularization.

3. Apply optimizer from `mlx_rs::optim` (e.g. Adam) to update `AlphaZeroNet` parameters.

This keeps training orthogonal to MCTS and RL env.

---

## 6. Integration with RL Environment

### 6.1 Usage in Self-Play

To run self-play with MCTS:

1. Construct env:

   ```rust
   use azul_rl_env::{AzulEnv, EnvConfig, BasicFeatureExtractor, RewardScheme};

   let config = EnvConfig {
       num_players: 2,
       reward_scheme: RewardScheme::TerminalOnly,
       include_full_state_in_step: true, // REQUIRED for MCTS
   };
   let features = BasicFeatureExtractor::new(2);
   let mut env = AzulEnv::new(config, features.clone());
   ```

2. Construct network and agent:

   ```rust
   use azul_rl_env::{AlphaZeroNet, AlphaZeroMctsAgent, MctsConfig};

   let obs_size = features.obs_size();
   let net = AlphaZeroNet::new(obs_size, 256);
   let mcts_config = MctsConfig::default();
   let mut agent = AlphaZeroMctsAgent::new(mcts_config, features, net);
   ```

3. Use `self_play_episode` as-is; it will now pass `state` to the agent’s `AgentInput`, and `AlphaZeroMctsAgent` will perform MCTS internally.

4. To log training data, update `self_play_episode` or write a variant that also stores `pi` from `run_search` and final reward `z` to a replay buffer.

### 6.2 Keeping RandomAgent Working

* `RandomAgent` ignores `AgentInput.state`; all its tests remain valid.
* We must update its tests to construct `AgentInput` with `state: step.state.as_ref()` — or simply `state: None` in isolated unit tests; both should compile and run.

### 6.3 Legal Action Mask Cross-Check

* `AzulEnv.build_legal_action_mask()` uses engine’s `legal_actions` and `ActionEncoder`, guaranteeing that env’s mask is consistent with engine.
* MCTS uses `legal_actions(&GameState)` directly; this is the same function, so the set of actions MCTS sees will match the mask passed in `AgentInput.legal_action_mask`.

---

## 7. Tests

We’ll add tests in `crates/rl-env/src/mcts_tests.rs` (or inside `mcts.rs` under `#[cfg(test)]`).

### 7.1 Unit Tests: Network

1. **Shape sanity check**

   ```rust
   #[test]
   fn test_alphazero_net_shapes() {
       let obs_size = 1234; // pretend number
       let net = AlphaZeroNet::new(obs_size, 256);

       let obs = mlx_rs::Array::zeros::<f32>(&[obs_size as i32]).unwrap();
       let (policy_logits, value) = net.predict_single(&obs);

       assert_eq!(policy_logits.shape(), &[ACTION_SPACE_SIZE as i32]);
       assert!(value >= -1.0 && value <= 1.0);
   }
   ```

2. **Determinism**

   ```rust
   #[test]
   fn test_alphazero_net_determinism() {
       let obs_size = 100;
       let net = AlphaZeroNet::new(obs_size, 64);
       let obs = mlx_rs::Array::zeros::<f32>(&[obs_size as i32]).unwrap();

       let (p1, v1) = net.predict_single(&obs);
       let (p2, v2) = net.predict_single(&obs);

       assert_eq!(p1.as_slice::<f32>(), p2.as_slice::<f32>());
       assert_eq!(v1, v2);
   }
   ```

### 7.2 Unit Tests: MCTS Core (with Dummy Net)

Create a **dummy** `PolicyValueNet` implementation that returns fixed priors and values:

```rust
struct DummyNet {
    priors: [f32; ACTION_SPACE_SIZE],
    value: f32,
}

impl PolicyValueNet for DummyNet {
    fn predict_single(&self, _obs: &Observation) -> (Array, f32) {
        let arr = Array::from_slice(&self.priors, &[ACTION_SPACE_SIZE as i32]);
        (arr, self.value)
    }

    fn predict_batch(&self, _obs_batch: &Array) -> (Array, Array) {
        unimplemented!()
    }
}
```

Tests:

1. **Root priors respected with 1 simulation**

   * Set `num_simulations = 1`, `cpuct` arbitrary.
   * Use `DummyNet` with priors highly peaked on a known legal action.
   * After `run_search`, check that the resulting π has its maximum at that action.

2. **Masking respected**

   * In a synthetic state, manually create `AgentInput` with some actions illegal (`legal_action_mask[id] = false`).
   * `DummyNet` uniform priors.
   * After selection, ensure agent never returns an illegal action.

3. **Search improves winning move**

   * Construct a toy Azul `GameState` where:

     * There is an immediate winning move (e.g., one action leads to `Phase::GameOver` with strongly positive final score).
   * Use `DummyNet` with uniform priors and 0 value.
   * In MCTS, we let terminal detection compute leaf value from final scores (for that specific test we can choose to override).
   * After many simulations, confirm that:

     * The winning action’s visit count is strictly greater than all others.
     * `AlphaZeroMctsAgent::select_action` returns that action.

### 7.3 Integration Tests: Agent + Env

1. **Agent selects legal actions**

   ```rust
   #[test]
   fn test_mcts_agent_selects_legal_action() {
       use crate::{AzulEnv, EnvConfig, BasicFeatureExtractor, Environment};
       use rand::rngs::StdRng;
       use rand::SeedableRng;

       let config = EnvConfig {
           num_players: 2,
           reward_scheme: RewardScheme::TerminalOnly,
           include_full_state_in_step: true,
       };
       let features = BasicFeatureExtractor::new(2);
       let mut env = AzulEnv::new(config, features.clone());

       let obs_size = features.obs_size();
       let dummy_net = DummyNet {
           priors: [1.0 / ACTION_SPACE_SIZE as f32; ACTION_SPACE_SIZE],
           value: 0.0,
       };
       let mcts_config = MctsConfig { num_simulations: 16, ..MctsConfig::default() };
       let mut agent = AlphaZeroMctsAgent::new(mcts_config, features, dummy_net);

       let mut rng = StdRng::seed_from_u64(42);
       let step = env.reset(&mut rng);

       let input = AgentInput {
           observation: &step.observations[step.current_player as usize],
           legal_action_mask: &step.legal_action_mask,
           current_player: step.current_player,
           state: step.state.as_ref(),
       };

       let action_id = agent.select_action(&input, &mut rng);
       assert!(step.legal_action_mask[action_id as usize]);
   }
   ```

2. **Full game smoke test**

   * Similar to existing integration test with `RandomAgent`, but using `AlphaZeroMctsAgent`.
   * Use small `num_simulations` (e.g., 16) to keep it fast.
   * Loop until `env.step` returns `done == true`.
   * At the end, check:

     * `env.game_state.phase == GameOver`.
     * `assert_tile_invariants(&env.game_state)` (reuse engine invariant checker via `azul_engine::assert_tile_invariants` in a `cfg(test)` context).

3. **Determinism with fixed seeds**

   * As with existing environment determinism test, but:

     * Fix env RNG seed.
     * Fix a separate RNG seed for MCTS agent.
     * Run one full episode, recording action sequence.
     * Re-run with same seeds and confirmed identical action sequence and final scores.

### 7.4 Tests for AgentInput Wiring

* Update `RandomAgent` tests to pass `state: step.state.as_ref()` (or `None`) when creating `AgentInput`.
* Add a small test that ensures `AgentInput.state` is `Some` when `EnvConfig.include_full_state_in_step == true`.

---

## 8. Implementation Plan

1. **Extend AgentInput**

   * Add `state: Option<&GameState>`.
   * Update `self_play_episode` and tests.

2. **Add `mcts.rs` module**

   * Implement `MctsConfig`, `ChildEdge`, `Node`, `MctsTree`.
   * Implement `AlphaZeroMctsAgent` with `run_search`, selection, expansion, backup, root policy extraction, and `Agent` impl.
   * Implement helpers:

     * `softmax` over legal logits.
     * `apply_temperature`.
     * `sample_from_policy`.
     * root Dirichlet noise using `rand_distr::Dirichlet`.

3. **Add `alphazero_net.rs`**

   * Implement `AlphaZeroNet` with MLX-RS `nn` primitives:

     * `Linear`, `Relu`, `Tanh`, `Sequential`.
     * Use `Array::reshape` and `Array::squeeze` to go between `[obs_size]` and `[1, obs_size]`.
   * Implement `PolicyValueNet` for `AlphaZeroNet`.

4. **Wire up re-exports**

   * In `crates/rl-env/src/lib.rs`:

     ```rust
     mod mcts;
     mod alphazero_net;

     pub use mcts::{AlphaZeroMctsAgent, MctsConfig, PolicyValueNet};
     pub use alphazero_net::AlphaZeroNet;
     ```

5. **Add tests**

   * Unit tests for network and MCTS as described.
   * Integration tests with `AzulEnv`.

6. **Run test suite**

   * `cargo test -p azul-rl-env`
   * `cargo test -p azul` for workspace-level check.

---

If you’d like, the next step can be an implementation sketch of `AlphaZeroMctsAgent::run_search` and `AlphaZeroNet::new` with concrete MLX-RS calls, but the spec above should be enough to start coding without surprises.

[1]: https://oxideai.github.io/mlx-rs/mlx_rs/nn/index.html "mlx_rs::nn - Rust"
