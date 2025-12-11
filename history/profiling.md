# Profiling & Performance Engineering Plan for `azul` AlphaZero Training

## 1. Goals and Scope

**Goal:** Make the `azul` training binary (`src/main.rs`) tractable to run and iterate on by:

1. Setting up **robust, repeatable profiling infrastructure** at several levels:
   - End‑to‑end wall‑clock metrics
   - In‑process timing + counters (domain-aware metrics)
   - Micro‑benchmarks targeting suspected hotspots
   - Integration with external profilers (perf/Flamegraph/Instruments)

2. Using that infrastructure to:
   - Isolate major bottlenecks (self‑play/MCTS vs training vs engine).
   - Quantify how hyperparameters (`num-iters`, `games-per-iter`, `mcts-sims`, `batch-size`, etc.) affect runtime.
   - Provide a baseline to evaluate future optimizations.

3. Call out **immediate low‑hanging performance wins** visible from the current code.

**Non-goals (for this spec):**

- Designing a “final” MLX autodiff training pipeline.
- Deep engine‑level optimization inside `azul-engine` (we’ll treat it as a black box for now).
- Algorithmic changes to the RL setup (e.g., different search algorithm).

---

## 2. Quick architecture + perf hypotheses

### 2.1 Rough pipeline

The training binary (`src/main.rs`) wires together:

- **Environment:** `AzulEnv<F>` (`crates/rl-env/src/environment.rs`)
- **MCTS agent:** `AlphaZeroMctsAgent<F, N>` (`crates/rl-env/src/mcts.rs`)
- **Neural net:** `AlphaZeroNet` (`crates/rl-env/src/alphazero_net.rs`)
- **Training loop:** `Trainer` (`crates/rl-env/src/alphazero/training.rs`)
- **CLI wiring:** `TrainableMctsAgent` implements `TrainableModel + MctsAgentExt` (`src/main.rs`)

`Trainer::run`:

```rust
for iter in 0..self.cfg.num_iters {
    // 1. Self-play: MCTS games
    for _ in 0..self.cfg.self_play_games_per_iter {
        let examples = self_play_game(...);
        self.replay.extend(examples);
    }

    // 2. Training: compute gradients and update model
    for _ in 0..self.cfg.training_steps_per_iter {
        if self.replay.len() < self.cfg.batch_size { break; }
        let batch = self.replay.sample(...).cloned().collect();
        let batch_refs = batch.iter().collect::<Vec<_>>();
        let _loss = self.training_step(&batch_refs)?;
    }

    // 3. Optional checkpoint
}
```

### 2.2 Suspected bottlenecks (from reading the code)

These are hypotheses to validate with profiling:

1. **Finite-difference gradients are likely dominating training time**

   In `crates/rl-env/src/alphazero/training.rs`:

   ```rust
   // training_step(...)
   let params = self.agent.parameters();
   let grads = compute_gradients_fd(&self.agent, &obs, &pi_target, &z_target, &params);
   self.agent.apply_gradients(self.cfg.learning_rate, &grads);
   ```

   `compute_gradients_fd` does central finite differences across parameters:

   ```rust
   let eps = 1e-4;
   let stride = (n / 100).max(1);      // sample up to ≈ 100 params per array
   for i in (0..n).step_by(stride) {
       // +eps
       let (logits_plus, values_plus) = model.forward_with_params(&perturbed_params, obs);
       // -eps
       let (logits_minus, values_minus) = model.forward_with_params(&perturbed_params, obs);

       grad_data[i] = (loss_plus - loss_minus) / (2.0 * eps);
   }
   ```

   For the current `TrainableMctsAgent`:
   - ~12 parameter arrays.
   - Effective ~50–150 sampled params per array.
   - **On the order of ~2,400 forward passes per training step.**

   With `--training-steps 50` and `--num-iters 10`, that’s roughly **120k forward passes** just for FD gradients, _in addition_ to MCTS and self‑play.

   Worse, `TrainableMctsAgent::apply_gradients` is a **no-op**, and `forward_with_params` ignores provided params and just creates a fresh `AlphaZeroNet` each time. So this entire FD gradient computation is **wasted work** right now (no learning actually happens).

2. **MCTS search + engine cloning**

   `AlphaZeroMctsAgent::run_search` (`crates/rl-env/src/mcts.rs`):
   - Builds a fresh `MctsTree` per move.
   - For each simulation:
     - Traverses the tree, decoding actions (`ActionEncoder::decode`) and applying them:

       ```rust
       let parent_state = tree.nodes[current_idx as usize].state.clone();
       let action = ActionEncoder::decode(edge.action_id);
       let step_result = apply_action(parent_state, action, rng)?;
       ```

     - Calls feature encoder + net evaluation at new nodes.

   With `mcts-sims 20` and ~O(100) moves/game, plus multiple games per iter, this is still significant, but likely smaller than FD gradients.

3. **Feature extraction**

   `BasicFeatureExtractor::encode` (`crates/rl-env/src/feature_extractor.rs`) walks the entire state and builds a fresh `Vec<f32>` every call. It’s used in:
   - `AzulEnv::build_observations` on each environment step.
   - MCTS `create_node` + leaf evaluation (encode state for `to_play`).

   It’s a good candidate for micro‑benchmarking and optimization.

4. **Action encode/decode**

   `ActionEncoder` (`crates/rl-env/src/action_encoder.rs`) uses arithmetic encode/decode, with multiple divisions and modulo operations in `decode`. This is called:
   - In `AzulEnv::build_legal_action_mask` for each legal action.
   - In MCTS for each expanded edge when we step a state.

   Given `ACTION_SPACE_SIZE=500`, a simple lookup table could remove that cost.

All of this will be validated / refined with the profiling infrastructure below.

---

## 3. Profiling strategy overview

We want a **4-layer** profiling stack:

1. **End-to-end timing & basic counters**
   - How long does a full run take?
   - How much time is self-play vs training vs checkpointing?
   - How many games, moves, MCTS simulations, training steps?

2. **In-process scoped timers / spans + counters**
   - Time per `self_play_game`, `training_step`, `AlphaZeroMctsAgent::run_search`, `FeatureExtractor::encode`, `AlphaZeroNet::forward_batch`, `compute_gradients_fd`, etc.
   - Counts for MCTS simulations, NN evaluations, environment steps.

3. **Micro-benchmarks** (Criterion or standard benches)
   - Isolate hot functions in a controlled harness.

4. **External profiling tools**
   - Flamegraphs (`cargo flamegraph` / `perf`) and/or Instruments to get full-stack view.

The spec below describes how to **build those layers into the repo**.

---

## 4. Implementation Plan

### 4.1 Add a profiling feature + dev dependencies

**Goal:** Make profiling opt‑in, so production runs aren’t polluted by instrumentation overhead.

**Changes:**

1. In `crates/rl-env/Cargo.toml` and top-level `Cargo.toml`:
   - Add an optional `profiling` feature:

   ```toml
   [features]
   default = []
   profiling = []
   ```

   - Add dev‑dependencies for micro‑benchmarks:

   ```toml
   [dev-dependencies]
   criterion = "0.5"   # or latest compatible
   ```

   (No need to use `tracing` unless we want more sophisticated spans; we can add it later.)

2. Convention: any instrumentation code is guarded behind:

   ```rust
   #[cfg(feature = "profiling")]
   use crate::profiling::*;
   ```

   and calls `profiling::` helpers.

### 4.2 Create a small profiling module

**File:** `crates/rl-env/src/profiling.rs` (only compiled with `feature = "profiling"`)

**Responsibility:**

- Provide:
  - **Global counters** for key metrics (with atomics).
  - **Scoped timers** (RAII) for wall‑clock durations of critical sections.
  - Simple reporting helpers to print a summary at the end.

**Sketch interface (no need to implement here, just spec):**

```rust
// crates/rl-env/src/profiling.rs

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[derive(Default)]
pub struct Counters {
    pub self_play_games: AtomicU64,
    pub self_play_moves: AtomicU64,
    pub mcts_searches: AtomicU64,
    pub mcts_simulations: AtomicU64,
    pub mcts_nodes_created: AtomicU64,
    pub mcts_nn_evals: AtomicU64,
    pub train_steps: AtomicU64,
    pub fd_forward_evals: AtomicU64,
    pub env_steps: AtomicU64,
    pub nn_batch_forwards: AtomicU64,
}

// A single global instance (behind cfg(feature="profiling"))
pub static PROF: Counters = Counters { /* ... */ };

pub struct Timer {
    start: Instant,
    dest: &'static AtomicU64, // nanos accumulator
}

impl Timer {
    pub fn new(dest: &'static AtomicU64) -> Self { /* ... */ }
}

impl Drop for Timer {
    fn drop(&mut self) { /* accumulate elapsed.as_nanos() */ }
}
```

We can keep **per‑component time accumulators** (as `AtomicU64` nanoseconds) in the global `Counters` struct, e.g.:

- `time_self_play_ns`
- `time_training_ns`
- `time_mcts_search_ns`
- `time_mcts_simulate_ns`
- `time_fd_grad_ns`
- etc.

**Pattern of use:**

```rust
#[cfg(feature = "profiling")]
let _t = profiling::Timer::new(&profiling::PROF.time_self_play_ns);
```

The RAII `Timer` will measure the scope duration and add it to that counter.

### 4.3 Instrument key pipeline boundaries

We’ll add scoped timers + counters around the most important parts:

#### 4.3.1 High-level timing in `Trainer::run`

File: `crates/rl-env/src/alphazero/training.rs`

Add around:

- Each **self-play phase** per iteration.
- Each **training phase** per iteration.

Example (spec):

```rust
pub fn run(&mut self) -> Result<(), TrainingError> {
    for iter in 0..self.cfg.num_iters {
        // 1. Self-play
        #[cfg(feature = "profiling")]
        let _t_sp = profiling::Timer::new(&profiling::PROF.time_self_play_ns);

        for _ in 0..self.cfg.self_play_games_per_iter {
            let examples = self_play_game( /* ... */ );
            #[cfg(feature = "profiling")]
            {
                profiling::PROF.self_play_games.fetch_add(1, Ordering::Relaxed);
                profiling::PROF.self_play_moves
                    .fetch_add(examples.len() as u64, Ordering::Relaxed);
            }
            self.replay.extend(examples);
        }
        drop(_t_sp); // implicit by going out of scope

        // 2. Training
        #[cfg(feature = "profiling")]
        let _t_train = profiling::Timer::new(&profiling::PROF.time_training_ns);

        for _ in 0..self.cfg.training_steps_per_iter {
            if self.replay.len() < self.cfg.batch_size { break; }

            #[cfg(feature = "profiling")]
            profiling::PROF.train_steps.fetch_add(1, Ordering::Relaxed);

            // existing training_step call
            let _loss = self.training_step(&batch_refs)?;
        }
    }
}
```

Also add a helper to print a summary at the end of `run` (only if profiling feature is enabled):

```rust
#[cfg(feature = "profiling")]
profiling::print_summary();
```

Where `print_summary()` computes human‑readable stats (seconds, qps, etc.) from the counters.

#### 4.3.2 Self-play game internals

File: `crates/rl-env/src/alphazero/training.rs::self_play_game`

Instrument:

- Whole `self_play_game` body to get time per game.
- Loop over moves to count steps.

Inside the main while loop:

```rust
while !step.done && move_idx < self_play_cfg.max_moves {
    // ...
    #[cfg(feature = "profiling")]
    profiling::PROF.env_steps.fetch_add(1, Ordering::Relaxed);
    // ...
}
```

This gives us **moves per second**, **env steps**, etc.

#### 4.3.3 MCTS search internals

File: `crates/rl-env/src/mcts.rs`

Instrument:

- `AlphaZeroMctsAgent::run_search`
  - Time per search.
  - Increment `mcts_searches`.

- `simulate`
  - Time per simulation.
  - Counter `mcts_simulations`.

- `create_node`
  - Counter `mcts_nodes_created` and maybe time.

- NN evaluations:
  - In `create_node`: network policy for root / new states.
  - In `simulate` leaf evaluation: `self.net.predict_single`.

For NN evals:

```rust
#[cfg(feature = "profiling")]
let _t_nn = profiling::Timer::new(&profiling::PROF.time_mcts_nn_eval_ns);
let (policy_logits, value) = self.net.predict_single(&obs);
#[cfg(feature = "profiling")]
profiling::PROF.mcts_nn_evals.fetch_add(1, Ordering::Relaxed);
```

This answers “How much wall time is spent in NN evaluation during search?” vs tree bookkeeping vs game engine.

#### 4.3.4 Training step internals

File: `crates/rl-env/src/alphazero/training.rs::training_step`

Instrument:

- Entire `training_step` time.
- `compute_gradients_fd` time in particular.

Inside `training_step`:

```rust
#[cfg(feature = "profiling")]
let _t_step = profiling::Timer::new(&profiling::PROF.time_training_step_ns);
```

Inside `compute_gradients_fd`:

```rust
#[cfg(feature = "profiling")]
let _t_fd = profiling::Timer::new(&profiling::PROF.time_fd_grad_ns);
```

And around each `forward_with_params`:

```rust
#[cfg(feature = "profiling")]
profiling::PROF.fd_forward_evals.fetch_add(1, Ordering::Relaxed);
```

This should quickly show whether FD gradient is dwarfing everything else (it almost certainly is).

#### 4.3.5 Feature extractor and environment

File: `crates/rl-env/src/feature_extractor.rs`:

Instrument `BasicFeatureExtractor::encode`:

```rust
#[cfg(feature = "profiling")]
let _t = profiling::Timer::new(&profiling::PROF.time_feature_encode_ns);
```

File: `crates/rl-env/src/environment.rs`:

Instrument:

- `AzulEnv::reset`
- `AzulEnv::step`: count steps and maybe time.

```rust
#[cfg(feature = "profiling")]
profiling::PROF.env_steps.fetch_add(1, Ordering::Relaxed);
```

---

### 4.4 Add micro-benchmarks (Criterion)

Create a `benches/` directory with focused benchmarks. These will allow us to run `cargo bench -p azul-rl-env` to measure specific hot paths.

#### 4.4.1 Bench: feature extraction

File: `crates/rl-env/benches/feature_extractor_bench.rs`

- Use `azul_engine::new_game` to create a sample `GameState`.
- Use `BasicFeatureExtractor::new(2)` and repeatedly call `encode`.

Measure:

- `encode` for a single player.
- `AzulEnv::build_observations` (calls encode for all players).

#### 4.4.2 Bench: MCTS search

File: `crates/rl-env/benches/mcts_bench.rs`

- Use `DummyNet` from `mcts.rs` (uniform priors, constant value) to remove NN cost.
- Use a fixed game state from `AzulEnv::reset`.
- Benchmark `AlphaZeroMctsAgent::run_search` with various `num_simulations` (e.g. 10, 20, 50, 100).

Outputs:

- `ms per search` vs `num_simulations`.
- Derived metric: `simulations per second`.

#### 4.4.3 Bench: NN forward

File: `crates/rl-env/benches/net_bench.rs`

- Instantiate `AlphaZeroNet` with `obs_size = BasicFeatureExtractor::new(2).obs_size()` and `hidden_size=128`.
- Create a random `Array` batch of shape `[batch_size, obs_size]` with varying `batch_size` (1, 32, 64, 256).
- Benchmark `forward_batch`.

This isolates how heavy MLX forward passes are relative to everything else.

#### 4.4.4 Bench: training_step (FD variant)

File: `crates/rl-env/benches/training_step_bench.rs`

- Use `Trainer` with:
  - Small replay buffer pre-populated with synthetic `TrainingExample`s.
  - `TrainableModel` stub (same as test `StubPolicyValueModel` or a simplified one).

- Benchmark `training_step` directly for a single batch.

This will very likely show FD gradients as the major hog.

---

### 4.5 External profiling: flamegraphs + OS profilers

We don’t need code changes for this, but we should document a “standard” workflow in `docs/PROFILING.md` (or similar):

1. **Build in release with profiling feature:**

   ```bash
   cargo build --release --features profiling
   ```

2. **Use `cargo flamegraph` (Linux, macOS with `perf` / DTrace installed):**

   ```bash
   cargo flamegraph --release --features profiling -- \
     --num-iters 2 --games-per-iter 1 --training-steps 5 --mcts-sims 20
   ```

   The idea is:
   - Use **small** hyperparameters to keep runs short but realistic.
   - Then scale up once we know where time goes.

3. On macOS, also note that Xcode Instruments can be attached to the `azul` binary, but that’s just a note in docs.

---

## 5. Low-hanging performance wins (already visible)

Even before running the profiler, some changes are “obvious wins” or at least very strong candidates. We should track these as separate issues, but the profiling infra will quantify the savings.

### 5.1 Short-term: kill or gate finite-difference gradients

Right now:

- `compute_gradients_fd` is extremely expensive.
- `TrainableMctsAgent::forward_with_params` **ignores params** and instantiates a new `AlphaZeroNet`.
- `TrainableMctsAgent::apply_gradients` is a no-op.

So **all FD gradient work is pure waste**.

**Immediate mitigations:**

1. **Add a CLI flag `--no-train` (or `--train=false`)** that sets `training_steps_per_iter = 0` in `parse_args`:
   - This lets us profile self-play/MCTS separately without the training cost in the loop.
   - It’s the first thing to do when trying to understand where time is going.

2. **Feature-gate FD gradients**:
   - Add a feature `fd-gradients` (default: off).
   - In `training_step`, if `cfg!(feature = "fd-gradients")` is false:
     - Compute the loss (maybe just forward once).
     - **Skip `compute_gradients_fd` entirely**.

   - This gives a huge speedup and aligns behavior with the current correctness story (since gradients weren’t actually updating the net anyway).

3. Longer term, replace FD with **proper MLX autodiff**. That’s a bigger project (out of scope here), but the profiling infra will let us compare.

### 5.2 Precompute ActionId → Action decode table

`ActionEncoder::decode` does arithmetic each time:

```rust
let d_idx = x % BOARD_SIZE as u16;
x /= BOARD_SIZE as u16;
let d_type = x % 2;
x /= 2;
// ...
```

Given:

- `ACTION_SPACE_SIZE = 500`.
- The mapping is **static** for a given set of engine constants.

We can:

- At startup (or in a `lazy_static!`/`once_cell`), precompute:

  ```rust
  static ACTION_LUT: [Action; ACTION_SPACE_SIZE];
  ```

- Then `decode(id)` becomes a simple `ACTION_LUT[id as usize]` lookup.

This will pay off, especially inside MCTS:

- Every expansion decodes an `ActionId`.
- `environment::step` uses decode each time as well.

**Low complexity, easy to test**, and almost certainly positive.

### 5.3 Cache / reuse zero observations

In `BasicFeatureExtractor::encode` and `AzulEnv::build_observations`:

- For players beyond `num_players`, we create a fresh zero observation with `create_zero_observation(self.features.obs_size())`.

Given default `num_players = 2` and `MAX_PLAYERS = 4`, we do this on every step.

Optimization:

- Preallocate a single zero observation per `AzulEnv` or `BasicFeatureExtractor` and reuse it.

Not huge, but nearly free.

### 5.4 Store `obs_size` in `BasicFeatureExtractor` instead of recomputing

`BasicFeatureExtractor::obs_size()` currently recomputes `calculate_obs_size()` every call. That’s not terrible, but:

- `encode` calls `obs_size()` → `calculate_obs_size()`.
- `create_zero_observation` uses `obs_size`.

Tiny but free win:

- Compute `obs_size` once in `BasicFeatureExtractor::new` and store it as a field.

### 5.5 Reduce cloning in MCTS

In `simulate`:

```rust
let parent_state = tree.nodes[current_idx as usize].state.clone();
let action = ActionEncoder::decode(edge.action_id);
let step_result = apply_action(parent_state, action, rng)?;
```

This clones `GameState` per expansion.

Potential low‑effort improvements:

- If `apply_action` takes `GameState` by value and produces a new one, you may be forced to clone; but you can:
  - Store `GameState` by `Box` or `Rc` and only clone when necessary.
  - Or have a separate “scratch” state for deterministic simulation.

However, this is more medium‑sized than trivial; profiling infra will tell us whether this is worth addressing early.

### 5.6 Fix `TrainableMctsAgent::forward_with_params` semantics

Currently in `src/main.rs`:

```rust
fn forward_with_params(&self, _params: &[Array], obs: &Array) -> (Array, Array) {
    let mut agent_copy = TrainableMctsAgent::new(2, 4, self.hidden_size);
    agent_copy.agent.net.forward_batch(obs)
}
```

- Ignores `params`.
- Creates a brand‑new network (with fresh random weights) on every call.

This is:

- Wrong from a learning perspective.
- Adding unnecessary allocation and initialization overhead to every FD gradient evaluation.

Short term, if we’re going to keep FD gradients at all, we should at least reuse the underlying net and avoid reinitialization. But since FD itself is going away or gated, this is less urgent; it’s still worth filing as technical debt.

---

## 6. Execution plan / milestones

To avoid boiling the ocean, here’s a concrete ordering:

### Milestone 1: “Turn off the bonfire & get baseline”

1. Add `--no-train` flag to the CLI and allow setting `training_steps_per_iter = 0`.

2. Add `profiling` feature + minimal profiling module with global timers and counters.

3. Instrument:
   - `Trainer::run` top-level.
   - `self_play_game`.
   - `AlphaZeroMctsAgent::run_search` and `simulate`.
   - `AzulEnv::reset` / `step`.

4. Run:

   ```bash
   cargo run --release --features profiling -- \
     --num-iters 10 --games-per-iter 2 --mcts-sims 20 --no-train
   ```

   Capture:
   - Total time.
   - Time split: self-play vs (now empty) training.
   - Counts: games, moves, env steps, MCTS searches, simulations.

This gives a **baseline for self-play alone** and confirms user’s perception (“way too slow”) is training, MCTS, or both.

### Milestone 2: “Profile the current training step”

1. Add `fd-gradients` feature and gate `compute_gradients_fd`:
   - Without `fd-gradients`: training_step just forward + loss.
   - With `fd-gradients`: do the current FD computation.

2. Instrument `training_step` and `compute_gradients_fd` as described.

3. Run:
   - `--no-train` off, but `fd-gradients` off:

     ```bash
     cargo run --release --features profiling -- \
       --num-iters 2 --games-per-iter 2 --training-steps 10 --mcts-sims 20
     ```

   - Then with `fd-gradients` on:

     ```bash
     cargo run --release --features "profiling,fd-gradients" -- \
       --num-iters 2 --games-per-iter 2 --training-steps 10 --mcts-sims 20
     ```

4. Compare:
   - `time_training_ns` with vs without FD.
   - Counters: `fd_forward_evals`, `nn_batch_forwards`.

This will quantify how much FD is costing (likely orders of magnitude).

### Milestone 3: Micro-benchmarks

1. Add Criterion benches for:
   - Feature extraction.
   - MCTS search (with `DummyNet`).
   - NN forward.

2. Run `cargo bench -p azul-rl-env` and record numbers.

This gives us a menu of per‑component costs.

### Milestone 4: Low-hanging optimizations

Based on the above, implement:

1. **Action decode LUT**.
2. **Cache zero observations & store obs_size in BasicFeatureExtractor**.
3. If FD is still used anywhere, at least make `forward_with_params` reuse the network instead of re‑constructing it.

Re-run benchmarks and compare.

### Milestone 5: External flamegraphs

Once instrumentation + micro-benchmarks are in place, use `cargo flamegraph` with realistic small workloads to:

- Confirm that the expected hot spots (MCTS, feature extraction, NN forward / FD training) match our counters.
- Find any unexpected “mystery” hotspots (e.g., allocator, logging, MLX internals).

---

## 7. Summary

Concretely, this spec gives you:

- A **feature‑gated profiling module** with timers & counters.
- Specific **instrumentation points** in `Trainer`, MCTS, env, feature extractor, and training.
- A suite of **micro‑benchmarks** to isolate core primitives.
- A documented workflow for **end‑to‑end timing** and **flamegraphs**.
- A prioritized list of **low‑hanging performance fixes**, the biggest one being:

  > Gate or remove finite-difference gradients that currently burn vast compute without updating any parameters.
