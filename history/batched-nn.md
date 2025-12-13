# Engineering Spec: Batched NN Inference for AlphaZero MCTS

**Status:** Proposed
**Primary files:**

* `crates/rl-env/src/mcts.rs` (main work)
* `crates/rl-env/src/alphazero_net.rs` (no required changes, optional helpers)
* `crates/rl-env/src/alphazero/training.rs` (config plumbing only)
* `crates/rl-env/src/lib.rs` (re-exports only, likely no change)

---

## 1) Context and problem statement

Your profiling shows:

* **Total time (20 games, 20 sims, self-play only):** 11.37s (1.76 games/sec)
* **NN inference:** ~8.0s (≈70% of total)
* **MCTS non-NN:** ~3.0s (≈26%)
* **Environment step:** negligible

Microbenchmarks indicate:

* `AlphaZeroNet::forward_batch` (B=1..256) ≈ **7 µs**
* `PolicyValueNet::predict_single` ≈ **165 µs**

And your MCTS currently calls the NN **twice per simulation-ish**:

* once in `create_node()` to get policy logits (but it discards the value),
* again in `simulate()` to get the value at the leaf.

That’s the first big correctness/perf smell: **policy and value are produced together but you pay twice**.

Even if you only fix the “double eval”, NN time should roughly halve. But the big lever is to stop using `predict_single` and instead **feed the NN as a true batch** with `predict_batch`, and do it in a way that is compatible with MCTS.

---

## 2) Goals

### Primary goals

1. **Replace per-position NN evaluation (`predict_single`) inside MCTS with batched evaluation (`predict_batch`).**
2. **Eliminate redundant NN calls** by expanding nodes using the same NN output that provides the value.
3. Increase throughput substantially on Apple Silicon + MLX Metal backend.

### Target acceptance criteria (measurable)

Using your baseline command:

```
cargo run --release --features profiling -- \
  --num-iters 1 \
  --games-per-iter 20 \
  --mcts-sims 20 \
  --no-train \
  --no-checkpoints
```

We should achieve all of:

* **NN eval calls inside MCTS become 0** (no `predict_single` in MCTS path).
* **`mcts_nn_batches` > 0** and **`mcts_nn_positions` ≈ number of expanded nodes**.
* **Self-play throughput improves ≥ 2.5×** (goal: **≥ 4.5 games/sec**) without changing gameplay rules.

> Note: even if NN becomes “almost free”, Amdahl’s law means you’ll eventually be limited by non-NN MCTS overhead. But batching should still give a big step-change and sets you up to parallelize further.

### Non-goals (for this spec)

* Multi-threading across CPU cores with a global inference server (that’s a “next spec” after we get batching working).
* Major engine/game-state refactors (e.g., incremental apply/undo instead of cloning `GameState`).
* GPU-side masked softmax for legal actions.

---

## 3) Proposed approach: Leaf-parallel batched MCTS inside `run_search`

We implement **leaf-parallel** MCTS (a standard AlphaZero technique used for GPU utilization):

* Run simulations in **groups** (“in-flight simulations”).
* For each simulation in the group, do **selection** down the tree to a leaf.
* Apply **virtual loss** along the selected path so other in-flight sims don’t pile onto the same path.
* Collect the leaf states for all in-flight sims and evaluate them in **one `predict_batch` call**.
* Use NN outputs to:

  * expand those leaf nodes (create priors / children),
  * and backup the value along each simulation’s path.
* Repeat until `num_simulations` completed.

This provides a principled way to batch NN evaluations while keeping MCTS behavior reasonable.

---

## 4) API and config changes

### 4.1 `MctsConfig` additions (in `crates/rl-env/src/mcts.rs`)

Add fields:

```rust
pub struct MctsConfig {
    pub num_simulations: u32,
    pub cpuct: f32,
    pub root_dirichlet_eps: f32,
    pub root_dirichlet_alpha: f32,
    pub temperature: f32,
    pub max_depth: u32,

    /// NEW: how many leaf positions to evaluate per NN batch.
    /// If 1 => sequential but still uses predict_batch(B=1) path.
    pub nn_batch_size: usize,

    /// NEW: virtual loss magnitude for in-flight simulations.
    /// Typical: 1.0 when values are in [-1, 1].
    pub virtual_loss: f32,
}
```

Update `Default`:

```rust
impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            num_simulations: 256,
            cpuct: 1.5,
            root_dirichlet_eps: 0.25,
            root_dirichlet_alpha: 0.3,
            temperature: 1.0,
            max_depth: 200,
            nn_batch_size: 32,   // good starting point
            virtual_loss: 1.0,   // good starting point
        }
    }
}
```

### 4.2 Profiling counters (optional but recommended)

If you have `PROF.mcts_nn_evals`, change semantics to “positions evaluated”, and add:

* `mcts_nn_batches`
* `mcts_nn_positions`

So you can track batching effectiveness.

---

## 5) Data structure changes (minimal)

### 5.1 Node should store NN value at expansion (optional but useful)

This lets you avoid re-evaluating value for expanded nodes, and is helpful for debugging.

In `Node`:

```rust
pub struct Node {
    pub state: GameState,
    pub to_play: PlayerIdx,
    pub is_terminal: bool,
    pub children: Vec<ChildEdge>,
    pub visit_count: u32,

    /// NEW (optional): value predicted when the node was expanded.
    pub nn_value: Option<f32>,
}
```

Initialize `nn_value: None` for stubs, `Some(value)` after expansion.

> You can skip this field if you want: the core algorithm works without storing it, because you only evaluate unexpanded leaves. But I recommend keeping it.

---

## 6) Detailed algorithm and code changes (by file)

### 6.1 `crates/rl-env/src/mcts.rs` — main implementation

We will **replace** the current `create_node()` + `simulate()` pattern with:

* stub node creation (no NN call),
* batched evaluation to expand nodes,
* leaf-parallel simulation batches with virtual loss.

#### 6.1.1 New helper structs

Add near the top:

```rust
use std::collections::HashMap;

struct PendingSim {
    path: Vec<PathStep>,
    leaf_idx: NodeIdx,
    // filled later:
    leaf_value: f32,
    // if leaf needs NN (non-terminal):
    eval_slot: Option<usize>,
}
```

You already have `PathStep { node_idx, child_idx }`.

#### 6.1.2 Replace `create_node()` with “stub creation” and “expand node”

**Stub node creation:**

```rust
fn create_node_stub(&self, tree: &mut MctsTree, state: GameState) -> NodeIdx {
    let to_play = state.current_player;
    let is_terminal = state.phase == Phase::GameOver;

    let node = Node {
        state,
        to_play,
        is_terminal,
        children: Vec::new(),
        visit_count: 0,
        nn_value: None, // if you add the field
    };
    let idx = tree.nodes.len() as NodeIdx;
    tree.nodes.push(node);
    idx
}
```

**Expand a node given NN outputs (policy logits row + value):**

```rust
fn expand_node_from_nn(
    &mut self,
    tree: &mut MctsTree,
    node_idx: NodeIdx,
    policy_logits_row: &[f32], // len == ACTION_SPACE_SIZE
    value: f32,
) {
    let node = &mut tree.nodes[node_idx as usize];

    if node.is_terminal {
        node.nn_value = Some(value);
        return;
    }

    // Only expand once
    if !node.children.is_empty() {
        return;
    }

    node.nn_value = Some(value);

    let actions = legal_actions(&node.state);

    let mut legal_ids_and_logits: Vec<(ActionId, f32)> = Vec::with_capacity(actions.len());
    for action in &actions {
        let id = ActionEncoder::encode(action);
        legal_ids_and_logits.push((id, policy_logits_row[id as usize]));
    }

    let priors = softmax(&legal_ids_and_logits);

    node.children = priors.into_iter().map(|(id, prior)| ChildEdge {
        action_id: id,
        prior,
        visit_count: 0,
        value_sum: 0.0,
        child: None,
    }).collect();
}
```

> This is exactly what your old `create_node()` did, but now it consumes logits from a batch output and stores the value too.

#### 6.1.3 Virtual loss helpers

Add two functions:

```rust
fn apply_virtual_loss(tree: &mut MctsTree, step: &PathStep, vloss: f32) {
    let node = &mut tree.nodes[step.node_idx as usize];
    let edge = &mut node.children[step.child_idx];

    // virtual visit
    edge.visit_count += 1;
    node.visit_count += 1;

    // penalize Q to discourage collisions
    edge.value_sum -= vloss;
}

fn revert_virtual_loss(tree: &mut MctsTree, step: &PathStep, vloss: f32) {
    let node = &mut tree.nodes[step.node_idx as usize];
    let edge = &mut node.children[step.child_idx];

    // revert virtual visit
    edge.visit_count -= 1;
    node.visit_count -= 1;

    // revert virtual penalty
    edge.value_sum += vloss;
}
```

Now create a new backup function that reverts virtual loss and applies the real backup:

```rust
fn backup_with_virtual_loss(tree: &mut MctsTree, path: &[PathStep], leaf_value: f32, vloss: f32) {
    let mut value = leaf_value;

    for step in path.iter().rev() {
        // remove virtual loss for this in-flight sim
        revert_virtual_loss(tree, step, vloss);

        // apply real backup
        let node = &mut tree.nodes[step.node_idx as usize];
        let edge = &mut node.children[step.child_idx];

        edge.visit_count += 1;
        edge.value_sum += value;
        node.visit_count += 1;

        value = -value;
    }
}
```

#### 6.1.4 Selection: run one “in-flight simulation” to a leaf

This function performs selection while applying virtual loss and creates stub children for unexpanded edges:

```rust
fn select_leaf(
    &mut self,
    tree: &mut MctsTree,
    root_idx: NodeIdx,
    rng: &mut impl Rng,
) -> PendingSim {
    let mut path: Vec<PathStep> = Vec::new();
    let mut current_idx = root_idx;

    for _depth in 0..(self.config.max_depth as usize) {
        let is_terminal;
        let has_children;
        {
            let node = &tree.nodes[current_idx as usize];
            is_terminal = node.is_terminal;
            has_children = !node.children.is_empty();
        }

        if is_terminal || !has_children {
            break;
        }

        // choose child via PUCT
        let child_idx = {
            let node = &tree.nodes[current_idx as usize];
            select_child(node, self.config.cpuct)
        };

        // record and apply virtual loss on the chosen edge
        let step = PathStep { node_idx: current_idx, child_idx };
        apply_virtual_loss(tree, &step, self.config.virtual_loss);
        path.push(step);

        // descend or expand stub
        let next_child_opt = tree.nodes[current_idx as usize].children[child_idx].child;
        if let Some(next_idx) = next_child_opt {
            current_idx = next_idx;
            continue;
        }

        // Expand edge by creating child stub node
        let parent_state = tree.nodes[current_idx as usize].state.clone();
        let action_id = tree.nodes[current_idx as usize].children[child_idx].action_id;
        let action = ActionEncoder::decode(action_id);

        let step_result = apply_action(parent_state, action, rng)
            .expect("MCTS should only expand legal actions");

        let new_idx = self.create_node_stub(tree, step_result.state);

        tree.nodes[current_idx as usize].children[child_idx].child = Some(new_idx);

        current_idx = new_idx;
        break;
    }

    PendingSim {
        path,
        leaf_idx: current_idx,
        leaf_value: 0.0,      // filled later
        eval_slot: None,      // filled later
    }
}
```

Key semantics:

* We treat a node with `children.is_empty()` as **unexpanded leaf**, and it becomes a candidate for NN evaluation.
* Terminal nodes also have empty children, but `is_terminal` distinguishes them.

#### 6.1.5 Batched evaluation + expansion + backup

We now add a function to process a group of `PendingSim`:

```rust
fn process_batch(
    &mut self,
    tree: &mut MctsTree,
    sims: &mut [PendingSim],
) {
    // 1) Identify which leaves need NN evaluation; compute terminal ones immediately.
    let mut unique_leafs: Vec<NodeIdx> = Vec::new();
    let mut leaf_to_slot: HashMap<NodeIdx, usize> = HashMap::new();

    for sim in sims.iter_mut() {
        let leaf = &tree.nodes[sim.leaf_idx as usize];
        if leaf.is_terminal {
            sim.leaf_value = compute_terminal_value(&leaf.state, leaf.to_play);
            sim.eval_slot = None;
        } else {
            let slot = *leaf_to_slot.entry(sim.leaf_idx).or_insert_with(|| {
                let s = unique_leafs.len();
                unique_leafs.push(sim.leaf_idx);
                s
            });
            sim.eval_slot = Some(slot);
        }
    }

    // 2) If any NN leaves exist, batch them.
    if !unique_leafs.is_empty() {
        let obs_size = self.features.obs_size();
        let b = unique_leafs.len();

        // Build [B, obs_size] contiguous buffer
        let mut obs_data: Vec<f32> = Vec::with_capacity(b * obs_size);
        for &node_idx in &unique_leafs {
            let node = &tree.nodes[node_idx as usize];
            let obs = self.features.encode(&node.state, node.to_play);
            obs_data.extend_from_slice(obs.as_slice::<f32>());
        }

        let obs_batch = Array::from_slice(&obs_data, &[b as i32, obs_size as i32]);

        // NN inference
        #[cfg(feature = "profiling")]
        let _t = Timer::new(&PROF.time_mcts_nn_eval_ns);
        #[cfg(feature = "profiling")]
        {
            PROF.mcts_nn_batches.fetch_add(1, Ordering::Relaxed);
            PROF.mcts_nn_evals.fetch_add(b as u64, Ordering::Relaxed);
        }

        let (policy_logits_batch, values_batch) = self.net.predict_batch(&obs_batch);

        let logits = policy_logits_batch.as_slice::<f32>();
        let values = values_batch.as_slice::<f32>();

        // 3) Expand each unique leaf using its logits row, and store leaf value for sims.
        for (slot, &node_idx) in unique_leafs.iter().enumerate() {
            let value = values[slot];
            let row_start = slot * ACTION_SPACE_SIZE;
            let row_end = row_start + ACTION_SPACE_SIZE;
            let logits_row = &logits[row_start..row_end];

            // Expand only if still unexpanded (should be)
            if tree.nodes[node_idx as usize].children.is_empty() {
                self.expand_node_from_nn(tree, node_idx, logits_row, value);
            }
        }

        // Fill leaf_value for each sim
        for sim in sims.iter_mut() {
            if let Some(slot) = sim.eval_slot {
                sim.leaf_value = values[slot];
            }
        }
    }

    // 4) Backup each simulation result (undo virtual loss and apply real value)
    for sim in sims.iter() {
        backup_with_virtual_loss(tree, &sim.path, sim.leaf_value, self.config.virtual_loss);
    }
}
```

**Optimization note:** `obs_data` is allocated per batch in the snippet above. In production, you should **reuse** a scratch buffer to avoid per-batch allocations (see §8).

#### 6.1.6 Rewrite `run_search()` to use the batched pipeline

Replace the body of `run_search()` with:

```rust
fn run_search(&mut self, root_state: &GameState, rng: &mut impl Rng) -> [f32; ACTION_SPACE_SIZE] {
    #[cfg(feature = "profiling")]
    let _t = Timer::new(&PROF.time_mcts_search_ns);
    #[cfg(feature = "profiling")]
    PROF.mcts_searches.fetch_add(1, Ordering::Relaxed);

    let mut tree = MctsTree::default();

    // 1) Root stub + root expansion (batch of 1)
    let root_idx = self.create_node_stub(&mut tree, root_state.clone());

    {
        // Expand root using predict_batch(B=1) to avoid predict_single overhead.
        let obs_size = self.features.obs_size();
        let root = &tree.nodes[root_idx as usize];
        let obs = self.features.encode(&root.state, root.to_play);
        let mut obs_data: Vec<f32> = Vec::with_capacity(obs_size);
        obs_data.extend_from_slice(obs.as_slice::<f32>());
        let obs_batch = Array::from_slice(&obs_data, &[1, obs_size as i32]);

        #[cfg(feature = "profiling")]
        let _t = Timer::new(&PROF.time_mcts_nn_eval_ns);
        #[cfg(feature = "profiling")]
        {
            PROF.mcts_nn_batches.fetch_add(1, Ordering::Relaxed);
            PROF.mcts_nn_evals.fetch_add(1, Ordering::Relaxed);
        }

        let (policy_logits_batch, values_batch) = self.net.predict_batch(&obs_batch);
        let logits = policy_logits_batch.as_slice::<f32>();
        let values = values_batch.as_slice::<f32>();

        let logits_row = &logits[0..ACTION_SPACE_SIZE];
        let value = values[0];

        self.expand_node_from_nn(&mut tree, root_idx, logits_row, value);
    }

    // Root Dirichlet noise
    if self.config.root_dirichlet_alpha > 0.0 && !tree.nodes[root_idx as usize].children.is_empty() {
        add_dirichlet_noise(
            &mut tree.nodes[root_idx as usize].children,
            self.config.root_dirichlet_alpha,
            self.config.root_dirichlet_eps,
            rng,
        );
    }

    // 2) Batched simulations
    let total = self.config.num_simulations as usize;
    let batch_size = self.config.nn_batch_size.max(1).min(total);

    let mut done = 0;
    let mut sims: Vec<PendingSim> = Vec::with_capacity(batch_size);

    while done < total {
        let n = (total - done).min(batch_size);

        sims.clear();
        for _ in 0..n {
            #[cfg(feature = "profiling")]
            PROF.mcts_simulations.fetch_add(1, Ordering::Relaxed);

            sims.push(self.select_leaf(&mut tree, root_idx, rng));
        }

        self.process_batch(&mut tree, &mut sims[..]);
        done += n;
    }

    // 3) Build policy from root visits
    let root = &tree.nodes[root_idx as usize];
    let mut counts = [0.0f32; ACTION_SPACE_SIZE];
    for edge in &root.children {
        counts[edge.action_id as usize] = edge.visit_count as f32;
    }

    apply_temperature(&counts, self.config.temperature)
}
```

At this point:

* `simulate()` and the old `create_node()` can be deleted or left behind a feature flag for regression comparison.
* All NN inference inside MCTS uses `predict_batch`.

#### 6.1.7 Remove the redundant second NN call

With the new pipeline, you never call the NN “for value” separately. Each evaluated leaf receives both logits and value in the same batch. So the “2 evals per sim” goes away automatically.

---

### 6.2 `crates/rl-env/src/alphazero_net.rs` — optional improvements (not required)

You don’t need to change anything to get batching working. But you can make it easier to keep `predict_single` from being a footgun:

**Optional: add a comment warning that MCTS must not use `predict_single`** because it’s slow.

**Optional: add a helper to build a batch without extra reshapes:**

```rust
impl AlphaZeroNet {
    pub fn predict_batch_from_flat(&mut self, flat: &[f32], batch: usize, obs_size: usize) -> (Array, Array) {
        let obs_batch = Array::from_slice(flat, &[batch as i32, obs_size as i32]);
        self.forward_batch(&obs_batch)
    }
}
```

This is “nice to have” but not necessary.

---

### 6.3 `crates/rl-env/src/alphazero/training.rs` — config plumbing

No changes required to training logic.

But you should ensure wherever you build `MctsConfig` (likely outside this subset) you can set:

* `nn_batch_size`
* `virtual_loss`

If training constructs `AlphaZeroMctsAgent::new(MctsConfig { num_simulations: ..., ..Default::default() })`, you’ll automatically get the default batching settings. Later you can expose CLI flags:

* `--mcts-nn-batch-size`
* `--mcts-virtual-loss`

---

### 6.4 `crates/rl-env/src/lib.rs`

No changes required unless you want to re-export new config fields (already part of `MctsConfig`).

---

## 7) Test changes required

### 7.1 Fix `DummyNet::predict_batch` in `mcts.rs` tests

Currently:

```rust
fn predict_batch(&mut self, _obs_batch: &Array) -> (Array, Array) {
    unimplemented!("DummyNet::predict_batch not implemented for tests")
}
```

This will now crash because MCTS uses `predict_batch`.

Implement:

```rust
fn predict_batch(&mut self, obs_batch: &Array) -> (Array, Array) {
    let batch_size = obs_batch.shape()[0] as usize;

    let mut policies = Vec::with_capacity(batch_size * ACTION_SPACE_SIZE);
    for _ in 0..batch_size {
        policies.extend_from_slice(&self.priors);
    }

    let values = vec![self.value; batch_size];

    let policy_arr = Array::from_slice(&policies, &[batch_size as i32, ACTION_SPACE_SIZE as i32]);
    let values_arr = Array::from_slice(&values, &[batch_size as i32]);
    (policy_arr, values_arr)
}
```

### 7.2 Update `test_mcts_config_default`

It now needs to assert defaults for `nn_batch_size` and `virtual_loss`.

### 7.3 Add one new invariant test (recommended)

Add a test that runs a search with `nn_batch_size > 1` and asserts:

* returned policy sums ~1 (after masking and renormalization in `select_action_and_policy`)
* action selected is legal (existing tests cover this)

---

## 8) Performance engineering details (critical once batching works)

The code above is correct but will still allocate per batch. Once NN cost drops, allocator noise may dominate.

### 8.1 Add scratch buffers to `AlphaZeroMctsAgent`

Add fields (or local reused vars in `run_search`) to reuse allocations:

* `Vec<f32> obs_scratch` sized `nn_batch_size * obs_size`
* `Vec<PendingSim> sims_scratch`
* `HashMap<NodeIdx, usize> leaf_to_slot` scratch (or a `Vec<Option<usize>>` indexed by node_idx for faster dedup)

For example, inside `run_search` you can maintain:

```rust
let mut obs_scratch: Vec<f32> = Vec::with_capacity(batch_size * obs_size);
let mut leaf_to_slot: HashMap<NodeIdx, usize> = HashMap::with_capacity(batch_size);
let mut unique_leafs: Vec<NodeIdx> = Vec::with_capacity(batch_size);
```

and then in `process_batch` use `clear()` instead of reallocating.

### 8.2 Avoid repeated `legal_actions` allocations if it shows up next

After batching, **MCTS non-NN** will matter more. The biggest remaining obvious costs will be:

* cloning `GameState` for apply_action
* `legal_actions(&state)` allocations
* CPU softmax per node expansion

This spec does not solve these, but you should re-profile after batching and then:

* consider storing legal action ids in node when first expanded,
* or implement a preallocated buffer approach if `azul_engine::legal_actions` supports it.

---

## 9) Rollout plan (low risk)

### Phase 1 (very small diff, immediate win)

Before you do leaf-parallel batching, do the two “obvious” fixes:

1. **Remove the second NN call**:

   * `create_node()` already gets `(policy_logits, value)` but discards value.
   * Store it and reuse it in `simulate()` so each expansion costs **one NN call**, not two.

2. **Stop calling `predict_single`** from MCTS even for single positions:

   * Build a `[1, obs_size]` `obs_batch` and call `predict_batch` (no squeeze/reshape).

This phase should already reduce NN overhead dramatically and is easy to validate.

### Phase 2 (this spec)

Implement leaf-parallel batching with:

* `select_leaf`
* `process_batch`
* virtual loss
* batch root expansion

### Phase 3 (next spec)

Once NN is cheap, MCTS non-NN will dominate. Then you likely want:

* parallelize games (multiple envs) + central inference queue to keep GPU fed
* or multi-threaded simulations with a batching inference worker

---

## 10) Risks and mitigations

### Risk: Search behavior changes vs strict sequential MCTS

Leaf-parallel MCTS is standard; it will change exact visitation distribution vs sequential. If this is a concern:

* support `nn_batch_size=1` mode (still uses `predict_batch(B=1)` fast path) for “reference behavior”.
* run quality checks comparing win rates against baseline.

### Risk: Virtual loss chosen poorly

If `virtual_loss` is too large, it can bias search badly. Start with `1.0` and tune; consider exposing it.

### Risk: Bugs from underflow in visit counts

`revert_virtual_loss` subtracts 1 from counts. Use `debug_assert!(edge.visit_count > 0)` and same for node counts.

---

## 11) Summary of exactly what to change

**In `mcts.rs`:**

* Add `nn_batch_size`, `virtual_loss` to `MctsConfig`.
* Add optional `nn_value: Option<f32>` to `Node`.
* Replace old `create_node()` and `simulate()` approach with:

  * `create_node_stub`
  * `expand_node_from_nn`
  * `apply_virtual_loss`, `revert_virtual_loss`
  * `select_leaf`
  * `process_batch`
  * rewrite `run_search` to drive batched sims
* Update profiling counters (optional).
* Implement `DummyNet::predict_batch` in tests.

**In `alphazero_net.rs`:**

* No required change.

**In `training.rs`:**

* No required change; optionally expose new MctsConfig knobs from CLI.

---

If you want, I can also provide a “patch-style” pseudo-diff for `mcts.rs` (function signatures and where each block goes) that matches your exact current layout, to make the implementation mostly mechanical.
