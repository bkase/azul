# Engineering Design: Parallel + Batched MCTS (CPU) with MLX GPU Inference Batching

**Audience:** repo maintainers / contributors working in `crates/rl-env/src/mcts.rs` and the training CLI
**Scope:** speed up *self-play* by (1) parallelizing MCTS simulation work on CPU and (2) batching NN inference on MLX/Metal to eliminate `predict_single` overhead.

---

## 0. Context and constraints

### What profiling told us (load-bearing facts)

* Self-play runtime is dominated by NN inference (~70%).
* `AlphaZeroNet::forward_batch` is ~7µs for batch sizes 1–256 (meaning the GPU work is cheap and/or amortized).
* `PolicyValueNet::predict_single` is **165µs** and called extremely frequently.
* MCTS overhead w/ dummy net is ~2.2µs/simulation → after we fix NN, **CPU MCTS overhead will become the bottleneck**.

### Important correctness/algorithm note in current MCTS code

In `mcts.rs`, **we do two NN evals per expanded leaf**:

* `create_node()` calls `self.net.predict_single(&obs)` and discards `_value`.
* The same simulation then reaches that newly created node and **calls predict_single again** to get value.

This is not how AlphaZero MCTS is typically implemented (policy+value come from a single eval at expansion). Fixing this is a near-free ~2× reduction in NN eval count.

---

## 1. Goals and success metrics

### Primary goals

1. **Reduce NN portion** of self-play time by ≥10× (via batching and eliminating duplicate evals).
2. **Scale CPU MCTS overhead** across cores (via parallel simulations).
3. Keep training compatibility: training remains fast, correctness preserved.

### Concrete success metrics (with the baseline you gave)

Baseline (20 games, 20 sims, no-train): **1.76 games/sec**, total 11.37s.

Targets:

* Phase 0 (single-eval fix): ≥ **2.5 games/sec** (expect ~1.8× in NN-heavy configs).
* Phase 1 (batched inference): ≥ **5–8 games/sec** (NN time collapses; MCTS overhead dominates).
* Phase 2 (CPU parallelism): ≥ **10 games/sec** (depends on core count and locking overhead).

---

## 2. Plan overview (phased)

### Phase 0 — Fix the wasted second NN eval (fastest win)

* Modify MCTS so expansion returns both policy priors and value and uses that value for backup.
* Expected: near 2× reduction in `mcts_nn_evals`, large wall-clock win immediately.

### Phase 1 — Batched NN eval (GPU/MLX) inside MCTS

* Stop calling `predict_single` in the search loop.
* Gather N leaf states, run `predict_batch` once, then expand/backup all N.
* This is the core path to turning 165µs calls into amortized single sync per batch.

### Phase 2 — CPU parallelism for MCTS simulations

* After Phase 1, CPU MCTS overhead is the limiter.
* Implement a parallel search mode that scales simulations across CPU cores.
* Recommended initial approach: **root-parallel + shared batched evaluator** (lower complexity).
* Optional later: **tree-parallel shared tree** (higher complexity, better search efficiency).

---

## 3. Detailed design

## 3.1 Phase 0: eliminate duplicate NN eval per expansion

### Current code path (problem)

In `simulate()`:

1. Expansion calls `create_node()` → NN eval #1 for (policy, value) but value discarded.
2. Evaluation of leaf calls `predict_single()` again → NN eval #2.

### Change

Make `create_node()` return the predicted value for the node it expanded, and have `simulate()` use it when expansion occurred.

#### Proposed signature change

In `crates/rl-env/src/mcts.rs`:

```rust
fn create_node(
    &mut self,
    tree: &mut MctsTree,
    state: GameState,
    rng: &mut impl Rng,
) -> (NodeIdx, f32)
```

* For terminal nodes: return `(idx, terminal_value)` (or 0; but better to return correct value for consistent backup).
* For non-terminal: return `(idx, value_pred)` from the same NN call used to get logits.

#### simulate() logic change (exact behavior)

* Add `let mut expanded_value: Option<f32> = None;`
* When expansion happens:

  * call `let (new_idx, v) = self.create_node(...)`
  * set `expanded_value = Some(v)`
* After traversal:

  * if `expanded_value.is_some()`: use it for `leaf_value` and **do not eval net again**.

This alone can cut NN evals roughly in half.

---

## 3.2 Phase 1: MLX GPU batching with `predict_batch`

This phase changes the *shape* of the search loop: instead of “simulate → single inference → backup”, we do “collect K leaves → one batched inference → expand+backup K”.

### Key insight

The real killer overhead is **sync** (`as_slice` / host readback) and per-call overhead, not math.
So we want:

* **one `policy_logits.as_slice()` per batch**, not per node
* **one `values.as_slice()` per batch**, not per node
* no `reshape/squeeze` per leaf (avoid `predict_single` path entirely)

### Core data structures

Add to `mcts.rs` (private to module):

```rust
struct LeafJob {
    // Path from root to the chosen edge we want to back up through.
    path: Vec<PathStep>,

    // If we expanded a new edge: where to attach the child node after inference.
    expand_parent: Option<(NodeIdx, usize /* child_idx */)>,

    // The leaf state to evaluate.
    state: GameState,
    to_play: PlayerIdx,

    // Legal action ids for this leaf state (needed to form priors from logits).
    legal_ids: Vec<ActionId>,

    // If terminal, we can skip NN and just use this value.
    terminal_value: Option<f32>,
}
```

### Collecting leaf jobs (single-threaded batched search)

Replace the inner simulation loop with:

* for `batch_size` times:

  * traverse tree with PUCT
  * if hit unexpanded edge:

    * compute next state via `apply_action`
    * compute legal actions for that next state (or do it later)
    * create a `LeafJob` with `expand_parent = Some((parent_idx, child_idx))`
    * **do not call the net here**
  * else if terminal: `terminal_value = Some(compute_terminal_value(...))`
  * else if max depth: treat as value-only leaf job

Then:

* Separate terminal jobs and NN jobs.
* Batch-evaluate NN jobs with `predict_batch`.

### Batched inference implementation

Inside `AlphaZeroMctsAgent` add helper:

```rust
fn eval_leaves_batch(&mut self, leaves: &mut [LeafJob]) {
    // 1) Build obs_data for non-terminal leaves
    // 2) Array::from_slice(obs_data, [B, obs_size])
    // 3) self.net.predict_batch(&obs_batch)
    // 4) logits_slice = policy_logits.as_slice::<f32>()  // once
    //    values_slice = values.as_slice::<f32>()          // once
    // 5) For each leaf i:
    //      - compute priors from logits_slice[i * ACTION_SPACE_SIZE..]
    //      - expand node if expand_parent present
    //      - backup value
}
```

#### Computing priors from logits (CPU)

Reuse your existing `softmax()` approach but take logits directly from the batched row:

```rust
fn priors_from_logits_row(legal_ids: &[ActionId], logits_row: &[f32]) -> Vec<(ActionId, f32)> {
    // max trick, exp, normalize
}
```

### Handling expansions

For each evaluated `LeafJob` with `expand_parent = Some((pidx, cidx))`:

* create node from state + priors (no NN call!)
* `tree.nodes[pidx].children[cidx].child = Some(new_idx);`
* then `backup(tree, &job.path, value)`

You’ll want a helper to create node from priors:

```rust
fn create_node_from_priors(tree: &mut MctsTree, state: GameState, priors: Vec<(ActionId, f32)>) -> NodeIdx
```

### (Optional but recommended) Virtual loss / “pending edge” within the batch

If you see many duplicates (same leaf selected multiple times within a batch), add a lightweight pending bit:

* Add to `ChildEdge`:

  ```rust
  pub pending: bool
  ```
* In `select_child`, skip pending edges (or penalize strongly).
* When you create a `LeafJob` for an unexpanded edge, set `pending = true`.
* After expansion, clear `pending = false`.

This keeps batch diversity higher, especially when batch_size > num_simulations is close.

### Why this is “GPU with MLX”

* All inference calls go through `predict_batch`, which uses MLX ops and Metal backend.
* We drastically reduce CPU-GPU synchronization points by calling `as_slice` once per batch.

---

## 3.3 Phase 2: CPU parallelization of MCTS simulations

Once Phase 1 is in, NN cost plummets and the 26% “MCTS non-NN” becomes the dominant cost. We need CPU parallelism to keep scaling.

There are two viable parallel strategies:

### Strategy A: Root-parallel (recommended first)

**Idea:** Run N independent MCTS searches from the same root, each with `num_simulations / N` sims, then sum root visit counts.

Pros:

* No shared-tree locking (simple and fast).
* Deterministic if seeds are deterministic.
* Easy to ship first.

Cons:

* Less search efficiency vs true tree-parallel (workers don’t share deeper stats).

#### Implementation plan

1. Extend `MctsConfig`:

   ```rust
   pub num_threads: usize,        // default 1
   pub nn_batch_size: usize,      // default 32 (or 64)
   pub parallel_mode: ParallelMode, // None | RootParallel
   ```
2. In `run_search`, if `num_threads > 1`, call `run_search_root_parallel`.
3. `run_search_root_parallel`:

   * Build root priors once (including Dirichlet noise).
   * Spawn N workers (Rayon scope; do NOT spawn OS threads per move).
   * Each worker:

     * clones root state
     * runs batched-MCTS locally for its sims (Phase 1 logic)
     * returns a root visit-count array `[u32; ACTION_SPACE_SIZE]`
   * Aggregate root visit counts (sum arrays) → apply temperature.

#### Concurrency note about `self.net`

If MLX modules are not `Send + Sync` (likely), **do not share net across workers**.

Instead, in root-parallel you have two options:

* **Option A1 (safe):** root-parallel *only on CPU portion*, but do inference on the calling thread via a centralized evaluation loop (workers send leaf jobs to main thread).
  This is more complex but avoids `Send`.
* **Option A2 (fast to prototype):** clone the network per worker.
  This is usually viable if you keep the net on GPU and don’t rebuild graphs every time, but may still serialize on the Metal queue and increase memory use.

**Recommendation:** Implement A1 if `AlphaZeroNet` is not Send/Sync.
Make it explicit in the first PR: add a small compile-time assert/test to check Send/Sync and choose path.

---

### Strategy B: Tree-parallel (shared tree) + batched evaluator (best long-term)

**Idea:** Multiple CPU workers run simulations concurrently on the *same* tree. When they hit a leaf needing eval, they queue it for batched GPU eval, then resume.

Pros:

* Best search efficiency (workers share stats).
* Matches how production AZ engines saturate GPU.

Cons:

* Requires thread-safe tree representation (atomics/locks).
* More engineering work.

#### Implementation sketch (if you go here)

* Shared `MctsTree` behind:

  * per-node expansion lock (Mutex)
  * atomic visit counts
  * atomic value sums (CAS loop over f32 bits)
* Workers:

  * selection uses atomic reads
  * apply virtual loss (atomic increments) to reduce collisions
  * leaf expansion: winner thread expands after evaluator returns priors
* Evaluator:

  * single thread owns `&mut net`
  * batches leaf states to `predict_batch`

This is a bigger project, but the architecture is the “correct” endpoint if you want to maximize throughput.

---

## 4. Concrete “changes by file”

## 4.1 `crates/rl-env/src/mcts.rs`

### Must-do changes

* **Phase 0:** change `create_node` to return `(NodeIdx, f32)` and remove duplicate NN eval.
* Add batched path:

  * new `LeafJob` struct
  * `collect_leaf_job(...)`
  * `eval_leaves_batch(...)` using `predict_batch`
  * `create_node_from_priors(...)`
* Add config fields:

  * `nn_batch_size`
  * `num_threads`
  * `parallel_mode` enum
* Update profiling counters:

  * `PROF.mcts_nn_evals` should increment **per leaf evaluated**, not per `predict_single` call.
  * Add `PROF.mcts_nn_batches` and `PROF.mcts_nn_batch_size_sum` to track batching efficiency.

### Optional changes (recommended)

* Add `pending: bool` to `ChildEdge` for in-batch dedupe/virtual loss.
* Add new tests:

  * `test_mcts_parallel_matches_single_thread_dummy_net` (Root-parallel only)
  * `test_batched_mcts_policy_sums_to_one`

## 4.2 `crates/rl-env/src/alphazero_net.rs`

* No functional change required if `predict_batch` is correct.
* Performance tweaks:

  * Add `forward_batch_inference(&mut self, obs_batch: &Array) -> (Array, Array)` wrapper that:

    * avoids extra squeezes/reshapes
    * calls `eval_params(...)` only when needed (don’t over-sync)
* If you suspect `as_slice` sync dominates, add an inference-only method that returns CPU vectors in one go:

  * `fn predict_batch_cpu(&mut self, obs_batch: &Array) -> (Vec<f32>, Vec<f32>)`
  * internally does `predict_batch` and then `as_slice` once.

## 4.3 `crates/rl-env/src/alphazero/training.rs`

* Add configuration plumbing for MCTS parallelism and batching.

  * You can put these fields either in `SelfPlayConfig` or directly into `MctsConfig`.
* Ensure training remains sequential after self-play (current behavior). That makes concurrency easier.

## 4.4 `src/main.rs`

Add CLI flags to control parallelism and batching:

* `--mcts-threads <N>` (default 1)
* `--mcts-nn-batch-size <B>` (default 32 or 64)
* `--mcts-parallel-mode <none|root>` (start with root only)
* (optional) `--rayon-threads <N>` or document `RAYON_NUM_THREADS`

When constructing `MctsConfig`, populate these.

---

## 5. Implementation checklist (actionable steps)

### PR 1 — Phase 0 “correctness + perf”

* [ ] Change `create_node` → returns `(NodeIdx, f32)`
* [ ] Update `simulate` to reuse expansion value (remove second eval)
* [ ] Update profiling counters; rerun baseline command and confirm:

  * `mcts_nn_evals` roughly halves
  * games/sec increases materially

### PR 2 — Phase 1 “batched inference”

* [ ] Add `nn_batch_size` to `MctsConfig`
* [ ] Implement `LeafJob` + `collect_leaf_job`
* [ ] Implement `eval_leaves_batch`:

  * build obs batch
  * call `predict_batch`
  * compute priors + values
  * expand + backup
* [ ] Update run_search loop to do simulations in batches:

  * `while sims_done < num_simulations { take = min(nn_batch_size, remaining); ... }`
* [ ] Add profiling metrics: batch sizes achieved
* [ ] Confirm `predict_single` is no longer used in the hot path (except maybe root init)
* [ ] Rerun benchmark; expect NN time to collapse

### PR 3 — Phase 2 “CPU parallelism (root-parallel)”

* [ ] Add `ParallelMode::RootParallel` and `num_threads` to `MctsConfig`
* [ ] Implement `run_search_root_parallel` with Rayon scope
* [ ] Deterministic seeding per worker:

  * derive `seed_i` from the input RNG in a deterministic sequence
* [ ] Aggregate root counts and apply temperature
* [ ] Add a test that checks:

  * output policy sums to 1
  * no illegal actions selected after masking
* [ ] Benchmark with `--mcts-threads` tuned to performance cores

### PR 4 — (Optional) virtual loss/pending edges for better batching

* [ ] Add `pending` bit to edges, skip pending in selection
* [ ] Measure if it increases effective batch diversity / reduces duplicate work

---

## 6. Practical tuning guidance for Apple Silicon

* Start with:

  * `--mcts-nn-batch-size 32` or `64`
  * `--mcts-threads` = number of **performance cores** (not total cores).
* Use `RAYON_NUM_THREADS` to keep Rayon from overscheduling onto efficiency cores.
* If you see GPU underutilized:

  * increase batch size
  * increase `mcts-threads`
* If you see CPU pegged but GPU idle:

  * batch size may be too small or too many sync points remain (`as_slice` per leaf somewhere)

---

## 7. Known risks and mitigations

1. **MLX thread-safety / Send + Sync issues**

   * Mitigation: keep *all* MLX calls on the calling thread (batched) and use Rayon only for pure CPU work, or implement a single-thread evaluator.
2. **Determinism changes**

   * Root-parallel can remain deterministic with fixed per-worker seeds.
   * Tree-parallel will not be deterministic; don’t ship that without accepting nondeterminism.
3. **Batching accidentally increases duplicates**

   * Add `pending` virtual loss or dedupe leaf selections inside the batch.

---

## 8. The two biggest “do this first” actions

1. **Fix the double NN eval** (Phase 0). It’s a correctness-aligned optimization and should be a big immediate win.
2. **Implement batched MCTS leaf evaluation using `predict_batch`** (Phase 1). This is the main lever to turn MLX/Metal into a throughput engine.

Once those are in, CPU parallelism becomes the next bottleneck to attack.

---

If you want, I can also provide a concrete pseudocode-to-Rust mapping for `LeafJob` collection + `eval_leaves_batch` + how to integrate it cleanly into the current `run_search` loop in `mcts.rs` (with minimal code churn).
