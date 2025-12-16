//! AlphaZero-style MCTS agent implementation
//!
//! This module provides:
//! - `PolicyValueNet` trait for neural network interface
//! - `MctsConfig` for search configuration
//! - `AlphaZeroMctsAgent` implementing MCTS search with neural network guidance
//! - Supporting types: `Node`, `ChildEdge`, `MctsTree`

use azul_engine::{apply_action, legal_actions, GameState, Phase, PlayerIdx};
use mlx_rs::Array;
use rand::Rng;
use rand_distr::{Distribution, Gamma};
use std::collections::{HashMap, VecDeque};
use std::sync::mpsc;
use std::thread;

use crate::{
    ActionEncoder, ActionId, Agent, AgentInput, FeatureExtractor, Observation, ACTION_SPACE_SIZE,
};

#[cfg(feature = "profiling")]
use crate::profiling::{Timer, PROF};
#[cfg(feature = "profiling")]
use std::sync::atomic::Ordering;

/// Batch of training examples for neural network training.
/// Arrays are stacked for efficient batch processing.
#[derive(Clone)]
pub struct Batch {
    /// Batched observations: [batch_size, obs_size]
    pub observations: Array,

    /// Batched policy targets: [batch_size, ACTION_SPACE_SIZE]
    pub policy_targets: Array,

    /// Batched value targets: [batch_size]
    pub value_targets: Array,
}

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
    fn predict_single(&mut self, obs: &Observation) -> (Array, f32);

    /// Optional batch interface for training.
    ///
    /// - obs_batch: [batch, obs_size]
    /// - policy_logits: [batch, ACTION_SPACE_SIZE]
    /// - values: [batch]
    fn predict_batch(&mut self, obs_batch: &Array) -> (Array, Array);
}

/// Optional hook for keeping a dedicated inference backend in sync.
pub trait InferenceSync {
    fn sync_inference_backend(&self);
}

/// Dedicated single-threaded inference worker to keep MLX/Metal usage serialized.
struct InferenceWorker<N>
where
    N: PolicyValueNet + Send + Clone + 'static,
{
    tx: mpsc::SyncSender<WorkerMsg<N>>,
}

enum WorkerMsg<N>
where
    N: PolicyValueNet + Send + Clone + 'static,
{
    Infer {
        obs: Vec<f32>,
        batch: usize,
        obs_size: usize,
        max_batch: usize,
        eval_count: usize,
        positions: usize,
        reply: mpsc::SyncSender<(Vec<f32>, Vec<f32>)>,
    },
    UpdateNet(N),
}

impl<N> InferenceWorker<N>
where
    N: PolicyValueNet + Send + Clone + 'static,
{
    fn new(mut net: N) -> Self {
        // Bounded channel provides backpressure and improves batch formation under load.
        const INFER_CHAN_CAP: usize = 1024;
        let (tx, rx) = mpsc::sync_channel::<WorkerMsg<N>>(INFER_CHAN_CAP);
        thread::spawn(move || {
            crate::configure_mlx_for_current_thread();

            // Drain-loop batching:
            // - Block for first request
            // - Drain as many queued requests as possible (up to max_batch)
            // - Optionally wait a tiny amount once if batch is very small
            const MIN_DISPATCH_BATCH: usize = 16;
            const SMALL_BATCH_WAIT_US: u64 = 100;

            struct InferReq {
                obs: Vec<f32>,
                batch: usize,
                #[cfg(feature = "profiling")]
                eval_count: usize,
                #[cfg(feature = "profiling")]
                positions: usize,
                reply: mpsc::SyncSender<(Vec<f32>, Vec<f32>)>,
            }

            let mut backlog: VecDeque<WorkerMsg<N>> = VecDeque::new();
            let mut batch_count = 0u64;
            // Clear MLX cache periodically to prevent Metal resource exhaustion.
            const CACHE_CLEAR_INTERVAL: u64 = 100;
            loop {
                let first = match backlog.pop_front() {
                    Some(msg) => msg,
                    None => match rx.recv() {
                        Ok(msg) => msg,
                        Err(_) => break,
                    },
                };

                match first {
                    WorkerMsg::Infer {
                        obs,
                        batch,
                        obs_size,
                        max_batch,
                        eval_count,
                        positions,
                        reply,
                    } => {
                        #[cfg(not(feature = "profiling"))]
                        let _ = (eval_count, positions);
                        #[cfg(feature = "profiling")]
                        let queue_start = std::time::Instant::now();
                        let mut reqs: Vec<InferReq> = Vec::with_capacity(8);
                        let mut total_req_batch = batch;
                        let mut max_batch = max_batch;
                        reqs.push(InferReq {
                            obs,
                            batch,
                            #[cfg(feature = "profiling")]
                            eval_count,
                            #[cfg(feature = "profiling")]
                            positions,
                            reply,
                        });

                        // Drain queued requests.
                        while total_req_batch < max_batch {
                            match rx.try_recv() {
                                Ok(next) => match next {
                                    WorkerMsg::Infer {
                                        obs,
                                        batch,
                                        obs_size: next_obs,
                                        max_batch: next_max,
                                        eval_count,
                                        positions,
                                        reply,
                                    } => {
                                        if next_obs == obs_size {
                                            let new_max = max_batch.min(next_max);
                                            if total_req_batch + batch <= new_max {
                                                max_batch = new_max;
                                                total_req_batch += batch;
                                                reqs.push(InferReq {
                                                    obs,
                                                    batch,
                                                    #[cfg(feature = "profiling")]
                                                    eval_count,
                                                    #[cfg(feature = "profiling")]
                                                    positions,
                                                    reply,
                                                });
                                            } else {
                                                backlog.push_back(WorkerMsg::Infer {
                                                    obs,
                                                    batch,
                                                    obs_size: next_obs,
                                                    max_batch: next_max,
                                                    eval_count,
                                                    positions,
                                                    reply,
                                                });
                                            }
                                        } else {
                                            backlog.push_back(WorkerMsg::Infer {
                                                obs,
                                                batch,
                                                obs_size: next_obs,
                                                max_batch: next_max,
                                                eval_count,
                                                positions,
                                                reply,
                                            });
                                        }
                                    }
                                    WorkerMsg::UpdateNet(new_net) => {
                                        net = new_net;
                                    }
                                },
                                Err(mpsc::TryRecvError::Empty) => break,
                                Err(mpsc::TryRecvError::Disconnected) => return,
                            }
                        }

                        // Optional tiny wait once if batch is very small and queue is empty.
                        if total_req_batch < MIN_DISPATCH_BATCH {
                            if let Ok(next) = rx
                                .recv_timeout(std::time::Duration::from_micros(SMALL_BATCH_WAIT_US))
                            {
                                match next {
                                    WorkerMsg::Infer {
                                        obs,
                                        batch,
                                        obs_size: next_obs,
                                        max_batch: next_max,
                                        eval_count,
                                        positions,
                                        reply,
                                    } => {
                                        if next_obs == obs_size {
                                            let new_max = max_batch.min(next_max);
                                            if total_req_batch + batch <= new_max {
                                                max_batch = new_max;
                                                total_req_batch += batch;
                                                reqs.push(InferReq {
                                                    obs,
                                                    batch,
                                                    #[cfg(feature = "profiling")]
                                                    eval_count,
                                                    #[cfg(feature = "profiling")]
                                                    positions,
                                                    reply,
                                                });
                                            } else {
                                                backlog.push_back(WorkerMsg::Infer {
                                                    obs,
                                                    batch,
                                                    obs_size: next_obs,
                                                    max_batch: next_max,
                                                    eval_count,
                                                    positions,
                                                    reply,
                                                });
                                            }
                                        } else {
                                            backlog.push_back(WorkerMsg::Infer {
                                                obs,
                                                batch,
                                                obs_size: next_obs,
                                                max_batch: next_max,
                                                eval_count,
                                                positions,
                                                reply,
                                            });
                                        }
                                    }
                                    WorkerMsg::UpdateNet(new_net) => {
                                        net = new_net;
                                    }
                                }
                            }
                        }

                        // Drain again after the optional wait.
                        while total_req_batch < max_batch {
                            match rx.try_recv() {
                                Ok(next) => match next {
                                    WorkerMsg::Infer {
                                        obs,
                                        batch,
                                        obs_size: next_obs,
                                        max_batch: next_max,
                                        eval_count,
                                        positions,
                                        reply,
                                    } => {
                                        if next_obs == obs_size {
                                            let new_max = max_batch.min(next_max);
                                            if total_req_batch + batch <= new_max {
                                                max_batch = new_max;
                                                total_req_batch += batch;
                                                reqs.push(InferReq {
                                                    obs,
                                                    batch,
                                                    #[cfg(feature = "profiling")]
                                                    eval_count,
                                                    #[cfg(feature = "profiling")]
                                                    positions,
                                                    reply,
                                                });
                                            } else {
                                                backlog.push_back(WorkerMsg::Infer {
                                                    obs,
                                                    batch,
                                                    obs_size: next_obs,
                                                    max_batch: next_max,
                                                    eval_count,
                                                    positions,
                                                    reply,
                                                });
                                            }
                                        } else {
                                            backlog.push_back(WorkerMsg::Infer {
                                                obs,
                                                batch,
                                                obs_size: next_obs,
                                                max_batch: next_max,
                                                eval_count,
                                                positions,
                                                reply,
                                            });
                                        }
                                    }
                                    WorkerMsg::UpdateNet(new_net) => {
                                        net = new_net;
                                    }
                                },
                                Err(mpsc::TryRecvError::Empty) => break,
                                Err(mpsc::TryRecvError::Disconnected) => return,
                            }
                        }

                        let total_batch = total_req_batch;
                        // Reduce shape churn by rounding up to a small set of batch sizes.
                        // MLX/Metal caching can accumulate resources keyed on tensor shapes.
                        let padded_batch = total_batch.next_power_of_two().min(max_batch);
                        #[cfg(feature = "profiling")]
                        let (total_positions, total_evals) =
                            reqs.iter().fold((0usize, 0usize), |(p, e), r| {
                                (p + r.positions, e + r.eval_count)
                            });
                        let mut obs_concat = Vec::with_capacity(padded_batch * obs_size);
                        for r in reqs.iter() {
                            obs_concat.extend_from_slice(&r.obs);
                        }
                        if padded_batch > total_batch {
                            obs_concat.resize(padded_batch * obs_size, 0.0);
                        }

                        let obs_batch =
                            Array::from_slice(&obs_concat, &[padded_batch as i32, obs_size as i32]);
                        #[cfg(feature = "profiling")]
                        let worker_start = std::time::Instant::now();
                        let (policy, values) = net.predict_batch(&obs_batch);
                        // MLX arrays are lazy; evaluate both outputs together to avoid redundant work
                        // and extra synchronization points.
                        mlx_rs::transforms::eval([&policy, &values])
                            .expect("Failed to eval NN outputs");
                        #[cfg(feature = "profiling")]
                        {
                            let worker_elapsed = worker_start.elapsed().as_nanos() as u64;
                            PROF.time_nn_worker_ns
                                .fetch_add(worker_elapsed, Ordering::Relaxed);
                        }
                        #[cfg(feature = "profiling")]
                        {
                            PROF.mcts_nn_batches.fetch_add(1, Ordering::Relaxed);
                            PROF.mcts_nn_positions
                                .fetch_add(total_positions as u64, Ordering::Relaxed);
                            PROF.mcts_nn_evals
                                .fetch_add(total_evals as u64, Ordering::Relaxed);
                        }
                        let logits_all = policy.as_slice::<f32>().to_vec();
                        let values_all = values.as_slice::<f32>().to_vec();

                        // Dispatch slices back
                        let mut logits_off = 0;
                        let mut values_off = 0;
                        for r in reqs.into_iter() {
                            let logits_len = r.batch * ACTION_SPACE_SIZE;
                            let logits_slice =
                                logits_all[logits_off..logits_off + logits_len].to_vec();
                            let values_slice =
                                values_all[values_off..values_off + r.batch].to_vec();
                            let _ = r.reply.send((logits_slice, values_slice));
                            logits_off += logits_len;
                            values_off += r.batch;
                        }

                        #[cfg(feature = "profiling")]
                        {
                            let queue_elapsed = queue_start.elapsed().as_nanos() as u64;
                            PROF.time_nn_worker_queue_ns
                                .fetch_add(queue_elapsed, Ordering::Relaxed);
                        }

                        // Periodically clear MLX cache to prevent Metal resource exhaustion
                        batch_count += 1;
                        if batch_count % CACHE_CLEAR_INTERVAL == 0 {
                            unsafe {
                                mlx_sys::mlx_clear_cache();
                            }
                        }
                    }
                    WorkerMsg::UpdateNet(new_net) => {
                        net = new_net;
                    }
                }
            }
        });
        Self { tx }
    }

    fn infer(
        &self,
        obs: Vec<f32>,
        batch: usize,
        obs_size: usize,
        max_batch: usize,
        eval_count: usize,
        positions: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let (reply_tx, reply_rx) = mpsc::sync_channel(1);
        self.tx
            .send(WorkerMsg::Infer {
                obs,
                batch,
                obs_size,
                max_batch,
                eval_count,
                positions,
                reply: reply_tx,
            })
            .expect("Inference worker channel closed");
        reply_rx.recv().expect("Inference worker dropped")
    }

    fn update_net(&self, net: N) {
        // Best-effort; if worker has shut down, let it panic upstream.
        let _ = self.tx.send(WorkerMsg::UpdateNet(net));
    }
}

/// Configuration for AlphaZero-style MCTS.
#[derive(Clone, Debug)]
pub struct MctsConfig {
    /// Number of simulations per root move.
    pub num_simulations: u32,

    /// PUCT exploration constant.
    pub cpuct: f32,

    /// Root Dirichlet noise epsilon (fraction of noise vs prior).
    pub root_dirichlet_eps: f32,

    /// Dirichlet concentration parameter alpha for root noise.
    /// Only used if > 0.0.
    pub root_dirichlet_alpha: f32,

    /// Softmax temperature for turning visit counts into a policy.
    /// - For training self-play: tau=1.0
    /// - For evaluation: tau->0 (argmax)
    pub temperature: f32,

    /// Maximum search depth in playouts (safety bound).
    pub max_depth: u32,

    /// How many leaf positions to evaluate per NN batch.
    /// If 1 => sequential but still uses predict_batch(B=1) path.
    pub nn_batch_size: usize,

    /// Virtual loss magnitude for in-flight simulations.
    /// Typical: 1.0 when values are in [-1, 1].
    pub virtual_loss: f32,
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
            nn_batch_size: 32,
            virtual_loss: 1.0,
        }
    }
}

/// Index into MCTS node arena.
pub type NodeIdx = u32;

/// Edge statistics for an action from a given node.
#[derive(Clone, Debug)]
pub struct ChildEdge {
    pub action_id: ActionId,
    pub prior: f32,       // P(s, a)
    pub visit_count: u32, // N(s, a)
    pub value_sum: f32,   // W(s, a), sum of backed-up values
    /// Immediate reward for this transition, from the parent node's perspective.
    /// Populated the first time the edge is expanded (child is created).
    pub reward: f32,
    pub child: Option<NodeIdx>, // None until first expansion along this edge
    /// True if this edge has a pending expansion in the current batch.
    /// Used to prevent multiple simulations from selecting the same unexpanded edge.
    pub pending: bool,
}

/// Node in the MCTS tree.
#[derive(Clone, Debug)]
pub struct Node {
    pub state: GameState,
    pub to_play: PlayerIdx, // state.current_player at this node
    pub is_terminal: bool,  // state.phase == GameOver

    /// Edges for all legal actions at this node.
    pub children: Vec<ChildEdge>,

    /// Cumulative visit count at this node (sum over children)
    pub visit_count: u32,

    /// Value predicted when the node was expanded.
    /// None for stub nodes (before NN evaluation), Some(value) after expansion.
    pub nn_value: Option<f32>,
}

/// The MCTS tree structure.
#[derive(Clone, Debug, Default)]
pub struct MctsTree {
    pub nodes: Vec<Node>,
}

/// Path step during tree traversal for backup.
#[derive(Clone, Debug)]
struct PathStep {
    node_idx: NodeIdx,
    child_idx: usize,
}

/// In-flight simulation tracking for batched MCTS.
/// Tracks path from root to leaf for backup after batch NN evaluation.
struct PendingSim {
    /// Path from root to leaf (for backup).
    path: Vec<PathStep>,
    /// Index of the leaf node reached.
    leaf_idx: NodeIdx,
    /// Value at the leaf (filled after NN eval or terminal computation).
    leaf_value: f32,
    /// Slot in the batch array if this leaf needs NN evaluation.
    /// None for terminal nodes.
    eval_slot: Option<usize>,
}

/// AlphaZero MCTS Agent that performs tree search guided by a neural network.
///
/// Clone is implemented to support parallel self-play games.
/// Each cloned agent maintains independent state (though the underlying neural
/// network weights are shared via MLX's copy-on-write semantics).
pub struct AlphaZeroMctsAgent<F, N>
where
    F: FeatureExtractor,
    N: PolicyValueNet + Send + Clone + 'static,
{
    pub config: MctsConfig,
    pub features: F,
    pub net: N,
    inference: std::sync::Arc<InferenceWorker<N>>,
}

impl<F, N> Clone for AlphaZeroMctsAgent<F, N>
where
    F: FeatureExtractor + Clone,
    N: PolicyValueNet + Clone + Send + 'static,
{
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            features: self.features.clone(),
            net: self.net.clone(),
            inference: self.inference.clone(),
        }
    }
}

impl<F, N> AlphaZeroMctsAgent<F, N>
where
    F: FeatureExtractor,
    N: PolicyValueNet + Send + Clone + 'static,
{
    pub fn new(config: MctsConfig, features: F, net: N) -> Self {
        let inference = std::sync::Arc::new(InferenceWorker::new(net.clone()));
        Self {
            config,
            features,
            net,
            inference,
        }
    }

    /// Refresh the inference worker with current network weights.
    pub fn sync_inference_backend(&self) {
        self.inference.update_net(self.net.clone());
    }

    /// Create a lightweight stub node without NN evaluation.
    /// Children remain empty until expand_node_from_nn is called.
    fn create_node_stub(&self, tree: &mut MctsTree, state: GameState) -> NodeIdx {
        let to_play = state.current_player;
        let is_terminal = state.phase == Phase::GameOver;

        let node = Node {
            state,
            to_play,
            is_terminal,
            children: Vec::new(),
            visit_count: 0,
            nn_value: None,
        };
        let idx = tree.nodes.len() as NodeIdx;
        tree.nodes.push(node);
        #[cfg(feature = "profiling")]
        PROF.mcts_nodes_created.fetch_add(1, Ordering::Relaxed);
        idx
    }

    /// Expand a node given NN outputs (policy logits row + value).
    /// This is idempotent - won't re-expand if children already exist.
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

        // Only expand once (idempotent)
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

        node.children = priors
            .into_iter()
            .map(|(id, prior)| ChildEdge {
                action_id: id,
                prior,
                visit_count: 0,
                value_sum: 0.0,
                reward: 0.0,
                child: None,
                pending: false,
            })
            .collect();
    }

    /// Select a leaf node for evaluation, applying virtual loss along the path.
    /// Creates stub nodes for unexpanded edges.
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

            // Stop at terminal or unexpanded nodes
            if is_terminal || !has_children {
                break;
            }

            // Choose child via PUCT
            let child_idx = {
                let node = &tree.nodes[current_idx as usize];
                select_child(node, self.config.cpuct)
            };

            // Record and apply virtual loss on the chosen edge
            let step = PathStep {
                node_idx: current_idx,
                child_idx,
            };
            apply_virtual_loss(tree, &step, self.config.virtual_loss);
            path.push(step);

            // Descend or expand stub
            let next_child_opt = tree.nodes[current_idx as usize].children[child_idx].child;
            if let Some(next_idx) = next_child_opt {
                current_idx = next_idx;
                continue;
            }

            // Expand edge by creating child stub node
            let (parent_state, parent_to_play, action_id, my_before, opp_before) = {
                let parent = &tree.nodes[current_idx as usize];
                let parent_to_play = parent.to_play;
                let n = parent.state.num_players as usize;
                debug_assert!(
                    n == 2,
                    "AlphaZeroMctsAgent v1 only supports 2 players, got {n}"
                );
                let my_before = parent.state.players[parent_to_play as usize].score as f32;
                let opp_before = parent.state.players[1 - parent_to_play as usize].score as f32;
                let action_id = parent.children[child_idx].action_id;
                (
                    parent.state.clone(),
                    parent_to_play,
                    action_id,
                    my_before,
                    opp_before,
                )
            };
            let action = ActionEncoder::decode(action_id);

            let step_result = apply_action(parent_state, action, rng)
                .expect("MCTS should only expand legal actions");

            // Immediate reward from parent perspective: Δ(score_diff)/20.
            let my_after = step_result.state.players[parent_to_play as usize].score as f32;
            let opp_after = step_result.state.players[1 - parent_to_play as usize].score as f32;
            let reward = ((my_after - my_before) - (opp_after - opp_before)) / 20.0;

            let new_idx = self.create_node_stub(tree, step_result.state);
            {
                let edge = &mut tree.nodes[current_idx as usize].children[child_idx];
                edge.child = Some(new_idx);
                edge.reward = reward;
                // Mark edge as pending to prevent other simulations from selecting same leaf
                edge.pending = true;
            }

            current_idx = new_idx;
            break;
        }

        PendingSim {
            path,
            leaf_idx: current_idx,
            leaf_value: 0.0, // filled later
            eval_slot: None, // filled later
        }
    }

    /// Process a batch of pending simulations with batched NN evaluation.
    /// Uses provided scratch buffers to avoid per-call allocations.
    fn process_batch(
        &mut self,
        tree: &mut MctsTree,
        sims: &mut [PendingSim],
        unique_leafs: &mut Vec<NodeIdx>,
        leaf_to_slot: &mut HashMap<NodeIdx, usize>,
        obs_scratch: &mut Vec<f32>,
    ) {
        // 1) Identify which leaves need NN evaluation; compute terminal ones immediately
        unique_leafs.clear();
        leaf_to_slot.clear();

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

        // 2) If any NN leaves exist, batch them
        if !unique_leafs.is_empty() {
            let obs_size = self.features.obs_size();
            let b = unique_leafs.len();

            // Build [B, obs_size] contiguous buffer using scratch
            obs_scratch.clear();
            for &node_idx in unique_leafs.iter() {
                let node = &tree.nodes[node_idx as usize];
                let obs = self.features.encode(&node.state, node.to_play);
                obs_scratch.extend_from_slice(obs.as_slice::<f32>());
            }

            // NN inference with profiling, executed on the dedicated worker thread.
            let (logits_vec, values_vec) = {
                #[cfg(feature = "profiling")]
                let _t = Timer::new(&PROF.time_mcts_nn_eval_ns);
                // Allow the shared inference worker to coalesce across parallel self-play games.
                // Bigger batches amortize MLX/Metal sync overhead.
                let max_batch = self
                    .config
                    .nn_batch_size
                    .max(b)
                    .saturating_mul(8)
                    .max(1)
                    .min(512);
                self.inference.infer(
                    obs_scratch.clone(),
                    b,
                    obs_size,
                    max_batch,
                    unique_leafs.len(),
                    b,
                )
            };
            let logits = &logits_vec;
            let values = &values_vec;

            // 3) Expand each unique leaf using its logits row, and store leaf value
            for (slot, &node_idx) in unique_leafs.iter().enumerate() {
                let value = values[slot];
                let row_start = slot * ACTION_SPACE_SIZE;
                let row_end = row_start + ACTION_SPACE_SIZE;
                let logits_row = &logits[row_start..row_end];

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

        // 4) Backup each simulation result and clear pending flags
        for sim in sims.iter() {
            backup_with_virtual_loss(tree, &sim.path, sim.leaf_value, self.config.virtual_loss);

            // Clear pending flag on the edge that led to this leaf
            if let Some(last_step) = sim.path.last() {
                tree.nodes[last_step.node_idx as usize].children[last_step.child_idx].pending =
                    false;
            }
        }
    }

    /// Main MCTS search entrypoint.
    /// Returns improved policy over ACTION_SPACE_SIZE.
    /// Uses batched NN inference for efficiency.
    fn run_search(
        &mut self,
        root_state: &GameState,
        rng: &mut impl Rng,
    ) -> [f32; ACTION_SPACE_SIZE] {
        crate::configure_mlx_for_current_thread();

        #[cfg(feature = "profiling")]
        let _t = Timer::new(&PROF.time_mcts_search_ns);
        #[cfg(feature = "profiling")]
        PROF.mcts_searches.fetch_add(1, Ordering::Relaxed);

        // Initialize tree with root stub
        let mut tree = MctsTree::default();
        let root_idx = self.create_node_stub(&mut tree, root_state.clone());

        // Expand root using predict_batch(B=1) to avoid predict_single overhead
        {
            let obs_size = self.features.obs_size();
            let root = &tree.nodes[root_idx as usize];
            let obs = self.features.encode(&root.state, root.to_play);
            let obs_data = obs.as_slice::<f32>().to_vec();
            let (logits_vec, values_vec) = {
                #[cfg(feature = "profiling")]
                let _t = Timer::new(&PROF.time_mcts_nn_eval_ns);
                // See process_batch(): keep headroom for cross-game coalescing.
                let max_batch = self
                    .config
                    .nn_batch_size
                    .max(1)
                    .saturating_mul(8)
                    .max(1)
                    .min(512);
                self.inference.infer(obs_data, 1, obs_size, max_batch, 1, 1)
            };

            let logits_row = &logits_vec[0..ACTION_SPACE_SIZE];
            let value = values_vec[0];

            self.expand_node_from_nn(&mut tree, root_idx, logits_row, value);
        }

        // Add Dirichlet noise to root priors
        if self.config.root_dirichlet_alpha > 0.0
            && !tree.nodes[root_idx as usize].children.is_empty()
        {
            add_dirichlet_noise(
                &mut tree.nodes[root_idx as usize].children,
                self.config.root_dirichlet_alpha,
                self.config.root_dirichlet_eps,
                rng,
            );
        }

        // Batched simulations - preallocate scratch buffers
        let total = self.config.num_simulations as usize;
        let batch_size = self.config.nn_batch_size.max(1).min(total);
        let obs_size = self.features.obs_size();

        // Scratch buffers reused across all batches
        let mut sims: Vec<PendingSim> = Vec::with_capacity(batch_size);
        let mut unique_leafs: Vec<NodeIdx> = Vec::with_capacity(batch_size);
        let mut leaf_to_slot: HashMap<NodeIdx, usize> = HashMap::with_capacity(batch_size);
        let mut obs_scratch: Vec<f32> = Vec::with_capacity(batch_size * obs_size);

        let mut done = 0;
        while done < total {
            let n = (total - done).min(batch_size);

            sims.clear();
            for _ in 0..n {
                #[cfg(feature = "profiling")]
                PROF.mcts_simulations.fetch_add(1, Ordering::Relaxed);

                sims.push(self.select_leaf(&mut tree, root_idx, rng));
            }

            self.process_batch(
                &mut tree,
                &mut sims[..],
                &mut unique_leafs,
                &mut leaf_to_slot,
                &mut obs_scratch,
            );
            done += n;
        }

        // Extract policy from root visit counts
        let root = &tree.nodes[root_idx as usize];
        let mut counts = [0.0f32; ACTION_SPACE_SIZE];
        for edge in &root.children {
            counts[edge.action_id as usize] = edge.visit_count as f32;
        }

        apply_temperature(&counts, self.config.temperature)
    }
}

impl<F, N> InferenceSync for AlphaZeroMctsAgent<F, N>
where
    F: FeatureExtractor,
    N: PolicyValueNet + Send + Clone + 'static,
{
    fn sync_inference_backend(&self) {
        self.inference.update_net(self.net.clone());
    }
}

/// PUCT selection: choose child with highest Q + U score.
/// Skips unexpanded edges with pending=true to improve batch diversity.
fn select_child(node: &Node, cpuct: f32) -> usize {
    let mut best_idx = 0;
    let mut best_score = f32::NEG_INFINITY;

    let parent_n = node.visit_count.max(1) as f32;

    for (i, edge) in node.children.iter().enumerate() {
        // Skip unexpanded edges that are already pending in this batch
        // This forces different simulations to explore different paths
        if edge.pending && edge.child.is_none() {
            continue;
        }

        let q = if edge.visit_count > 0 {
            edge.value_sum / edge.visit_count as f32
        } else {
            0.0
        };

        let u = cpuct * edge.prior * (parent_n.sqrt() / (1.0 + edge.visit_count as f32));

        let score = q + u;
        if score > best_score {
            best_score = score;
            best_idx = i;
        }
    }
    best_idx
}

/// Apply virtual loss to an edge to discourage other in-flight simulations
/// from selecting the same path.
fn apply_virtual_loss(tree: &mut MctsTree, step: &PathStep, vloss: f32) {
    let node = &mut tree.nodes[step.node_idx as usize];
    let edge = &mut node.children[step.child_idx];

    // Virtual visit
    edge.visit_count += 1;
    node.visit_count += 1;

    // Penalize Q to discourage collisions
    edge.value_sum -= vloss;
}

/// Revert virtual loss from an edge.
fn revert_virtual_loss(tree: &mut MctsTree, step: &PathStep, vloss: f32) {
    let node = &mut tree.nodes[step.node_idx as usize];
    let edge = &mut node.children[step.child_idx];

    // Catch underflow bugs in debug
    debug_assert!(edge.visit_count > 0, "virtual loss underflow on edge");
    debug_assert!(node.visit_count > 0, "virtual loss underflow on node");

    // Revert virtual visit
    edge.visit_count -= 1;
    node.visit_count -= 1;

    // Revert virtual penalty
    edge.value_sum += vloss;
}

/// Backup with virtual loss: first revert virtual loss, then apply real backup.
fn backup_with_virtual_loss(tree: &mut MctsTree, path: &[PathStep], leaf_value: f32, vloss: f32) {
    // Value is always interpreted as "from the perspective of to_play at the
    // current node in the traversal".
    //
    // Azul does *not* guarantee strict alternation of turns across all moves
    // (the first-player marker can cause the same player to act twice across
    // the round boundary). Therefore we must only negate when the player to
    // move changes between parent and child.
    let mut value = leaf_value;

    for step in path.iter().rev() {
        // Remove virtual loss for this in-flight sim.
        revert_virtual_loss(tree, step, vloss);

        // Read parent/child to_play without holding a mutable borrow.
        let (parent_to_play, child_to_play, reward) = {
            let parent = &tree.nodes[step.node_idx as usize];
            let edge = &parent.children[step.child_idx];
            let child_idx = edge
                .child
                .expect("backup path must reference an expanded child node");
            (
                parent.to_play,
                tree.nodes[child_idx as usize].to_play,
                edge.reward,
            )
        };

        // Transform from child perspective -> parent perspective.
        if parent_to_play != child_to_play {
            value = -value;
        }

        // Include dense reward on this edge.
        value = (reward + value).clamp(-1.0, 1.0);

        // Apply real backup (value is now from parent perspective).
        let node = &mut tree.nodes[step.node_idx as usize];
        let edge = &mut node.children[step.child_idx];
        edge.visit_count += 1;
        edge.value_sum += value;
        node.visit_count += 1;
    }
}

/// Terminal node value for dense-reward MCTS.
///
/// When backing up `Q(s,a) = r(s,a) + V(s')`, all score changes (including endgame
/// bonuses) are captured by the per-step reward on the final transition, so the
/// remaining value at a terminal state is 0.
fn compute_terminal_value(_state: &GameState, _to_play: PlayerIdx) -> f32 {
    0.0
}

/// Softmax over legal logits to produce priors.
fn softmax(logits: &[(ActionId, f32)]) -> Vec<(ActionId, f32)> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max_logit = logits
        .iter()
        .map(|(_, l)| *l)
        .fold(f32::NEG_INFINITY, f32::max);

    let mut exps: Vec<(ActionId, f32)> = Vec::with_capacity(logits.len());
    let mut sum = 0.0;

    for (id, l) in logits {
        let e = (l - max_logit).exp();
        exps.push((*id, e));
        sum += e;
    }

    let sum = sum.max(1e-8);
    exps.into_iter().map(|(id, e)| (id, e / sum)).collect()
}

/// Apply temperature to visit counts to get policy.
fn apply_temperature(counts: &[f32; ACTION_SPACE_SIZE], tau: f32) -> [f32; ACTION_SPACE_SIZE] {
    let mut pi = [0.0f32; ACTION_SPACE_SIZE];

    if tau <= 1e-6 {
        // Argmax: set pi[a*] = 1, others 0
        let mut best_idx = 0;
        let mut best_count = f32::NEG_INFINITY;
        for (i, &c) in counts.iter().enumerate() {
            if c > best_count {
                best_count = c;
                best_idx = i;
            }
        }
        if best_count > 0.0 {
            pi[best_idx] = 1.0;
        }
    } else {
        // pi(a) proportional to N(a)^(1/tau)
        let inv_tau = 1.0 / tau;
        let mut sum = 0.0;
        for (i, &c) in counts.iter().enumerate() {
            if c > 0.0 {
                let p = c.powf(inv_tau);
                pi[i] = p;
                sum += p;
            }
        }
        if sum > 0.0 {
            for p in &mut pi {
                *p /= sum;
            }
        }
    }

    pi
}

/// Sample an action from the policy distribution.
fn sample_from_policy(pi: &[f32; ACTION_SPACE_SIZE], tau: f32, rng: &mut impl Rng) -> usize {
    if tau <= 1e-6 {
        // Argmax
        pi.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    } else {
        // Sample according to distribution
        let r: f32 = rng.random();
        let mut cumsum = 0.0;
        for (i, &p) in pi.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                return i;
            }
        }
        // Fallback to last non-zero
        pi.iter()
            .enumerate()
            .rev()
            .find(|(_, &p)| p > 0.0)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

/// Add Dirichlet noise to root priors for exploration.
/// Uses gamma sampling method: sample x_i ~ Gamma(alpha, 1), normalize to get Dir(alpha).
fn add_dirichlet_noise(edges: &mut [ChildEdge], alpha: f32, eps: f32, rng: &mut impl Rng) {
    if edges.is_empty() || alpha <= 0.0 {
        return;
    }

    // Sample Dirichlet using gamma distribution method
    let gamma = match Gamma::new(alpha as f64, 1.0) {
        Ok(g) => g,
        Err(_) => return, // Skip noise if Gamma creation fails
    };

    let mut noise: Vec<f64> = Vec::with_capacity(edges.len());
    let mut sum = 0.0;
    for _ in 0..edges.len() {
        let sample = gamma.sample(rng);
        noise.push(sample);
        sum += sample;
    }

    // Normalize to get Dirichlet samples
    if sum > 0.0 {
        for x in &mut noise {
            *x /= sum;
        }
    }

    // Mix noise with priors: P'(s,a) = (1-eps)*P(s,a) + eps*eta_a
    for (edge, &eta) in edges.iter_mut().zip(noise.iter()) {
        edge.prior = (1.0 - eps) * edge.prior + eps * (eta as f32);
    }
}

impl<F, N> Agent for AlphaZeroMctsAgent<F, N>
where
    F: FeatureExtractor,
    N: PolicyValueNet + Clone + Send + 'static,
{
    fn select_action(&mut self, input: &AgentInput, rng: &mut impl Rng) -> ActionId {
        let state = input
            .state
            .expect("AlphaZeroMctsAgent requires full GameState in AgentInput.state");
        assert_eq!(
            state.num_players, 2,
            "AlphaZeroMctsAgent v1 only supports 2 players"
        );

        // Run search to get improved policy over ACTION_SPACE_SIZE
        let pi = self.run_search(state, rng);

        // Mask illegal actions using input.legal_action_mask
        let mut masked_pi = [0.0f32; ACTION_SPACE_SIZE];
        for (id, &prob) in pi.iter().enumerate() {
            if id < input.legal_action_mask.len() && input.legal_action_mask[id] {
                masked_pi[id] = prob;
            }
        }

        // Re-normalize
        let sum: f32 = masked_pi.iter().sum();
        if sum > 0.0 {
            for p in &mut masked_pi {
                *p /= sum;
            }
        }

        // Sample according to temperature
        let action_id = sample_from_policy(&masked_pi, self.config.temperature, rng);
        action_id as ActionId
    }
}

impl<F, N> crate::alphazero::training::MctsAgentExt for AlphaZeroMctsAgent<F, N>
where
    F: FeatureExtractor,
    N: PolicyValueNet + Clone + Send + 'static,
{
    fn select_action_and_policy(
        &mut self,
        input: &AgentInput,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> crate::alphazero::training::MctsSearchResult {
        let state = input
            .state
            .expect("AlphaZeroMctsAgent requires full GameState in AgentInput.state");
        assert_eq!(
            state.num_players, 2,
            "AlphaZeroMctsAgent v1 only supports 2 players"
        );

        // Temporarily override temperature for this search
        let original_temp = self.config.temperature;
        self.config.temperature = temperature;

        // Run search to get improved policy over ACTION_SPACE_SIZE
        let pi = self.run_search(state, rng);

        // Restore original temperature
        self.config.temperature = original_temp;

        // Mask illegal actions using input.legal_action_mask
        let mut masked_pi = [0.0f32; ACTION_SPACE_SIZE];
        for (id, &prob) in pi.iter().enumerate() {
            if id < input.legal_action_mask.len() && input.legal_action_mask[id] {
                masked_pi[id] = prob;
            }
        }

        // Re-normalize
        let sum: f32 = masked_pi.iter().sum();
        if sum > 0.0 {
            for p in &mut masked_pi {
                *p /= sum;
            }
        }

        // Sample according to temperature
        let action_id = sample_from_policy(&masked_pi, temperature, rng);

        crate::alphazero::training::MctsSearchResult {
            action: action_id as ActionId,
            policy: masked_pi.to_vec(),
        }
    }
}

impl<F, N> crate::alphazero::training::TrainableModel for AlphaZeroMctsAgent<F, N>
where
    F: FeatureExtractor,
    N: PolicyValueNet + crate::alphazero::training::TrainableModel + Send + Clone + 'static,
{
    fn param_count(&self) -> usize {
        crate::alphazero::training::TrainableModel::param_count(&self.net)
    }

    fn parameters(&self) -> Vec<Array> {
        crate::alphazero::training::TrainableModel::parameters(&self.net)
    }

    fn forward(&mut self, obs: &Array) -> (Array, Array) {
        crate::alphazero::training::TrainableModel::forward(&mut self.net, obs)
    }

    fn apply_gradients(&mut self, learning_rate: f32, grads: &[Array]) {
        crate::alphazero::training::TrainableModel::apply_gradients(
            &mut self.net,
            learning_rate,
            grads,
        );
    }

    fn eval_parameters(&self) {
        crate::alphazero::training::TrainableModel::eval_parameters(&self.net)
    }

    fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        crate::alphazero::training::TrainableModel::save(&self.net, path)
    }

    fn load(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        let res = crate::alphazero::training::TrainableModel::load(&mut self.net, path);
        if res.is_ok() {
            self.inference.update_net(self.net.clone());
        }
        res
    }
}

impl<F, N> mlx_rs::module::ModuleParameters for AlphaZeroMctsAgent<F, N>
where
    F: FeatureExtractor,
    N: PolicyValueNet + mlx_rs::module::ModuleParameters + Send + Clone + 'static,
{
    fn num_parameters(&self) -> usize {
        mlx_rs::module::ModuleParameters::num_parameters(&self.net)
    }

    fn parameters(&self) -> mlx_rs::module::ModuleParamRef<'_> {
        mlx_rs::module::ModuleParameters::parameters(&self.net)
    }

    fn parameters_mut(&mut self) -> mlx_rs::module::ModuleParamMut<'_> {
        mlx_rs::module::ModuleParameters::parameters_mut(&mut self.net)
    }

    fn trainable_parameters(&self) -> mlx_rs::module::ModuleParamRef<'_> {
        mlx_rs::module::ModuleParameters::trainable_parameters(&self.net)
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        mlx_rs::module::ModuleParameters::freeze_parameters(&mut self.net, recursive)
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        mlx_rs::module::ModuleParameters::unfreeze_parameters(&mut self.net, recursive)
    }

    fn all_frozen(&self) -> Option<bool> {
        mlx_rs::module::ModuleParameters::all_frozen(&self.net)
    }

    fn any_frozen(&self) -> Option<bool> {
        mlx_rs::module::ModuleParameters::any_frozen(&self.net)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphazero::training::MctsAgentExt;
    use crate::{AzulEnv, BasicFeatureExtractor, EnvConfig, Environment, RewardScheme};
    use rand::SeedableRng;

    /// Dummy neural network for MCTS testing.
    /// Returns fixed priors and value, useful for testing MCTS logic independently.
    #[derive(Clone)]
    pub struct DummyNet {
        pub priors: [f32; ACTION_SPACE_SIZE],
        pub value: f32,
    }

    impl DummyNet {
        pub fn uniform(value: f32) -> Self {
            Self {
                priors: [1.0 / ACTION_SPACE_SIZE as f32; ACTION_SPACE_SIZE],
                value,
            }
        }
    }

    impl PolicyValueNet for DummyNet {
        fn predict_single(&mut self, _obs: &Observation) -> (Array, f32) {
            let arr = Array::from_slice(&self.priors, &[ACTION_SPACE_SIZE as i32]);
            (arr, self.value)
        }

        fn predict_batch(&mut self, obs_batch: &Array) -> (Array, Array) {
            let batch_size = obs_batch.shape()[0] as usize;

            // Tile priors across batch
            let mut policies = Vec::with_capacity(batch_size * ACTION_SPACE_SIZE);
            for _ in 0..batch_size {
                policies.extend_from_slice(&self.priors);
            }

            // Same value for all positions
            let values = vec![self.value; batch_size];

            let policy_arr =
                Array::from_slice(&policies, &[batch_size as i32, ACTION_SPACE_SIZE as i32]);
            let values_arr = Array::from_slice(&values, &[batch_size as i32]);
            (policy_arr, values_arr)
        }
    }

    #[test]
    fn test_mcts_config_default() {
        let config = MctsConfig::default();
        assert_eq!(config.num_simulations, 256);
        assert_eq!(config.cpuct, 1.5);
        assert_eq!(config.temperature, 1.0);
        assert_eq!(config.max_depth, 200);
        // Batching defaults
        assert_eq!(config.nn_batch_size, 32);
        assert_eq!(config.virtual_loss, 1.0);
    }

    #[test]
    fn test_softmax_basic() {
        let logits = vec![(0, 1.0f32), (1, 2.0), (2, 3.0)];
        let priors = softmax(&logits);

        assert_eq!(priors.len(), 3);
        let sum: f32 = priors.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Higher logit should have higher prior
        assert!(priors[2].1 > priors[1].1);
        assert!(priors[1].1 > priors[0].1);
    }

    #[test]
    fn test_softmax_empty() {
        let logits: Vec<(ActionId, f32)> = vec![];
        let priors = softmax(&logits);
        assert!(priors.is_empty());
    }

    #[test]
    fn test_apply_temperature_argmax() {
        let mut counts = [0.0f32; ACTION_SPACE_SIZE];
        counts[10] = 100.0;
        counts[20] = 50.0;
        counts[30] = 25.0;

        let pi = apply_temperature(&counts, 0.0);

        assert_eq!(pi[10], 1.0);
        assert_eq!(pi[20], 0.0);
        assert_eq!(pi[30], 0.0);
    }

    #[test]
    fn test_apply_temperature_uniform() {
        let mut counts = [0.0f32; ACTION_SPACE_SIZE];
        counts[10] = 100.0;
        counts[20] = 100.0;

        let pi = apply_temperature(&counts, 1.0);

        assert!((pi[10] - 0.5).abs() < 1e-5);
        assert!((pi[20] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_sample_from_policy_argmax() {
        let mut pi = [0.0f32; ACTION_SPACE_SIZE];
        pi[42] = 1.0;

        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
        let action = sample_from_policy(&pi, 0.0, &mut rng);

        assert_eq!(action, 42);
    }

    #[test]
    fn test_mcts_agent_selects_legal_action() {
        let config = EnvConfig {
            num_players: 2,
            reward_scheme: RewardScheme::TerminalOnly,
            include_full_state_in_step: true,
        };
        let features = BasicFeatureExtractor::new(2);
        let mut env = AzulEnv::new(config, features.clone());

        let dummy_net = DummyNet::uniform(0.0);
        let mcts_config = MctsConfig {
            num_simulations: 16,
            ..MctsConfig::default()
        };
        let mut agent = AlphaZeroMctsAgent::new(mcts_config, features, dummy_net);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let step = env.reset(&mut rng);

        let input = AgentInput {
            observation: &step.observations[step.current_player as usize],
            legal_action_mask: &step.legal_action_mask,
            current_player: step.current_player,
            state: step.state.as_ref(),
        };

        let action_id = agent.select_action(&input, &mut rng);
        assert!(
            step.legal_action_mask[action_id as usize],
            "MCTS agent should only select legal actions"
        );
    }

    #[test]
    fn test_mcts_never_selects_illegal_action() {
        let config = EnvConfig {
            num_players: 2,
            reward_scheme: RewardScheme::TerminalOnly,
            include_full_state_in_step: true,
        };
        let features = BasicFeatureExtractor::new(2);
        let mut env = AzulEnv::new(config, features.clone());

        let dummy_net = DummyNet::uniform(0.0);
        let mcts_config = MctsConfig {
            num_simulations: 8,
            ..MctsConfig::default()
        };
        let mut agent = AlphaZeroMctsAgent::new(mcts_config, features, dummy_net);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let step = env.reset(&mut rng);

        let input = AgentInput {
            observation: &step.observations[step.current_player as usize],
            legal_action_mask: &step.legal_action_mask,
            current_player: step.current_player,
            state: step.state.as_ref(),
        };

        // Run multiple times to increase confidence
        for _ in 0..20 {
            let action_id = agent.select_action(&input, &mut rng);
            assert!(
                step.legal_action_mask[action_id as usize],
                "MCTS should never select illegal action {}",
                action_id
            );
        }
    }

    #[test]
    fn test_mcts_full_game_smoke() {
        let config = EnvConfig {
            num_players: 2,
            reward_scheme: RewardScheme::TerminalOnly,
            include_full_state_in_step: true,
        };
        let features = BasicFeatureExtractor::new(2);
        let mut env = AzulEnv::new(config, features.clone());

        let dummy_net = DummyNet::uniform(0.0);
        let mcts_config = MctsConfig {
            num_simulations: 4, // Keep small for speed
            ..MctsConfig::default()
        };
        let mut agent = AlphaZeroMctsAgent::new(mcts_config, features, dummy_net);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut step = env.reset(&mut rng);

        let mut steps = 0;
        while !step.done && steps < 500 {
            let p = step.current_player as usize;
            let input = AgentInput {
                observation: &step.observations[p],
                legal_action_mask: &step.legal_action_mask,
                current_player: step.current_player,
                state: step.state.as_ref(),
            };

            let action_id = agent.select_action(&input, &mut rng);
            step = env.step(action_id, &mut rng).expect("step should succeed");
            steps += 1;
        }

        assert!(step.done, "Game should complete");
        assert_eq!(
            env.game_state.phase,
            azul_engine::Phase::GameOver,
            "Game should be over"
        );
    }

    #[test]
    fn test_mcts_determinism() {
        let config = EnvConfig {
            num_players: 2,
            reward_scheme: RewardScheme::TerminalOnly,
            include_full_state_in_step: true,
        };

        // Run 1
        let features1 = BasicFeatureExtractor::new(2);
        let mut env1 = AzulEnv::new(config.clone(), features1.clone());
        let dummy_net1 = DummyNet::uniform(0.0);
        let mcts_config = MctsConfig {
            num_simulations: 8,
            root_dirichlet_alpha: 0.0, // Disable noise for determinism
            ..MctsConfig::default()
        };
        let mut agent1 = AlphaZeroMctsAgent::new(mcts_config.clone(), features1, dummy_net1);

        let mut rng1 = rand::rngs::StdRng::seed_from_u64(42);
        let mut step1 = env1.reset(&mut rng1);
        let mut actions1 = Vec::new();

        for _ in 0..10 {
            if step1.done {
                break;
            }
            let p = step1.current_player as usize;
            let input = AgentInput {
                observation: &step1.observations[p],
                legal_action_mask: &step1.legal_action_mask,
                current_player: step1.current_player,
                state: step1.state.as_ref(),
            };

            let action_id = agent1.select_action(&input, &mut rng1);
            actions1.push(action_id);
            step1 = env1
                .step(action_id, &mut rng1)
                .expect("step should succeed");
        }

        // Run 2 with same seeds
        let features2 = BasicFeatureExtractor::new(2);
        let mut env2 = AzulEnv::new(config, features2.clone());
        let dummy_net2 = DummyNet::uniform(0.0);
        let mut agent2 = AlphaZeroMctsAgent::new(mcts_config, features2, dummy_net2);

        let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);
        let mut step2 = env2.reset(&mut rng2);
        let mut actions2 = Vec::new();

        for _ in 0..10 {
            if step2.done {
                break;
            }
            let p = step2.current_player as usize;
            let input = AgentInput {
                observation: &step2.observations[p],
                legal_action_mask: &step2.legal_action_mask,
                current_player: step2.current_player,
                state: step2.state.as_ref(),
            };

            let action_id = agent2.select_action(&input, &mut rng2);
            actions2.push(action_id);
            step2 = env2
                .step(action_id, &mut rng2)
                .expect("step should succeed");
        }

        assert_eq!(
            actions1, actions2,
            "MCTS should be deterministic with same seeds"
        );
    }

    #[test]
    fn test_agent_input_state_is_some() {
        let config = EnvConfig {
            num_players: 2,
            reward_scheme: RewardScheme::TerminalOnly,
            include_full_state_in_step: true,
        };
        let features = BasicFeatureExtractor::new(2);
        let mut env = AzulEnv::new(config, features);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let step = env.reset(&mut rng);

        assert!(
            step.state.is_some(),
            "EnvStep.state should be Some when include_full_state_in_step is true"
        );
    }

    #[test]
    fn test_batched_mcts_invariants() {
        let config = EnvConfig {
            num_players: 2,
            reward_scheme: RewardScheme::TerminalOnly,
            include_full_state_in_step: true,
        };
        let features = BasicFeatureExtractor::new(2);
        let mut env = AzulEnv::new(config, features.clone());

        // Use explicit batching config
        let mcts_config = MctsConfig {
            num_simulations: 50,
            nn_batch_size: 8, // Batched!
            virtual_loss: 1.0,
            ..MctsConfig::default()
        };

        let dummy_net = DummyNet::uniform(0.0);
        let mut agent = AlphaZeroMctsAgent::new(mcts_config, features, dummy_net);

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let step = env.reset(&mut rng);

        let input = AgentInput {
            observation: &step.observations[step.current_player as usize],
            legal_action_mask: &step.legal_action_mask,
            current_player: step.current_player,
            state: step.state.as_ref(),
        };

        let result = agent.select_action_and_policy(&input, 1.0, &mut rng);

        // Policy sums to ~1 (after masking and renormalization)
        let policy_sum: f32 = result.policy.iter().sum();
        assert!(
            (policy_sum - 1.0).abs() < 0.01,
            "Policy should sum to 1, got {}",
            policy_sum
        );

        // Action is legal
        assert!(
            step.legal_action_mask[result.action as usize],
            "Selected action {} must be legal",
            result.action
        );
    }

    #[test]
    fn test_backup_flips_value_when_to_play_changes() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let state = azul_engine::new_game(2, 0, &mut rng);

        let mut tree = MctsTree::default();
        tree.nodes.push(Node {
            state: state.clone(),
            to_play: 0,
            is_terminal: false,
            children: vec![ChildEdge {
                action_id: 0,
                prior: 1.0,
                visit_count: 0,
                value_sum: 0.0,
                reward: 0.0,
                child: Some(1),
                pending: false,
            }],
            visit_count: 0,
            nn_value: None,
        });
        tree.nodes.push(Node {
            state,
            to_play: 1,
            is_terminal: false,
            children: Vec::new(),
            visit_count: 0,
            nn_value: None,
        });

        let step = PathStep {
            node_idx: 0,
            child_idx: 0,
        };
        apply_virtual_loss(&mut tree, &step, 1.0);
        backup_with_virtual_loss(&mut tree, &[step], 0.5, 1.0);

        let edge = &tree.nodes[0].children[0];
        assert_eq!(edge.visit_count, 1);
        assert!((edge.value_sum - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_backup_keeps_value_when_to_play_same() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let state = azul_engine::new_game(2, 0, &mut rng);

        let mut tree = MctsTree::default();
        tree.nodes.push(Node {
            state: state.clone(),
            to_play: 0,
            is_terminal: false,
            children: vec![ChildEdge {
                action_id: 0,
                prior: 1.0,
                visit_count: 0,
                value_sum: 0.0,
                reward: 0.0,
                child: Some(1),
                pending: false,
            }],
            visit_count: 0,
            nn_value: None,
        });
        tree.nodes.push(Node {
            state,
            to_play: 0,
            is_terminal: false,
            children: Vec::new(),
            visit_count: 0,
            nn_value: None,
        });

        let step = PathStep {
            node_idx: 0,
            child_idx: 0,
        };
        apply_virtual_loss(&mut tree, &step, 1.0);
        backup_with_virtual_loss(&mut tree, &[step], 0.5, 1.0);

        let edge = &tree.nodes[0].children[0];
        assert_eq!(edge.visit_count, 1);
        assert!((edge.value_sum - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_mcts_prefers_higher_dense_reward_in_terminal_puzzle() {
        use azul_engine::{Action, Color, DraftDestination, DraftSource, Token, WALL_DEST_COL};

        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let mut state = azul_engine::new_game(2, 0, &mut rng);

        // Make this the (already-triggered) final round so the game ends after this round resolves.
        state.final_round_triggered = true;

        // Ensure it's player 0's turn.
        state.current_player = 0;

        // Clear factories and leave exactly one tile in the center so this move ends the round.
        for f in 0..state.factories.num_factories as usize {
            state.factories.factories[f].len = 0;
        }
        state.center.len = 1;
        state.center.items[0] = Token::Tile(Color::Red);

        // Block rows 1–3 for Red so we only compare row0 vs row4 (plus Floor).
        for row in 1..=3usize {
            let col = WALL_DEST_COL[row][Color::Red as usize] as usize;
            state.players[0].wall[row][col] = Some(Color::Red);
        }

        // Set up a strong incentive to place on row 4:
        // - Pattern line row4 already has 4 Red tiles, so placing 1 more completes it.
        // - Wall row4 is filled except the Red column, so the placement creates a complete row,
        //   yielding high placement score + final scoring bonus.
        state.players[0].pattern_lines[4].color = Some(Color::Red);
        state.players[0].pattern_lines[4].count = 4;

        let red_col_row4 = WALL_DEST_COL[4][Color::Red as usize] as usize;
        for col in 0..azul_engine::BOARD_SIZE {
            if col != red_col_row4 {
                state.players[0].wall[4][col] = Some(Color::Blue);
            }
        }

        let bad_action = Action {
            source: DraftSource::Center,
            color: Color::Red,
            dest: DraftDestination::PatternLine(0),
        };
        let good_action = Action {
            source: DraftSource::Center,
            color: Color::Red,
            dest: DraftDestination::PatternLine(4),
        };

        let legal = azul_engine::legal_actions(&state);
        assert!(legal.contains(&bad_action), "bad_action should be legal");
        assert!(legal.contains(&good_action), "good_action should be legal");

        fn reward_from_parent_perspective(
            parent: &GameState,
            child: &GameState,
            to_play: u8,
        ) -> f32 {
            let my_before = parent.players[to_play as usize].score as f32;
            let opp_before = parent.players[1 - to_play as usize].score as f32;
            let my_after = child.players[to_play as usize].score as f32;
            let opp_after = child.players[1 - to_play as usize].score as f32;
            ((my_after - my_before) - (opp_after - opp_before)) / 20.0
        }

        // Sanity-check the puzzle: good_action must yield a higher immediate reward.
        let mut r1 = rand::rngs::StdRng::seed_from_u64(1);
        let bad_child = apply_action(state.clone(), bad_action, &mut r1).unwrap();
        let r_bad = reward_from_parent_perspective(&state, &bad_child.state, 0);

        let mut r2 = rand::rngs::StdRng::seed_from_u64(2);
        let good_child = apply_action(state.clone(), good_action, &mut r2).unwrap();
        let r_good = reward_from_parent_perspective(&state, &good_child.state, 0);

        assert!(
            r_good > r_bad,
            "expected good_action reward > bad_action reward, got {r_good} vs {r_bad}"
        );

        // With a uniform prior/value network, MCTS should still choose the higher-reward move.
        let features = BasicFeatureExtractor::new(2);
        let dummy_net = DummyNet::uniform(0.0);
        let mcts_config = MctsConfig {
            num_simulations: 64,
            temperature: 0.0,
            root_dirichlet_alpha: 0.0,
            ..MctsConfig::default()
        };
        let mut agent = AlphaZeroMctsAgent::new(mcts_config, features.clone(), dummy_net);

        let obs = features.encode(&state, state.current_player);
        let mut legal_mask = vec![false; ACTION_SPACE_SIZE];
        for a in &legal {
            let id = ActionEncoder::encode(a);
            legal_mask[id as usize] = true;
        }

        let input = AgentInput {
            observation: &obs,
            legal_action_mask: &legal_mask,
            current_player: state.current_player,
            state: Some(&state),
        };

        let mut mcts_rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = agent.select_action_and_policy(&input, 0.0, &mut mcts_rng);

        let good_id = ActionEncoder::encode(&good_action);
        assert_eq!(result.action, good_id);
    }

    /// Test C: MCTS avoids dominated floor moves.
    ///
    /// This test verifies that with immediate floor penalties applied in the engine,
    /// MCTS will prefer a zero-overflow pattern line placement over dumping to floor.
    /// Before the immediate penalty fix, MCTS couldn't distinguish these at depth 1.
    #[test]
    fn test_mcts_avoids_dominated_floor() {
        use azul_engine::{Action, Color, DraftDestination, DraftSource, Token, WALL_DEST_COL};

        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let mut state = azul_engine::new_game(2, 0, &mut rng);

        // Ensure it's player 0's turn.
        state.current_player = 0;

        // Clear factories and center, then place exactly 1 tile in center (plus FP marker).
        for f in 0..state.factories.num_factories as usize {
            state.factories.factories[f].len = 0;
        }
        state.center.len = 2;
        state.center.items[0] = Token::FirstPlayerMarker;
        state.center.items[1] = Token::Tile(Color::Blue);

        // Ensure pattern line row 0 (capacity 1) is empty and Blue not on wall row 0.
        state.players[0].pattern_lines[0].color = None;
        state.players[0].pattern_lines[0].count = 0;
        let blue_col_row0 = WALL_DEST_COL[0][Color::Blue as usize] as usize;
        state.players[0].wall[0][blue_col_row0] = None;

        // Ensure floor is empty (starting fresh).
        state.players[0].floor.len = 0;
        state.players[0].score = 0;

        // Define the two actions to compare.
        let floor_action = Action {
            source: DraftSource::Center,
            color: Color::Blue,
            dest: DraftDestination::Floor,
        };
        let pattern_line_action = Action {
            source: DraftSource::Center,
            color: Color::Blue,
            dest: DraftDestination::PatternLine(0),
        };

        let legal = azul_engine::legal_actions(&state);
        assert!(
            legal.contains(&floor_action),
            "floor_action should be legal"
        );
        assert!(
            legal.contains(&pattern_line_action),
            "pattern_line_action should be legal"
        );

        // Verify immediate reward difference:
        // - Floor: tile + FP marker both to floor = slots 0,1 = penalties -1 -1 = -2
        // - Pattern line: tile to pattern line row 0 (completes it), FP marker to floor = -1 penalty,
        //   but since this is the last move of the round, end-of-round wall tiling happens,
        //   placing the tile on wall (+1 for isolated tile), so net = -1 + 1 = 0.
        fn score_delta(parent: &GameState, child: &GameState, player: u8) -> i16 {
            child.players[player as usize].score - parent.players[player as usize].score
        }

        let mut r1 = rand::rngs::StdRng::seed_from_u64(1);
        let floor_child = apply_action(state.clone(), floor_action, &mut r1).unwrap();
        let floor_delta = score_delta(&state, &floor_child.state, 0);
        // Floor: 1 Blue tile + FP marker = slots 0,1 = penalties -1 -1 = -2
        assert_eq!(floor_delta, -2, "floor should have -2 penalty");

        let mut r2 = rand::rngs::StdRng::seed_from_u64(2);
        let pattern_child = apply_action(state.clone(), pattern_line_action, &mut r2).unwrap();
        let pattern_delta = score_delta(&state, &pattern_child.state, 0);
        // Pattern line completes row 0, wall tile scores +1, FP marker to floor slot 0 = -1
        // Net: +1 - 1 = 0
        assert_eq!(pattern_delta, 0, "pattern line should have 0 net (+1 wall, -1 FP marker)");

        // Pattern line action is strictly better (less penalty).
        assert!(
            pattern_delta > floor_delta,
            "pattern_line_action should have better score delta: {} vs {}",
            pattern_delta,
            floor_delta
        );

        // With a uniform prior/value network and no exploration noise,
        // MCTS should choose the pattern line action due to the immediate reward signal.
        let features = BasicFeatureExtractor::new(2);
        let dummy_net = DummyNet::uniform(0.0);
        let mcts_config = MctsConfig {
            num_simulations: 200, // Enough to see the reward signal clearly
            temperature: 0.0,
            root_dirichlet_alpha: 0.0,
            ..MctsConfig::default()
        };
        let mut agent = AlphaZeroMctsAgent::new(mcts_config, features.clone(), dummy_net);

        let obs = features.encode(&state, state.current_player);
        let mut legal_mask = vec![false; ACTION_SPACE_SIZE];
        for a in &legal {
            let id = ActionEncoder::encode(a);
            legal_mask[id as usize] = true;
        }

        let input = AgentInput {
            observation: &obs,
            legal_action_mask: &legal_mask,
            current_player: state.current_player,
            state: Some(&state),
        };

        let mut mcts_rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = agent.select_action_and_policy(&input, 0.0, &mut mcts_rng);

        let pattern_line_id = ActionEncoder::encode(&pattern_line_action);
        assert_eq!(
            result.action, pattern_line_id,
            "MCTS should choose pattern line (id {}) over floor; got action id {}",
            pattern_line_id, result.action
        );
    }

    #[test]
    #[ignore]
    #[should_panic(expected = "Resource limit")]
    fn repro_mlx_metal_resource_limit_fast() {
        use rand::{Rng, SeedableRng};

        // IMPORTANT: run this test single-threaded:
        // RUST_TEST_THREADS=1 cargo test -p azul-rl-env repro_mlx_metal_resource_limit_fast -- --ignored --nocapture

        // Use a real model for fidelity.
        let features = crate::BasicFeatureExtractor::new(2);
        let obs_size = features.obs_size();

        // Keep the net small-ish so the loop runs fast, but still uses real MLX ops.
        // If the repro is too slow, reduce hidden_size. If it doesn't repro, increase it.
        let hidden_size = 64;
        let mut net = crate::AlphaZeroNet::new(obs_size, hidden_size);

        // --- Option A: bypass the worker and hammer eval directly.
        let max_batch: usize = 512;

        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let mut obs = vec![0.0f32; max_batch * obs_size];
        for x in obs.iter_mut() {
            *x = rng.random::<f32>();
        }

        // Loop count: tune this down/up on your machine.
        // You want it to fail in seconds/minutes, not hours.
        let iters: usize = std::env::var("MLX_STRESS_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(200_000);

        for i in 0..iters {
            // Mix batch sizes to mimic real coalescing (and produce shape variety).
            // Shape variety is often what explodes caches.
            let b = match i & 7 {
                0 => 1,
                1 => 8,
                2 => 16,
                3 => 32,
                4 => 64,
                5 => 128,
                6 => 256,
                _ => 512,
            };

            let obs_slice = &obs[..(b * obs_size)];
            let obs_batch =
                mlx_rs::Array::from_slice(obs_slice, &[b as i32, obs_size as i32]);

            let (policy, values) = net.predict_batch(&obs_batch);

            // This is the exact operation that panics in the worker when Metal hits the limit.
            mlx_rs::transforms::eval([&policy, &values]).unwrap();

            // Force materialization / host touch.
            let _ = policy.as_slice::<f32>();
            let _ = values.as_slice::<f32>();

            if i % 10_000 == 0 {
                eprintln!("stress iters = {i}/{iters}");
            }
        }
    }

    #[test]
    #[ignore]
    fn stress_inference_worker_mlx_metal_fast() {
        use rand::{Rng, SeedableRng};
        use std::sync::Arc;

        // Run single-threaded test harness, but we spawn our own threads here to feed the worker.
        // RUST_TEST_THREADS=1 cargo test -p azul-rl-env stress_inference_worker_mlx_metal_fast -- --ignored --nocapture

        let features = crate::BasicFeatureExtractor::new(2);
        let obs_size = features.obs_size();
        let net = crate::AlphaZeroNet::new(obs_size, 64);
        let worker = Arc::new(InferenceWorker::new(net));

        let max_batch: usize = std::env::var("MLX_WORKER_MAX_BATCH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256);
        let threads: usize = std::env::var("MLX_WORKER_STRESS_THREADS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8);
        let iters_per_thread: usize = std::env::var("MLX_WORKER_STRESS_ITERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(50_000);

        std::thread::scope(|scope| {
            for t in 0..threads {
                let worker = Arc::clone(&worker);
                scope.spawn(move || {
                    let mut rng = rand::rngs::StdRng::seed_from_u64(t as u64);
                    let mut obs = vec![0.0f32; max_batch * obs_size];
                    for x in obs.iter_mut() {
                        *x = rng.random::<f32>();
                    }

                    for i in 0..iters_per_thread {
                        // Simulate shape variability across requests.
                        let b = match (t + i) & 7 {
                            0 => 1,
                            1 => 4,
                            2 => 8,
                            3 => 16,
                            4 => 32,
                            5 => 24,
                            6 => 12,
                            _ => 32,
                        };

                        let obs_slice = &obs[..(b * obs_size)];
                        let (logits, values) = worker.infer(
                            obs_slice.to_vec(),
                            b,
                            obs_size,
                            max_batch,
                            1,
                            b,
                        );

                        // Touch outputs to force any lazy sync/copies.
                        let _ = (logits, values);

                        if i % 10_000 == 0 && t == 0 {
                            eprintln!("worker stress iters = {i}/{iters_per_thread}");
                        }
                    }
                });
            }
        });
    }
}
