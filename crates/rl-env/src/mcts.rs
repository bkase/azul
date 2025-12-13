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

/// Index into MCTS node arena.
pub type NodeIdx = u32;

/// Edge statistics for an action from a given node.
#[derive(Clone, Debug)]
pub struct ChildEdge {
    pub action_id: ActionId,
    pub prior: f32,             // P(s, a)
    pub visit_count: u32,       // N(s, a)
    pub value_sum: f32,         // W(s, a), sum of backed-up values
    pub child: Option<NodeIdx>, // None until first expansion along this edge
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

/// AlphaZero MCTS Agent that performs tree search guided by a neural network.
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
        Self {
            config,
            features,
            net,
        }
    }

    /// Main MCTS search entrypoint.
    /// Returns improved policy over ACTION_SPACE_SIZE.
    fn run_search(
        &mut self,
        root_state: &GameState,
        rng: &mut impl Rng,
    ) -> [f32; ACTION_SPACE_SIZE] {
        #[cfg(feature = "profiling")]
        let _t = Timer::new(&PROF.time_mcts_search_ns);
        #[cfg(feature = "profiling")]
        PROF.mcts_searches.fetch_add(1, Ordering::Relaxed);

        // Initialize tree with root node
        let mut tree = MctsTree::default();

        // Create root node
        let root_idx = self.create_node(&mut tree, root_state.clone(), rng);

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

        // Run simulations
        for _ in 0..self.config.num_simulations {
            self.simulate(&mut tree, root_idx, rng);
        }

        // Extract policy from root visit counts
        let root = &tree.nodes[root_idx as usize];
        let mut counts = [0.0f32; ACTION_SPACE_SIZE];
        for edge in &root.children {
            counts[edge.action_id as usize] = edge.visit_count as f32;
        }

        apply_temperature(&counts, self.config.temperature)
    }

    /// Create a new node in the tree, expanding it with neural network evaluation.
    fn create_node(
        &mut self,
        tree: &mut MctsTree,
        state: GameState,
        _rng: &mut impl Rng,
    ) -> NodeIdx {
        #[cfg(feature = "profiling")]
        PROF.mcts_nodes_created.fetch_add(1, Ordering::Relaxed);

        let to_play = state.current_player;
        let is_terminal = state.phase == Phase::GameOver;

        // Terminal nodes have no children
        if is_terminal {
            let node = Node {
                state,
                to_play,
                is_terminal,
                children: Vec::new(),
                visit_count: 0,
            };
            let idx = tree.nodes.len() as NodeIdx;
            tree.nodes.push(node);
            return idx;
        }

        // Get legal actions from engine
        let actions = legal_actions(&state);

        // Encode state and get network prediction
        let obs = self.features.encode(&state, to_play);
        let (policy_logits, _value) = {
            #[cfg(feature = "profiling")]
            let _t = Timer::new(&PROF.time_mcts_nn_eval_ns);
            #[cfg(feature = "profiling")]
            PROF.mcts_nn_evals.fetch_add(1, Ordering::Relaxed);
            self.net.predict_single(&obs)
        };
        let policy_logits_slice = policy_logits.as_slice::<f32>();

        // Build (action_id, logit) pairs for legal actions
        let mut legal_ids_and_logits: Vec<(ActionId, f32)> = Vec::with_capacity(actions.len());
        for action in &actions {
            let id = ActionEncoder::encode(action);
            let logit = policy_logits_slice[id as usize];
            legal_ids_and_logits.push((id, logit));
        }

        // Compute softmax priors over legal actions
        let priors = softmax(&legal_ids_and_logits);

        // Create child edges
        let children = priors
            .into_iter()
            .map(|(id, prior)| ChildEdge {
                action_id: id,
                prior,
                visit_count: 0,
                value_sum: 0.0,
                child: None,
            })
            .collect();

        let node = Node {
            state,
            to_play,
            is_terminal,
            children,
            visit_count: 0,
        };
        let idx = tree.nodes.len() as NodeIdx;
        tree.nodes.push(node);
        idx
    }

    /// Run one MCTS simulation from root.
    fn simulate(&mut self, tree: &mut MctsTree, root_idx: NodeIdx, rng: &mut impl Rng) {
        #[cfg(feature = "profiling")]
        let _t = Timer::new(&PROF.time_mcts_simulate_ns);
        #[cfg(feature = "profiling")]
        PROF.mcts_simulations.fetch_add(1, Ordering::Relaxed);

        let mut path: Vec<PathStep> = Vec::new();
        let mut current_idx = root_idx;

        // Selection: traverse tree using PUCT until we reach a leaf
        loop {
            let node = &tree.nodes[current_idx as usize];

            if node.is_terminal {
                break;
            }

            if node.children.is_empty() {
                break;
            }

            // Check if we've reached max depth
            if path.len() >= self.config.max_depth as usize {
                break;
            }

            let child_idx = select_child(node, self.config.cpuct);
            let edge = &tree.nodes[current_idx as usize].children[child_idx];

            path.push(PathStep {
                node_idx: current_idx,
                child_idx,
            });

            if let Some(next_idx) = edge.child {
                current_idx = next_idx;
            } else {
                // Expansion: create child node
                let parent_state = tree.nodes[current_idx as usize].state.clone();
                let action = ActionEncoder::decode(edge.action_id);

                let step_result = apply_action(parent_state, action, rng)
                    .expect("MCTS should only expand legal actions");

                let new_idx = self.create_node(tree, step_result.state, rng);
                tree.nodes[current_idx as usize].children[child_idx].child = Some(new_idx);
                current_idx = new_idx;
                break;
            }
        }

        // Evaluation: get value for leaf node
        let leaf_node = &tree.nodes[current_idx as usize];
        let leaf_value = if leaf_node.is_terminal {
            // Compute terminal value from game scores
            compute_terminal_value(&leaf_node.state, leaf_node.to_play)
        } else {
            // Use network value
            let obs = self.features.encode(&leaf_node.state, leaf_node.to_play);
            let (_policy, value) = {
                #[cfg(feature = "profiling")]
                let _t = Timer::new(&PROF.time_mcts_nn_eval_ns);
                #[cfg(feature = "profiling")]
                PROF.mcts_nn_evals.fetch_add(1, Ordering::Relaxed);
                self.net.predict_single(&obs)
            };
            value
        };

        // Backup: propagate value back through path
        backup(tree, &path, leaf_value);
    }
}

/// PUCT selection: choose child with highest Q + U score.
fn select_child(node: &Node, cpuct: f32) -> usize {
    let mut best_idx = 0;
    let mut best_score = f32::NEG_INFINITY;

    let parent_n = node.visit_count.max(1) as f32;

    for (i, edge) in node.children.iter().enumerate() {
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

/// Backup value through the path, flipping sign for 2-player zero-sum.
fn backup(tree: &mut MctsTree, path: &[PathStep], leaf_value: f32) {
    let mut value = leaf_value;

    for step in path.iter().rev() {
        let node = &mut tree.nodes[step.node_idx as usize];
        let edge = &mut node.children[step.child_idx];

        edge.visit_count += 1;
        edge.value_sum += value;
        node.visit_count += 1;

        // Flip value for parent perspective (2-player zero-sum)
        value = -value;
    }
}

/// Compute terminal value from game result (from perspective of to_play).
fn compute_terminal_value(state: &GameState, to_play: PlayerIdx) -> f32 {
    // For 2-player, compute normalized reward
    let my_score = state.players[to_play as usize].score as f32;
    let opp_idx = 1 - to_play as usize;
    let opp_score = state.players[opp_idx].score as f32;

    // Return value in [-1, 1] range
    let diff = my_score - opp_score;
    // Clamp to prevent extreme values
    (diff / 50.0).clamp(-1.0, 1.0)
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
    N: PolicyValueNet,
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
    N: PolicyValueNet,
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
    N: PolicyValueNet + crate::alphazero::training::TrainableModel,
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
        )
    }

    fn eval_parameters(&self) {
        crate::alphazero::training::TrainableModel::eval_parameters(&self.net)
    }

    fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        crate::alphazero::training::TrainableModel::save(&self.net, path)
    }

    fn load(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        crate::alphazero::training::TrainableModel::load(&mut self.net, path)
    }
}

impl<F, N> mlx_rs::module::ModuleParameters for AlphaZeroMctsAgent<F, N>
where
    F: FeatureExtractor,
    N: PolicyValueNet + mlx_rs::module::ModuleParameters,
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
    use crate::{AzulEnv, BasicFeatureExtractor, EnvConfig, Environment, RewardScheme};
    use rand::SeedableRng;

    /// Dummy neural network for MCTS testing.
    /// Returns fixed priors and value, useful for testing MCTS logic independently.
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

        fn predict_batch(&mut self, _obs_batch: &Array) -> (Array, Array) {
            unimplemented!("DummyNet::predict_batch not implemented for tests")
        }
    }

    #[test]
    fn test_mcts_config_default() {
        let config = MctsConfig::default();
        assert_eq!(config.num_simulations, 256);
        assert_eq!(config.cpuct, 1.5);
        assert_eq!(config.temperature, 1.0);
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
}
