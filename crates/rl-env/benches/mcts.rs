//! Benchmark for MCTS search
//!
//! Measures the performance of AlphaZeroMctsAgent::run_search with DummyNet

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::SeedableRng;

use azul_engine::new_game;
use azul_rl_env::{
    Agent, AlphaZeroMctsAgent, BasicFeatureExtractor, MctsConfig, Observation, PolicyValueNet,
    ACTION_SPACE_SIZE,
};
use mlx_rs::Array;

/// Dummy network that returns uniform policy and zero value.
/// Removes NN cost to isolate MCTS overhead.
#[derive(Clone)]
struct DummyNet;

impl PolicyValueNet for DummyNet {
    fn predict_single(&mut self, _obs: &Observation) -> (Array, f32) {
        let policy = vec![1.0 / ACTION_SPACE_SIZE as f32; ACTION_SPACE_SIZE];
        let policy_arr = Array::from_slice(&policy, &[ACTION_SPACE_SIZE as i32]);
        (policy_arr, 0.0)
    }

    fn predict_batch(&mut self, obs_batch: &Array) -> (Array, Array) {
        let batch_size = obs_batch.shape()[0] as usize;
        let policy = vec![1.0 / ACTION_SPACE_SIZE as f32; batch_size * ACTION_SPACE_SIZE];
        let values = vec![0.0f32; batch_size];
        let policy_arr = Array::from_slice(&policy, &[batch_size as i32, ACTION_SPACE_SIZE as i32]);
        let values_arr = Array::from_slice(&values, &[batch_size as i32]);
        (policy_arr, values_arr)
    }
}

fn bench_mcts_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("mcts_search");

    for num_sims in [10, 20, 50, 100] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_sims),
            &num_sims,
            |b, &num_sims| {
                let mut rng = rand::rngs::StdRng::seed_from_u64(42);
                let state = new_game(2, 0, &mut rng);
                let features = BasicFeatureExtractor::new(2);
                let net = DummyNet;
                let config = MctsConfig {
                    num_simulations: num_sims,
                    root_dirichlet_alpha: 0.0, // Disable noise for reproducibility
                    ..Default::default()
                };
                let mut agent = AlphaZeroMctsAgent::new(config, features, net);

                b.iter(|| {
                    let input = azul_rl_env::AgentInput {
                        observation: &Array::zeros::<f32>(&[100]).unwrap(), // Dummy obs
                        legal_action_mask: &[true; ACTION_SPACE_SIZE],
                        current_player: 0,
                        state: Some(&state),
                    };
                    let result = agent.select_action(black_box(&input), &mut rng);
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_mcts_search);
criterion_main!(benches);
