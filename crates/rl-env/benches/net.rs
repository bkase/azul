//! Benchmark for neural network forward pass
//!
//! Measures the performance of AlphaZeroNet::forward_batch

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use azul_rl_env::{AlphaZeroNet, BasicFeatureExtractor, FeatureExtractor, PolicyValueNet};
use mlx_rs::Array;

fn bench_net_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("net_forward_batch");

    let features = BasicFeatureExtractor::new(2);
    let obs_size = features.obs_size();
    let hidden_size = 128;

    for batch_size in [1, 32, 64, 256] {
        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &batch_size| {
                let mut net = AlphaZeroNet::new(obs_size, hidden_size);
                // Create random input batch
                let obs = Array::zeros::<f32>(&[batch_size, obs_size as i32]).unwrap();

                b.iter(|| {
                    let (policy, value) = net.forward_batch(black_box(&obs));
                    black_box((policy, value))
                })
            },
        );
    }

    group.finish();
}

fn bench_net_predict_single(c: &mut Criterion) {
    let features = BasicFeatureExtractor::new(2);
    let obs_size = features.obs_size();
    let hidden_size = 128;
    let mut net = AlphaZeroNet::new(obs_size, hidden_size);

    c.bench_function("net_predict_single", |b| {
        let obs = Array::zeros::<f32>(&[obs_size as i32]).unwrap();

        b.iter(|| {
            let (policy, value) = net.predict_single(black_box(&obs));
            black_box((policy, value))
        })
    });
}

criterion_group!(benches, bench_net_forward, bench_net_predict_single);
criterion_main!(benches);
