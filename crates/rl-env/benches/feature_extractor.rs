//! Benchmark for feature extraction
//!
//! Measures the performance of BasicFeatureExtractor::encode()

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::SeedableRng;

use azul_engine::new_game;
use azul_rl_env::{BasicFeatureExtractor, FeatureExtractor};

fn bench_feature_encode(c: &mut Criterion) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let state = new_game(2, 0, &mut rng);
    let features = BasicFeatureExtractor::new(2);

    c.bench_function("feature_extractor_encode", |b| {
        b.iter(|| {
            let obs = features.encode(black_box(&state), 0);
            black_box(obs)
        })
    });
}

fn bench_feature_encode_all_players(c: &mut Criterion) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    let state = new_game(2, 0, &mut rng);
    let features = BasicFeatureExtractor::new(2);

    c.bench_function("feature_extractor_encode_all_players", |b| {
        b.iter(|| {
            let obs0 = features.encode(black_box(&state), 0);
            let obs1 = features.encode(black_box(&state), 1);
            black_box((obs0, obs1))
        })
    });
}

criterion_group!(benches, bench_feature_encode, bench_feature_encode_all_players);
criterion_main!(benches);
