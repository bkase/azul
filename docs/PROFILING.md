# Profiling Guide for Azul AlphaZero

This guide documents the profiling infrastructure and workflow for measuring and optimizing performance of the AlphaZero training pipeline.

## Built-in Profiling Counters

The crate includes lightweight profiling instrumentation behind a feature flag.

### Enable Profiling

```bash
cargo build --release --features profiling
```

### Run with Profiling

```bash
# Self-play only (isolate MCTS/NN overhead)
cargo run --release --features profiling -- \
  --num-iters 1 \
  --games-per-iter 20 \
  --mcts-sims 20 \
  --no-train \
  --no-checkpoints

# Full training loop
cargo run --release --features profiling -- \
  --num-iters 2 \
  --games-per-iter 5 \
  --training-steps 50 \
  --mcts-sims 20
```

### Profiling Output

The profiling summary shows:

- **Event Counts**: Games, moves, MCTS searches/simulations, NN evaluations, training steps
- **Time Breakdown**: Total time spent in each phase (self-play, training, MCTS, NN eval)
- **Derived Metrics**: Games/sec, simulations/sec, average NN eval time

Example output:
```
=== Profiling Summary ===

Event Counts:
  Self-play games:               20
  Self-play moves:             1313
  MCTS searches:               1313
  MCTS simulations:           26260
  MCTS NN evaluations:        52245
  Training steps:                 0

Time Breakdown:
  Self-play total:          11.374 s
  MCTS search total:        11.368 s
  MCTS NN eval total:        8.009 s

Derived Metrics:
  Games/sec:                  1.76
  Simulations/sec:            2310
  Avg NN eval time:          153.3 us
```

## Criterion Benchmarks

Micro-benchmarks isolate per-component costs:

```bash
# All benchmarks
cargo bench -p azul-rl-env

# Specific benchmark
cargo bench -p azul-rl-env --bench feature_extractor
cargo bench -p azul-rl-env --bench mcts
cargo bench -p azul-rl-env --bench net
```

### Available Benchmarks

| Benchmark | What it measures |
|-----------|------------------|
| `feature_extractor_encode` | GameState -> observation encoding (~248ns) |
| `mcts_search/{10,20,50,100}` | MCTS search with DummyNet (~2.2us/sim) |
| `net_forward_batch/{1,32,64,256}` | NN forward pass batch (~7us) |
| `net_predict_single` | NN single prediction (~165us) |

## Flamegraph Profiling

For deeper analysis, use `cargo-flamegraph`:

### Installation

```bash
cargo install flamegraph
```

### macOS Setup

On macOS, you may need to:
1. Run with `sudo` for dtrace permissions, or
2. Use Instruments.app as an alternative

### Generate Flamegraph

```bash
# Recommended: small workload for readable flamegraph
cargo flamegraph -- \
  --num-iters 2 \
  --games-per-iter 1 \
  --training-steps 5 \
  --mcts-sims 20 \
  --no-checkpoints

# Opens flamegraph.svg in browser
open flamegraph.svg
```

### Interpreting Flamegraphs

Key hotspots to look for:
- `run_search` / `simulate` - MCTS tree search
- `predict_single` / `forward_batch` - Neural network inference
- `encode` - Feature extraction
- `training_step` - Gradient computation and weight updates
- MLX internals (`mlx_rs::*`, `metal::*`) - GPU operations

Unexpected hotspots might indicate:
- Excessive allocations (`alloc::*`)
- Lock contention (`std::sync::*`)
- Unintended copies or clones

## macOS Instruments Alternative

For detailed macOS profiling:

1. Build release binary:
   ```bash
   cargo build --release
   ```

2. Open Instruments.app

3. Use "Time Profiler" template

4. Record `target/release/azul` with appropriate arguments

5. Analyze call tree for hotspots

## Performance Baseline (Dec 2024)

Current baseline on Apple Silicon (M-series):

| Metric | Value |
|--------|-------|
| Games/sec (self-play only) | ~1.76 |
| Simulations/sec | ~2310 |
| Avg NN eval time | ~153us |
| NN as % of total time | ~70% |

The neural network inference dominates runtime. Optimization opportunities:
- Batching NN evaluations across MCTS nodes
- Reducing NN model size for faster inference
- Async/parallel self-play games
