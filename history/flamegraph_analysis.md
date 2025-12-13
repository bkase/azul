# Flamegraph & Profiling Analysis

## Summary

Profiling analysis performed December 2024 using:
1. Built-in profiling counters (`--features profiling`)
2. Criterion micro-benchmarks
3. macOS Instruments Time Profiler (via `cargo flamegraph`)

## Key Findings

### 1. Performance Bottleneck: Neural Network Inference

**~70% of total runtime is spent in NN inference.**

| Metric | Value |
|--------|-------|
| Avg NN eval time | 153-165 µs |
| NN evals per simulation | ~2 (one for expansion, one for value) |
| Total NN time / Total time | 70% |

The neural network's `predict_single` method is the dominant hotspot. Each MCTS simulation requires at least one NN evaluation.

### 2. MCTS Pure Overhead is Minimal

With a DummyNet (no actual NN), MCTS runs at:
- ~2.2 µs per simulation
- This is ~75x faster than with real NN

MCTS tree operations (selection, expansion, backup) are efficient.

### 3. Feature Extraction is Fast

- ~248 ns per encode
- Negligible compared to NN inference

### 4. Training is Now Fast

After replacing FD gradients with MLX autodiff:
- ~970 training steps/sec
- Training is <1% of total time
- No longer a bottleneck

## Optimization Opportunities

### High Impact (address NN bottleneck)

1. **Batch NN evaluations across MCTS leaves**
   - Current: 1 eval per node expansion
   - Opportunity: Batch multiple pending expansions
   - Expected: Reduce NN call count by 10-50x

2. **Virtual loss / parallel MCTS**
   - Run multiple simulations concurrently
   - Share NN batches across parallel searches

3. **Smaller/faster neural network**
   - Current hidden_size=128 may be overkill
   - Quantization or distillation

### Low Impact (already optimized)

- Action decode LUT: ~1.6% improvement (implemented)
- Cached obs_size: negligible (implemented)
- Cached zero observations: negligible (implemented)

## Benchmark Baseline (Dec 2024)

### Micro-benchmarks (Criterion)

| Benchmark | Time |
|-----------|------|
| feature_extractor_encode | 244 ns |
| net_forward_batch/1 | 6.8 µs |
| net_forward_batch/256 | 7.1 µs |
| net_predict_single | 165 µs |
| mcts_search/20 (DummyNet) | 45.5 µs |
| mcts_search/100 (DummyNet) | 219 µs |

### Full System (20 games, 20 MCTS sims)

| Metric | Value |
|--------|-------|
| Total time | 11.37 s |
| Games/sec | 1.76 |
| Moves/game | 65.65 |
| Simulations/sec | 2310 |
| NN evals/sec | 6524 |
| Avg NN eval time | 153 µs |

## Tools Used

1. **Built-in profiling** (`--features profiling`)
   - Atomic counters for events
   - RAII timers for durations
   - Zero overhead when disabled

2. **Criterion benchmarks**
   - Isolated per-component timing
   - Statistical analysis
   - Regression detection

3. **cargo flamegraph / Instruments**
   - macOS Time Profiler
   - Call tree analysis
   - Validates counter findings

## Conclusion

The AlphaZero implementation is well-optimized for CPU-side operations. The main bottleneck is neural network inference, which requires algorithmic changes (batching) rather than micro-optimizations.
