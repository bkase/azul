# Azul AlphaZero

A complete AlphaZero implementation for learning to play the board game [Azul](https://boardgamegeek.com/boardgame/230802/azul) through self-play reinforcement learning. No human knowledge required—the AI learns entirely from playing against itself.

## What is Azul?

Azul is a 2-4 player turn-based tile-placement board game where players:
- Draft colored tiles from factory displays
- Place tiles in pattern lines on their player board
- Complete patterns to score points on a 5×5 wall
- Manage risk—tiles that overflow go to the floor line and incur penalties

## Features

- **Pure Rust implementation** of the complete Azul game engine
- **AlphaZero training pipeline** with MCTS + neural network guidance
- **Apple Silicon optimized** via MLX for GPU-accelerated training
- **Interactive play mode** to play against trained models
- **Diagnostic tools** for debugging and inspecting learned policies

## Project Structure

```
azul/
├── crates/
│   ├── engine/      # Core game engine (pure, deterministic game logic)
│   └── rl-env/      # RL pipeline (environment, MCTS, neural net, training)
└── src/
    ├── main.rs      # Training binary
    ├── bin/play.rs  # Interactive play against AI
    └── bin/inspect.rs # Debug/inspection tool
```

## Requirements

- Rust 1.70+ (edition 2021)
- macOS with Apple Silicon (M1/M2/M3+) for MLX/Metal GPU acceleration

## Building

```bash
# Standard release build
cargo build --release

# With profiling instrumentation
cargo build --release --features profiling
```

## Usage

### Training

```bash
# Quick experiment
cargo run --release -- \
  --num-iters 10 \
  --games-per-iter 2 \
  --mcts-sims 50 \
  --checkpoint-dir ./checkpoints

# Full training run
cargo run --release -- \
  --num-iters 500 \
  --games-per-iter 75 \
  --training-steps 100 \
  --mcts-sims 200 \
  --checkpoint-dir ./checkpoints

# Resume from checkpoint
cargo run --release -- \
  --resume ./checkpoints/checkpoint_000100.safetensors \
  --checkpoint-dir ./checkpoints
```

**Training options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--num-iters` | 100 | Training iterations |
| `--games-per-iter` | 5 | Self-play games per iteration |
| `--training-steps` | 50 | Gradient steps per iteration |
| `--batch-size` | 64 | Training batch size |
| `--mcts-sims` | 50 | MCTS simulations per move |
| `--mcts-nn-batch-size` | 32 | Neural network batch size |
| `--checkpoint-dir` | - | Where to save checkpoints |
| `--resume` | - | Resume from checkpoint file |

### Playing Against the AI

```bash
cargo run --release --bin play -- \
  --checkpoint ./checkpoints/checkpoint_000500.safetensors \
  --mcts-sims 200

# Let AI play first
cargo run --release --bin play -- \
  --checkpoint ./checkpoints/checkpoint_000500.safetensors \
  --ai-first
```

### Inspecting the Model

Debug tool to visualize network predictions vs MCTS search:

```bash
cargo run --release --bin inspect -- \
  --checkpoint ./checkpoints/checkpoint_000500.safetensors \
  --mcts-sims 500
```

## Architecture

### Game Engine (`azul-engine`)

Pure, stateless implementation of Azul rules:
- Complete game state representation (factories, player boards, scores)
- Legal action generation
- Score computation with wall bonuses (rows, columns, color sets)
- 5 tile colors, 5×5 wall with Latin-square pattern

### RL Environment (`azul-rl-env`)

AlphaZero training components:

**Neural Network:**
- Fully-connected architecture using MLX
- Shared trunk → policy head (500 actions) + value head ([-1, 1])
- Safetensors checkpoint format

**MCTS:**
- UCB-based tree search with neural network guidance
- Batched inference via dedicated worker thread
- Dirichlet noise for root exploration
- Temperature-based action selection

**Training Loop:**
- Parallel self-play generation (Rayon)
- Replay buffer with configurable capacity
- Combined policy (cross-entropy) + value (MSE) loss

## Performance

On Apple Silicon (M-series):
- ~1.76 games/second (self-play)
- ~2,310 MCTS simulations/second
- ~7μs per batched neural network evaluation

## How AlphaZero Works

1. **Self-play**: The current neural network plays games against itself using MCTS
2. **Data collection**: Game states, MCTS visit counts, and outcomes are stored
3. **Training**: The network learns to predict MCTS policies and game outcomes
4. **Repeat**: The improved network generates better self-play data

The key insight is that MCTS search provides a "policy improvement operator"—the search distribution is stronger than the raw network output, so training the network to match MCTS improves it over time.

## License

MIT
