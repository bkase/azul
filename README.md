# Azul AlphaZero

A complete AlphaZero-style implementation for learning to play the board game [Azul](https://boardgamegeek.com/boardgame/230802/azul) through self-play reinforcement learning. No human knowledge required—the agent learns entirely by playing against itself.

## What is Azul?

Azul is a 2–4 player turn-based tile-placement board game where players:
- Draft colored tiles from factory displays
- Place tiles in pattern lines on their player board
- Complete patterns to score points on a 5×5 wall
- Manage risk—tiles that overflow go to the floor line and incur penalties

## Scope / Status

- The **game engine** supports **2–4 players**.
- The **AlphaZero pipeline** (MCTS + net + training + CLI tools) currently targets **2-player** games.

## Features

- **Pure Rust implementation** of the complete Azul game engine
- **AlphaZero training pipeline** with MCTS + neural network guidance (self-play)
- **Arena gating** that maintains a `best.safetensors` via candidate-vs-best evaluation
- **Optional teacher games** against a fixed checkpoint during training
- **Apple Silicon optimized** via MLX for GPU-accelerated training
- **Interactive play mode** to play against trained models
- **Inspection tool** to compare raw network priors vs MCTS search

## Project Structure

```
azul/
├── .cargo/config.toml    # cargo aliases + env (see below)
├── crates/
│   ├── engine/           # Core game engine (pure, deterministic game logic)
│   └── rl-env/           # RL pipeline (env, MCTS, net, training)
├── docs/
│   └── PROFILING.md      # profiling workflows (counters, benches, flamegraphs)
└── src/
    ├── main.rs           # Training binary (azul)
    ├── lib.rs            # Shared CLI utilities
    └── bin/
        ├── play.rs       # Interactive play against AI
        └── inspect.rs    # Debug/inspection tool
```

## Requirements

- Rust (stable, edition 2021)
- macOS + Apple Silicon (MLX/Metal GPU acceleration)

## Cargo aliases

This repo defines aliases in `.cargo/config.toml` to:
1) keep profiling and non-profiling builds in separate `target` directories, and
2) avoid flaky parallel tests with MLX/Metal by forcing single-threaded tests (`RUST_TEST_THREADS=1`).

| Command | Purpose |
|---------|---------|
| `cargo build-fast` | Build into `target/` (no features) |
| `cargo run-fast` | Run into `target/` (no features) |
| `cargo test-fast` | Test into `target/` (no features) |
| `cargo build-prof` | Build into `target-profiling/` with `--features profiling` |
| `cargo run-prof` | Run into `target-profiling/` with `--features profiling` |
| `cargo test-prof` | Test into `target-profiling/` with `--features profiling` |
| `cargo train` | Release training with profiling: `cargo run --bin azul --features profiling --target-dir target-profiling --release -- ...args...` |

## Building

```bash
# Standard release build (no profiling)
cargo build --release

# Release build with profiling counters/timers enabled
cargo build-prof --release
```

## Usage

### Training (binary: `azul`)

`cargo train` is the quickest way to run training in release mode with profiling enabled (the alias already includes the `--` separator, so you pass flags directly).

```bash
# Quick experiment (checkpoints written every --eval-interval iters)
cargo train \
  --num-iters 10 \
  --games-per-iter 2 \
  --mcts-sims 64 \
  --checkpoint-dir ./checkpoints

# Full training run
cargo train \
  --num-iters 500 \
  --games-per-iter 75 \
  --training-steps 100 \
  --mcts-sims 200 \
  --checkpoint-dir ./checkpoints

# Resume from checkpoint (continues from iter+1 based on filename)
cargo train \
  --resume ./checkpoints/checkpoint_000100.safetensors \
  --checkpoint-dir ./checkpoints

# Self-play only (useful for profiling MCTS/NN)
cargo train \
  --num-iters 1 \
  --games-per-iter 20 \
  --mcts-sims 64 \
  --no-train \
  --no-checkpoints
```

For the authoritative list of flags, run `cargo train --help`.

**Training options (current defaults):**
| Flag | Default | Description |
|------|---------|-------------|
| `--num-iters` | 100 | Training iterations |
| `--games-per-iter` | 25 | Self-play games per iteration |
| `--training-steps` | 25 | Gradient steps per iteration |
| `--batch-size` | 128 | Training batch size |
| `--mcts-sims` | 128 | MCTS simulations per move (self-play) |
| `--mcts-nn-batch-size` | 32 | Leaf positions per NN batch (batched inference) |
| `--mcts-virtual-loss` | 1.0 | Virtual loss magnitude for in-flight sims |
| `--eval-interval` | 10 | Save checkpoint + run arena eval every N iters |
| `--selfplay-dirichlet-alpha` | 0.3 | Root noise concentration (self-play) |
| `--selfplay-dirichlet-eps` | 0.25 | Root noise mixing fraction (self-play) |
| `--selfplay-temp-cutoff-move` | 200 | Sample (tau=1) before this move, argmax after |
| `--arena-games` | 20 | Candidate-vs-best games per eval (0 disables gating) |
| `--arena-mcts-sims` | 800 | MCTS sims per move during arena eval |
| `--arena-threshold` | 0.55 | Promotion threshold on (wins + 0.5 * ties) / games |
| `--arena-best-checkpoint` | - | Initialize `best.safetensors` from this checkpoint (if missing) |
| `--teacher-checkpoint` | - | Fixed teacher checkpoint for extra training games |
| `--teacher-games-per-iter` | 0 | Teacher games per iteration (0 disables teacher) |
| `--teacher-mcts-sims` | 800 | MCTS sims per move for teacher |
| `--checkpoint-dir` | - | Where to write checkpoints (required unless `--no-checkpoints`) |
| `--no-checkpoints` | false | Disable checkpointing + arena entirely |
| `--no-train` | false | Disable training steps (self-play only) |
| `--resume` | - | Resume from a `checkpoint_XXXXXX.safetensors` file |

### Playing Against the AI (binary: `play`)

`play` loads a checkpoint and lets you play interactively against the MCTS agent.

If `--checkpoint` is omitted, it searches `./checkpoints/`:
1) prefers `./checkpoints/best.safetensors` if present, otherwise
2) uses the latest `*.safetensors` file in that directory.

```bash
# Use best (or latest) checkpoint from ./checkpoints
cargo run --release --bin play

# Explicit checkpoint
cargo run --release --bin play -- \
  --checkpoint ./checkpoints/best.safetensors \
  --mcts-sims 800

# Let AI play first
cargo run --release --bin play -- \
  --checkpoint ./checkpoints/best.safetensors \
  --ai-first
```

### Inspecting the Model (binary: `inspect`)

Inspection tool to visualize the network’s raw priors + value and compare them to MCTS.

```bash
cargo run --release --bin inspect -- \
  --checkpoint ./checkpoints/best.safetensors \
  --mcts-sims 500 \
  --controlled
```

## Checkpoints

- Checkpoints are written to `--checkpoint-dir` every `--eval-interval` iterations as `checkpoint_{iter:06}.safetensors`.
- Arena gating maintains `best.safetensors` in the same directory (unless `--arena-games 0` or `--no-checkpoints`).
- `--resume` expects the `checkpoint_{iter:06}.safetensors` naming pattern and continues from `iter + 1`.

## Profiling

- Enable lightweight counters/timers with `--features profiling` (or just use `cargo train`).
- See `docs/PROFILING.md` for workflows (counters, criterion benches, flamegraphs).

## Web UI (GitHub Pages)

- Source lives in `web/`, published assets live in `docs/` for GitHub Pages.
- Build + publish locally: `scripts/publish-web.sh` (copies latest `best.safetensors` into `docs/`).
- In GitHub: Settings → Pages → Deploy from a branch → `main` / `docs`.

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
- Shared trunk → policy head (`ACTION_SPACE_SIZE = 500`) + value head (tanh to `[-1, 1]`)
- Safetensors checkpoint format

**MCTS:**
- UCB-based tree search with neural network guidance
- Batched inference via a dedicated worker thread (serializes MLX/Metal usage)
- Dirichlet noise for root exploration (self-play)
- Virtual loss for in-flight simulations

**Training Loop:**
- Parallel self-play generation (Rayon)
- Replay buffer with configurable capacity
- Combined policy (cross-entropy) + value (MSE) loss
- Periodic checkpointing + arena evaluation (best-model gating)

## License

MIT
