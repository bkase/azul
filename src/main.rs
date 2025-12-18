//! AlphaZero training CLI for Azul
//!
//! This binary runs the self-play training loop for learning to play Azul.

use std::path::PathBuf;

use clap::Parser;
use rand::SeedableRng;

use azul_rl_env::{
    alphazero::training::{TeacherConfig, TrainableModel, Trainer, TrainerConfig},
    AlphaZeroMctsAgent, AlphaZeroNet, AzulEnv, BasicFeatureExtractor, EnvConfig, FeatureExtractor,
    RewardScheme,
};

/// Parse iteration number from checkpoint filename.
///
/// Expected format: `checkpoint_NNNNNN.safetensors` where NNNNNN is a zero-padded iteration number.
/// Returns `None` if the filename doesn't match the expected pattern.
fn parse_iteration_from_checkpoint(path: &std::path::Path) -> Option<usize> {
    let filename = path.file_name()?.to_str()?;
    let stem = filename.strip_suffix(".safetensors")?;
    let iter_str = stem.strip_prefix("checkpoint_")?;
    iter_str.parse().ok()
}

/// AlphaZero training for Azul
#[derive(Parser, Debug)]
#[command(name = "azul")]
#[command(about = "Self-play training loop for learning to play Azul", long_about = None)]
struct Args {
    /// Number of training iterations
    #[arg(long, default_value_t = 100)]
    num_iters: usize,

    /// Self-play games per iteration
    #[arg(long, default_value_t = 25)]
    games_per_iter: usize,

    /// Training steps per iteration
    #[arg(long, default_value_t = 25)]
    training_steps: usize,

    /// Batch size for training
    #[arg(long, default_value_t = 128)]
    batch_size: usize,

    /// MCTS simulations per move
    #[arg(long, default_value_t = 128)]
    mcts_sims: usize,

    /// MCTS neural network batch size (for batched inference)
    #[arg(long, default_value_t = 32)]
    mcts_nn_batch_size: usize,

    /// MCTS virtual loss for parallel path selection
    #[arg(long, default_value_t = 1.0)]
    mcts_virtual_loss: f32,

    /// Checkpoint + arena evaluation interval (in iterations)
    #[arg(long, default_value_t = 10)]
    eval_interval: usize,

    /// Self-play Dirichlet alpha (root noise concentration)
    #[arg(long, default_value_t = 0.3)]
    selfplay_dirichlet_alpha: f32,

    /// Self-play Dirichlet epsilon (mixing fraction vs prior)
    #[arg(long, default_value_t = 0.25)]
    selfplay_dirichlet_eps: f32,

    /// Self-play temperature cutoff move (tau=1 before, tau=0 after)
    #[arg(long, default_value_t = 200)]
    selfplay_temp_cutoff_move: usize,

    /// Arena evaluation games per eval interval (0 disables gating)
    #[arg(long, default_value_t = 20)]
    arena_games: usize,

    /// Arena MCTS simulations per move
    #[arg(long, default_value_t = 800)]
    arena_mcts_sims: usize,

    /// Arena promotion threshold on (wins + 0.5 * ties) / games
    #[arg(long, default_value_t = 0.55)]
    arena_threshold: f32,

    /// Initialize `best.safetensors` from this checkpoint (if best is missing)
    #[arg(long)]
    arena_best_checkpoint: Option<PathBuf>,

    /// Optional fixed teacher checkpoint to play against during training (candidate-vs-teacher games)
    #[arg(long)]
    teacher_checkpoint: Option<PathBuf>,

    /// Teacher games per iteration (0 disables teacher games)
    #[arg(long, default_value_t = 0)]
    teacher_games_per_iter: usize,

    /// Teacher MCTS simulations per move
    #[arg(long, default_value_t = 800)]
    teacher_mcts_sims: usize,

    /// Directory for saving checkpoints (required unless --no-checkpoints is set)
    #[arg(long, required_unless_present = "no_checkpoints")]
    checkpoint_dir: Option<PathBuf>,

    /// Disable checkpointing entirely (for quick experiments/profiling)
    #[arg(long, default_value_t = false)]
    no_checkpoints: bool,

    /// Disable training (for profiling self-play/MCTS separately)
    #[arg(long, default_value_t = false)]
    no_train: bool,

    /// Resume training from a checkpoint file (e.g., checkpoints/checkpoint_000050.safetensors)
    /// If provided, loads weights and resumes from the iteration encoded in the filename.
    #[arg(long)]
    resume: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Validate checkpoint configuration
    if !args.no_checkpoints && args.checkpoint_dir.is_none() {
        eprintln!("error: --checkpoint-dir is required unless --no-checkpoints is set");
        std::process::exit(1);
    }

    // If --no-train is set, override training_steps to 0
    let training_steps = if args.no_train {
        0
    } else {
        args.training_steps
    };

    // Determine start iteration from resume checkpoint
    let start_iter = if let Some(ref checkpoint_path) = args.resume {
        match parse_iteration_from_checkpoint(checkpoint_path) {
            Some(iter) => {
                eprintln!("Resuming from checkpoint: {checkpoint_path:?} (iteration {iter})");
                iter + 1 // Start from the next iteration
            }
            None => {
                eprintln!(
                    "Warning: Could not parse iteration from checkpoint filename {checkpoint_path:?}, starting from 0"
                );
                0
            }
        }
    } else {
        0
    };

    let mut self_play = azul_rl_env::alphazero::training::SelfPlayConfig::default();
    self_play.mcts.num_simulations = args.mcts_sims as u32;
    self_play.mcts.nn_batch_size = args.mcts_nn_batch_size;
    self_play.mcts.virtual_loss = args.mcts_virtual_loss;
    self_play.mcts.root_dirichlet_alpha = args.selfplay_dirichlet_alpha;
    self_play.mcts.root_dirichlet_eps = args.selfplay_dirichlet_eps;
    self_play.temp_cutoff_move = args.selfplay_temp_cutoff_move;

    let mut arena = azul_rl_env::alphazero::training::ArenaConfig {
        games: args.arena_games,
        ..Default::default()
    };
    arena.mcts.num_simulations = args.arena_mcts_sims as u32;
    arena.win_rate_threshold = args.arena_threshold;
    arena.initial_best_checkpoint = args.arena_best_checkpoint.clone();

    let teacher = args
        .teacher_checkpoint
        .clone()
        .map(|checkpoint| TeacherConfig {
            checkpoint,
            games_per_iter: args.teacher_games_per_iter,
            mcts: azul_rl_env::MctsConfig {
                num_simulations: args.teacher_mcts_sims as u32,
                nn_batch_size: args.mcts_nn_batch_size,
                virtual_loss: args.mcts_virtual_loss,
                ..Default::default()
            },
        });

    let config = TrainerConfig {
        num_iters: args.num_iters,
        start_iter,
        self_play_games_per_iter: args.games_per_iter,
        training_steps_per_iter: training_steps,
        batch_size: args.batch_size,
        self_play,
        replay_capacity: 50_000,
        eval_interval: args.eval_interval,
        arena,
        teacher,
        checkpoint_dir: args.checkpoint_dir,
        ..Default::default()
    };

    eprintln!("AlphaZero Training Configuration:");
    eprintln!(
        "  Iterations: {} (starting from {})",
        config.num_iters, config.start_iter
    );
    eprintln!("  Games per iteration: {}", config.self_play_games_per_iter);
    eprintln!(
        "  Training steps per iteration: {}",
        config.training_steps_per_iter
    );
    eprintln!("  Batch size: {}", config.batch_size);
    eprintln!(
        "  MCTS simulations: {}",
        config.self_play.mcts.num_simulations
    );
    eprintln!(
        "  Self-play noise: alpha={} eps={}",
        config.self_play.mcts.root_dirichlet_alpha, config.self_play.mcts.root_dirichlet_eps
    );
    eprintln!(
        "  Self-play temp cutoff: {}",
        config.self_play.temp_cutoff_move
    );
    eprintln!("  MCTS NN batch size: {}", args.mcts_nn_batch_size);
    eprintln!("  MCTS virtual loss: {}", args.mcts_virtual_loss);
    eprintln!("  Replay buffer capacity: {}", config.replay_capacity);
    eprintln!(
        "  Arena: games={} sims={} threshold={}",
        config.arena.games, config.arena.mcts.num_simulations, config.arena.win_rate_threshold
    );
    if let Some(ref init_best) = config.arena.initial_best_checkpoint {
        eprintln!("  Arena initial best: {init_best:?}");
    }
    if let Some(ref teacher) = config.teacher {
        eprintln!(
            "  Teacher: {:?} games/iter={} sims={}",
            teacher.checkpoint, teacher.games_per_iter, teacher.mcts.num_simulations
        );
    }
    eprintln!("  Eval interval: {}", config.eval_interval);
    if let Some(ref dir) = config.checkpoint_dir {
        eprintln!("  Checkpoint directory: {dir:?}");
    } else {
        eprintln!("  Checkpointing: disabled");
    }
    eprintln!();

    // Create environment
    let env_config = EnvConfig {
        num_players: config.num_players,
        reward_scheme: RewardScheme::DenseScoreDelta,
        include_full_state_in_step: true,
    };
    let features = BasicFeatureExtractor::new(config.num_players);
    let env = AzulEnv::new(env_config, features.clone());

    // Create agent with AlphaZeroNet
    let hidden_size = 128;
    let obs_size = features.obs_size();
    let mut net = AlphaZeroNet::new(obs_size, hidden_size);

    // Load checkpoint if resuming
    if let Some(ref checkpoint_path) = args.resume {
        eprintln!("Loading weights from checkpoint...");
        net.load(checkpoint_path)?;
        eprintln!("Weights loaded successfully.");
    }

    let agent = AlphaZeroMctsAgent::new(config.self_play.mcts.clone(), features, net);

    // Create RNG
    let rng = rand::rngs::StdRng::seed_from_u64(42);

    // Create trainer
    let mut trainer = Trainer::new(env, agent, config, rng);

    // Run training
    eprintln!("Starting training...");
    trainer.run()?;

    eprintln!("Training complete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_parse_iteration_from_checkpoint_valid() {
        let path = Path::new("checkpoints/checkpoint_000050.safetensors");
        assert_eq!(parse_iteration_from_checkpoint(path), Some(50));

        let path = Path::new("checkpoint_000000.safetensors");
        assert_eq!(parse_iteration_from_checkpoint(path), Some(0));

        let path = Path::new("/some/deep/path/checkpoint_001234.safetensors");
        assert_eq!(parse_iteration_from_checkpoint(path), Some(1234));
    }

    #[test]
    fn test_parse_iteration_from_checkpoint_invalid() {
        // Wrong extension
        let path = Path::new("checkpoint_000050.bin");
        assert_eq!(parse_iteration_from_checkpoint(path), None);

        // Wrong prefix
        let path = Path::new("model_000050.safetensors");
        assert_eq!(parse_iteration_from_checkpoint(path), None);

        // No number
        let path = Path::new("checkpoint_.safetensors");
        assert_eq!(parse_iteration_from_checkpoint(path), None);

        // Invalid number
        let path = Path::new("checkpoint_abc.safetensors");
        assert_eq!(parse_iteration_from_checkpoint(path), None);
    }

    #[test]
    fn test_parse_iteration_handles_large_numbers() {
        let path = Path::new("checkpoint_999999.safetensors");
        assert_eq!(parse_iteration_from_checkpoint(path), Some(999999));
    }
}
