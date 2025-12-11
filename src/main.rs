//! AlphaZero training CLI for Azul
//!
//! This binary runs the self-play training loop for learning to play Azul.

use std::path::PathBuf;

use clap::Parser;
use rand::SeedableRng;

use azul_rl_env::{
    alphazero::training::{Trainer, TrainerConfig},
    AlphaZeroMctsAgent, AlphaZeroNet, AzulEnv, BasicFeatureExtractor, EnvConfig, FeatureExtractor,
    MctsConfig, RewardScheme,
};

/// AlphaZero training for Azul
#[derive(Parser, Debug)]
#[command(name = "azul")]
#[command(about = "Self-play training loop for learning to play Azul", long_about = None)]
struct Args {
    /// Number of training iterations
    #[arg(long, default_value_t = 100)]
    num_iters: usize,

    /// Self-play games per iteration
    #[arg(long, default_value_t = 5)]
    games_per_iter: usize,

    /// Training steps per iteration
    #[arg(long, default_value_t = 50)]
    training_steps: usize,

    /// Batch size for training
    #[arg(long, default_value_t = 64)]
    batch_size: usize,

    /// MCTS simulations per move
    #[arg(long, default_value_t = 50)]
    mcts_sims: usize,

    /// Directory for saving checkpoints (required unless --no-checkpoints is set)
    #[arg(long, required_unless_present = "no_checkpoints")]
    checkpoint_dir: Option<PathBuf>,

    /// Disable checkpointing entirely (for quick experiments/profiling)
    #[arg(long, default_value_t = false)]
    no_checkpoints: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Validate checkpoint configuration
    if !args.no_checkpoints && args.checkpoint_dir.is_none() {
        eprintln!("error: --checkpoint-dir is required unless --no-checkpoints is set");
        std::process::exit(1);
    }

    let config = TrainerConfig {
        num_iters: args.num_iters,
        self_play_games_per_iter: args.games_per_iter,
        training_steps_per_iter: args.training_steps,
        batch_size: args.batch_size,
        self_play: azul_rl_env::alphazero::training::SelfPlayConfig {
            mcts_simulations: args.mcts_sims,
            ..Default::default()
        },
        replay_capacity: 50_000,
        eval_interval: 10,
        checkpoint_dir: args.checkpoint_dir,
        ..Default::default()
    };

    eprintln!("AlphaZero Training Configuration:");
    eprintln!("  Iterations: {}", config.num_iters);
    eprintln!("  Games per iteration: {}", config.self_play_games_per_iter);
    eprintln!(
        "  Training steps per iteration: {}",
        config.training_steps_per_iter
    );
    eprintln!("  Batch size: {}", config.batch_size);
    eprintln!("  MCTS simulations: {}", config.self_play.mcts_simulations);
    eprintln!("  Replay buffer capacity: {}", config.replay_capacity);
    if let Some(ref dir) = config.checkpoint_dir {
        eprintln!("  Checkpoint directory: {dir:?}");
    } else {
        eprintln!("  Checkpointing: disabled");
    }
    eprintln!();

    // Create environment
    let env_config = EnvConfig {
        num_players: config.num_players,
        reward_scheme: RewardScheme::TerminalOnly,
        include_full_state_in_step: true,
    };
    let features = BasicFeatureExtractor::new(config.num_players);
    let env = AzulEnv::new(env_config, features.clone());

    // Create agent with AlphaZeroNet
    let hidden_size = 128;
    let obs_size = features.obs_size();
    let net = AlphaZeroNet::new(obs_size, hidden_size);
    let mcts_config = MctsConfig {
        num_simulations: config.self_play.mcts_simulations as u32,
        ..Default::default()
    };
    let agent = AlphaZeroMctsAgent::new(mcts_config, features, net);

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
