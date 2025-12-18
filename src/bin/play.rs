//! Interactive CLI to play Azul against the AlphaZero agent
//!
//! Usage: cargo run --bin play [--checkpoint path/to/checkpoint.safetensors]

use std::io::{self, Write};
use std::path::PathBuf;

use clap::Parser;
use rand::SeedableRng;

use azul::display::{display_board, format_action, BOLD, DIM, RESET};
use azul_engine::{apply_action, legal_actions, new_game, Action, GameState, Phase};
use azul_rl_env::{
    alphazero::training::TrainableModel, ActionEncoder, Agent, AgentInput, AlphaZeroMctsAgent,
    AlphaZeroNet, BasicFeatureExtractor, FeatureExtractor, MctsConfig,
};

/// Play Azul against the AlphaZero agent
#[derive(Parser, Debug)]
#[command(name = "play")]
#[command(about = "Play Azul against the trained AI", long_about = None)]
struct Args {
    /// Path to checkpoint file (defaults to latest in checkpoints/)
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    /// MCTS simulations per AI move (more = stronger but slower)
    #[arg(long, default_value_t = 800)]
    mcts_sims: usize,

    /// Play as player 1 (AI goes first) instead of player 0
    #[arg(long)]
    ai_first: bool,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn get_human_action(state: &GameState) -> Action {
    let actions = legal_actions(state);

    println!("{BOLD}Your legal moves:{RESET}");
    for (i, action) in actions.iter().enumerate() {
        println!("  {}: {}", i, format_action(action));
    }

    loop {
        print!("\n{BOLD}Enter move number:{RESET} ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => {
                // EOF
                println!("\nGoodbye!");
                std::process::exit(0);
            }
            Err(_) => {
                println!("Error reading input, try again.");
                continue;
            }
            Ok(_) => {}
        }

        let input = input.trim();
        if input == "q" || input == "quit" {
            println!("Goodbye!");
            std::process::exit(0);
        }

        match input.parse::<usize>() {
            Ok(idx) if idx < actions.len() => return actions[idx],
            Ok(_) => println!("Invalid move number. Enter 0-{}", actions.len() - 1),
            Err(_) => println!("Please enter a number (or 'q' to quit)"),
        }
    }
}

fn find_latest_checkpoint() -> Option<PathBuf> {
    let checkpoint_dir = PathBuf::from("checkpoints");
    if !checkpoint_dir.exists() {
        return None;
    }

    // Prefer arena-gated best model if present.
    let best = checkpoint_dir.join("best.safetensors");
    if best.exists() {
        return Some(best);
    }

    let mut checkpoints: Vec<_> = std::fs::read_dir(&checkpoint_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
        .collect();

    checkpoints.sort_by_key(|e| e.path());
    checkpoints.last().map(|e| e.path())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Find checkpoint
    let checkpoint_path = args.checkpoint.or_else(find_latest_checkpoint);

    let checkpoint_path = match checkpoint_path {
        Some(p) => p,
        None => {
            eprintln!("No checkpoint found. Run training first or specify --checkpoint");
            std::process::exit(1);
        }
    };

    println!("{BOLD}Loading checkpoint:{RESET} {:?}", checkpoint_path);

    // Set up agent
    let features = BasicFeatureExtractor::new(2);
    let hidden_size = 128;
    let obs_size = features.obs_size();
    let mut net = AlphaZeroNet::new(obs_size, hidden_size);
    net.load(&checkpoint_path)?;

    let mcts_config = MctsConfig {
        num_simulations: args.mcts_sims as u32,
        temperature: 0.0,          // Argmax for stronger play
        root_dirichlet_alpha: 0.0, // No exploration noise
        root_dirichlet_eps: 0.0,   // No exploration noise
        ..Default::default()
    };

    let mut agent = AlphaZeroMctsAgent::new(mcts_config, features.clone(), net);

    // Set up game
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);
    let human_player: u8 = if args.ai_first { 1 } else { 0 };
    let starting_player = 0;

    let mut state = new_game(2, starting_player, &mut rng);

    println!("\n{BOLD}Welcome to Azul!{RESET}");
    println!(
        "You are Player {} ({})",
        human_player,
        if human_player == 0 { "first" } else { "second" }
    );
    println!("Type 'q' to quit at any time.\n");

    // Game loop
    while state.phase != Phase::GameOver {
        display_board(&state, Some(human_player));

        let action = if state.current_player == human_player {
            get_human_action(&state)
        } else {
            println!("{DIM}AI is thinking...{RESET}");

            // Build AgentInput
            let obs = features.encode(&state, state.current_player);
            let legal_mask = build_legal_action_mask(&state);

            let input = AgentInput {
                observation: &obs,
                legal_action_mask: &legal_mask,
                current_player: state.current_player,
                state: Some(&state),
            };

            let action_id = agent.select_action(&input, &mut rng);
            let action = ActionEncoder::decode(action_id);

            println!("AI plays: {}", format_action(&action));
            action
        };

        let result = apply_action(state, action, &mut rng).expect("Action should be legal");
        state = result.state;

        if let Some(scores) = result.final_scores {
            display_board(&state, Some(human_player));
            println!("\n{BOLD}═══════════════════════════════════════{RESET}");
            println!("{BOLD}                GAME OVER{RESET}");
            println!("{BOLD}═══════════════════════════════════════{RESET}");
            println!("Your score: {}", scores[human_player as usize]);
            println!("AI score:   {}", scores[1 - human_player as usize]);

            if scores[human_player as usize] > scores[1 - human_player as usize] {
                println!("\n{BOLD}🎉 YOU WIN! 🎉{RESET}");
            } else if scores[human_player as usize] < scores[1 - human_player as usize] {
                println!("\n{DIM}AI wins. Better luck next time!{RESET}");
            } else {
                println!("\n{BOLD}It's a tie!{RESET}");
            }
            break;
        }
    }

    Ok(())
}

fn build_legal_action_mask(state: &GameState) -> Vec<bool> {
    let actions = legal_actions(state);
    let mut mask = vec![false; 500];
    for action in actions {
        let id = ActionEncoder::encode(&action);
        mask[id as usize] = true;
    }
    mask
}
