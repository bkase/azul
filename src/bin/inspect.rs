//! Inspection tool to debug what the trained network is thinking
//!
//! This tool loads a checkpoint and runs the agent on a controlled game state
//! to see what the network's raw priors and value predictions are, and compare
//! them to what MCTS produces after search.
//!
//! Usage: cargo run --bin inspect -- --checkpoint path/to/checkpoint.safetensors

use std::path::PathBuf;

use clap::Parser;
use rand::SeedableRng;

use azul::display::{display_board, format_action_compact, BOLD, RESET};
use azul_engine::{new_game, Color, DraftDestination, GameState};
use azul_rl_env::{
    alphazero::training::{MctsAgentExt, MctsSearchResult, TrainableModel},
    ActionEncoder, AgentInput, AlphaZeroMctsAgent, AlphaZeroNet, BasicFeatureExtractor,
    FeatureExtractor, MctsConfig, PolicyValueNet, ACTION_SPACE_SIZE,
};

#[derive(Parser, Debug)]
#[command(name = "inspect")]
#[command(about = "Inspect network predictions and MCTS behavior", long_about = None)]
struct Args {
    /// Path to checkpoint file
    #[arg(long)]
    checkpoint: PathBuf,

    /// Number of MCTS simulations
    #[arg(long, default_value_t = 500)]
    mcts_sims: usize,

    /// Random seed for game generation
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Use a controlled scenario instead of random game state
    #[arg(long)]
    controlled: bool,
}

/// Create a controlled "sanity check" scenario where there's an obvious good move.
/// Player 0 has an empty pattern line (row 0, capacity 1).
/// Factory 0 has [Red, Blue, Blue, Blue] - so taking Red gives exactly 1 tile.
/// Taking Red from Factory 0 to Line 1 is clearly the best move (no overflow to floor).
fn create_controlled_scenario(rng: &mut impl rand::Rng) -> GameState {
    let mut state = new_game(2, 0, rng);

    // Clear factory 0 and fill with 1 Red + 3 Blue tiles
    // This ensures taking Red gives exactly 1 tile (fits Line 1 perfectly)
    state.factories.factories[0].len = 4;
    state.factories.factories[0].tiles = [Color::Red, Color::Blue, Color::Blue, Color::Blue];

    // Clear player 0's pattern line row 0
    state.players[0].pattern_lines[0].count = 0;
    state.players[0].pattern_lines[0].color = None;

    // Ensure it's player 0's turn
    state.current_player = 0;

    state
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum.max(1e-8)).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Load network
    let features = BasicFeatureExtractor::new(2);
    let hidden_size = 128;
    let obs_size = features.obs_size();
    let mut net = AlphaZeroNet::new(obs_size, hidden_size);

    println!("Loading checkpoint: {:?}", args.checkpoint);
    net.load(&args.checkpoint)?;

    // Set up MCTS agent with high simulations and no noise for analysis
    let mcts_config = MctsConfig {
        num_simulations: args.mcts_sims as u32,
        temperature: 0.0,          // Argmax for deterministic selection
        root_dirichlet_alpha: 0.0, // No exploration noise
        ..Default::default()
    };

    let mut agent = AlphaZeroMctsAgent::new(mcts_config, features.clone(), net);
    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);

    // Create game state
    let state = if args.controlled {
        println!("\n{BOLD}=== CONTROLLED SCENARIO ==={RESET}");
        println!(
            "Expectation: Agent should take Red from F0 to Line 1 (exactly 1 tile, perfect fit)"
        );
        create_controlled_scenario(&mut rng)
    } else {
        println!("\n{BOLD}=== RANDOM GAME STATE ==={RESET}");
        new_game(2, 0, &mut rng)
    };

    // Display the full board using the shared display module
    display_board(&state, Some(state.current_player));

    // Get raw network predictions
    println!("\n=== RAW NETWORK OUTPUTS ===");
    let obs = features.encode(&state, state.current_player);
    let (policy_logits, value) = agent.net.predict_single(&obs);

    println!("Network Value Prediction: {:.4}", value);
    println!("  Note: Value is the predicted remaining advantage; near 0 can be normal.");

    // Compute softmax probabilities from logits
    let logits_slice = policy_logits.as_slice::<f32>();
    let probs = softmax(logits_slice);

    // Build legal action mask
    let legal_actions = azul_engine::legal_actions(&state);
    let mut legal_mask = vec![false; ACTION_SPACE_SIZE];
    for action in &legal_actions {
        let id = ActionEncoder::encode(action);
        legal_mask[id as usize] = true;
    }

    // Find top network priors among legal actions
    let mut legal_priors: Vec<(usize, f32)> = probs
        .iter()
        .enumerate()
        .filter(|(i, _)| legal_mask[*i])
        .map(|(i, &p)| (i, p))
        .collect();
    legal_priors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop 10 Network Priors (legal actions only):");
    println!("{:<6} {:<25} {:<12}", "Rank", "Action", "Net Prior");
    println!("{}", "-".repeat(45));
    for (rank, (action_id, prior)) in legal_priors.iter().take(10).enumerate() {
        let action = ActionEncoder::decode(*action_id as u16);
        let action_str = format_action_compact(&action);
        println!("{:<6} {:<25} {:.6}", rank + 1, action_str, prior);
    }

    // Check if priors are roughly uniform (indicating no learning)
    let prior_variance: f32 = {
        let mean = legal_priors.iter().map(|(_, p)| p).sum::<f32>() / legal_priors.len() as f32;
        legal_priors
            .iter()
            .map(|(_, p)| (p - mean).powi(2))
            .sum::<f32>()
            / legal_priors.len() as f32
    };
    let prior_std = prior_variance.sqrt();
    println!("\nPrior statistics:");
    println!("  Std dev: {:.6}", prior_std);
    if prior_std < 0.01 {
        println!("  WARNING: Priors are nearly uniform - network may not have learned!");
    }

    // Count floor actions in top 5
    let floor_in_top5 = legal_priors
        .iter()
        .take(5)
        .filter(|(id, _)| {
            let action = ActionEncoder::decode(*id as u16);
            matches!(action.dest, DraftDestination::Floor)
        })
        .count();
    if floor_in_top5 > 0 {
        println!(
            "  WARNING: {} floor action(s) in top 5 priors!",
            floor_in_top5
        );
    }

    // Run MCTS search
    println!("\n=== MCTS SEARCH ({} simulations) ===", args.mcts_sims);
    let input = AgentInput {
        observation: &obs,
        legal_action_mask: &legal_mask,
        current_player: state.current_player,
        state: Some(&state),
    };

    let result: MctsSearchResult = agent.select_action_and_policy(&input, 0.0, &mut rng);

    // Get top MCTS moves
    let mut mcts_moves: Vec<(usize, f32)> = result
        .policy
        .iter()
        .enumerate()
        .filter(|(_, &p)| p > 0.001)
        .map(|(i, &p)| (i, p))
        .collect();
    mcts_moves.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop MCTS Actions (by visit count policy):");
    println!(
        "{:<6} {:<25} {:<12} {:<12}",
        "Rank", "Action", "MCTS Prob", "Net Prior"
    );
    println!("{}", "-".repeat(60));
    for (rank, (action_id, mcts_prob)) in mcts_moves.iter().take(10).enumerate() {
        let action = ActionEncoder::decode(*action_id as u16);
        let action_str = format_action_compact(&action);
        let net_prior = probs[*action_id];
        println!(
            "{:<6} {:<25} {:.6}     {:.6}",
            rank + 1,
            action_str,
            mcts_prob,
            net_prior
        );
    }

    // Check if selected action is a floor dump
    let selected_action = ActionEncoder::decode(result.action);
    println!("\n{BOLD}=== SELECTED ACTION ==={RESET}");
    println!("{}", format_action_compact(&selected_action));

    if matches!(selected_action.dest, DraftDestination::Floor) {
        println!("\nWARNING: Agent selected a FLOOR action!");
        println!("This suggests the network may have learned a degenerate policy.");
    }

    // Summary diagnostics
    println!("\n=== DIAGNOSTIC SUMMARY ===");
    let issues = vec![
        (prior_std < 0.01, "Nearly uniform priors"),
        (floor_in_top5 > 0, "Floor actions in top priors"),
        (
            matches!(selected_action.dest, DraftDestination::Floor),
            "MCTS selected floor action",
        ),
    ];

    let mut found_issues = false;
    for (is_issue, msg) in issues {
        if is_issue {
            println!("  [!] {}", msg);
            found_issues = true;
        }
    }

    if !found_issues {
        println!("  [OK] No obvious issues detected");
    } else {
        println!("\nPossible causes:");
        println!("  1. Reward signal too weak (try increasing reward scaling)");
        println!("  2. Network not receiving gradients (check training loop)");
        println!("  3. Not enough training iterations");
        println!("  4. MCTS backup logic issue (especially for multi-player)");
    }

    Ok(())
}
