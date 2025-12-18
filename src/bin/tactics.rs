//! Tactical regression evaluator for trained Azul agents.
//!
//! Runs a small suite of hand-constructed "must block" scenarios (e.g. denying +10 color sets)
//! and reports the agent's success rate.

use std::path::PathBuf;

use clap::Parser;
use rand::SeedableRng;

use azul_engine::{new_game, Color, Token, ALL_COLORS, WALL_DEST_COL};
use azul_rl_env::{
    alphazero::training::{MctsAgentExt, MctsSearchResult, TrainableModel},
    ActionEncoder, AgentInput, AlphaZeroMctsAgent, AlphaZeroNet, BasicFeatureExtractor,
    FeatureExtractor, MctsConfig, ACTION_SPACE_SIZE,
};

#[derive(Parser, Debug)]
#[command(name = "tactics")]
#[command(about = "Run tactical regression checks on a checkpoint", long_about = None)]
struct Args {
    /// Path to checkpoint file (defaults to latest in checkpoints/)
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    /// MCTS simulations per move for tactic evaluation (more = stronger but slower)
    #[arg(long, default_value_t = 800)]
    mcts_sims: usize,

    /// Random seed (used only for initial GameState construction)
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn find_latest_checkpoint() -> Option<PathBuf> {
    let checkpoint_dir = PathBuf::from("checkpoints");
    if !checkpoint_dir.exists() {
        return None;
    }

    let mut checkpoints: Vec<_> = std::fs::read_dir(&checkpoint_dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
        .collect();

    checkpoints.sort_by_key(|e| e.path());
    checkpoints.last().map(|e| e.path())
}

fn make_color_set_denial_state(
    threat_color: Color,
    distractor_color: Color,
    rng: &mut impl rand::Rng,
) -> azul_engine::GameState {
    assert_ne!(threat_color, distractor_color);

    let mut state = new_game(2, 0, rng);
    state.current_player = 0;

    // Clear factories (draft from center only).
    for f in 0..state.factories.num_factories as usize {
        state.factories.factories[f].len = 0;
    }

    // Center: FP marker + threat + distractor.
    state.center.len = 3;
    state.center.items[0] = Token::FirstPlayerMarker;
    state.center.items[1] = Token::Tile(threat_color);
    state.center.items[2] = Token::Tile(distractor_color);

    // Clear per-player transient state.
    for p in 0..2 {
        state.players[p].pattern_lines[0].color = None;
        state.players[p].pattern_lines[0].count = 0;
        state.players[p].floor.len = 0;
        state.players[p].score = 0;
    }

    // Force game end at end-of-round by giving player 0 a complete row (row 4).
    for color in ALL_COLORS {
        let col = WALL_DEST_COL[4][color as usize] as usize;
        state.players[0].wall[4][col] = Some(color);
    }

    // Player 1 has 4/5 of threat_color already (rows 1..4).
    for (dest_cols, wall_row) in WALL_DEST_COL
        .iter()
        .zip(state.players[1].wall.iter_mut())
        .skip(1)
    {
        let col = dest_cols[threat_color as usize] as usize;
        wall_row[col] = Some(threat_color);
    }
    // Missing tile in row 0.
    let missing_col_row0 = WALL_DEST_COL[0][threat_color as usize] as usize;
    state.players[1].wall[0][missing_col_row0] = None;

    state
}

fn run_single_tactic(
    agent: &mut AlphaZeroMctsAgent<BasicFeatureExtractor, AlphaZeroNet>,
    features: &BasicFeatureExtractor,
    state: &azul_engine::GameState,
    rng: &mut impl rand::Rng,
) -> MctsSearchResult {
    let obs = features.encode(state, state.current_player);
    let legal = azul_engine::legal_actions(state);
    let mut legal_mask = vec![false; ACTION_SPACE_SIZE];
    for a in &legal {
        let id = ActionEncoder::encode(a);
        legal_mask[id as usize] = true;
    }

    let input = AgentInput {
        observation: &obs,
        legal_action_mask: &legal_mask,
        current_player: state.current_player,
        state: Some(state),
    };

    agent.select_action_and_policy(&input, 0.0, rng)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let checkpoint_path = args
        .checkpoint
        .or_else(find_latest_checkpoint)
        .ok_or_else(|| {
            "No checkpoint found. Run training first or specify --checkpoint".to_string()
        })?;

    eprintln!("Loading checkpoint: {checkpoint_path:?}");

    let features = BasicFeatureExtractor::new(2);
    let obs_size = features.obs_size();
    let hidden_size = 128;
    let mut net = AlphaZeroNet::new(obs_size, hidden_size);
    net.load(&checkpoint_path)?;

    let mcts_config = MctsConfig {
        num_simulations: args.mcts_sims as u32,
        ..Default::default()
    };
    let mut agent = AlphaZeroMctsAgent::new(mcts_config, features.clone(), net);

    let mut rng = rand::rngs::StdRng::seed_from_u64(args.seed);

    // Suite 1: deny +10 color-set completion (one case per color).
    let mut successes = 0usize;
    let mut total = 0usize;

    for (i, &threat) in ALL_COLORS.iter().enumerate() {
        let distractor = ALL_COLORS[(i + 1) % ALL_COLORS.len()];
        let state = make_color_set_denial_state(threat, distractor, &mut rng);

        let result = run_single_tactic(&mut agent, &features, &state, &mut rng);
        let chosen = ActionEncoder::decode(result.action);

        let ok = chosen.color == threat;
        total += 1;
        if ok {
            successes += 1;
        }

        println!(
            "deny_color_set[{threat:?}]: chose {chosen:?} -> {}",
            if ok { "OK" } else { "MISS" }
        );
    }

    println!();
    println!(
        "Summary: {successes}/{total} ({:.1}%)",
        100.0 * (successes as f32) / (total.max(1) as f32)
    );

    Ok(())
}
