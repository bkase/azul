//! Profiling infrastructure for measuring training performance.
//!
//! This module provides:
//! - Global counters for key metrics (self-play games, MCTS simulations, etc.)
//! - Time accumulators for different phases of training
//! - RAII-style scoped timers for measuring durations
//! - Summary reporting helpers
//!
//! All functionality is gated behind the `profiling` feature flag.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// Global counters for profiling metrics.
pub struct Counters {
    // Event counters
    pub self_play_games: AtomicU64,
    pub self_play_moves: AtomicU64,
    pub mcts_searches: AtomicU64,
    pub mcts_simulations: AtomicU64,
    pub mcts_nodes_created: AtomicU64,
    pub mcts_nn_evals: AtomicU64,
    pub mcts_nn_batches: AtomicU64,    // number of predict_batch calls
    pub mcts_nn_positions: AtomicU64,  // total positions evaluated (sum of batch sizes)
    pub train_steps: AtomicU64,
    pub fd_forward_evals: AtomicU64,
    pub env_steps: AtomicU64,
    pub nn_batch_forwards: AtomicU64,

    // Time accumulators (in nanoseconds)
    pub time_self_play_ns: AtomicU64,
    pub time_training_ns: AtomicU64,
    pub time_mcts_search_ns: AtomicU64,
    pub time_mcts_simulate_ns: AtomicU64,
    pub time_mcts_nn_eval_ns: AtomicU64,
    pub time_training_step_ns: AtomicU64,
    pub time_fd_grad_ns: AtomicU64,
    pub time_feature_encode_ns: AtomicU64,
    pub time_env_step_ns: AtomicU64,
    pub time_env_reset_ns: AtomicU64,
}

impl Counters {
    /// Create a new Counters instance with all values at zero.
    pub const fn new() -> Self {
        Self {
            // Event counters
            self_play_games: AtomicU64::new(0),
            self_play_moves: AtomicU64::new(0),
            mcts_searches: AtomicU64::new(0),
            mcts_simulations: AtomicU64::new(0),
            mcts_nodes_created: AtomicU64::new(0),
            mcts_nn_evals: AtomicU64::new(0),
            mcts_nn_batches: AtomicU64::new(0),
            mcts_nn_positions: AtomicU64::new(0),
            train_steps: AtomicU64::new(0),
            fd_forward_evals: AtomicU64::new(0),
            env_steps: AtomicU64::new(0),
            nn_batch_forwards: AtomicU64::new(0),

            // Time accumulators
            time_self_play_ns: AtomicU64::new(0),
            time_training_ns: AtomicU64::new(0),
            time_mcts_search_ns: AtomicU64::new(0),
            time_mcts_simulate_ns: AtomicU64::new(0),
            time_mcts_nn_eval_ns: AtomicU64::new(0),
            time_training_step_ns: AtomicU64::new(0),
            time_fd_grad_ns: AtomicU64::new(0),
            time_feature_encode_ns: AtomicU64::new(0),
            time_env_step_ns: AtomicU64::new(0),
            time_env_reset_ns: AtomicU64::new(0),
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.self_play_games.store(0, Ordering::Relaxed);
        self.self_play_moves.store(0, Ordering::Relaxed);
        self.mcts_searches.store(0, Ordering::Relaxed);
        self.mcts_simulations.store(0, Ordering::Relaxed);
        self.mcts_nodes_created.store(0, Ordering::Relaxed);
        self.mcts_nn_evals.store(0, Ordering::Relaxed);
        self.mcts_nn_batches.store(0, Ordering::Relaxed);
        self.mcts_nn_positions.store(0, Ordering::Relaxed);
        self.train_steps.store(0, Ordering::Relaxed);
        self.fd_forward_evals.store(0, Ordering::Relaxed);
        self.env_steps.store(0, Ordering::Relaxed);
        self.nn_batch_forwards.store(0, Ordering::Relaxed);

        self.time_self_play_ns.store(0, Ordering::Relaxed);
        self.time_training_ns.store(0, Ordering::Relaxed);
        self.time_mcts_search_ns.store(0, Ordering::Relaxed);
        self.time_mcts_simulate_ns.store(0, Ordering::Relaxed);
        self.time_mcts_nn_eval_ns.store(0, Ordering::Relaxed);
        self.time_training_step_ns.store(0, Ordering::Relaxed);
        self.time_fd_grad_ns.store(0, Ordering::Relaxed);
        self.time_feature_encode_ns.store(0, Ordering::Relaxed);
        self.time_env_step_ns.store(0, Ordering::Relaxed);
        self.time_env_reset_ns.store(0, Ordering::Relaxed);
    }
}

/// Global profiling counters instance.
pub static PROF: Counters = Counters::new();

/// RAII-style scoped timer that accumulates elapsed time on drop.
///
/// # Example
///
/// ```ignore
/// use azul_rl_env::profiling::{Timer, PROF};
///
/// fn expensive_operation() {
///     let _t = Timer::new(&PROF.time_self_play_ns);
///     // ... work happens here ...
/// } // timer adds elapsed time to PROF.time_self_play_ns on drop
/// ```
pub struct Timer {
    start: Instant,
    dest: &'static AtomicU64,
}

impl Timer {
    /// Create a new timer that will add elapsed time to `dest` on drop.
    #[inline]
    pub fn new(dest: &'static AtomicU64) -> Self {
        Self {
            start: Instant::now(),
            dest,
        }
    }
}

impl Drop for Timer {
    #[inline]
    fn drop(&mut self) {
        let elapsed_ns = self.start.elapsed().as_nanos() as u64;
        self.dest.fetch_add(elapsed_ns, Ordering::Relaxed);
    }
}

/// Print a human-readable summary of profiling counters and timings.
pub fn print_summary() {
    let games = PROF.self_play_games.load(Ordering::Relaxed);
    let moves = PROF.self_play_moves.load(Ordering::Relaxed);
    let mcts_searches = PROF.mcts_searches.load(Ordering::Relaxed);
    let mcts_sims = PROF.mcts_simulations.load(Ordering::Relaxed);
    let mcts_nodes = PROF.mcts_nodes_created.load(Ordering::Relaxed);
    let mcts_nn_evals = PROF.mcts_nn_evals.load(Ordering::Relaxed);
    let mcts_nn_batches = PROF.mcts_nn_batches.load(Ordering::Relaxed);
    let mcts_nn_positions = PROF.mcts_nn_positions.load(Ordering::Relaxed);
    let train_steps = PROF.train_steps.load(Ordering::Relaxed);
    let fd_forward_evals = PROF.fd_forward_evals.load(Ordering::Relaxed);
    let env_steps = PROF.env_steps.load(Ordering::Relaxed);
    let nn_batch_forwards = PROF.nn_batch_forwards.load(Ordering::Relaxed);

    let time_self_play_ns = PROF.time_self_play_ns.load(Ordering::Relaxed);
    let time_training_ns = PROF.time_training_ns.load(Ordering::Relaxed);
    let time_mcts_search_ns = PROF.time_mcts_search_ns.load(Ordering::Relaxed);
    let time_mcts_simulate_ns = PROF.time_mcts_simulate_ns.load(Ordering::Relaxed);
    let time_mcts_nn_eval_ns = PROF.time_mcts_nn_eval_ns.load(Ordering::Relaxed);
    let time_training_step_ns = PROF.time_training_step_ns.load(Ordering::Relaxed);
    let time_fd_grad_ns = PROF.time_fd_grad_ns.load(Ordering::Relaxed);
    let time_env_step_ns = PROF.time_env_step_ns.load(Ordering::Relaxed);
    let time_env_reset_ns = PROF.time_env_reset_ns.load(Ordering::Relaxed);

    // Convert ns to seconds
    let ns_to_sec = |ns: u64| ns as f64 / 1_000_000_000.0;

    eprintln!("\n=== Profiling Summary ===\n");

    eprintln!("Event Counts:");
    eprintln!("  Self-play games:     {:>12}", games);
    eprintln!("  Self-play moves:     {:>12}", moves);
    eprintln!("  MCTS searches:       {:>12}", mcts_searches);
    eprintln!("  MCTS simulations:    {:>12}", mcts_sims);
    eprintln!("  MCTS nodes created:  {:>12}", mcts_nodes);
    eprintln!("  MCTS NN evaluations: {:>12}", mcts_nn_evals);
    eprintln!("  MCTS NN batches:     {:>12}", mcts_nn_batches);
    eprintln!("  MCTS NN positions:   {:>12}", mcts_nn_positions);
    eprintln!("  Training steps:      {:>12}", train_steps);
    eprintln!("  FD forward evals:    {:>12}", fd_forward_evals);
    eprintln!("  Environment steps:   {:>12}", env_steps);
    eprintln!("  NN batch forwards:   {:>12}", nn_batch_forwards);

    eprintln!("\nTime Breakdown:");
    eprintln!("  Self-play total:      {:>10.3} s", ns_to_sec(time_self_play_ns));
    eprintln!("  Training total:       {:>10.3} s", ns_to_sec(time_training_ns));
    eprintln!("  MCTS search total:    {:>10.3} s", ns_to_sec(time_mcts_search_ns));
    eprintln!("  MCTS simulate total:  {:>10.3} s", ns_to_sec(time_mcts_simulate_ns));
    eprintln!("  MCTS NN eval total:   {:>10.3} s", ns_to_sec(time_mcts_nn_eval_ns));
    eprintln!("  Training step total:  {:>10.3} s", ns_to_sec(time_training_step_ns));
    eprintln!("  FD gradient total:    {:>10.3} s", ns_to_sec(time_fd_grad_ns));
    eprintln!("  Env step total:       {:>10.3} s", ns_to_sec(time_env_step_ns));
    eprintln!("  Env reset total:      {:>10.3} s", ns_to_sec(time_env_reset_ns));

    eprintln!("\nDerived Metrics:");

    // Games per second
    if time_self_play_ns > 0 && games > 0 {
        let games_per_sec = games as f64 / ns_to_sec(time_self_play_ns);
        eprintln!("  Games/sec:            {:>10.2}", games_per_sec);
    }

    // Moves per game
    if games > 0 {
        let moves_per_game = moves as f64 / games as f64;
        eprintln!("  Moves/game:           {:>10.2}", moves_per_game);
    }

    // Simulations per search
    if mcts_searches > 0 {
        let sims_per_search = mcts_sims as f64 / mcts_searches as f64;
        eprintln!("  Simulations/search:   {:>10.2}", sims_per_search);
    }

    // Simulations per second
    if time_mcts_search_ns > 0 && mcts_sims > 0 {
        let sims_per_sec = mcts_sims as f64 / ns_to_sec(time_mcts_search_ns);
        eprintln!("  Simulations/sec:      {:>10.0}", sims_per_sec);
    }

    // NN evals per second
    if time_mcts_nn_eval_ns > 0 && mcts_nn_evals > 0 {
        let nn_evals_per_sec = mcts_nn_evals as f64 / ns_to_sec(time_mcts_nn_eval_ns);
        eprintln!("  NN evals/sec:         {:>10.0}", nn_evals_per_sec);
    }

    // Average NN eval time
    if mcts_nn_evals > 0 {
        let avg_nn_eval_us = (time_mcts_nn_eval_ns as f64 / mcts_nn_evals as f64) / 1000.0;
        eprintln!("  Avg NN eval time:     {:>10.1} us", avg_nn_eval_us);
    }

    // Average batch size (batched MCTS)
    if mcts_nn_batches > 0 {
        let avg_batch_size = mcts_nn_positions as f64 / mcts_nn_batches as f64;
        eprintln!("  Avg NN batch size:    {:>10.2}", avg_batch_size);
    }

    // Positions per second (batched MCTS)
    if time_mcts_nn_eval_ns > 0 && mcts_nn_positions > 0 {
        let positions_per_sec = mcts_nn_positions as f64 / ns_to_sec(time_mcts_nn_eval_ns);
        eprintln!("  NN positions/sec:     {:>10.0}", positions_per_sec);
    }

    // Training steps per second
    if time_training_ns > 0 && train_steps > 0 {
        let steps_per_sec = train_steps as f64 / ns_to_sec(time_training_ns);
        eprintln!("  Training steps/sec:   {:>10.2}", steps_per_sec);
    }

    eprintln!();
}
