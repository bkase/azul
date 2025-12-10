//! RL Environment trait and AzulEnv implementation

use crate::{apply_action, legal_actions, new_game, GameState, Phase, MAX_PLAYERS};
use rand::Rng;

use super::{
    create_zero_observation, ActionEncoder, ActionId, EnvConfig, EnvStep, FeatureExtractor,
    Observation, Reward, RewardScheme, StepError, ACTION_SPACE_SIZE,
};

/// Generic environment interface for RL
pub trait Environment {
    /// Type used to represent observations
    type ObservationType;

    /// Type used to represent actions
    type ActionType;

    /// Type used to represent rewards
    type RewardType;

    /// Reset the environment to a fresh episode (new game).
    ///
    /// Returns the first EnvStep, representing the initial state prior
    /// to any actions.
    fn reset(&mut self, rng: &mut impl Rng) -> EnvStep<Self::ObservationType, Self::RewardType>;

    /// Apply an action for the current player, advance the environment
    /// by one step, and return the resulting EnvStep.
    fn step(
        &mut self,
        action: Self::ActionType,
        rng: &mut impl Rng,
    ) -> Result<EnvStep<Self::ObservationType, Self::RewardType>, StepError>;
}

/// Azul RL Environment
pub struct AzulEnv<F: FeatureExtractor> {
    /// Underlying Azul engine state
    pub game_state: GameState,

    /// Environment configuration
    pub config: EnvConfig,

    /// Feature extractor for building observations
    pub features: F,

    /// Last action applied (ActionId), if any
    pub last_action: Option<ActionId>,

    /// Whether this episode has ended
    pub done: bool,
}

impl<F: FeatureExtractor> AzulEnv<F> {
    /// Create a new AzulEnv with the given configuration and feature extractor.
    ///
    /// The environment is not initialized until reset() is called.
    pub fn new(config: EnvConfig, features: F) -> Self {
        assert!(
            (2..=4).contains(&config.num_players),
            "num_players must be 2, 3, or 4"
        );

        Self {
            game_state: GameState::default(),
            config,
            features,
            last_action: None,
            done: true, // Not initialized until reset
        }
    }

    /// Build the legal action mask for the current state
    pub fn build_legal_action_mask(&self) -> Vec<bool> {
        let mut mask = vec![false; ACTION_SPACE_SIZE];
        let actions = legal_actions(&self.game_state);

        for action in actions {
            let id = ActionEncoder::encode(&action) as usize;
            if id < ACTION_SPACE_SIZE {
                mask[id] = true;
            }
        }

        mask
    }

    /// Build observations for all players
    fn build_observations(&self) -> [Observation; MAX_PLAYERS] {
        std::array::from_fn(|idx| {
            if (idx as u8) < self.game_state.num_players {
                self.features.encode(&self.game_state, idx as u8)
            } else {
                create_zero_observation(self.features.obs_size())
            }
        })
    }
}

impl<F: FeatureExtractor> Environment for AzulEnv<F> {
    type ObservationType = Observation;
    type ActionType = ActionId;
    type RewardType = Reward;

    fn reset(&mut self, rng: &mut impl Rng) -> EnvStep<Self::ObservationType, Self::RewardType> {
        // 1. Build a fresh GameState
        self.game_state = new_game(self.config.num_players, 0, rng);
        self.last_action = None;
        self.done = false;

        // 2. Build per-player observations
        let observations = self.build_observations();

        // 3. Rewards are all zero on reset
        let rewards = [0.0f32; MAX_PLAYERS];

        // 4. Build legal action mask for the current player
        let legal_action_mask = self.build_legal_action_mask();

        // 5. Optional full state
        let state_clone = if self.config.include_full_state_in_step {
            Some(self.game_state.clone())
        } else {
            None
        };

        EnvStep {
            observations,
            rewards,
            done: false,
            current_player: self.game_state.current_player,
            legal_action_mask,
            last_action: None,
            state: state_clone,
        }
    }

    fn step(
        &mut self,
        action_id: Self::ActionType,
        rng: &mut impl Rng,
    ) -> Result<EnvStep<Self::ObservationType, Self::RewardType>, StepError> {
        // 1. Check if episode is done
        if self.done {
            return Err(StepError::EpisodeDone);
        }

        // 2. Validate action_id range
        if action_id as usize >= ACTION_SPACE_SIZE {
            return Err(StepError::InvalidActionId(action_id));
        }

        // 3. Check legality
        let legal_mask = self.build_legal_action_mask();
        if !legal_mask[action_id as usize] {
            return Err(StepError::IllegalAction(action_id));
        }

        // 4. Decode action
        let action = ActionEncoder::decode(action_id);

        // 5. Record previous scores
        let mut prev_scores = [0i16; MAX_PLAYERS];
        #[allow(clippy::needless_range_loop)]
        for p in 0..self.game_state.num_players as usize {
            prev_scores[p] = self.game_state.players[p].score;
        }

        // 6. Apply action via engine
        let apply_result = apply_action(self.game_state.clone(), action, rng)
            .expect("Engine-level errors should be impossible if masking is correct");

        self.game_state = apply_result.state;

        // 7. Compute rewards based on RewardScheme
        let mut rewards = [0.0f32; MAX_PLAYERS];

        match self.config.reward_scheme {
            RewardScheme::DenseScoreDelta => {
                for p in 0..self.game_state.num_players as usize {
                    let new_score = self.game_state.players[p].score;
                    let delta = (new_score - prev_scores[p]) as f32;
                    rewards[p] = delta;
                }
            }

            RewardScheme::TerminalOnly => {
                if self.game_state.phase == Phase::GameOver {
                    let mut final_scores = [0.0f32; MAX_PLAYERS];
                    #[allow(clippy::needless_range_loop)]
                    for p in 0..self.game_state.num_players as usize {
                        final_scores[p] = self.game_state.players[p].score as f32;
                    }
                    let n = self.game_state.num_players as usize;
                    let mean = final_scores[0..n].iter().sum::<f32>() / (n as f32);
                    for p in 0..n {
                        rewards[p] = final_scores[p] - mean;
                    }
                }
                // Non-terminal steps have zero reward (already initialized)
            }
        }

        // 8. Update done flag
        self.done = self.game_state.phase == Phase::GameOver;

        // 9. Build new observations
        let observations = self.build_observations();

        // 10. Build new legal action mask (or all-false if done)
        let legal_action_mask = if self.done {
            vec![false; ACTION_SPACE_SIZE]
        } else {
            self.build_legal_action_mask()
        };

        // 11. Update last_action
        self.last_action = Some(action_id);

        // 12. Optional state clone
        let state_clone = if self.config.include_full_state_in_step {
            Some(self.game_state.clone())
        } else {
            None
        };

        Ok(EnvStep {
            observations,
            rewards,
            done: self.done,
            current_player: self.game_state.current_player,
            legal_action_mask,
            last_action: self.last_action,
            state: state_clone,
        })
    }
}

/// Run a self-play episode with the given agent
pub fn self_play_episode<F: FeatureExtractor, A: super::Agent>(
    env: &mut AzulEnv<F>,
    agent: &mut A,
    rng: &mut impl Rng,
    replay_buffer: &mut Vec<super::Transition>,
) {
    use super::{AgentInput, Transition};

    // 1. Reset environment
    let mut step = env.reset(rng);

    while !step.done {
        let p = step.current_player as usize;

        // 2. Build input for current player
        let agent_input = AgentInput {
            observation: &step.observations[p],
            legal_action_mask: &step.legal_action_mask,
            current_player: step.current_player,
        };

        // 3. Select action
        let action_id = agent.select_action(&agent_input, rng);

        // 4. Step environment
        let next_step = env.step(action_id, rng).expect("step failed unexpectedly");

        // 5. Log transition for training (replay buffer)
        let transition = Transition {
            player: step.current_player,
            observation_before: step.observations[p].clone(),
            action_id,
            reward: next_step.rewards[p],
            observation_after: next_step.observations[p].clone(),
            done: next_step.done,
        };
        replay_buffer.push(transition);

        // 6. Move on
        step = next_step;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl_env::{Agent, BasicFeatureExtractor};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_azul_env_new() {
        let config = EnvConfig::default();
        let features = BasicFeatureExtractor::new(2);
        let env = AzulEnv::new(config, features);

        assert!(env.done); // Not initialized until reset
    }

    #[test]
    fn test_azul_env_reset() {
        let config = EnvConfig::default();
        let features = BasicFeatureExtractor::new(2);
        let mut env = AzulEnv::new(config, features);

        let mut rng = StdRng::seed_from_u64(42);
        let step = env.reset(&mut rng);

        assert!(!step.done);
        assert!(!env.done);
        assert!(step.last_action.is_none());
        assert!(!step.legal_action_mask.is_empty());
        assert!(step.legal_action_mask.iter().any(|&x| x)); // At least one legal action
    }

    #[test]
    fn test_azul_env_step() {
        let config = EnvConfig::default();
        let features = BasicFeatureExtractor::new(2);
        let mut env = AzulEnv::new(config, features);

        let mut rng = StdRng::seed_from_u64(42);
        let step = env.reset(&mut rng);

        // Find a legal action
        let action_id = step
            .legal_action_mask
            .iter()
            .enumerate()
            .find(|(_, &legal)| legal)
            .map(|(id, _)| id as ActionId)
            .expect("Should have at least one legal action");

        let next_step = env.step(action_id, &mut rng).expect("Step should succeed");

        assert_eq!(next_step.last_action, Some(action_id));
    }

    #[test]
    fn test_azul_env_episode_done_error() {
        let config = EnvConfig::default();
        let features = BasicFeatureExtractor::new(2);
        let mut env = AzulEnv::new(config, features);

        let mut rng = StdRng::seed_from_u64(42);
        env.reset(&mut rng);

        // Manually set done
        env.done = true;

        let result = env.step(0, &mut rng);
        assert!(matches!(result, Err(StepError::EpisodeDone)));
    }

    #[test]
    fn test_azul_env_illegal_action_error() {
        let config = EnvConfig::default();
        let features = BasicFeatureExtractor::new(2);
        let mut env = AzulEnv::new(config, features);

        let mut rng = StdRng::seed_from_u64(42);
        let step = env.reset(&mut rng);

        // Find an illegal action
        let illegal_action = step
            .legal_action_mask
            .iter()
            .enumerate()
            .find(|(_, &legal)| !legal)
            .map(|(id, _)| id as ActionId);

        if let Some(action_id) = illegal_action {
            let result = env.step(action_id, &mut rng);
            assert!(matches!(result, Err(StepError::IllegalAction(_))));
        }
    }

    #[test]
    fn test_legal_action_mask_consistency() {
        let config = EnvConfig::default();
        let features = BasicFeatureExtractor::new(2);
        let mut env = AzulEnv::new(config, features);

        let mut rng = StdRng::seed_from_u64(42);
        env.reset(&mut rng);

        let mask = env.build_legal_action_mask();
        let legal_actions_from_engine = legal_actions(&env.game_state);

        // All engine legal actions should be in mask
        for action in &legal_actions_from_engine {
            let id = ActionEncoder::encode(action) as usize;
            if id < ACTION_SPACE_SIZE {
                assert!(mask[id], "Legal action should be in mask");
            }
        }

        // All mask true entries should correspond to legal actions
        for (id, &legal) in mask.iter().enumerate() {
            if legal {
                let action = ActionEncoder::decode(id as ActionId);
                assert!(
                    legal_actions_from_engine.contains(&action),
                    "Mask true entry should be a legal action"
                );
            }
        }
    }

    #[test]
    fn test_dense_score_delta_reward_correctness() {
        // Test that cumulative rewards equal final scores
        let config = EnvConfig {
            num_players: 2,
            reward_scheme: RewardScheme::DenseScoreDelta,
            include_full_state_in_step: false,
        };
        let features = BasicFeatureExtractor::new(2);
        let mut env = AzulEnv::new(config, features);
        let mut agent = super::super::RandomAgent::new();

        let mut rng = StdRng::seed_from_u64(42);

        // Run 5 episodes
        for _ in 0..5 {
            let mut cum_rewards = [0.0f32; crate::MAX_PLAYERS];
            let mut step = env.reset(&mut rng);

            while !step.done {
                let p = step.current_player as usize;
                let input = super::super::AgentInput {
                    observation: &step.observations[p],
                    legal_action_mask: &step.legal_action_mask,
                    current_player: step.current_player,
                };

                let action_id = agent.select_action(&input, &mut rng);
                let next_step = env.step(action_id, &mut rng).expect("step should succeed");

                // Accumulate rewards
                for player in 0..env.game_state.num_players as usize {
                    cum_rewards[player] += next_step.rewards[player];
                }

                step = next_step;
            }

            // Verify cumulative rewards equal final scores
            for player in 0..env.game_state.num_players as usize {
                let final_score = env.game_state.players[player].score as f32;
                assert!(
                    (cum_rewards[player] - final_score).abs() < 0.001,
                    "Cumulative reward {} should equal final score {} for player {}",
                    cum_rewards[player],
                    final_score,
                    player
                );
            }
        }
    }

    #[test]
    fn test_terminal_only_reward_correctness() {
        // Test TerminalOnly reward scheme
        let config = EnvConfig {
            num_players: 2,
            reward_scheme: RewardScheme::TerminalOnly,
            include_full_state_in_step: false,
        };
        let features = BasicFeatureExtractor::new(2);
        let mut env = AzulEnv::new(config, features);
        let mut agent = super::super::RandomAgent::new();

        let mut rng = StdRng::seed_from_u64(42);

        // Run 5 episodes
        for _ in 0..5 {
            let mut step = env.reset(&mut rng);
            let mut non_terminal_rewards_are_zero = true;

            while !step.done {
                let p = step.current_player as usize;
                let input = super::super::AgentInput {
                    observation: &step.observations[p],
                    legal_action_mask: &step.legal_action_mask,
                    current_player: step.current_player,
                };

                let action_id = agent.select_action(&input, &mut rng);
                let next_step = env.step(action_id, &mut rng).expect("step should succeed");

                // Check non-terminal rewards are zero
                if !next_step.done {
                    for player in 0..env.game_state.num_players as usize {
                        if next_step.rewards[player] != 0.0 {
                            non_terminal_rewards_are_zero = false;
                        }
                    }
                }

                step = next_step;
            }

            assert!(
                non_terminal_rewards_are_zero,
                "Non-terminal rewards should be zero"
            );

            // Verify terminal rewards
            // 1. Sum to zero (zero-sum property)
            let n = env.game_state.num_players as usize;
            let reward_sum: f32 = step.rewards[0..n].iter().sum();
            assert!(
                reward_sum.abs() < 0.001,
                "Terminal rewards should sum to zero, got {}",
                reward_sum
            );

            // 2. Equal to score minus mean
            let final_scores: Vec<f32> = (0..n)
                .map(|p| env.game_state.players[p].score as f32)
                .collect();
            let mean = final_scores.iter().sum::<f32>() / n as f32;

            for player in 0..n {
                let expected = final_scores[player] - mean;
                assert!(
                    (step.rewards[player] - expected).abs() < 0.001,
                    "Terminal reward {} should equal {} for player {}",
                    step.rewards[player],
                    expected,
                    player
                );
            }
        }
    }

    #[test]
    fn test_environment_determinism() {
        // Test that same seed + same actions = same results
        // We use separate RNGs for agent and env to ensure env behavior is deterministic
        let config = EnvConfig {
            num_players: 2,
            reward_scheme: RewardScheme::DenseScoreDelta,
            include_full_state_in_step: true,
        };

        // Run 1: collect actions and results
        let features1 = BasicFeatureExtractor::new(2);
        let mut env1 = AzulEnv::new(config.clone(), features1);
        let mut agent1 = super::super::RandomAgent::new();
        let mut agent_rng1 = StdRng::seed_from_u64(100); // Separate RNG for agent
        let mut env_rng1 = StdRng::seed_from_u64(42); // RNG for env

        let mut actions_taken: Vec<ActionId> = Vec::new();
        let mut rewards_1: Vec<[f32; crate::MAX_PLAYERS]> = Vec::new();
        let mut observations_1: Vec<Vec<f32>> = Vec::new();

        let mut step1 = env1.reset(&mut env_rng1);
        observations_1.push(step1.observations[0].as_slice().to_vec());

        while !step1.done {
            let p = step1.current_player as usize;
            let input = super::super::AgentInput {
                observation: &step1.observations[p],
                legal_action_mask: &step1.legal_action_mask,
                current_player: step1.current_player,
            };

            let action_id = agent1.select_action(&input, &mut agent_rng1);
            actions_taken.push(action_id);

            let next = env1
                .step(action_id, &mut env_rng1)
                .expect("step should succeed");
            rewards_1.push(next.rewards);
            observations_1.push(next.observations[0].as_slice().to_vec());
            step1 = next;
        }

        // Run 2: replay same actions with same env seed
        let features2 = BasicFeatureExtractor::new(2);
        let mut env2 = AzulEnv::new(config, features2);
        let mut env_rng2 = StdRng::seed_from_u64(42); // Same seed as env_rng1

        let step2_init = env2.reset(&mut env_rng2);

        // Verify initial observation matches
        assert_eq!(
            step2_init.observations[0].as_slice().to_vec(),
            observations_1[0],
            "Initial observations should match"
        );

        let mut step2 = step2_init;
        for (idx, &action_id) in actions_taken.iter().enumerate() {
            let next = env2
                .step(action_id, &mut env_rng2)
                .expect("step should succeed");

            // Verify rewards match
            assert_eq!(
                next.rewards, rewards_1[idx],
                "Rewards should be deterministic at step {}",
                idx
            );

            // Verify observations match
            assert_eq!(
                next.observations[0].as_slice().to_vec(),
                observations_1[idx + 1],
                "Observations should be deterministic at step {}",
                idx
            );

            step2 = next;
        }

        // Verify episode terminates at same point
        assert!(step2.done, "Episode should be done");

        // Verify final game states match
        assert_eq!(
            env1.game_state.players[0].score, env2.game_state.players[0].score,
            "Final scores should match"
        );
        assert_eq!(
            env1.game_state.players[1].score, env2.game_state.players[1].score,
            "Final scores should match for player 1"
        );
    }

    #[test]
    fn test_integration_smoke_100_episodes() {
        // Integration test: run 100 complete episodes
        let config = EnvConfig {
            num_players: 2,
            reward_scheme: RewardScheme::DenseScoreDelta,
            include_full_state_in_step: false,
        };
        let features = BasicFeatureExtractor::new(2);
        let mut env = AzulEnv::new(config, features);
        let mut agent = super::super::RandomAgent::new();

        let mut rng = StdRng::seed_from_u64(12345);

        let mut total_steps = 0u64;
        let mut total_score_p0 = 0i32;
        let mut total_score_p1 = 0i32;

        for episode in 0..100 {
            let mut steps_this_episode = 0;
            let mut step = env.reset(&mut rng);

            // Verify initial observations are valid
            assert_eq!(
                step.observations[0].shape(),
                &[env.features.obs_size() as i32],
                "Observation shape should be correct"
            );

            while !step.done {
                let p = step.current_player as usize;
                let input = super::super::AgentInput {
                    observation: &step.observations[p],
                    legal_action_mask: &step.legal_action_mask,
                    current_player: step.current_player,
                };

                let action_id = agent.select_action(&input, &mut rng);
                let next = env.step(action_id, &mut rng).expect("step should succeed");

                steps_this_episode += 1;
                step = next;
            }

            // Episode should terminate
            assert!(step.done, "Episode {} should terminate", episode);

            total_steps += steps_this_episode;
            total_score_p0 += env.game_state.players[0].score as i32;
            total_score_p1 += env.game_state.players[1].score as i32;
        }

        // Print statistics
        let avg_steps = total_steps as f64 / 100.0;
        let avg_score_p0 = total_score_p0 as f64 / 100.0;
        let avg_score_p1 = total_score_p1 as f64 / 100.0;

        println!("Integration test results:");
        println!("  Average steps per episode: {:.1}", avg_steps);
        println!("  Average score P0: {:.1}", avg_score_p0);
        println!("  Average score P1: {:.1}", avg_score_p1);

        // Basic sanity checks
        assert!(avg_steps > 10.0, "Episodes should take more than 10 steps");
        assert!(
            avg_steps < 500.0,
            "Episodes should take fewer than 500 steps"
        );
        assert!(avg_score_p0 > 0.0, "Average score should be positive");
        assert!(avg_score_p1 > 0.0, "Average score should be positive");
    }
}
