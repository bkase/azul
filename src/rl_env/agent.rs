//! Agent API for action selection

use rand::Rng;

use super::{ActionId, Observation};
use crate::PlayerIdx;

/// Inputs provided to an agent when selecting an action
pub struct AgentInput<'a> {
    /// Observation vector for the player whose turn it is
    pub observation: &'a Observation,

    /// Mask over action IDs:
    /// legal_action_mask[id] == true if the action is legal
    pub legal_action_mask: &'a [bool],

    /// Index of the player whose turn it is
    pub current_player: PlayerIdx,
}

/// Trait for anything that can choose actions in the environment:
/// random policy, neural-net-based policy, or human input.
pub trait Agent {
    /// Choose a legal action given an observation and legal-action mask.
    ///
    /// Requirement:
    /// - Must only return ActionIds for which legal_action_mask[id as usize] == true.
    /// - May use rng for exploration.
    fn select_action(&mut self, input: &AgentInput, rng: &mut impl Rng) -> ActionId;
}

/// Random agent that uniformly samples from legal actions
#[derive(Clone, Debug, Default)]
pub struct RandomAgent;

impl RandomAgent {
    pub fn new() -> Self {
        Self
    }
}

impl Agent for RandomAgent {
    fn select_action(&mut self, input: &AgentInput, rng: &mut impl Rng) -> ActionId {
        let legal_ids: Vec<ActionId> = input
            .legal_action_mask
            .iter()
            .enumerate()
            .filter(|(_, &legal)| legal)
            .map(|(id, _)| id as ActionId)
            .collect();

        assert!(
            !legal_ids.is_empty(),
            "No legal actions available for agent"
        );

        let idx = rng.random_range(0..legal_ids.len() as u32) as usize;
        legal_ids[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl_env::{AzulEnv, BasicFeatureExtractor, EnvConfig, Environment};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_random_agent_selects_legal_action() {
        let config = EnvConfig::default();
        let features = BasicFeatureExtractor::new(2);
        let mut env = AzulEnv::new(config, features);

        let mut rng = StdRng::seed_from_u64(42);
        let step = env.reset(&mut rng);

        let mut agent = RandomAgent::new();
        let input = AgentInput {
            observation: &step.observations[step.current_player as usize],
            legal_action_mask: &step.legal_action_mask,
            current_player: step.current_player,
        };

        let action = agent.select_action(&input, &mut rng);

        // Action should be legal
        assert!(
            step.legal_action_mask[action as usize],
            "Random agent should select legal action"
        );
    }

    #[test]
    fn test_random_agent_many_selections() {
        let config = EnvConfig::default();
        let features = BasicFeatureExtractor::new(2);
        let mut env = AzulEnv::new(config, features);

        let mut rng = StdRng::seed_from_u64(42);
        let step = env.reset(&mut rng);

        let mut agent = RandomAgent::new();
        let input = AgentInput {
            observation: &step.observations[step.current_player as usize],
            legal_action_mask: &step.legal_action_mask,
            current_player: step.current_player,
        };

        // Select many times and verify all are legal
        for _ in 0..100 {
            let action = agent.select_action(&input, &mut rng);
            assert!(step.legal_action_mask[action as usize]);
        }
    }
}
