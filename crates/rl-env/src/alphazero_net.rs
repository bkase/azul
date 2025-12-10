//! Neural network for AlphaZero policy and value prediction
//!
//! Implements a simple fully-connected architecture using mlx_rs::nn primitives.

use mlx_rs::module::Module;
use mlx_rs::nn::{Linear, Relu, Sequential, Tanh};
use mlx_rs::Array;

use crate::mcts::PolicyValueNet;
use crate::{Observation, ACTION_SPACE_SIZE};

/// AlphaZero neural network with separate policy and value heads.
///
/// Architecture:
/// - Trunk: Linear(obs_size, hidden_size) -> Relu -> Linear(hidden_size, hidden_size) -> Relu
/// - Policy head: Linear(hidden_size, hidden_size) -> Relu -> Linear(hidden_size, ACTION_SPACE_SIZE)
/// - Value head: Linear(hidden_size, hidden_size) -> Relu -> Linear(hidden_size, 1) -> Tanh
pub struct AlphaZeroNet {
    pub obs_size: usize,
    pub hidden_size: usize,
    pub trunk: Sequential,
    pub policy_head: Sequential,
    pub value_head: Sequential,
}

impl AlphaZeroNet {
    /// Create a new AlphaZeroNet with the given observation and hidden sizes.
    pub fn new(obs_size: usize, hidden_size: usize) -> Self {
        // Build trunk: Linear -> Relu -> Linear -> Relu
        let trunk = Sequential::new()
            .append(Linear::new(obs_size as i32, hidden_size as i32).expect("Failed to create trunk linear 1"))
            .append(Relu)
            .append(Linear::new(hidden_size as i32, hidden_size as i32).expect("Failed to create trunk linear 2"))
            .append(Relu);

        // Build policy head: Linear -> Relu -> Linear (no softmax - raw logits)
        let policy_head = Sequential::new()
            .append(Linear::new(hidden_size as i32, hidden_size as i32).expect("Failed to create policy linear 1"))
            .append(Relu)
            .append(Linear::new(hidden_size as i32, ACTION_SPACE_SIZE as i32).expect("Failed to create policy linear 2"));

        // Build value head: Linear -> Relu -> Linear -> Tanh
        let value_head = Sequential::new()
            .append(Linear::new(hidden_size as i32, hidden_size as i32).expect("Failed to create value linear 1"))
            .append(Relu)
            .append(Linear::new(hidden_size as i32, 1).expect("Failed to create value linear 2"))
            .append(Tanh);

        Self {
            obs_size,
            hidden_size,
            trunk,
            policy_head,
            value_head,
        }
    }

    /// Forward pass for a batched input.
    /// `obs_batch` shape: [B, obs_size]
    /// Returns: (policy_logits [B, ACTION_SPACE_SIZE], values [B])
    pub fn forward_batch(&mut self, obs_batch: &Array) -> (Array, Array) {
        // Run through trunk
        let h = self.trunk.forward(obs_batch).expect("Trunk forward failed");

        // Run through policy head
        let policy_logits = self.policy_head.forward(&h).expect("Policy head forward failed");

        // Run through value head - output is [B, 1], squeeze to [B]
        let value_2d = self.value_head.forward(&h).expect("Value head forward failed");
        let value = value_2d.squeeze().expect("Failed to squeeze value");

        (policy_logits, value)
    }

    /// Forward pass for a single observation [obs_size].
    /// Returns: (policy_logits [ACTION_SPACE_SIZE], value scalar)
    pub fn forward_single(&mut self, obs: &Observation) -> (Array, f32) {
        // Reshape to [1, obs_size]
        let obs_batch = obs
            .reshape(&[1, self.obs_size as i32])
            .expect("Failed to reshape observation");

        let (policy_logits_batch, value_batch) = self.forward_batch(&obs_batch);

        // Squeeze [1, ACTION_SPACE_SIZE] -> [ACTION_SPACE_SIZE]
        let policy_logits = policy_logits_batch
            .squeeze()
            .expect("Failed to squeeze policy logits");

        // Get scalar value from [1] array
        let value_slice = value_batch.as_slice::<f32>();
        let value_scalar = value_slice[0];

        (policy_logits, value_scalar)
    }
}

impl PolicyValueNet for AlphaZeroNet {
    fn predict_single(&mut self, obs: &Observation) -> (Array, f32) {
        self.forward_single(obs)
    }

    fn predict_batch(&mut self, obs_batch: &Array) -> (Array, Array) {
        self.forward_batch(obs_batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alphazero_net_shapes() {
        let obs_size = 100;
        let hidden_size = 64;
        let mut net = AlphaZeroNet::new(obs_size, hidden_size);

        // Create a test observation
        let obs = Array::zeros::<f32>(&[obs_size as i32]).expect("Failed to create test observation");

        let (policy_logits, value) = net.predict_single(&obs);

        // Check policy shape
        assert_eq!(
            policy_logits.shape(),
            &[ACTION_SPACE_SIZE as i32],
            "Policy logits should have shape [ACTION_SPACE_SIZE]"
        );

        // Check value is in valid range (Tanh output)
        assert!(
            value >= -1.0 && value <= 1.0,
            "Value {} should be in [-1, 1]",
            value
        );
    }

    #[test]
    fn test_alphazero_net_batch() {
        let obs_size = 100;
        let hidden_size = 64;
        let batch_size = 4;
        let mut net = AlphaZeroNet::new(obs_size, hidden_size);

        // Create batch of observations
        let obs_batch = Array::zeros::<f32>(&[batch_size, obs_size as i32])
            .expect("Failed to create batch observation");

        let (policy_logits, values) = net.predict_batch(&obs_batch);

        // Check policy shape: [batch, ACTION_SPACE_SIZE]
        assert_eq!(
            policy_logits.shape(),
            &[batch_size, ACTION_SPACE_SIZE as i32],
            "Policy logits batch should have shape [batch, ACTION_SPACE_SIZE]"
        );

        // Check values shape: [batch]
        assert_eq!(
            values.shape(),
            &[batch_size],
            "Values batch should have shape [batch]"
        );

        // Check values are in valid range
        let values_slice = values.as_slice::<f32>();
        for &v in values_slice {
            assert!(
                v >= -1.0 && v <= 1.0,
                "Value {} should be in [-1, 1]",
                v
            );
        }
    }

    #[test]
    fn test_alphazero_net_determinism() {
        let obs_size = 100;
        let mut net = AlphaZeroNet::new(obs_size, 64);

        // Same input should give same output
        let obs = Array::zeros::<f32>(&[obs_size as i32]).expect("Failed to create test observation");

        let (p1, v1) = net.predict_single(&obs);
        let (p2, v2) = net.predict_single(&obs);

        let p1_slice = p1.as_slice::<f32>();
        let p2_slice = p2.as_slice::<f32>();

        assert_eq!(p1_slice, p2_slice, "Policy should be deterministic");
        assert_eq!(v1, v2, "Value should be deterministic");
    }
}
