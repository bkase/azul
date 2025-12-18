//! Neural network for AlphaZero policy and value prediction
//!
//! Implements a simple fully-connected architecture using mlx_rs::nn primitives.

use mlx_rs::macros::ModuleParameters;
use mlx_rs::module::{Module, ModuleParameters};
use mlx_rs::nn::{Linear, Relu, Sequential, Tanh};
use mlx_rs::transforms::eval_params;
use mlx_rs::Array;

use crate::mcts::PolicyValueNet;
use crate::{Observation, ACTION_SPACE_SIZE};

/// AlphaZero neural network with separate policy and value heads.
///
/// Architecture:
/// - Trunk: Linear(obs_size, hidden_size) -> Relu -> Linear(hidden_size, hidden_size) -> Relu
/// - Policy head: Linear(hidden_size, hidden_size) -> Relu -> Linear(hidden_size, ACTION_SPACE_SIZE)
/// - Value head: Linear(hidden_size, hidden_size) -> Relu -> Linear(hidden_size, 1) -> Tanh
///
/// Clone is implemented to support parallel self-play games. MLX arrays use
/// copy-on-write semantics, so cloning is efficient for read-only inference.
#[derive(Debug, ModuleParameters)]
#[module(root = mlx_rs)]
pub struct AlphaZeroNet {
    pub obs_size: usize,
    pub hidden_size: usize,
    #[param]
    pub trunk: Sequential,
    #[param]
    pub policy_head: Sequential,
    #[param]
    pub value_head: Sequential,
}

// Safe to mark Send because we run MLX inference on a single dedicated thread.
unsafe impl Send for AlphaZeroNet {}

impl Clone for AlphaZeroNet {
    fn clone(&self) -> Self {
        // Create a new network with the same architecture
        let mut new_net = Self::new(self.obs_size, self.hidden_size);

        // Copy parameters from self to new_net
        let src_params = ModuleParameters::parameters(self).flatten();
        let mut dst_params = ModuleParameters::parameters_mut(&mut new_net).flatten();

        // Copy each parameter array
        for (key, src_arr) in src_params {
            if let Some(dst_arr) = dst_params.get_mut(&key) {
                // MLX arrays use copy-on-write, so this is efficient
                **dst_arr = src_arr.clone();
            }
        }

        new_net
    }
}

impl AlphaZeroNet {
    /// Create a new AlphaZeroNet with the given observation and hidden sizes.
    pub fn new(obs_size: usize, hidden_size: usize) -> Self {
        // Build trunk: Linear -> Relu -> Linear -> Relu
        let trunk = Sequential::new()
            .append(
                Linear::new(obs_size as i32, hidden_size as i32)
                    .expect("Failed to create trunk linear 1"),
            )
            .append(Relu)
            .append(
                Linear::new(hidden_size as i32, hidden_size as i32)
                    .expect("Failed to create trunk linear 2"),
            )
            .append(Relu);

        // Build policy head: Linear -> Relu -> Linear (no softmax - raw logits)
        let policy_head = Sequential::new()
            .append(
                Linear::new(hidden_size as i32, hidden_size as i32)
                    .expect("Failed to create policy linear 1"),
            )
            .append(Relu)
            .append(
                Linear::new(hidden_size as i32, ACTION_SPACE_SIZE as i32)
                    .expect("Failed to create policy linear 2"),
            );

        // Build value head: Linear -> Relu -> Linear -> Tanh
        let value_head = Sequential::new()
            .append(
                Linear::new(hidden_size as i32, hidden_size as i32)
                    .expect("Failed to create value linear 1"),
            )
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
        let policy_logits = self
            .policy_head
            .forward(&h)
            .expect("Policy head forward failed");

        // Run through value head - output is [B, 1], squeeze to [B]
        let value_2d = self
            .value_head
            .forward(&h)
            .expect("Value head forward failed");
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

impl crate::alphazero::training::TrainableModel for AlphaZeroNet {
    fn param_count(&self) -> usize {
        // Use the ModuleParameters trait to get actual parameter count
        ModuleParameters::num_parameters(self)
    }

    fn parameters(&self) -> Vec<Array> {
        // Extract actual parameters from the model using ModuleParameters trait
        // Sort by key for consistent ordering (same order as apply_gradients)
        let flattened = ModuleParameters::parameters(self).flatten();
        let mut items: Vec<_> = flattened.into_iter().collect();
        items.sort_by(|a, b| a.0.cmp(&b.0));
        items.into_iter().map(|(_, arr)| arr.clone()).collect()
    }

    fn forward(&mut self, obs: &Array) -> (Array, Array) {
        self.forward_batch(obs)
    }

    fn apply_gradients(&mut self, learning_rate: f32, grads: &[Array]) {
        // Get mutable access to parameters
        let mut params_mut = ModuleParameters::parameters_mut(self).flatten();

        // Collect keys sorted for deterministic ordering
        let mut keys: Vec<_> = params_mut.keys().cloned().collect();
        keys.sort();

        // Create learning rate as a scalar array
        let lr_array = Array::from_slice(&[learning_rate], &[1]);

        // Apply SGD update: param = param - learning_rate * grad
        for (i, key) in keys.iter().enumerate() {
            if i < grads.len() {
                if let Some(param) = params_mut.get_mut(key) {
                    // Only apply if shapes match
                    if param.shape() == grads[i].shape() {
                        let scaled_grad = grads[i].multiply(&lr_array).expect("multiply failed");
                        let updated = param.subtract(&scaled_grad).expect("subtract failed");
                        **param = updated;
                    }
                }
            }
        }
    }

    fn eval_parameters(&self) {
        // Force evaluation of all model parameters
        let _ = eval_params(ModuleParameters::parameters(self));
    }

    fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        // Use mlx-rs safetensors format for saving
        use mlx_rs::module::ModuleParametersExt;
        self.save_safetensors(path)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }

    fn load(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        // Use mlx-rs safetensors format for loading
        use mlx_rs::module::ModuleParametersExt;
        self.load_safetensors(path)
            .map_err(|e| std::io::Error::other(e.to_string()))
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
        let obs =
            Array::zeros::<f32>(&[obs_size as i32]).expect("Failed to create test observation");

        let (policy_logits, value) = net.predict_single(&obs);

        // Check policy shape
        assert_eq!(
            policy_logits.shape(),
            &[ACTION_SPACE_SIZE as i32],
            "Policy logits should have shape [ACTION_SPACE_SIZE]"
        );

        // Check value is in valid range (Tanh output)
        assert!(
            (-1.0..=1.0).contains(&value),
            "Value {value} should be in [-1, 1]"
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
            assert!((-1.0..=1.0).contains(&v), "Value {v} should be in [-1, 1]");
        }
    }

    #[test]
    fn test_alphazero_net_determinism() {
        let obs_size = 100;
        let mut net = AlphaZeroNet::new(obs_size, 64);

        // Same input should give same output
        let obs =
            Array::zeros::<f32>(&[obs_size as i32]).expect("Failed to create test observation");

        let (p1, v1) = net.predict_single(&obs);
        let (p2, v2) = net.predict_single(&obs);

        let p1_slice = p1.as_slice::<f32>();
        let p2_slice = p2.as_slice::<f32>();

        assert_eq!(p1_slice, p2_slice, "Policy should be deterministic");
        assert_eq!(v1, v2, "Value should be deterministic");
    }

    #[test]
    fn test_module_parameters_returns_real_weights() {
        use crate::alphazero::training::TrainableModel;

        let obs_size = 50;
        let hidden_size = 32;
        let net = AlphaZeroNet::new(obs_size, hidden_size);

        // Get actual parameters
        let params = TrainableModel::parameters(&net);

        // Should have 12 parameter tensors (6 layers * 2 for weight + bias)
        assert_eq!(params.len(), 12, "Should have 12 parameter tensors");

        // At least one parameter should be non-zero (weights are initialized randomly)
        let has_nonzero = params.iter().any(|p| {
            let sum: f32 = p
                .abs()
                .expect("abs failed")
                .sum(None)
                .expect("sum failed")
                .item();
            sum > 1e-6
        });
        assert!(
            has_nonzero,
            "At least one parameter should have non-zero weights"
        );
    }

    #[test]
    fn test_apply_gradients_changes_model() {
        use crate::alphazero::training::TrainableModel;

        let obs_size = 50;
        let hidden_size = 32;
        let mut net = AlphaZeroNet::new(obs_size, hidden_size);

        // Get initial output
        let obs = Array::zeros::<f32>(&[1, obs_size as i32]).expect("Failed to create obs");
        let (policy_before, value_before) = net.forward_batch(&obs);
        let policy_before_slice: Vec<f32> = policy_before.as_slice::<f32>().to_vec();
        let value_before_scalar = value_before.as_slice::<f32>()[0];

        // Create non-zero gradients (using ones scaled to be significant)
        let params = TrainableModel::parameters(&net);
        let grads: Vec<Array> = params
            .iter()
            .map(|p| {
                let shape = p.shape();
                mlx_rs::ops::ones::<f32>(shape).expect("ones failed")
            })
            .collect();

        // Apply gradients with significant learning rate
        TrainableModel::apply_gradients(&mut net, 0.1, &grads);
        TrainableModel::eval_parameters(&net);

        // Get output after gradient application
        let (policy_after, value_after) = net.forward_batch(&obs);
        let policy_after_slice: Vec<f32> = policy_after.as_slice::<f32>().to_vec();
        let value_after_scalar = value_after.as_slice::<f32>()[0];

        // Output should have changed
        let policy_changed = policy_before_slice
            .iter()
            .zip(policy_after_slice.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);

        assert!(
            policy_changed || (value_before_scalar - value_after_scalar).abs() > 1e-6,
            "Model output should change after applying gradients. \
             Policy before: {:?}, after: {:?}. Value before: {}, after: {}",
            &policy_before_slice[..5.min(policy_before_slice.len())],
            &policy_after_slice[..5.min(policy_after_slice.len())],
            value_before_scalar,
            value_after_scalar
        );
    }

    #[test]
    fn test_param_count_matches_actual_parameters() {
        use crate::alphazero::training::TrainableModel;

        let obs_size = 50;
        let hidden_size = 32;
        let net = AlphaZeroNet::new(obs_size, hidden_size);

        let param_count = TrainableModel::param_count(&net);
        let params = TrainableModel::parameters(&net);

        assert_eq!(
            param_count,
            params.len(),
            "param_count() should match actual number of parameter tensors"
        );
    }

    #[test]
    fn test_eval_parameters_does_not_panic() {
        use crate::alphazero::training::TrainableModel;

        let net = AlphaZeroNet::new(50, 32);

        // This should not panic
        TrainableModel::eval_parameters(&net);
    }

    #[test]
    fn test_save_load_safetensors_roundtrip() {
        use crate::alphazero::training::TrainableModel;

        let obs_size = 50;
        let hidden_size = 32;
        let mut net = AlphaZeroNet::new(obs_size, hidden_size);

        // Get output before saving
        let obs = Array::zeros::<f32>(&[obs_size as i32]).expect("Failed to create obs");
        let (policy_before, value_before) = net.predict_single(&obs);
        let policy_before_slice: Vec<f32> = policy_before.as_slice::<f32>().to_vec();

        // Save to temp file
        let temp_dir = std::env::temp_dir();
        let checkpoint_path = temp_dir.join("test_checkpoint.safetensors");
        TrainableModel::save(&net, &checkpoint_path).expect("Failed to save checkpoint");

        // Create a new network and load the checkpoint
        let mut net2 = AlphaZeroNet::new(obs_size, hidden_size);

        // New network has different random weights (we don't need to verify this)

        // Load checkpoint into new network
        TrainableModel::load(&mut net2, &checkpoint_path).expect("Failed to load checkpoint");

        // Get output after loading
        let (policy_after, value_after) = net2.predict_single(&obs);
        let policy_after_slice: Vec<f32> = policy_after.as_slice::<f32>().to_vec();

        // Outputs should match the original network
        for (i, (before, after)) in policy_before_slice
            .iter()
            .zip(policy_after_slice.iter())
            .enumerate()
        {
            assert!(
                (before - after).abs() < 1e-5,
                "Policy logit {i} differs: before={before}, after={after}"
            );
        }
        assert!(
            (value_before - value_after).abs() < 1e-5,
            "Value differs: before={value_before}, after={value_after}"
        );

        // Clean up
        let _ = std::fs::remove_file(&checkpoint_path);
    }
}
