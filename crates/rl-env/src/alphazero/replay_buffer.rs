//! Replay buffer implementation for AlphaZero training
//!
//! A ring buffer that stores training examples for sampling during training.

use super::TrainingExample;
use rand::Rng;

/// Ring buffer for storing training examples.
///
/// When the buffer reaches capacity, new examples overwrite the oldest ones.
pub struct ReplayBuffer {
    /// Maximum number of examples to store
    capacity: usize,

    /// Storage for training examples
    data: Vec<TrainingExample>,

    /// Index where the next example will be written
    write_index: usize,
}

impl ReplayBuffer {
    /// Create a new replay buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            data: Vec::with_capacity(capacity),
            write_index: 0,
        }
    }

    /// Return the number of examples currently stored.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Add a single example to the buffer.
    ///
    /// When the buffer is full, this overwrites the oldest example.
    pub fn push(&mut self, example: TrainingExample) {
        if self.data.len() < self.capacity {
            // Buffer not yet full - append
            self.data.push(example);
            // Reset write_index to 0 when we reach capacity
            if self.data.len() == self.capacity {
                self.write_index = 0;
            }
        } else {
            // Buffer full - overwrite oldest
            self.data[self.write_index] = example;
            self.write_index = (self.write_index + 1) % self.capacity;
        }
    }

    /// Add many examples to the buffer.
    pub fn extend<I: IntoIterator<Item = TrainingExample>>(&mut self, it: I) {
        for example in it {
            self.push(example);
        }
    }

    /// Uniformly sample `batch_size` examples with replacement.
    ///
    /// # Panics
    /// Panics if the buffer is empty.
    pub fn sample<'a>(&'a self, rng: &mut impl Rng, batch_size: usize) -> Vec<&'a TrainingExample> {
        assert!(!self.is_empty(), "Cannot sample from empty replay buffer");

        let len = self.data.len();
        (0..batch_size)
            .map(|_| {
                let idx = rng.random_range(0..len as u32) as usize;
                &self.data[idx]
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ACTION_SPACE_SIZE;
    use mlx_rs::Array;
    use rand::SeedableRng;

    fn make_example(value: f32) -> TrainingExample {
        TrainingExample {
            observation: Array::zeros::<f32>(&[10]).unwrap(),
            policy: vec![0.0; ACTION_SPACE_SIZE],
            value,
        }
    }

    #[test]
    fn test_replay_buffer_push_and_len() {
        let mut buffer = ReplayBuffer::new(3);

        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());

        buffer.push(make_example(1.0));
        assert_eq!(buffer.len(), 1);

        buffer.push(make_example(2.0));
        assert_eq!(buffer.len(), 2);

        buffer.push(make_example(3.0));
        assert_eq!(buffer.len(), 3);
    }

    #[test]
    fn test_replay_buffer_overwrite_oldest() {
        let mut buffer = ReplayBuffer::new(2);

        // Push A, B
        buffer.push(make_example(1.0)); // A
        buffer.push(make_example(2.0)); // B

        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.data[0].value, 1.0);
        assert_eq!(buffer.data[1].value, 2.0);

        // Push C - should overwrite A
        buffer.push(make_example(3.0)); // C

        assert_eq!(buffer.len(), 2);
        // Buffer should contain {C, B} or {B, C} depending on write position
        // Since write_index was at 0 after filling, C overwrites position 0
        assert_eq!(buffer.data[0].value, 3.0); // C
        assert_eq!(buffer.data[1].value, 2.0); // B
    }

    #[test]
    fn test_replay_buffer_sample_returns_valid_refs() {
        let mut buffer = ReplayBuffer::new(5);

        buffer.push(make_example(1.0));
        buffer.push(make_example(2.0));
        buffer.push(make_example(3.0));

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        for _ in 0..5 {
            let samples = buffer.sample(&mut rng, 10);
            assert_eq!(samples.len(), 10);

            for sample in samples {
                // Each sample should be one of our stored values
                assert!(
                    sample.value == 1.0 || sample.value == 2.0 || sample.value == 3.0,
                    "Sample value {} should be one of 1.0, 2.0, 3.0",
                    sample.value
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "Cannot sample from empty replay buffer")]
    fn test_replay_buffer_sample_empty_panics() {
        let buffer = ReplayBuffer::new(5);
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        buffer.sample(&mut rng, 1);
    }

    #[test]
    fn test_replay_buffer_extend() {
        let mut buffer = ReplayBuffer::new(10);

        let examples = vec![make_example(1.0), make_example(2.0), make_example(3.0)];

        buffer.extend(examples);

        assert_eq!(buffer.len(), 3);
    }
}
