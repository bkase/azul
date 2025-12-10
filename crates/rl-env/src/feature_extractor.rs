//! Feature extraction: GameState → Observation
//!
//! Converts full game state into fixed-length array observation
//! from the perspective of a given player.

use azul_engine::{
    GameState, PlayerIdx, Token, ALL_COLORS, BOARD_SIZE, FLOOR_CAPACITY, MAX_FACTORIES,
    MAX_PLAYERS, TILE_COLORS,
};

use super::Observation;

/// Converts a full GameState into a fixed-length array observation
/// from the perspective of a given player.
pub trait FeatureExtractor: Clone {
    /// Returns the length of the flattened observation vector.
    ///
    /// For all states and players, encode(...) must return an Observation
    /// with shape [obs_size].
    fn obs_size(&self) -> usize;

    /// Encode state from the perspective of `player` into a 1D array.
    ///
    /// Requirements:
    /// - Deterministic given (state, player).
    /// - Shape is [obs_size()], rank-1, dtype f32.
    /// - No side effects or internal randomness.
    fn encode(&self, state: &GameState, player: PlayerIdx) -> Observation;
}

/// Create a zero-filled observation of the given size
pub fn create_zero_observation(obs_size: usize) -> Observation {
    mlx_rs::Array::zeros::<f32>(&[obs_size as i32]).expect("Failed to create zero observation")
}

/// Basic feature extractor implementation
///
/// Encodes:
/// - All factory displays (colors as counts)
/// - Center pool contents (color counts + first player marker)
/// - For each player (rotated so current player is first):
///   - Pattern lines (5 rows, color + count)
///   - Wall (5×5 binary)
///   - Floor line (7 slots)
/// - Current player indicator
/// - Scores (normalized by 100)
#[derive(Clone, Debug)]
pub struct BasicFeatureExtractor {
    num_players: u8,
}

impl BasicFeatureExtractor {
    pub fn new(num_players: u8) -> Self {
        assert!((2..=4).contains(&num_players));
        Self { num_players }
    }

    /// Calculate the observation size for this extractor
    fn calculate_obs_size(&self) -> usize {
        // Factories: MAX_FACTORIES * TILE_COLORS (count of each color)
        let factory_features = MAX_FACTORIES * TILE_COLORS;

        // Center: TILE_COLORS (counts) + 1 (first player marker present)
        let center_features = TILE_COLORS + 1;

        // Per player:
        // - Pattern lines: 5 rows * (TILE_COLORS one-hot color + 1 count/capacity)
        // - Wall: BOARD_SIZE * BOARD_SIZE
        // - Floor: color counts + FP marker + total count
        // - Score: 1 (normalized)
        let pattern_line_features = BOARD_SIZE * (TILE_COLORS + 1);
        let wall_features = BOARD_SIZE * BOARD_SIZE;
        let floor_features = TILE_COLORS + 1 + 1; // counts per color + FP marker + total count
        let score_feature = 1;
        let per_player_features =
            pattern_line_features + wall_features + floor_features + score_feature;

        // All players
        let all_players_features = per_player_features * MAX_PLAYERS;

        // Current player one-hot
        let current_player_features = MAX_PLAYERS;

        // Starting player next round one-hot
        let starting_player_features = MAX_PLAYERS;

        // Round number (normalized)
        let round_feature = 1;

        factory_features
            + center_features
            + all_players_features
            + current_player_features
            + starting_player_features
            + round_feature
    }
}

impl FeatureExtractor for BasicFeatureExtractor {
    fn obs_size(&self) -> usize {
        self.calculate_obs_size()
    }

    fn encode(&self, state: &GameState, player: PlayerIdx) -> Observation {
        let obs_size = self.obs_size();
        let mut features = vec![0.0f32; obs_size];
        let mut idx = 0;

        // Helper to write features and advance index
        macro_rules! write_feature {
            ($val:expr) => {
                features[idx] = $val;
                idx += 1;
            };
        }

        // 1. Encode factories (MAX_FACTORIES * TILE_COLORS)
        for f in 0..MAX_FACTORIES {
            if f < state.factories.num_factories as usize {
                let factory = &state.factories.factories[f];
                // Count each color in this factory
                let mut color_counts = [0u8; TILE_COLORS];
                for i in 0..factory.len as usize {
                    color_counts[factory.tiles[i] as usize] += 1;
                }
                for &count in &color_counts {
                    write_feature!(count as f32 / 4.0); // Normalize by max (4)
                }
            } else {
                // Empty factory (not in use for this player count)
                for _ in 0..TILE_COLORS {
                    write_feature!(0.0);
                }
            }
        }

        // 2. Encode center pool (TILE_COLORS + 1)
        let mut center_color_counts = [0u8; TILE_COLORS];
        let mut fp_marker_in_center = false;
        for i in 0..state.center.len as usize {
            match state.center.items[i] {
                Token::Tile(color) => {
                    center_color_counts[color as usize] += 1;
                }
                Token::FirstPlayerMarker => {
                    fp_marker_in_center = true;
                }
            }
        }
        for &count in &center_color_counts {
            write_feature!(count as f32 / 20.0); // Normalize by max possible
        }
        write_feature!(if fp_marker_in_center { 1.0 } else { 0.0 });

        // 3. Encode all players (rotated so requesting player is first)
        for p_offset in 0..MAX_PLAYERS {
            let p = ((player as usize) + p_offset) % (self.num_players as usize);
            let is_active_player = p_offset < self.num_players as usize;

            if is_active_player {
                let player_state = &state.players[p];

                // Pattern lines: 5 rows * (TILE_COLORS one-hot + fill ratio)
                for r in 0..BOARD_SIZE {
                    let line = &player_state.pattern_lines[r];
                    let capacity = (r + 1) as f32;

                    // One-hot color encoding
                    for color in ALL_COLORS {
                        write_feature!(if line.color == Some(color) { 1.0 } else { 0.0 });
                    }
                    // Fill ratio
                    write_feature!(line.count as f32 / capacity);
                }

                // Wall: BOARD_SIZE * BOARD_SIZE binary
                for row in 0..BOARD_SIZE {
                    for col in 0..BOARD_SIZE {
                        write_feature!(if player_state.wall[row][col].is_some() {
                            1.0
                        } else {
                            0.0
                        });
                    }
                }

                // Floor: color counts + FP marker + total
                let mut floor_color_counts = [0u8; TILE_COLORS];
                let mut fp_on_floor = false;
                for i in 0..player_state.floor.len as usize {
                    match player_state.floor.slots[i] {
                        Token::Tile(color) => {
                            floor_color_counts[color as usize] += 1;
                        }
                        Token::FirstPlayerMarker => {
                            fp_on_floor = true;
                        }
                    }
                }
                for &count in &floor_color_counts {
                    write_feature!(count as f32 / FLOOR_CAPACITY as f32);
                }
                write_feature!(if fp_on_floor { 1.0 } else { 0.0 });
                write_feature!(player_state.floor.len as f32 / FLOOR_CAPACITY as f32);

                // Score (normalized by 100)
                write_feature!(player_state.score as f32 / 100.0);
            } else {
                // Inactive player slot - zero out
                let pattern_line_features = BOARD_SIZE * (TILE_COLORS + 1);
                let wall_features = BOARD_SIZE * BOARD_SIZE;
                let floor_features = TILE_COLORS + 1 + 1;
                let score_feature = 1;
                let per_player_features =
                    pattern_line_features + wall_features + floor_features + score_feature;

                for _ in 0..per_player_features {
                    write_feature!(0.0);
                }
            }
        }

        // 4. Current player one-hot
        for p in 0..MAX_PLAYERS {
            write_feature!(if p == state.current_player as usize {
                1.0
            } else {
                0.0
            });
        }

        // 5. Starting player next round one-hot
        for p in 0..MAX_PLAYERS {
            write_feature!(if p == state.starting_player_next_round as usize {
                1.0
            } else {
                0.0
            });
        }

        // 6. Round number (normalized)
        write_feature!(state.round as f32 / 10.0); // Typically games last 5-6 rounds

        debug_assert_eq!(idx, obs_size, "Feature count mismatch");

        mlx_rs::Array::from_slice(&features, &[obs_size as i32])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use azul_engine::new_game;
    use crate::ObservationExt;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_basic_extractor_obs_size() {
        let extractor = BasicFeatureExtractor::new(2);
        let obs_size = extractor.obs_size();
        assert!(obs_size > 0);
        println!("Observation size: {}", obs_size);
    }

    #[test]
    fn test_basic_extractor_encode() {
        let mut rng = StdRng::seed_from_u64(42);
        let state = new_game(2, 0, &mut rng);
        let extractor = BasicFeatureExtractor::new(2);

        let obs = extractor.encode(&state, 0);
        assert_eq!(obs.shape(), &[extractor.obs_size() as i32]);
    }

    #[test]
    fn test_basic_extractor_determinism() {
        let mut rng = StdRng::seed_from_u64(42);
        let state = new_game(2, 0, &mut rng);
        let extractor = BasicFeatureExtractor::new(2);

        let obs1 = extractor.encode(&state, 0);
        let obs2 = extractor.encode(&state, 0);

        // Convert to vectors for comparison
        let data1: Vec<f32> = obs1.as_f32_slice().to_vec();
        let data2: Vec<f32> = obs2.as_f32_slice().to_vec();

        assert_eq!(data1, data2, "Encoding should be deterministic");
    }

    #[test]
    fn test_different_player_perspectives() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut state = new_game(2, 0, &mut rng);
        let extractor = BasicFeatureExtractor::new(2);

        // At game start, both players have identical boards, so observations would be the same
        // except for the player rotation. Let's modify one player's board to make them different.
        state.players[0].pattern_lines[0].color = Some(azul_engine::Color::Blue);
        state.players[0].pattern_lines[0].count = 1;

        let obs0 = extractor.encode(&state, 0);
        let obs1 = extractor.encode(&state, 1);

        let data0: Vec<f32> = obs0.as_f32_slice().to_vec();
        let data1: Vec<f32> = obs1.as_f32_slice().to_vec();

        // Observations should differ because:
        // 1. Player 0 sees their own board (with the Blue tile) first
        // 2. Player 1 sees their own board (empty) first
        assert_ne!(
            data0, data1,
            "Different players should see different observations"
        );
    }

    #[test]
    fn test_create_zero_observation() {
        let obs = create_zero_observation(100);
        assert_eq!(obs.shape(), &[100]);

        let data: Vec<f32> = obs.as_f32_slice().to_vec();
        assert!(data.iter().all(|&x| x == 0.0));
    }
}
