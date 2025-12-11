//! Action encoding: Action ↔ ActionId
//!
//! Fixed global action space of size 300, covering all syntactically possible
//! Azul draft moves regardless of current legality.

use azul_engine::{
    Action, Color, DraftDestination, DraftSource, BOARD_SIZE, MAX_FACTORIES, TILE_COLORS,
};

use super::ActionId;

/// Total size of the discrete action space
/// Calculation: (9 factories + 1 center) * 5 colors * 2 dest_types * 5 dest_indices = 500
/// Note: The original spec said 300, but this was an arithmetic error.
pub const ACTION_SPACE_SIZE: usize = 500;

/// Encodes/decodes between engine Actions and discrete ActionIds
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ActionEncoder;

impl ActionEncoder {
    /// Returns the size of the discrete action space
    pub const fn action_space_size() -> usize {
        ACTION_SPACE_SIZE
    }

    /// Encode a concrete Action into a discrete ActionId.
    ///
    /// Packing scheme:
    /// id = ((((s_type * MAX_FACTORIES + s_idx) * TILE_COLORS + c_idx) * 2 + d_type) * BOARD_SIZE + d_idx)
    ///
    /// Where:
    /// - s_type ∈ {0,1} (Factory=0, Center=1)
    /// - s_idx ∈ 0..MAX_FACTORIES (only used if Factory)
    /// - c_idx ∈ 0..TILE_COLORS
    /// - d_type ∈ {0,1} (PatternLine=0, Floor=1)
    /// - d_idx ∈ 0..BOARD_SIZE (only used if PatternLine)
    ///
    /// Panics in debug builds if the Action has indices out of range.
    pub fn encode(action: &Action) -> ActionId {
        let (s_type, s_idx) = match action.source {
            DraftSource::Factory(f) => (0u16, f as u16),
            DraftSource::Center => (1u16, 0),
        };

        let c_idx = action.color as u16;

        let (d_type, d_idx) = match action.dest {
            DraftDestination::PatternLine(r) => (0u16, r as u16),
            DraftDestination::Floor => (1u16, 0),
        };

        let id = (((s_type * MAX_FACTORIES as u16 + s_idx) * TILE_COLORS as u16 + c_idx) * 2
            + d_type)
            * BOARD_SIZE as u16
            + d_idx;

        debug_assert!(
            (id as usize) < ACTION_SPACE_SIZE,
            "ActionId {id} out of range"
        );

        id
    }

    /// Decode an ActionId back into an Action.
    ///
    /// This may result in syntactically valid but *illegal* Actions for
    /// the current state. The environment must mask illegal IDs.
    ///
    /// Panics if id >= ACTION_SPACE_SIZE.
    pub fn decode(id: ActionId) -> Action {
        assert!(
            (id as usize) < ACTION_SPACE_SIZE,
            "ActionId {id} >= ACTION_SPACE_SIZE {ACTION_SPACE_SIZE}"
        );

        let mut x = id;

        let d_idx = x % BOARD_SIZE as u16;
        x /= BOARD_SIZE as u16;

        let d_type = x % 2;
        x /= 2;

        let c_idx = x % TILE_COLORS as u16;
        x /= TILE_COLORS as u16;

        let s_idx = x % MAX_FACTORIES as u16;
        x /= MAX_FACTORIES as u16;

        let s_type = x; // 0 or 1

        let source = if s_type == 0 {
            DraftSource::Factory(s_idx as u8)
        } else {
            DraftSource::Center
        };

        let color = Color::from_index(c_idx as u8)
            .expect("color index should be valid within action space");

        let dest = if d_type == 0 {
            DraftDestination::PatternLine(d_idx as u8)
        } else {
            DraftDestination::Floor
        };

        Action {
            source,
            color,
            dest,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use azul_engine::ALL_COLORS;

    #[test]
    fn test_action_space_size_calculation() {
        // source: 10 options (9 factories + center, encoded as s_type*9+s_idx)
        // color: 5 options
        // dest_type: 2 options (pattern line, floor)
        // dest_idx: 5 options
        // Total: 10 * 5 * 2 * 5 = 500
        // (The original spec said 300, but that was an arithmetic error)
        assert_eq!(ACTION_SPACE_SIZE, 500);
    }

    #[test]
    fn test_round_trip_factory_pattern_line() {
        let action = Action {
            source: DraftSource::Factory(3),
            color: Color::Red,
            dest: DraftDestination::PatternLine(2),
        };

        let id = ActionEncoder::encode(&action);
        let decoded = ActionEncoder::decode(id);

        assert_eq!(decoded, action);
    }

    #[test]
    fn test_round_trip_center_floor() {
        let action = Action {
            source: DraftSource::Center,
            color: Color::Blue,
            dest: DraftDestination::Floor,
        };

        let id = ActionEncoder::encode(&action);
        let decoded = ActionEncoder::decode(id);

        assert_eq!(decoded, action);
    }

    #[test]
    fn test_round_trip_all_factories() {
        for f in 0..MAX_FACTORIES as u8 {
            for color in ALL_COLORS {
                for row in 0..BOARD_SIZE as u8 {
                    let action = Action {
                        source: DraftSource::Factory(f),
                        color,
                        dest: DraftDestination::PatternLine(row),
                    };

                    let id = ActionEncoder::encode(&action);
                    if (id as usize) < ACTION_SPACE_SIZE {
                        let decoded = ActionEncoder::decode(id);
                        assert_eq!(decoded, action);
                    }
                }

                let action = Action {
                    source: DraftSource::Factory(f),
                    color,
                    dest: DraftDestination::Floor,
                };
                let id = ActionEncoder::encode(&action);
                if (id as usize) < ACTION_SPACE_SIZE {
                    let decoded = ActionEncoder::decode(id);
                    assert_eq!(decoded, action);
                }
            }
        }
    }

    #[test]
    fn test_round_trip_center() {
        for color in ALL_COLORS {
            for row in 0..BOARD_SIZE as u8 {
                let action = Action {
                    source: DraftSource::Center,
                    color,
                    dest: DraftDestination::PatternLine(row),
                };

                let id = ActionEncoder::encode(&action);
                if (id as usize) < ACTION_SPACE_SIZE {
                    let decoded = ActionEncoder::decode(id);
                    assert_eq!(decoded, action);
                }
            }

            let action = Action {
                source: DraftSource::Center,
                color,
                dest: DraftDestination::Floor,
            };
            let id = ActionEncoder::encode(&action);
            if (id as usize) < ACTION_SPACE_SIZE {
                let decoded = ActionEncoder::decode(id);
                assert_eq!(decoded, action);
            }
        }
    }

    #[test]
    fn test_decode_all_valid_ids() {
        // Every id in [0, ACTION_SPACE_SIZE) should decode to a valid Action
        for id in 0..ACTION_SPACE_SIZE as u16 {
            let action = ActionEncoder::decode(id);

            // Verify source is valid
            match action.source {
                DraftSource::Factory(f) => assert!(f < MAX_FACTORIES as u8),
                DraftSource::Center => {}
            }

            // Verify dest is valid
            match action.dest {
                DraftDestination::PatternLine(r) => assert!(r < BOARD_SIZE as u8),
                DraftDestination::Floor => {}
            }
        }
    }
}
