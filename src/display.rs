//! Shared display utilities for rendering Azul game state in the terminal
//!
//! Provides colorized, human-readable output for game boards, factories, and actions.

use azul_engine::{Action, Color, DraftDestination, DraftSource, GameState, Token, BOARD_SIZE, WALL_PATTERN};

// ANSI color codes for tile display
pub const BLUE: &str = "\x1b[94m";
pub const YELLOW: &str = "\x1b[93m";
pub const RED: &str = "\x1b[91m";
pub const BLACK: &str = "\x1b[90m";
pub const TEAL: &str = "\x1b[96m";
pub const RESET: &str = "\x1b[0m";
pub const BOLD: &str = "\x1b[1m";
pub const DIM: &str = "\x1b[2m";

pub fn color_code(color: Color) -> &'static str {
    match color {
        Color::Blue => BLUE,
        Color::Yellow => YELLOW,
        Color::Red => RED,
        Color::Black => BLACK,
        Color::Teal => TEAL,
    }
}

pub fn color_char(color: Color) -> char {
    match color {
        Color::Blue => 'B',
        Color::Yellow => 'Y',
        Color::Red => 'R',
        Color::Black => 'K',
        Color::Teal => 'T',
    }
}

pub fn color_name(color: Color) -> &'static str {
    match color {
        Color::Blue => "Blue",
        Color::Yellow => "Yellow",
        Color::Red => "Red",
        Color::Black => "Black",
        Color::Teal => "Teal",
    }
}

pub fn display_tile(color: Color) -> String {
    format!("{}{}{}", color_code(color), color_char(color), RESET)
}

pub fn display_token(token: Token) -> String {
    match token {
        Token::Tile(c) => display_tile(c),
        Token::FirstPlayerMarker => format!("{}1{}", BOLD, RESET),
    }
}

/// Format an action for display
pub fn format_action(action: &Action) -> String {
    let source = match action.source {
        DraftSource::Factory(f) => format!("F{}", f),
        DraftSource::Center => "Center".to_string(),
    };
    let color = format!(
        "{}{}{}",
        color_code(action.color),
        color_char(action.color),
        RESET
    );
    let dest = match action.dest {
        DraftDestination::PatternLine(r) => format!("Line {}", r + 1),
        DraftDestination::Floor => "Floor".to_string(),
    };
    format!("{} {} -> {}", source, color, dest)
}

/// Format an action in compact form (for tables)
pub fn format_action_compact(action: &Action) -> String {
    let source = match action.source {
        DraftSource::Factory(f) => format!("F{}", f),
        DraftSource::Center => "C".to_string(),
    };
    let dest = match action.dest {
        DraftDestination::PatternLine(r) => format!("L{}", r + 1),
        DraftDestination::Floor => "Floor".to_string(),
    };
    format!("{} {} {}", source, color_name(action.color), dest)
}

/// Display the full game board with all player information
///
/// If `highlight_player` is Some, that player's board will be shown with emphasis.
pub fn display_board(state: &GameState, highlight_player: Option<u8>) {
    println!("\n{BOLD}══════════════════════════════════════════════════════════════{RESET}");
    println!(
        "{BOLD}  Round {}{RESET}   |   Current Player: {}",
        state.round + 1,
        state.current_player
    );
    println!("{BOLD}══════════════════════════════════════════════════════════════{RESET}\n");

    // Factories
    println!("{BOLD}FACTORIES:{RESET}");
    for f in 0..state.factories.num_factories as usize {
        let factory = &state.factories.factories[f];
        print!("  F{}: ", f);
        if factory.len == 0 {
            print!("{DIM}(empty){RESET}");
        } else {
            for i in 0..factory.len as usize {
                print!("{} ", display_tile(factory.tiles[i]));
            }
        }
        println!();
    }

    // Center
    print!("\n{BOLD}CENTER:{RESET} ");
    if state.center.len == 0 {
        print!("{DIM}(empty){RESET}");
    } else {
        for i in 0..state.center.len as usize {
            print!("{} ", display_token(state.center.items[i]));
        }
    }
    println!("\n");

    // Player boards
    for p in 0..state.num_players as usize {
        let player = &state.players[p];
        let is_highlighted = highlight_player == Some(p as u8);
        let header = if is_highlighted {
            format!("{BOLD}PLAYER {} (Score: {}){RESET}", p, player.score)
        } else {
            format!("{DIM}PLAYER {} (Score: {}){RESET}", p, player.score)
        };
        println!("{}", header);
        println!("  Pattern Lines          Wall");

        for row in 0..BOARD_SIZE {
            // Pattern line (right-aligned)
            let line = &player.pattern_lines[row];
            let cap = row + 1;
            let empty = cap - line.count as usize;

            print!("  ");
            // Leading spaces for alignment
            for _ in 0..(BOARD_SIZE - cap) {
                print!("  ");
            }
            // Empty slots
            for _ in 0..empty {
                print!("{DIM}.{RESET} ");
            }
            // Filled slots
            if let Some(color) = line.color {
                for _ in 0..line.count {
                    print!("{} ", display_tile(color));
                }
            }

            print!(" -> ");

            // Wall
            for col in 0..BOARD_SIZE {
                if let Some(color) = player.wall[row][col] {
                    print!("{} ", display_tile(color));
                } else {
                    // Show expected color dimmed
                    let expected = WALL_PATTERN[row][col];
                    print!("{}{}{} ", DIM, color_char(expected), RESET);
                }
            }
            println!();
        }

        // Floor
        print!("  Floor: ");
        if player.floor.len == 0 {
            print!("{DIM}(empty){RESET}");
        } else {
            for i in 0..player.floor.len as usize {
                print!("{} ", display_token(player.floor.slots[i]));
            }
        }
        println!("\n");
    }
}

/// Display a compact summary of the game state (for debugging)
pub fn display_board_summary(state: &GameState) {
    println!("\n{BOLD}=== GAME STATE SUMMARY ==={RESET}");
    println!("Round: {}, Current Player: {}", state.round + 1, state.current_player);

    println!("\n{BOLD}Factories:{RESET}");
    for f in 0..state.factories.num_factories as usize {
        let factory = &state.factories.factories[f];
        if factory.len > 0 {
            print!("  F{}: ", f);
            for i in 0..factory.len as usize {
                print!("{} ", display_tile(factory.tiles[i]));
            }
            println!();
        }
    }

    println!("\n{BOLD}Center ({} tiles):{RESET}", state.center.len);
    if state.center.len > 0 {
        print!("  ");
        for i in 0..state.center.len as usize {
            print!("{} ", display_token(state.center.items[i]));
        }
        println!();
    }

    for p in 0..state.num_players as usize {
        let player = &state.players[p];
        println!("\n{BOLD}Player {} (Score: {}):{RESET}", p, player.score);
        println!("  Pattern Lines:");
        for row in 0..BOARD_SIZE {
            let line = &player.pattern_lines[row];
            if line.count > 0 {
                let color_str = line.color.map(|c| color_name(c)).unwrap_or("-");
                println!("    Row {}: {}/{} {}", row + 1, line.count, row + 1, color_str);
            }
        }
        if player.floor.len > 0 {
            println!("  Floor: {} tiles", player.floor.len);
        }
    }
}
