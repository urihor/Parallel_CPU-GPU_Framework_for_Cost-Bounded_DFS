//
// Created by uriel on 04/11/2025.
//

/// Simple sliding-tile puzzle environment (4x4 15-puzzle).
/// Provides actions, transitions, goal test and solvability test.
///

#include "puzzle_env.h"

// Local helpers to convert a linear index [0..15] into row/column.
// These just forward to puzzle15_state::row / col.
namespace {
    int rowOf(const int idx) noexcept {
        return puzzle15_state::row(idx);
    }

    int colOf(const int idx) noexcept {
        return puzzle15_state::col(idx);
    }
}

// Return all legal moves (actions) from the current state.
// We look at the blank position and check which directions are inside the board.
// Order of actions: Up, Down, Left, Right (when applicable).
std::vector<StpEnv::Action> StpEnv::GetActions(const State &s) {
    std::vector<Action> moves;
    moves.reserve(4);

    const int b = s.blankPos;
    const int r = rowOf(b);
    const int c = colOf(b);

    if (r > 0) {
        moves.push_back(Action::Up);
    }
    if (r + 1 < State::rows()) {
        moves.push_back(Action::Down);
    }
    if (c > 0) {
        moves.push_back(Action::Left);
    }
    if (c + 1 < State::cols()) {
        moves.push_back(Action::Right);
    }

    return moves;
}

// Apply a single action to the state in-place.
// The action always moves the blank in the given direction by swapping it
// with the neighbor tile. NeighborIndex() computes the tile index to swap with.
void StpEnv::ApplyAction(State &s, const Action a) {
    const puzzle15_state::Index nb = NeighborIndex(s, a);
    // O(1) swap: also updates s.blankPos internally.
    s.swapBlankWith(nb);
}

// Undo an action by applying the inverse move.
// For example, UndoAction(..., Up) is equivalent to ApplyAction(..., Down).
void StpEnv::UndoAction(State &s, Action a) {
    // Undo = apply the opposite move.
    ApplyAction(s, inverse(a));
}

// Check whether the state is the goal state.
// Goal layout: tiles[0..14] = 1..15, tiles[15] = 0 (blank in bottom-right corner).
bool StpEnv::IsGoal(const State &s) const {
    for (int i = 0; i < State::size() - 1; i++) {
        if (s.tiles[i] != static_cast<puzzle15_state::Tile>(i + 1))
            return false;
    }
    return true;
}

// Check whether a given 4x4 puzzle configuration is solvable.
// Classic 15-puzzle rule: count inversions and consider the blank row.
// Here we:
//   * Count inversions (ignoring the blank).
//   * Compute the blank row from the bottom.
// Then we use a parity condition on (blank_row_from_bottom + inversions).
bool StpEnv::IsSolvable(const State &s) const {
    const auto &tiles = s.tiles;

    int inversions = 0;
    for (int i = 0; i < State::size(); ++i) {
        if (tiles[i] == 0)
            continue;
        for (int j = i + 1; j < State::size(); ++j) {
            if (tiles[j] == 0)
                continue;
            if (tiles[i] > tiles[j])
                ++inversions;
        }
    }

    // Example: row index from bottom for a 4x4 board.
    const int blank_row_from_bottom = 4 - (s.blankPos / 4);
    const bool solvable = ((blank_row_from_bottom + inversions) % 2 == 1);
    return solvable;
}

// Compute the index of the tile that will be swapped with the blank
// when applying action a to state s.
// Assumes that a is legal for s.
puzzle15_state::Index StpEnv::NeighborIndex(const State &s, const Action a) {
    const puzzle15_state::Index b = s.blankPos;
    puzzle15_state::Index nb = 0;
    switch (a) {
        case Action::Up:
            nb = b - State::cols();
            break;
        case Action::Down:
            nb = b + State::cols();
            break;
        case Action::Left:
            nb = b - 1;
            break;
        case Action::Right:
            nb = b + 1;
            break;
    }
    // At this point nb is guaranteed to be in the valid range [0, Size).
    return nb;
}
