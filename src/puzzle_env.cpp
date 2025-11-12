//
// Created by uriel on 04/11/2025.
//
#include "puzzle_env.h"

// עוזרים מקומיים לשורות/עמודות
namespace {
    int rowOf(const int idx) noexcept {
        return puzzle15_state::row(idx);
    }

    int colOf(const int idx) noexcept {
        return puzzle15_state::col(idx);
    }
}

std::vector<StpEnv::Action> StpEnv::GetActions(const State &s) {
    std::vector<Action> moves;
    moves.reserve(4);

    const int b = s.blankPos;
    const int r = rowOf(b);
    const int c = colOf(b);

    if (r > 0) {
        moves.push_back(Action::Up);
    }
    if (r + 1 < s.rows()) {
        moves.push_back(Action::Down);
    }
    if (c > 0) {
        moves.push_back(Action::Left);
    }
    if (c + 1 < s.cols()) {
        moves.push_back(Action::Right);
    }

    return moves;
}

void StpEnv::ApplyAction(State &s, const Action a) {
    const int nb = NeighborIndex(s, a);
    s.swapBlankWith(nb); // החלפה O(1): מעדכן גם את blankPos
}

void StpEnv::UndoAction(State &s, Action a) {
    // ביטול = יישום המהלך ההפוך
    ApplyAction(s, inverse(a));
}

bool StpEnv::IsGoal(const State &s) const {
    // tiles[0..14] = 1..15 ; tiles[15] = 0
    for (int i = 0; i < s.size() - 1; i++) {
        if (s.tiles[i] != static_cast<puzzle15_state::Tile>(i + 1))
            return false;
    }
    return true;
}

puzzle15_state::Index StpEnv::NeighborIndex(const State &s, const Action a) {

    const int b = s.blankPos; // להימנע מ-underflow
    int nb = 0;
    switch (a) {
        case Action::Up: nb = b - s.cols();
            break;
        case Action::Down: nb = b + s.cols();
            break;
        case Action::Left: nb = b - 1;
            break;
        case Action::Right: nb = b + 1;
            break;
    }
    // כאן nb תקין בטווח [0, Size)
    return static_cast<puzzle15_state::Index>(nb);
}
