//
// Created by uriel on 04/11/2025.
//

#pragma once
#include <vector>
#include "i_env.h"
#include "puzzle15_state.h"
#include "puzzle_actions.h"

struct StpEnv final : i_env<puzzle15_state, StpMove> {
    using State = puzzle15_state;
    using Action = StpMove;

    // return all the legal moves according to the blank position
    std::vector<Action> GetActions(const State &s) override;

    // apply one move on the state, move the blank to the relevant neighbor
    void ApplyAction(State &s, Action a) override;

    // undo a move : apply the inverse move
    void UndoAction(State &s, Action a) override;

    // is the state is the goal state
    [[nodiscard]] bool IsGoal(const State &s) const override;

    // is the board is solvable
    [[nodiscard]] bool IsSolvable(const State &s) const override;

private:
    // return the index of the neighbor that we swap with (assume the move is legal)
    [[nodiscard]] static puzzle15_state::Index NeighborIndex(const State &s, Action a);
};
