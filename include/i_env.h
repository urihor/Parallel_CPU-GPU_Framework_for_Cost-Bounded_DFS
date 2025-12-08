//
// Created by uriel on 04/11/2025.
//

#pragma once
#include <vector>

/// interface of environment
template<class State, class Action>
struct i_env {
    virtual std::vector<Action> GetActions(const State &s) = 0; // legal moves from state s
    virtual void ApplyAction(State &s, Action a) = 0; // apply one move (in-place)
    virtual void UndoAction(State &s, Action a) = 0; // undo a move
    virtual bool IsGoal(const State &s) const = 0; // is s the goal state
    virtual bool IsSolvable(const State &s) const = 0; // is the puzzle solvable

    virtual ~i_env() = default;
};
