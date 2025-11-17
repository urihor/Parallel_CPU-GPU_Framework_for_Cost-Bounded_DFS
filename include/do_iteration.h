//
// Created by Owner on 16/11/2025.
//

//
// Algorithm 4: Subtree expansion (DoIteration) for Batch IDA* / CB-DFS.
// Generic version that works with any Env (15-puzzle, Rubik's cube, etc.).
//
#pragma once

#include "work.h"

namespace batch_ida {

// Generic heuristic function pointer type.
// Env must define a nested State type.
template<class Env>
using HeuristicFn = int (*)(const typename Env::State&);

/**
 * One CB-DFS / Batch IDA* expansion step for a single subtree (Work).
 *
 * - env        : search domain (must provide State, Action, GetActions, ApplyAction, IsGoal).
 * - work       : subtree descriptor and DFS stack.
 * - bound      : current IDA* threshold on f = g + h.
 * - heuristic  : heuristic function h(s).
 * - next_bound : in/out. Must be initialised by the caller to +INF.
 *                For every pruned node with f > bound, we update:
 *                    next_bound = min(next_bound, f).
 *
 * Returns true iff a goal state was found with f <= bound.
 */
template<class Env, class Heuristic>
bool DoIteration(Env& env,
                 WorkFor<Env>& work,
                 int bound,
                 Heuristic heuristic,
                 int& next_bound)
{
    using State  = typename Env::State;
    using Action = typename Env::Action;
    using Node   = typename WorkFor<Env>::Node;

    // If this subtree is already exhausted, there is nothing to do.
    if (work.is_done()) {
        return false;
    }

    // Ensure that the DFS stack is initialized with the subtree root.
    work.ensure_initialized();

    while (true) {
        if (work.is_done()) {
            return false; // no nodes left in this subtree
        }

        Node node = work.pop_node();

        State& s_mut = node.state;
        const State& s = s_mut;
        const int g = node.g;

        const int h = heuristic(s);
        const int f = g + h;

        if (f > bound) {
            // Pruned node; its f-value is a candidate for the next threshold.
            if (f < next_bound) {
                next_bound = f;
            }
            // Try another node in this subtree.
            continue;
        }

        // Now we know f <= bound: candidate for expansion / goal.
        if (env.IsGoal(s)) {
            // Found a solution within the current bound.
            return true;
        }

    auto actions = env.GetActions(s);

    for (Action a : actions) {
        // For a generic Env we assume ApplyAction returns the successor state.
        State child = s_mut;
        env.ApplyAction(child, a);

            const int child_g = g + 1; // assume unit-cost edges
            work.push_node(child, child_g);
        }

        // Only a single node is expanded on each DoIteration call.
        return false;
    }
}

// Convenience overload for raw function-pointer heuristics.
template<class Env>
bool DoIteration(Env& env,
                 WorkFor<Env>& work,
                 int bound,
                 HeuristicFn<Env> heuristic,
                 int& next_bound)
{
    return DoIteration<Env, HeuristicFn<Env>>(env, work, bound, heuristic, next_bound);
}

} // namespace batch_ida
