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
 * Algorithm 4: DoIteration
 *
 * Executes a single subtree-expansion step on the given Work.
 * The code is fully generic in Env and Heuristic, so it can be used
 * for different domains (15-puzzle, Rubik's cube, etc.).
 *
 * Requirements on Env:
 *   using State;
 *   using Action;
 *   std::vector<Action> GetActions(const State&) const;
 *   State ApplyAction(const State&, Action) const;
 *
 * Requirements on Heuristic:
 *   int operator()(const State&)  or  int(const State&)
 */
template<class Env, class Heuristic>
void DoIteration(Env& env,
                 WorkFor<Env>& work,
                 int bound,
                 Heuristic heuristic)
{
    using State    = typename Env::State;
    using Action   = typename Env::Action;
    using WorkType = WorkFor<Env>;
    using Node     = typename WorkType::Node;

    // If this subtree is already exhausted, there is nothing to do.
    if (work.is_done()) {
        return;
    }

    // Ensure that the DFS stack is initialized with the subtree root.
    work.ensure_initialized();

    bool newStatesFound = false;
    Node node_to_expand{};

    // Look for the next node whose f-value is strictly below the current bound.
    while (!newStatesFound) {
        if (work.is_done()) {
            // No more nodes left in this subtree.
            return;
        }

        Node node = work.pop_node();
        const State& s = node.state;
        const int g    = node.g;

        const int h = heuristic(s);
        const int f = g + h;

        if (f < bound) {
            node_to_expand = node;
            newStatesFound = true;
        } else {
            // Node is pruned. In the full Batch IDA* algorithm this f-value
            // should be used to update the next cost threshold.
        }
    }

    const State& s = node_to_expand.state;
    const int g    = node_to_expand.g;

    auto actions = env.GetActions(s);

    for (Action a : actions) {
        // For a generic Env we assume ApplyAction returns the successor state.
        State child = env.ApplyAction(s, a);

        // For now, we assume unit edge cost for all domains.
        const int child_g = g + 1;
        work.push_node(child, child_g);

        // In the original paper, TensorRepresentation(child) would be
        // appended to a GPU batch. We intentionally omit this here.
    }
}

// Convenience overload for raw function pointer heuristics.
template<class Env>
void DoIteration(Env& env,
                 WorkFor<Env>& work,
                 int bound,
                 HeuristicFn<Env> heuristic)
{
    DoIteration<Env, HeuristicFn<Env>>(env, work, bound, heuristic);
}

} // namespace batch_ida
