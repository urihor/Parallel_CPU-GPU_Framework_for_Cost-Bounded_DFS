#pragma once

#include <limits>
#include "work.h"

namespace batch_ida {

    template<class Env>
    using HeuristicFn = int (*)(const typename Env::State &);

    template<class Env, class Heuristic>
    bool DoIteration(Env &env,
                     WorkFor<Env> &work,
                     int bound,
                     Heuristic heuristic,
                     int &next_bound)
    {
        using State  = typename Env::State;
        using Action = typename Env::Action;

        constexpr int INF = std::numeric_limits<int>::max();
        next_bound = INF;

        if (work.is_done() || work.goal_found()) {
            return work.goal_found();
        }

        work.ensure_initialized();

        while (true) {
            if (!work.has_current()) {
                // No more nodes left in this subtree.
                return false;
            }

            auto &frame = work.current_frame();

            if (!frame.expanded) {
                const State &s = frame.state;
                const int g = frame.g;
                const int h = heuristic(s);
                const int f = g + h;

                work.increment_expanded();

                if (f > bound) {
                    if (f < next_bound) {
                        next_bound = f;
                    }
                    work.pop_frame();
                    continue;
                }

                if (env.IsGoal(s)) {
                    work.mark_goal_current();
                    return true;
                }

                frame.actions = env.GetActions(s);
                frame.next_child_index = 0;
                frame.expanded = true;
            }

            if (frame.next_child_index < frame.actions.size()) {
                const Action a = frame.actions[frame.next_child_index++];

                State child = frame.state;
                env.ApplyAction(child, a);

                const int child_g = frame.g + 1; // uniform cost
                work.push_child(child, child_g, a);

                // Only one new node per call.
                return false;
            }
            else {
                // No more children for this node: backtrack.
                work.pop_frame();
                // loop again to find another node or finish the subtree
            }
        }
    }

    template<class Env>
    bool DoIteration(Env &env,
                     WorkFor<Env> &work,
                     int bound,
                     HeuristicFn<Env> heuristic,
                     int &next_bound)
    {
        return DoIteration<Env, HeuristicFn<Env>>(env, work, bound, heuristic, next_bound);
    }

} // namespace batch_ida
