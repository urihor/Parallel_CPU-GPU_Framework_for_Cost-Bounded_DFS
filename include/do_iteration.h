#pragma once
#include "neural_batch_service.h"
#include "work.h"
#include "nvtx_helpers.h"

namespace batch_ida {
    // Type alias for a simple heuristic function pointer:
    //   int h(const Env::State&);
    template<class Env>
    using HeuristicFn = int (*)(const typename Env::State &);

    // Subtree expansion (Algorithm 4-style when neural batching is enabled).
    //
    // When batch_ida::neural_batch_enabled() == false, this function behaves like
    // the original version: synchronous heuristic evaluation and expanding a
    // single child per call.
    //
    // When batch_ida::neural_batch_enabled() == true, the heuristic values h(s)
    // are obtained from NeuralBatchService in a non-blocking way:
    //
    //   * If h(s) is not ready yet, we enqueue s into the batch and return false.
    //     The calling CB-DFS thread will then try a different Work.
    //   * Once h(s) is ready, we expand s, generate all its children, push them
    //     onto the local DFS stack, and enqueue each child into the batch so that
    //     its heuristic can be computed while other subtrees are explored.
    //
    template<class Env, class Heuristic>
    bool DoIteration(
        Env &env,
        WorkFor<Env> &work,
        const int bound,
        Heuristic heuristic,
        int &next_bound
    ) {
        using State = typename Env::State;
        using Action = typename Env::Action;

        if (work.is_done() || work.goal_found()) {
            return work.goal_found();
        }
        work.ensure_initialized();

        if (!work.has_current()) {
            return false;
        }

        auto &frame = work.current_frame();
        State s = frame.state;
        const int g = frame.g;

        // Decide whether we are in "batched neural" mode or plain synchronous mode.
        NeuralBatchService *batch_service = nullptr;
        if (neural_batch_enabled()) {
            batch_service = &NeuralBatchService::instance();
        }

        int h = 0;

        if (batch_service && batch_service->is_running()) {
            int h_dummy = 0;
            const auto st = batch_service->request_h(s, h_dummy);
            // Non-blocking: try to read h(s) from the batch service.
            if (st == NeuralBatchService::HRequestStatus::Ready) {
                NVTX_MARK("DoIteration: request_h HIT");
                h = h_dummy;
            } else if (st == NeuralBatchService::HRequestStatus::Pending) {
                NVTX_MARK("DoIteration: request_h MISS -> enqueue+yield");
                return false;
            }
            h = heuristic(s);
        } else {
            // Fallback: normal synchronous heuristic evaluation.
            h = heuristic(s);
        }

        const int f = g + h;

        // If f > bound, update next_bound and prune.
        if (f > bound) {
            next_bound = (std::min)(next_bound, f);
            work.pop_frame();
            return false;
        }

        // Goal test.
        if (env.IsGoal(s)) {
            work.mark_goal_current();
            return true;
        }

        // Prepare expansion if needed.
        if (!frame.expanded) {
            frame.actions.clear();
            frame.actions = env.GetActions(s);
            frame.next_child_index = 0;
            frame.expanded = true;
            work.increment_expanded();
            if (batch_service && batch_service->is_running()) {
                for (const auto &action: frame.actions) {
                    State child = s;
                    env.ApplyAction(child, action);
                    batch_service->enqueue(child);
                }
            }
        }

        // Expansion step.
        if (frame.next_child_index < frame.actions.size()) {
            // Expand only one child per call for better interleaving between
            // works when using a purely synchronous heuristic.

            const auto action = frame.actions[frame.next_child_index++];

            State child = s;
            env.ApplyAction(child, action);
            const int child_g = g + 1; // unit-cost move in STP

            work.push_child(std::move(child), child_g, action);
        } else {
            // No more children => backtrack.
            work.pop_frame();
        }

        return false;
    }
} // namespace batch_ida
