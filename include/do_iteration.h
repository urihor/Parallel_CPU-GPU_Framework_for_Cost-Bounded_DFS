#pragma once
#include "neural_batch_service.h"
#include "work.h"
#include "nvtx_helpers.h"
#include <iostream>

namespace batch_ida {
    // Type alias for a simple heuristic function pointer:
    //   int h(const Env::State&);
    template<class Env>
    using HeuristicFn = int (*)(const typename Env::State &);

    // Subtree expansion (Algorithm-4 style when neural batching is enabled).
    //
    // Semantics:
    //   * Returns true  – if a goal was found in this subtree (and Work marks it).
    //   * Returns false – otherwise (either pruned, still pending NN result, or
    //                     just progressed one child).
    //
    // Behavior controlled by feature flags:
    //   * neural_batch_enabled() == false:
    //       - Pure synchronous IDA*: heuristic(s) is called directly (e.g., PDB).
    //       - One child is expanded per call, like the original DoIteration.
    //
    //   * neural_batch_enabled() == true:
    //       - Heuristic values h(s) come from NeuralBatchService in a non-blocking way.
    //       - If the NN value for s is not available yet:
    //           * The state is enqueued for batching.
    //           * The function returns false immediately, so the caller can switch
    //             to a different Work (another subtree).
    //       - Once the NN value is ready, we compute f = g + h_prune, expand s,
    //         generate its children, push one child onto the DFS stack, and enqueue
    //         children to the NeuralBatchService so their h-values are computed
    //         while other subtrees are being explored.
    //
    //   * guide_batch_enabled():
    //       - NN is used only for “guiding” (ordering / scheduling) but not for
    //         pruning. The actual pruning heuristic remains the synchronous one
    //         (e.g., PDB). The gating still avoids doing heavy PDB work until
    //         the NN value for s is ready, so CPU work can better overlap GPU work.
    //
    template<class Env, class Heuristic>
    bool DoIteration(
        Env &env,
        WorkFor<Env> &work,
        const int bound,
        Heuristic heuristic, // synchronous heuristic used for pruning (e.g., PDB)
        int &next_bound
    ) {
        using State = typename Env::State;
        using Action = typename Env::Action;

        // If this Work is already finished or goal has been found, nothing to do.
        if (work.is_done() || work.goal_found()) return work.goal_found();

        // Ensure Work has a root frame; if not, there is nothing to expand yet.
        work.ensure_initialized();
        if (!work.has_current()) return false;

        // Current DFS frame (node) and its state / g-cost.
        auto &frame = work.current_frame();
        State s = frame.state;
        const int g = frame.g;

        // -----------------------------
        // 1) PRUNE / GUIDE gating
        // -----------------------------
        // Try to use the global NeuralBatchService. If it is not running,
        // we fall back to the purely synchronous heuristic path.
        NeuralBatchService *svc =
                NeuralBatchService::instance().is_running()
                    ? &NeuralBatchService::instance()
                    : nullptr;

        int h_prune = 0; // heuristic used in f = g + h_prune and for pruning
        int h_nn = 0; // NN heuristic value (can be used for pruning or just guiding)

        const bool use_nn_prune = batch_ida::neural_batch_enabled();
        const bool use_nn_guide = batch_ida::guide_batch_enabled();
        const bool gate_on_nn = (svc && (use_nn_prune || use_nn_guide));

        // Key for caching the NN heuristic at this frame (avoid re-requesting).
        const uint64_t key = s.pack();

        if (gate_on_nn) {
            // (a) Fast path: we already have cached NN result for this exact state
            //     on this DFS frame → no interaction with the batch service.
            if (frame.h_cached_ready && frame.h_cached_key == key) {
                h_nn = frame.h_cached_value;
                // If we use NN for pruning, h_prune = h_nn.
                // Otherwise, we still use the synchronous heuristic (e.g., PDB)
                // for pruning, and NN is only used for guiding.
                h_prune = use_nn_prune ? h_nn : heuristic(s);
            } else {
                // (b) Ask NeuralBatchService for h(s).
                //     request_h() can:
                //       * enqueue s and return Pending,
                //       * return Ready with a value,
                //       * or fall back (e.g., service shutting down).
                int tmp = 0;
                auto st = svc->request_h(s, tmp);

                if (st == NeuralBatchService::HRequestStatus::Pending) {
                    // NN value is not ready yet. We do not compute the synchronous
                    // heuristic here; we simply return false and let the calling
                    // CB-DFS thread pick another Work. This allows GPU work to run
                    // in parallel without blocking on a heavy PDB call.
                    return false; // gate: no PDB here
                }

                if (st == NeuralBatchService::HRequestStatus::Ready) {
                    // NN value arrived → cache it on this frame for reuse.
                    frame.h_cached_ready = true;
                    frame.h_cached_key = key;
                    frame.h_cached_value = tmp;

                    h_nn = tmp;
                    h_prune = use_nn_prune ? h_nn : heuristic(s);
                } else {
                    // Service did not give a value (e.g. disabled or error).
                    // Fall back to synchronous heuristic only.
                    h_prune = heuristic(s);
                }
            }
        } else {
            // No NN gating/guide → standard synchronous heuristic.
            h_prune = heuristic(s);
        }

        // IDA*: f = g + h. If f > bound, update next_bound and prune this node.
        const int f = g + h_prune;
        if (f > bound) {
            next_bound = (std::min)(next_bound, f);
            work.pop_frame();
            return false;
        }

        // Check for goal after passing the f-bound test.
        if (env.IsGoal(s)) {
            work.mark_goal_current();
            return true;
        }

        // -----------------------------
        // 2) Expand + enqueue children
        // -----------------------------
        if (!frame.expanded) {
            // First time we see this node: generate all applicable actions.
            frame.actions = env.GetActions(s);
            frame.next_child_index = 0;
            frame.expanded = true;
            work.increment_expanded();

            // If NN batching is active (prune or guide), enqueue all children so
            // their h-values can be computed asynchronously while other subtrees
            // are being explored by the CB-DFS threads.
            if (svc && (use_nn_prune || use_nn_guide)) {
                for (const auto &action: frame.actions) {
                    State child = s;
                    env.ApplyAction(child, action);
                    svc->enqueue(child);
                }
            }
        }

        // -----------------------------
        // 3) One-child DFS step
        // -----------------------------
        if (frame.next_child_index < frame.actions.size()) {
            // Take the next action, generate the child and push it onto
            // the local DFS stack. The caller will later continue from there.
            const auto action = frame.actions[frame.next_child_index++];

            State child = s;
            env.ApplyAction(child, action);
            const int child_g = g + 1;

            work.push_child(std::move(child), child_g, action);
        } else {
            // No more children from this node → backtrack in the DFS stack.
            work.pop_frame();
        }

        // No solution found yet on this call.
        return false;
    }
} // namespace batch_ida
