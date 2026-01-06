//
// Created by Owner on 19/11/2025.
//
#pragma once

#include <vector>
#include <limits>
#include <thread>

#include "work.h"
#include "heuristic_router.h"
#include "cb-dfs.h"
#include "do_iteration.h"
#include "generate_work.h"   // assumed existing header for Algorithm 2

namespace batch_ida {
    /**
     * Batch IDA* (Algorithm 1) - multy-threaded version.
     *
     * This function runs an IDA* loop using:
     *   - Algorithm 2 (GenerateWork) to build a pool of subtree roots ("works")
     *     up to depth d_init from the global root.
     *   - Algorithm 3 (CB-DFS) to perform a cost-bounded DFS over those works
     *     with the current threshold 'bound'.
     *   - Algorithm 4 (DoIteration) as the primitive expansion step.
     *
     * Template parameters:
     *   Env       - environment type, must provide:
     *                 using State;
     *                 using Action;
     *                 bool IsGoal(const State&) const;
     *   Heuristic - any callable with signature:
     *                 int h(const Env::State&);
     *
     * Parameters:
     *   env           - problem environment (15-puzzle, cube, etc.).
     *   start         - start state.
     *   heuristic     - admissible heuristic h(s) (e.g., PDB).
     *   d_init        - depth limit for GenerateWork (Algorithm 2).
     *   work_num      - logical number of stacks (workNum in Algorithm 3).
     *   solution_cost - [out] on success, the optimal solution cost (depth);
     *                   on failure, set to std::numeric_limits<int>::max().
     *
     * Returns:
     *   true  - if a solution was found; solution_cost holds its cost.
     *   false - if no solution was found (or the search space is exhausted
     *           without reaching a goal).
     */
    template<class Env, class Heuristic>
    bool BatchIDA(Env &env,
                  typename Env::State &start,
                  Heuristic heuristic,
                  int d_init,
                  int work_num,
                  int &solution_cost,
                  std::vector<typename Env::Action> &solution,
                  const int num_threads_hint = 0) {
        using Action = typename Env::Action;
        using State = typename Env::State;

        constexpr int INF = std::numeric_limits<int>::max();
        solution_cost = INF;
        solution.clear();


        if (!env.IsSolvable(start)) {
            // Unsolvable instance: no need to run IDA* at all.
            std::cerr << "Board is not solvable" << std::endl;
            return false;
        }

        // Trivial case: start is already the goal.
        if (env.IsGoal(start)) {
            solution_cost = 0;
            return true;
        }

        // Initial IDA* threshold.
        const bool svc_on = NeuralBatchService::instance().is_running();
        const bool use_nn_prune = batch_ida::neural_batch_enabled();
        const bool use_nn_guide = batch_ida::guide_batch_enabled();

        if (svc_on && (use_nn_prune || use_nn_guide)) {
            NeuralBatchService::instance().enqueue(start); // warm-up
        }

        int bound = use_nn_prune && svc_on
                        ? HeuristicRouter::instance().h_sync(start)
                        : heuristic(start);


        if (bound >= INF) {
            // Heuristic says "infinite" / unreachable.
            return false;
        }

        // --------------------------------------------------
        // 1) Generate initial works ONCE (Algorithm 2).
        // --------------------------------------------------
        std::vector<WorkFor<Env> > works;
        works.reserve(1024); // arbitrary; can be tuned or removed

        std::vector<Action> empty_history;

        int best_len = INF;
        std::vector<Action> best_sol;
        /*GenerateWork(env,
                         start,
                         d_init,
                         empty_history,
                         works,
                         best_len,
                         best_sol);*/
        auto key_fn = [](const State &s) {
            return s.pack(); // State must have pack()
        };
        std::unordered_set<std::size_t> seen;
        GenerateWorkDedup(env,
                          start,
                          d_init,
                          empty_history,
                          works,
                          seen,
                          key_fn,
                          best_len,
                          best_sol);

        //std::cout << "num of works: " << works.size() << std::endl;
        // If GenerateWork itself found a solution with cost <= bound,
        // we can stop immediately.
        if (best_len <= bound) {
            solution_cost = best_len;
            solution = best_sol;
            return true;
        }
        // If there are no works at all and GenerateWork did not find a
        // solution, then there is nothing left to search.
        if (works.empty()) {
            return false;
        }
        // Decide how many OS threads to use once, based on the hint + hardware.
        std::size_t num_threads;
        if (num_threads_hint > 0) {
            num_threads = static_cast<std::size_t>(num_threads_hint);
        } else {
            unsigned int hw = std::thread::hardware_concurrency();
            num_threads = (hw == 0u) ? 1u : static_cast<std::size_t>(hw);
        }


        // Do not run more threads than Works.
        if (num_threads > works.size()) {
            num_threads = works.size();
        }

        //std::cout << "num of threads: "<<num_threads << std::endl;

        // --------------------------------------------------
        // 2) IDA* outer loop: reuse the same works each time.
        // --------------------------------------------------
        while (true) {
            // If neural batching is enabled, clear the cached entries between bounds.
            // This drops all states from the previous IDA* iteration and lets the
            // GPU cache only states relevant to the current threshold.
            /*if (neural_batch_enabled()) {
                NeuralBatchService::instance().reset_for_new_bound();
            }*/

            // Reset all works' search state for the new IDA* iteration.
            for (auto &w: works) {
                w.reset_for_new_iteration();
            }

            // Run CB-DFS (Algorithm 3) on the generated works
            // with the current threshold 'bound'.
            int next_bound = INF;

            if (CB_DFS(env,
                       works,
                       work_num,
                       bound,
                       heuristic,
                       next_bound,
                       num_threads)) {
                // With an admissible heuristic and the standard IDA* bound
                // update rule, the first solution found has cost == bound.
                // Find which Work contains the goal and reconstruct its path.
                for (auto &w: works) {
                    if (w.goal_found()) {
                        solution.clear();
                        w.reconstruct_full_path(solution);
                        solution_cost = static_cast<int>(solution.size());

                        std::uint64_t total_expanded = 0;
                        for (const auto &ww: works) {
                            total_expanded += ww.expanded_nodes();
                        }

                        //std::cout << "Nodes expanded for this board: "
                        //       << total_expanded << std::endl;

                        return true;
                    }
                }
            }

            // No solution within 'bound'. If next_bound stayed INF, it means
            // there were no nodes with f > bound, so increasing the bound
            // will not reveal new nodes → no solution exists.
            if (next_bound == INF) {
                return false;
            }

            // 3) Update the threshold and start a new IDA* iteration.
            bound = next_bound;
        }

        // Should not reach here.
        return false;
    }

    /**
     * Convenience overload for function-pointer heuristics:
     *
     *   int h(const Env::State&);
     */
    template
    <class Env>
    bool BatchIDA(Env &env,
                  typename Env::State &start,
                  HeuristicFn<Env> heuristic,
                  int d_init,
                  int work_num,
                  int &solution_cost,
                  std::vector<typename Env::Action> &solution,
                  int num_threads_hint = 0) {
        return BatchIDA<Env, HeuristicFn<Env> >(env,
                                                start,
                                                heuristic,
                                                d_init,
                                                work_num,
                                                solution_cost,
                                                solution,
                                                num_threads_hint);
    }
} // namespace batch_ida
