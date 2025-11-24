//
// Created by Owner on 19/11/2025.
//
#pragma once

#include <vector>
#include <limits>

#include "work.h"
#include "cb-dfs.h"
#include "do_iteration.h"
#include "generate_work.h"   // assumed existing header for Algorithm 2

namespace batch_ida {
    /**
     * Batch IDA* (Algorithm 1) - single-threaded version.
     *
     * This function runs an IDA* loop using:
     *   - Algorithm 2 (GenerateWork) to build a pool of subtree roots ("works")
     *     up to depth d_init from the global root.
     *   - Algorithm 3 (CB-DFS) to perform a cost-bounded DFS over those works
     *     with the current threshold 'bound'.
     *   - Algorithm 4 (DoIteration) as the primitive expansion step.
     *
     * For now, we do NOT reconstruct the actual solution path. We only return
     * the solution cost (depth) if a solution is found. Path reconstruction
     * can be added later by extending Work/Node to store parent pointers.
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
                  std::vector<typename Env::Action> &solution) {
        using Action = typename Env::Action;

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
        int bound = heuristic(start);
        if (bound >= INF) {
            // Heuristic says "infinite" / unreachable.
            return false;
        }

        // IDA* outer loop.
        while (true) {
            // 1) Generate initial works (Algorithm 2).
            std::vector<WorkFor<Env> > works;
            works.reserve(1024); // arbitrary; can be tuned or removed

            std::vector<Action> empty_history;

            // These are used only inside GenerateWork. We ignore the returned
            // path for now and rely on IDA* thresholds for the solution cost.
            int best_len = INF;
            std::vector<Action> best_sol;

            /*GenerateWork(env,
                         start,
                         d_init,
                         empty_history,
                         works,
                         best_len,
                         best_sol);*/

            std::unordered_set<std::size_t> seen;
            GenerateWorkDedup(env,
                         start,
                         d_init,
                         empty_history,
                         works,
                         seen,
                         [](const puzzle15_state& s){ return std::hash<puzzle15_state>{}(s); },
                         best_len,
                         best_sol);

            // If GenerateWork itself found a solution with cost <= bound,
            // we can stop immediately.
            if (best_len < INF && best_len <= bound) {
                solution_cost = best_len;
                solution = best_sol;
                return true;
            }

            // If there are no works at all and GenerateWork did not find a
            // solution, then there is nothing left to search.
            if (works.empty()) {
                return false;
            }

            // 2) Run CB-DFS (Algorithm 3) on the generated works
            //    with the current threshold 'bound'.
            int next_bound = INF;
            if (CB_DFS(env,
                       works,
                       work_num,
                       bound,
                       heuristic,
                       next_bound)) {
                // With an admissible heuristic and the standard IDA* bound
                // update rule, the first solution found has cost == bound.
                // Find which Work contains the goal and reconstruct its path.
                for (auto &w: works) {
                    if (w.goal_found()) {
                        solution.clear();
                        w.reconstruct_full_path(solution);
                        solution_cost = static_cast<int>(solution.size());
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
                  std::vector<typename Env::Action> &solution) {
        return BatchIDA<Env, HeuristicFn<Env> >(env,
                                                start,
                                                heuristic,
                                                d_init,
                                                work_num,
                                                solution_cost,
                                                solution);
    }
} // namespace batch_ida
