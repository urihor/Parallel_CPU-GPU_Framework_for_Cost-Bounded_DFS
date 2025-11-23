//
// Created by uriel on 17/11/2025.
//
#pragma once

#include <vector>
#include <limits>
#include <algorithm>

#include "work.h"
#include "work_scheduler.h"
#include "do_iteration.h"  // DoIteration, HeuristicFn

namespace batch_ida {

/**
 * Single-threaded CB-DFS (Algorithm 3) for one IDA* iteration.
 *
 * This implementation mirrors the structure of the pseudocode:
 *
 *   - There is a global pool of Works (subtrees), created by GenerateWork.
 *   - We simulate 'work_num' logical stacks:
 *         stack[0], stack[1], ..., stack[work_num-1]
 *     In the original pseudocode, each stack[i] is a CB-DFS stack assigned
 *     to a logical worker / core.
 *
 *   - Each stack[i] in this C++ code is represented by a pointer to a Work
 *     (subtree) in the global 'works' vector:
 *         WorkFor<Env>* stack[i];
 *
 *   - terminated[i] indicates that stack[i] will never receive more work
 *     (its current subtree is done and the global pool is exhausted).
 *
 * Control variables:
 *
 *   - miss    : number of stacks that have entered the terminated state.
 *   - counter : round-robin index over stacks:
 *                   i = counter mod num_stacks
 *
 *   When miss == num_stacks, all stacks are in the terminated state and
 *   there is no more work in this IDA* iteration.
 *
 * Parameters:
 *   env        - problem environment (15-puzzle, cube, etc.).
 *   works      - global pool of Work items (subtrees), from GenerateWork.
 *   work_num   - number of logical stacks to simulate (workNum in pseudocode).
 *   bound      - current IDA* threshold on f = g + h.
 *   heuristic  - heuristic function h(s).
 *   next_bound - [out] minimal f(n) > bound over all pruned nodes in this
 *                iteration, or std::numeric_limits<int>::max() if none.
 *
 * Returns:
 *   true  - if a goal state was found with f <= bound.
 *   false - otherwise.
 */
template<class Env, class Heuristic>
bool CB_DFS(Env& env,
            std::vector<WorkFor<Env>>& works,
            int work_num,
            int bound,
            Heuristic heuristic,
            int& next_bound)
{
    using std::numeric_limits;
    using WorkType = WorkFor<Env>;

    next_bound = numeric_limits<int>::max();

    if (works.empty() || work_num <= 0) {
        return false;
    }

    // We cannot have more logical stacks than Works.
    const std::size_t num_stacks =
        std::min<std::size_t>(static_cast<std::size_t>(work_num), works.size());

    WorkScheduler<Env> scheduler(works);

    // stack[i] in the pseudocode: which Work is assigned to logical stack i.
    std::vector<WorkType*> stack(num_stacks, nullptr);

    // terminated[i] == 1 → this stack will never get more work.
    std::vector<unsigned char> terminated(num_stacks, 0);

    int miss = 0; // number of stacks that have entered the terminated state

    // Initial assignment: try to give each stack[i] an initial Work.
    for (std::size_t i = 0; i < num_stacks; ++i) {
        WorkType* w = nullptr;
        if (scheduler.acquire(w)) {
            stack[i] = w;
            terminated[i] = 0;
        } else {
            stack[i] = nullptr;
            terminated[i] = 1;
            ++miss; // this stack starts in the terminated state
        }
    }

    // If all stacks are terminated from the start, there is nothing to do.
    if (miss == static_cast<int>(num_stacks)) {
        return false;
    }

    bool found_solution = false;

    std::size_t counter = 0;

    // Stop when:
    //   - we find a goal, or
    //   - miss == num_stacks (all stacks are terminated).
    while (!found_solution && miss < static_cast<int>(num_stacks)) {
        const std::size_t i = counter % num_stacks;

        if (terminated[i]) {
            // This stack is permanently out of work; already counted in 'miss'.
            ++counter;
            continue;
        }

        WorkType* w = stack[i];

        // If this stack[i] has no Work or its Work is finished, try to acquire a new Work.
        if (w == nullptr || w->is_done()) {
            WorkType* new_w = nullptr;
            if (scheduler.acquire(new_w)) {
                stack[i] = new_w;
                w = new_w;
            } else {
                // No more Works in global pool → this stack is now terminated.
                terminated[i] = 1;
                ++miss;       // we have one more terminated stack
                ++counter;
                continue;
            }
        }

        // At this point we have a non-null Work that is not done.
        found_solution = DoIteration(env, *w, bound, heuristic, next_bound);
        ++counter;
    }

    return found_solution;
}

/**
 * Convenience overload for function-pointer heuristics:
 *
 *   int h(const Env::State&);
 */
template<class Env>
bool CB_DFS(Env& env,
            std::vector<WorkFor<Env>>& works,
            int work_num,
            int bound,
            HeuristicFn<Env> heuristic,
            int& next_bound)
{
    return CB_DFS<Env, HeuristicFn<Env>>(env, works, work_num, bound, heuristic, next_bound);
}

} // namespace batch_ida
