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
 *   - We simulate 'work_num' logical workers. Each worker i holds at most
 *     one active Work in slot[i].
 *   - terminated[i] indicates that worker i will never receive more work
 *     (its current subtree is finished and the global pool is exhausted).
 *
 * Control variables:
 *
 *   - miss    : number of workers that have entered the terminated state.
 *   - counter : round-robin index over workers (for the parallel version;
 *               here we simulate it in a single thread).
 *
 * Loop (high level):
 *
 *   miss    ← 0
 *   counter ← 0
 *   while miss < num_slots and not goal found:
 *       i ← counter mod num_slots
 *       if terminated[i] == 1 then
 *           // worker i is permanently out of work
 *           // miss is NOT changed here (it was already counted)
 *       else
 *           if slot[i] is empty or its Work is done then
 *               try to acquire new Work from global pool
 *               if acquire fails then
 *                   terminated[i] ← 1
 *                   miss ← miss + 1
 *               else
 *                   // acquired new Work; perform one expansion step
 *                   DoIteration(slot[i], bound, heuristic, next_bound)
 *           else
 *               // slot[i] already holds an active non-done Work
 *               DoIteration(slot[i], bound, heuristic, next_bound)
 *       counter ← counter + 1
 *
 *   When miss == num_slots, all workers are in the terminated state and
 *   there is no more work in this IDA* iteration.
 *
 * Parameters:
 *   env        - problem environment (15-puzzle, cube, etc.).
 *   works      - global pool of Work items (subtrees), from GenerateWork.
 *   work_num   - number of logical workers / slots to simulate.
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

    // We cannot have more slots than Works.
    const std::size_t num_slots =
        std::min<std::size_t>(static_cast<std::size_t>(work_num), works.size());

    WorkScheduler<Env> scheduler(works);

    // slot[i] holds the currently assigned Work for worker i (or nullptr).
    std::vector<WorkType*> slots(num_slots, nullptr);

    // terminated[i] == 1 → this worker will never get more work.
    std::vector<unsigned char> terminated(num_slots, 0);

    int miss = 0; // number of workers that have entered the terminated state

    // Initial assignment: try to give each slot an initial Work.
    for (std::size_t i = 0; i < num_slots; ++i) {
        WorkType* w = nullptr;
        std::size_t idx = 0;
        if (scheduler.acquire(w, idx)) {
            slots[i] = w;
            terminated[i] = 0;
        } else {
            slots[i] = nullptr;
            terminated[i] = 1;
            ++miss; // this worker starts in the terminated state
        }
    }

    // If all workers are terminated from the start, there is nothing to do.
    if (miss == static_cast<int>(num_slots)) {
        return false;
    }

    bool found_solution = false;

    std::size_t counter = 0;

    // Stop when:
    //   - we find a goal, or
    //   - miss == num_slots (all workers are terminated).
    while (!found_solution && miss < static_cast<int>(num_slots)) {
        const std::size_t i = counter % num_slots;

        if (terminated[i]) {
            // This worker is permanently out of work; already counted in 'miss'.
            ++counter;
            continue;
        }

        WorkType* w = slots[i];

        // If this slot has no Work or its Work is finished, try to acquire a new Work.
        if (w == nullptr || w->is_done()) {
            WorkType* new_w = nullptr;
            std::size_t idx = 0;
            if (scheduler.acquire(new_w, idx)) {
                slots[i] = new_w;
                w = new_w;
            } else {
                // No more Works in global pool → this worker is now terminated.
                terminated[i] = 1;
                ++miss;       // we have one more terminated worker
                ++counter;
                continue;
            }
        }

        // At this point we have a non-null Work that is not done.
        bool found = DoIteration(env, *w, bound, heuristic, next_bound);
        if (found) {
            found_solution = true;
        }

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

