//
// Created by uriel on 17/11/2025.
// Parallel CB-DFS (CPU only) following Algorithm 3 style.
// Each OS thread runs a local CB-DFS with `work_num` logical stacks,
// and WorkScheduler is used to acquire new subtrees (Work objects) as needed.
//
#pragma once

#include <vector>
#include <limits>
#include <algorithm>
#include  <thread>
#include <atomic>
#include <cstddef>

#include "work.h"
#include "work_scheduler.h"
#include "do_iteration.h"  // DoIteration, HeuristicFn

namespace batch_ida {

/**
 * Multy-threaded CB-DFS (Algorithm 3) for one IDA* iteration.
 *
 * This implementation mirrors the structure of the pseudocode:
 *
 *   - There is a global pool of Works (subtrees), created by GenerateWork.
 *   - We simulate 'work_num' logical stacks:
 *         stack[0], stack[1], ..., stack[work_num-1]
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
            int& next_bound,
            int num_threads = 0)
{
    using WorkType = WorkFor<Env>;
    constexpr int INF = std::numeric_limits<int>::max();

    next_bound = INF;

    if (works.empty() || work_num <= 0) {
        return false;
    }

    // Thread-safe scheduler over the global pool of Work objects.
    WorkScheduler<Env> scheduler(works);

    // Shared state between all CPU threads.
    std::atomic<bool> found_solution(false);
    std::atomic<int>  global_next_bound(INF);

    // Decide how many OS threads to use.
    std::size_t num_of_threads;
    if (num_threads > 0) {
        num_of_threads = static_cast<std::size_t>(num_threads);
    } else {
        unsigned int hw = std::thread::hardware_concurrency();
        num_of_threads = (hw == 0u) ? 1u : static_cast<std::size_t>(hw);
    }

    if (num_of_threads > works.size()) {
        num_of_threads = works.size();
    }
    if (num_of_threads == 0) {
        return false;
    }

    // Worker function: each OS thread runs a local copy of Algorithm 3.
    auto worker_fn = [&](int /*thread_id*/) {
        const auto num_stacks = static_cast<std::size_t>(work_num);

        // Each thread has its own array of logical stacks (Work pointers).
        std::vector<WorkType*> stacks(num_stacks, nullptr);
        std::vector<bool> terminated(num_stacks, false);

        int local_miss = 0;
        std::size_t counter = 0;

        // Initial acquire: for every stack slot, try to get a Work from the scheduler.
        // This corresponds to "Initiate stack[workNum]" in the pseudocode.
        for (std::size_t i = 0; i < num_stacks; ++i) {
            WorkType* w = nullptr;
            if (!scheduler.acquire(w)) {
                // No global Work available for this slot → treat as permanently terminated.
                terminated[i] = true;
                ++local_miss;
            } else {
                stacks[i] = w;
            }
        }

        // Main CB-DFS loop for this thread.
        //
        // Stop when:
        //   - a solution was found by any thread (found_solution == true), or
        //   - all local stacks have entered the terminated state (local_miss == num_stacks).
        while (!found_solution.load(std::memory_order_acquire) &&
               local_miss < static_cast<int>(num_stacks)) {

            const std::size_t i = counter % num_stacks;
            ++counter;

            if (terminated[i]) {
                // This stack is permanently out of work; already counted in 'local_miss'.
                continue;
            }

            WorkType*& w = stacks[i];

            // If this stack currently has no Work, or the assigned Work is fully explored,
            // try to acquire a new Work from the global scheduler.
            if (w == nullptr || w->is_done()) {
                WorkType* new_w = nullptr;
                if (!scheduler.acquire(new_w)) {
                    // Scheduler has no more Work items to hand out at this point.
                    // This logical stack is now terminated forever.
                    terminated[i] = true;
                    ++local_miss;
                    continue;
                }
                w = new_w;
                // Keep terminated[i] == false and continue with the new subtree
            }

            // At this point, 'w' points to an active subtree.
            int local_next = INF;
            const bool found_here = DoIteration(env, *w, bound, heuristic, local_next);

            if (found_here) {
                // Try to become the first thread that reports a solution.
                bool expected = false;
                if (found_solution.compare_exchange_strong(
                        expected, true,
                        std::memory_order_acq_rel,
                        std::memory_order_relaxed)) {
                    // We are the first to mark that a goal was found.
                    // The Work object 'w' now remembers the goal node.
                    // BatchIDA will reconstruct and print the full path later.
                }
                return; // Stop this thread.
            }

            // No goal found in this step: update the global next_bound candidate
            // with the local minimum f-value that exceeded 'bound'.
            if (local_next < INF) {
                int current = global_next_bound.load(std::memory_order_relaxed);
                while (local_next < current &&
                       !global_next_bound.compare_exchange_weak(
                               current,           // overwritten on failure
                               local_next,
                               std::memory_order_acq_rel,
                               std::memory_order_relaxed)) {
                    // Loop while local_next is still smaller than the current global value.
                }
            }

            // Note: if this particular Work becomes fully explored (w->is_done())
            // after this DoIteration call, we will detect it at the top of the loop
            // and try to acquire a new Work for this stack slot.
        }
    };

    // Spawn the CPU workers.
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (std::size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker_fn, static_cast<int>(t));
    }

    // Join all workers.
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    // Report the minimal f > bound that any thread observed (or INF if none).
    next_bound = global_next_bound.load(std::memory_order_relaxed);

    // Return whether at least one thread found a goal in this iteration.
    return found_solution.load(std::memory_order_relaxed);
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
            int& next_bound,
            int num_threads = 0)
{
    return CB_DFS<Env, HeuristicFn<Env>>(env, works, work_num, bound, heuristic, next_bound, num_threads);
}

} // namespace batch_ida
