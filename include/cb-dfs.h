//
// Created by uriel on 17/11/2025.
// Parallel CB-DFS (CPU only) following Algorithm 3 style.
// Each OS thread runs a local CB-DFS with `work_num` logical stacks,
// and WorkScheduler is used to acquire new subtrees (Work objects) as needed.
//
#pragma once

#include <vector>
#include <limits>
#include <thread>
#include <atomic>

#include "work.h"
#include "work_scheduler.h"
#include "do_iteration.h"  // DoIteration, HeuristicFn

namespace batch_ida {

/**
 * Parallel CB-DFS (Algorithm 3) for a single IDA* iteration.
 *
 * Each CPU thread runs a local instance of CB-DFS, with its own:
 *   - stack[work_num]
 *   - terminated[work_num]
 *   - miss
 *   - counter
 *
 * All threads share:
 *   - a single WorkScheduler over the global 'works' array,
 *   - the 'found_solution' flag,
 *   - and a global accumulator for 'next_bound'.
 *
 * Parameters:
 *   env        - problem environment (e.g., 15-puzzle).
 *   works      - global pool of Work items, created by GenerateWork(…).
 *   work_num   - logical number of stacks per thread (workNum in the paper).
 *   bound      - current IDA* threshold on f = g + h.
 *   heuristic  - heuristic function h(s).
 *   next_bound - [out] minimal f(n) > bound seen in this iteration,
 *                or std::numeric_limits<int>::max() if none.
 *   num_threads_hint - optional number of OS threads to use; if <= 0,
 *                      std::thread::hardware_concurrency() is used.
 *
 * Returns:
 *   true  - if at least one thread found a goal with f <= bound.
 *   false - otherwise.
 */
template<class Env, class Heuristic>
bool CB_DFS(Env& env,
            std::vector<WorkFor<Env>>& works,
            int work_num,
            int bound,
            Heuristic heuristic,
            int& next_bound,
            std::size_t num_threads)
{
    using WorkType = WorkFor<Env>;
    constexpr int INF = std::numeric_limits<int>::max();

    next_bound = INF;

    if (works.empty() || work_num <= 0) {
        return false;
    }

    // Shared scheduler over the global Works vector.
    // All threads pull distinct Work items from this scheduler.
    WorkScheduler<Env> scheduler(works);

    // Shared flags between threads.
    std::atomic<bool> found_solution(false);

    // Global accumulator for next_bound (minimal f > bound).
    std::atomic<int> global_next_bound(INF);

    // Use the number of threads decided by the caller (BatchIDA).
    std::size_t num_threads_effective = num_threads;

    if (num_threads_effective == 0) {
        num_threads_effective = 1;
    }

    if (num_threads_effective > works.size()) {
        num_threads_effective = works.size();
    }


    auto worker_fn = [&](int /*thread_id*/) {
        const std::size_t num_stacks = static_cast<std::size_t>(work_num);

        // Local logical stacks for this thread.
        std::vector<WorkType*> stacks(num_stacks, nullptr);
        std::vector<bool>      terminated(num_stacks, false);

        // miss: number of logical stacks that have entered the terminated state
        // in THIS thread (local Algorithm 3 instance).
        int local_miss = 0;

        // counter: round-robin index over local stacks.
        std::size_t counter = 0;

        // ---- Initial acquire: fill stack[0..work_num-1] once ----
        // This follows the pseudocode "Initiate stack[workNum]".
        for (std::size_t i = 0; i < num_stacks; ++i) {
            WorkType* w = nullptr;
            if (!scheduler.acquire(w)) {
                // No more global Work available for this slot.
                terminated[i] = true;
                ++local_miss;
            } else {
                stacks[i] = w;
            }
        }

        // If this thread did not get any active stack, it has nothing to do.
        if (local_miss >= static_cast<int>(num_stacks)) {
            return;
        }

        // ---- Main CB-DFS loop for this thread ----
        //
        // Stop when:
        //   - another thread found a solution (found_solution == true), or
        //   - all local stacks in this thread have become terminated.
        while (!found_solution.load(std::memory_order_acquire) &&
               local_miss < static_cast<int>(num_stacks)) {

            const std::size_t i = counter % num_stacks;
            ++counter;

            if (terminated[i]) {
                // This stack is permanently dead in this thread.
                continue;
            }

            WorkType*& w = stacks[i];

            // If there is no Work assigned to this stack, or the current Work
            // has been fully explored, try to acquire a new subtree from the
            // global scheduler.
            if (w == nullptr || w->is_done()) {
                WorkType* new_w = nullptr;
                if (!scheduler.acquire(new_w)) {
                    // No more global Work to give to this local stack.
                    terminated[i] = true;
                    ++local_miss;
                    continue;
                }
                w = new_w;
            }

            // At this point, w points to an active subtree assigned to stack i.
            int local_next = INF;
            bool found_here = DoIteration(env, *w, bound, heuristic, local_next);

            if (found_here) {
                // Try to be the first thread that reports a solution.
                bool expected = false;
                if (found_solution.compare_exchange_strong(
                        expected,
                        true,
                        std::memory_order_acq_rel,
                        std::memory_order_relaxed)) {
                    // We are the first to mark that a goal was found.
                    // The Work object 'w' has already recorded the goal node
                    // (e.g. via mark_goal_current()).
                }
                return; // stop this worker
            }

            // No solution in this step: update the global next_bound candidate
            // with the local minimal f-value that exceeded 'bound'.
            if (local_next < INF) {
                int current = global_next_bound.load(std::memory_order_relaxed);
                while (local_next < current &&
                       !global_next_bound.compare_exchange_weak(
                           current,
                           local_next,
                           std::memory_order_acq_rel,
                           std::memory_order_relaxed)) {
                    // On failure, 'current' has been updated with the latest value.
                    // We retry while local_next is still smaller than 'current'.
                }
            }
        }
    };

    // Spawn OS threads.
    std::vector<std::thread> threads;
    threads.reserve(num_threads_effective);
    for (std::size_t t = 0; t < num_threads_effective; ++t) {
        threads.emplace_back(worker_fn, static_cast<int>(t));
    }


    // Join all threads.
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }

    // Final next_bound is the global minimal f > bound over all pruned nodes.
    next_bound = global_next_bound.load(std::memory_order_relaxed);
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
            std::size_t num_threads)
{
    return CB_DFS<Env, HeuristicFn<Env>>(env,
                                         works,
                                         work_num,
                                         bound,
                                         heuristic,
                                         next_bound,
                                         num_threads);
}

} // namespace batch_ida
