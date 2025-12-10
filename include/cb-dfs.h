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
 * Parameters:
 *   env        - problem environment (e.g., 15-puzzle).
 *   works      - global pool of Work items, created by GenerateWork(…).
 *   work_num   - logical number of stacks per thread (workNum in the paper).
 *   bound      - current IDA* threshold on f = g + h.
 *   heuristic  - heuristic function h(s).
 *   next_bound - [out] minimal f(n) > bound seen in this iteration,
 *                or std::numeric_limits<int>::max() if none.
 *   num_threads_hint - if > 0, use exactly this many OS threads;
 *                      otherwise, use std::thread::hardware_concurrency().
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
            int num_threads)
{
    using WorkType = WorkFor<Env>;
    constexpr int INF = std::numeric_limits<int>::max();

    next_bound = INF;

    if (works.empty() || work_num <= 0) {
        return false;
    }

    // Shared scheduler over the global Works vector.
    WorkScheduler<Env> scheduler(works);

    // Shared flags between threads.
    std::atomic<bool> found_solution(false);

    // Global accumulator for next_bound (minimal f > bound).
    std::atomic<int> global_next_bound(INF);


    auto worker_fn = [&](int /*thread_id*/) {
        const std::size_t num_stacks = static_cast<std::size_t>(work_num);

        // Local logical stacks for this thread.
        std::vector<WorkType*> stacks(num_stacks, nullptr);
        std::vector<bool> terminated(num_stacks, false);

        // miss: number of logical stacks that have entered the terminated
        // state in THIS thread (local Algorithm 3 instance).
        int local_miss = 0;

        // counter: round-robin index over local stacks.
        std::size_t counter = 0;

        // ---- Initial acquire: fill stack[0..work_num-1] once ----
        for (std::size_t i = 0; i < num_stacks; ++i) {
            WorkType* w = nullptr;
            if (!scheduler.acquire(w)) {
                terminated[i] = true;
                ++local_miss;
            } else {
                stacks[i] = w;
            }
        }

        if (local_miss >= static_cast<int>(num_stacks)) {
            return;
        }

        // ---- Main CB-DFS loop for this thread ----
        while (!found_solution.load(std::memory_order_acquire) &&
               local_miss < static_cast<int>(num_stacks)) {

            const std::size_t i = counter % num_stacks;
            ++counter;

            if (terminated[i]) {
                continue; // this logical stack is permanently dead
            }

            WorkType*& w = stacks[i];

            // If there is no Work assigned to this stack, or the current Work
            // has been fully explored, try to acquire a new subtree.
            if (w == nullptr || w->is_done()) {
                WorkType* new_w = nullptr;
                if (!scheduler.acquire(new_w)) {
                    terminated[i] = true;
                    ++local_miss;
                    continue;
                }
                w = new_w;
            }

            int  local_next = INF;
            bool found_here = DoIteration(env, *w, bound, heuristic, local_next);

            if (found_here) {
                bool expected = false;
                if (found_solution.compare_exchange_strong(
                        expected,
                        true,
                        std::memory_order_acq_rel,
                        std::memory_order_relaxed)) {
                    // first thread to report a solution
                }
                return;
            }

            if (local_next < INF) {
                int current = global_next_bound.load(std::memory_order_relaxed);
                while (local_next < current &&
                       !global_next_bound.compare_exchange_weak(
                           current,
                           local_next,
                           std::memory_order_acq_rel,
                           std::memory_order_relaxed)) {
                    // retry while local_next is still smaller than 'current'
                }
            }
        }
    };

    // Spawn OS threads.
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (std::size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back(worker_fn, static_cast<int>(t));
    }

    // Join all threads.
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }

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
            int num_threads_hint)
{
    return CB_DFS<Env, HeuristicFn<Env>>(env,
                                         works,
                                         work_num,
                                         bound,
                                         heuristic,
                                         next_bound,
                                         num_threads_hint);
}

} // namespace batch_ida
