//
// Created by uriel on 17/11/2025.
//
#pragma once

#include <vector>
#include <cstddef>
#include <atomic>

#include "work.h"

// WorkScheduler is a small thread-safe helper that hands out Work items
// (subtrees) to worker threads without duplication.
//
// It does NOT create or own the Work objects. Instead, it holds a reference
// to a vector<WorkFor<Env>> that was created externally (e.g. by GenerateWork).
//
// Each call to acquire() returns a unique Work index for this scheduler
// instance. Once an index has been handed out, it will not be given again.
//
// Typical usage pattern (per IDA* iteration / per bound):
//
//   std::vector<WorkFor<Env>> works;
//   GenerateWork(..., works, ...);
//
//   WorkScheduler<Env> scheduler(works);
//
//   // in each worker thread:
//   WorkFor<Env>* work = nullptr;
//   std::size_t idx = 0;
//   while (scheduler.acquire(work, idx)) {
//       // process *work using DoIteration / CB-DFS for this bound
//   }
//
template<class Env>
class WorkScheduler {
public:
    using WorkType = WorkFor<Env>;

    explicit WorkScheduler(std::vector<WorkType>& works)
        : works_(works),
          next_index_(0)
    {}

    // Reset the internal index so that Work items can be handed out again.
    // Call this at the beginning of a new IDA* iteration (new bound),
    // after you have reset the state of each Work, if necessary.
    void reset() noexcept {
        next_index_.store(0, std::memory_order_relaxed);
    }

    // Try to acquire the next available Work item.
    //
    // Returns:
    //   true  - if a new Work was assigned. 'out_work' and 'out_index'
    //           are filled with its address and index.
    //   false - if there are no more Work items to assign
    //           (all works have already been handed out).
    //
    // This function is thread-safe: multiple threads may call acquire()
    // concurrently, and each Work index will be returned at most once.
    bool acquire(WorkType*& out_work, std::size_t& out_index) {
        std::size_t i = next_index_.fetch_add(1, std::memory_order_relaxed);
        if (i >= works_.size()) {
            return false; // no more works
        }
        out_index = i;
        out_work  = &works_[i];
        return true;
    }

    // Number of Work items managed by this scheduler.
    std::size_t size() const noexcept {
        return works_.size();
    }

private:
    std::vector<WorkType>& works_;
    std::atomic<std::size_t> next_index_;
};


