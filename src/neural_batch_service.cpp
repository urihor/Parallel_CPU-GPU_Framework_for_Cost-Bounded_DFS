#include "neural_batch_service.h"

#include <algorithm>
#include <stdexcept>
#include <iostream>

//
// NeuralBatchService
// ------------------
// A singleton service that collects states from multiple threads and
// evaluates their neural heuristic in batches (e.g. on the GPU).
//
// Typical usage pattern:
//
//   1. Call NeuralBatchService::instance().start(fn, max_batch_size, max_wait)
//      at program initialization, with a BatchComputeFn that runs the model
//      on a batch of states.
//
//   2. In the search code:
//        - call enqueue(state) to request a batched evaluation for a state
//        - later call try_get_h(state, h_out) to see if the result is ready
//
//   3. On shutdown, call NeuralBatchService::instance().shutdown().
//
// If the service is not running (start() was never called, or shutdown()
// has already been invoked), enqueue() and try_get_h() are effectively
// no-ops / always-fail, and the caller is expected to fall back to the
// synchronous heuristic path.
//

// Get the global singleton instance of the service.
NeuralBatchService& NeuralBatchService::instance() {
    static NeuralBatchService inst;
    return inst;
}

// Start the batching service, spawning the worker thread.
//
// Parameters:
//   fn             - batch evaluation function: takes a vector of States and
//                    fills a vector<int> with the corresponding heuristic values.
//   max_batch_size - maximum number of states to process in a single batch.
//   max_wait       - maximum time the worker waits to accumulate more states
//                    before it processes a smaller batch.
//
// If called more than once while already running, subsequent calls are ignored.
// Throws std::invalid_argument if fn is empty.
void NeuralBatchService::start(BatchComputeFn fn,
                               std::size_t max_batch_size,
                               std::chrono::nanoseconds max_wait) {
    if (running_) {
        // Already started; ignore subsequent calls.
        return;
    }
    if (!fn) {
        throw std::invalid_argument(
            "NeuralBatchService::start() called with empty BatchComputeFn");
    }

    max_batch_size_ = max_batch_size;
    max_wait_ = max_wait;
    stop_ = false;
    running_ = true;

    // Launch the worker thread that will perform batched evaluations.
    worker_thread_ = std::thread(&NeuralBatchService::worker_loop, this, fn);
}

// Stop the batching service and join the worker thread.
// Safe to call multiple times; calls after the first are no-ops.
void NeuralBatchService::shutdown() {
    if (!running_) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    running_ = false;
}

void NeuralBatchService::reset_for_new_bound() {
    // If the service is not running, there is nothing to clear.
    if (!running_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
}


// Ensure the worker thread is stopped when the service is destroyed.
NeuralBatchService::~NeuralBatchService() {
    shutdown();
}

// Enqueue a state for batched neural evaluation.
//
// The state is inserted into an internal map keyed by its packed representation.
// If the state is already present (either waiting to be processed or already
// computed), the call is a no-op.
//
// If the service is not running, this function does nothing. The caller should
// then fall back to the synchronous heuristic.
void NeuralBatchService::enqueue(const State& s) {
    if (!running_) {
        // If batching is not running, do nothing. The caller will fall back
        // to the synchronous heuristic path.
        return;
    }

    const Key key = s.pack();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = entries_.find(key);
        if (it != entries_.end()) {
            // Already known (either waiting or ready) -> no-op.
            return;
        }

        Entry e;
        e.state = s;
        entries_.emplace(key, std::move(e));
    }

    // Wake the worker so it can consider forming a new batch.
    cv_.notify_one();
}

// Try to retrieve a heuristic value for a given state if it is ready.
//
// Returns:
//   true  - if the heuristic value for this state was available; h_out is set
//           and the internal entry is removed.
//   false - if the service is not running, or the state was never enqueued,
//           or its value is not ready yet.
//
// This is a non-blocking call; it never waits for the worker to finish.
bool NeuralBatchService::try_get_h(const State& s, int& h_out) {
    if (!running_) {
        return false;
    }

    const Key key = s.pack();

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(key);
    if (it == entries_.end() || !it->second.ready) {
        return false;
    }

    h_out = it->second.h_value;
    // Once the value is consumed we can forget the entry to keep memory small.
    //entries_.erase(it);
    return true;
}

/*
 * Background worker thread loop.
 *
 * This thread continuously:
 *  1. Waits until there is at least one pending state in `entries_`.
 *  2. Starts collecting a mini-batch of unscheduled states into `local_batch`,
 *     up to `max_batch_size_`.
 *  3. Stops collecting when:
 *       - the batch is full, OR
 *       - `max_wait_` time has elapsed since the first state was collected.
 *  4. Releases the mutex and calls the user-provided `BatchComputeFn` (typically
 *     a GPU-based heuristic) on the batch.
 *  5. Stores the results back into `entries_` and notifies any waiting threads.
 *
 * This ensures that:
 *  - We never exceed `max_batch_size_` in a single GPU call.
 *  - We do not wait longer than `max_wait_` for the batch to fill up.
 *  - If only a few states arrive, we still flush a partial batch after `max_wait_`.
 */
void NeuralBatchService::worker_loop(BatchComputeFn fn) {
    using clock = std::chrono::steady_clock;

    while (!stop_) {
        std::vector<State> local_batch;
        std::vector<Key>   local_keys;
        clock::time_point  start_collect;

        {
            std::unique_lock<std::mutex> lock(mutex_);

            // 1) Wait until there is at least one pending entry or we are stopping.
            cv_.wait(lock, [&] {
                return stop_ || !entries_.empty();
            });

            if (stop_) {
                return;
            }

            // We are about to start collecting a batch.
            start_collect = clock::now();
            auto deadline = start_collect + max_wait_;

            // 2) Collect up to max_batch_size_ unscheduled entries.
            while (!stop_) {
                // Pull unscheduled entries into local_batch.
                for (auto &kv : entries_) {
                    Entry &e = kv.second;
                    if (!e.scheduled) {
                        e.scheduled = true;
                        local_batch.push_back(e.state);
                        local_keys.push_back(kv.first);

                        if (local_batch.size() >= max_batch_size_) {
                            break; // batch is full
                        }
                    }
                }

                if (!local_batch.empty()) {
                    auto now = clock::now();

                    // Case A: batch is full -> stop collecting and process it.
                    if (local_batch.size() >= max_batch_size_) {
                        break;
                    }

                    // Case B: timeout elapsed -> process partial batch.
                    if (now >= deadline) {
                        break;
                    }

                    // Case C: partial batch and still time left.
                    // Wait for more entries or until deadline.
                    std::size_t known_entries = entries_.size();
                    cv_.wait_until(lock, deadline, [&] {
                        return stop_ || entries_.size() > known_entries;
                    });

                    // Re-check in the next iteration (either new entries arrived
                    // or the deadline is close/has passed).
                    continue;
                }

                // local_batch is still empty here.
                // Wait until new entries arrive or until the deadline.
                std::size_t known_entries = entries_.size();
                cv_.wait_until(lock, deadline, [&] {
                    return stop_ || entries_.size() > known_entries;
                });

                if (stop_) {
                    return;
                }

                // If the deadline has passed and we still have no batch, give
                // the outer loop a chance to re-check (maybe stop_ changed).
                if (clock::now() >= deadline && local_batch.empty()) {
                    break;
                }
            }

            if (stop_) {
                return;
            }

            if (local_batch.empty()) {
                // Nothing collected before timeout / stop; go back and wait again.
                continue;
            }
        } // mutex_ is released here while running the GPU computation.

        // 3) Measure how long we waited before sending the batch to the GPU.
        auto end_collect = clock::now();
        auto waited_ms = std::chrono::duration<double, std::milli>(
                             end_collect - start_collect).count();

        std::cout << "[NeuralBatchService] batch size: " << local_batch.size()
                  << ", waited: " << waited_ms << " ms before GPU call\n";

        // 4) Compute heuristics for the whole batch (typically on the GPU).
        std::vector<int> hs(local_batch.size());
        try {
            fn(local_batch, hs);
        } catch (const std::exception &ex) {
            std::cerr << "[NeuralBatchService] batch_fn threw exception: "
                      << ex.what() << std::endl;
            // Optionally stop the service so callers can fall back to a safe path.
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stop_ = true;
            }
            cv_.notify_all();
            return;
        }

        // 5) Store the results back into the shared map and mark entries as ready.
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const std::size_t n = std::min(local_batch.size(), hs.size());
            for (std::size_t i = 0; i < n; ++i) {
                const Key key = local_keys[i];
                auto it = entries_.find(key);
                if (it == entries_.end()) {
                    // Entry might have been removed meanwhile (e.g., consumer gave up).
                    continue;
                }
                it->second.ready   = true;
                it->second.h_value = hs[i];
            }
        }

        // Notify any threads waiting for results (try_get_h or others).
        cv_.notify_all();
    }
}

