#include "neural_batch_service.h"

#include <algorithm>
#include <stdexcept>
#include <iostream>
#include "nvtx_helpers.h"

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
NeuralBatchService &NeuralBatchService::instance() {
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

NeuralBatchService::HRequestStatus
NeuralBatchService::request_h(const State &s, int &h_out) {
    if (!running_) {
        return HRequestStatus::NotRunning;
    }

    const Key key = s.pack();
    bool notify_worker = false;
    HRequestStatus status = HRequestStatus::Pending;

    {
        std::lock_guard<std::mutex> lock(mutex_);

        // One lookup: insert-if-missing
        auto [it, inserted] = entries_.try_emplace(key);

        if (inserted) {
            it->second.state = s;

            // Since you already did (1) pending queue:
            pending_.push_back(key);

            notify_worker = true;
            status = HRequestStatus::Pending;
        } else {
            if (it->second.ready) {
                h_out = it->second.h_value;
                status = HRequestStatus::Ready;
            } else {
                status = HRequestStatus::Pending;
            }
        }
    }

    if (notify_worker) {
        cv_.notify_one();
    }

    return status;
}


void NeuralBatchService::reset_for_new_bound() {
    // If the service is not running, there is nothing to clear.
    if (!running_) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    pending_.clear();
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
void NeuralBatchService::enqueue(const State &s) {
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
        pending_.push_back(key);
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
bool NeuralBatchService::try_get_h(const State &s, int &h_out) {
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
    entries_.erase(it);
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
    std::vector<State> local_batch;
    std::vector<Key> local_keys;
    std::vector<int> hs;
    local_batch.reserve(max_batch_size_);
    local_keys.reserve(max_batch_size_);
    hs.reserve(max_batch_size_);
    NVTX_RANGE("NBS: collect batch");

    while (!stop_) {
        local_batch.clear();
        local_keys.clear();
        hs.clear();
        clock::time_point start_collect;

        {
            std::unique_lock<std::mutex> lock(mutex_);

            // Wait until we have something pending (NOT just entries_ non-empty).
            cv_.wait(lock, [&] {
                return stop_ || !pending_.empty();
            });

            if (stop_) {
                return;
            }

            local_batch.reserve(max_batch_size_);
            local_keys.reserve(max_batch_size_);

            // 1) Pop first valid item (skip stale keys if reset/erase happened).
            while (!stop_ && local_batch.empty()) {
                while (!pending_.empty() && local_batch.empty()) {
                    const Key key = pending_.front();
                    pending_.pop_front();

                    auto it = entries_.find(key);
                    if (it == entries_.end()) {
                        continue; // stale key (e.g., reset_for_new_bound cleared map)
                    }

                    Entry &e = it->second;
                    if (e.ready || e.scheduled) {
                        continue; // should be rare, but keep it safe
                    }

                    e.scheduled = true;
                    local_batch.push_back(e.state);
                    local_keys.push_back(key);
                }

                if (local_batch.empty()) {
                    // pending_ had only stale keys; wait for new ones.
                    cv_.wait(lock, [&] { return stop_ || !pending_.empty(); });
                }
            }

            if (stop_) {
                return;
            }

            // Timer starts from first collected item.
            start_collect = clock::now();
            auto deadline = start_collect + max_wait_;

            // 2) Keep collecting until full or timeout.
            while (!stop_ && local_batch.size() < max_batch_size_) {
                while (!pending_.empty() && local_batch.size() < max_batch_size_) {
                    const Key key = pending_.front();
                    pending_.pop_front();

                    auto it = entries_.find(key);
                    if (it == entries_.end()) {
                        continue;
                    }

                    Entry &e = it->second;
                    if (e.ready || e.scheduled) {
                        continue;
                    }

                    e.scheduled = true;
                    local_batch.push_back(e.state);
                    local_keys.push_back(key);
                }

                if (local_batch.size() >= max_batch_size_) {
                    break;
                }

                if (clock::now() >= deadline) {
                    break;
                }

                // Wait for more pending keys (or until deadline).
                cv_.wait_until(lock, deadline, [&] {
                    return stop_ || !pending_.empty();
                });
            }

            if (stop_) {
                return;
            }
        } // unlock mutex_ during GPU call

        // Debug timing (same idea as you had).
        auto end_collect = clock::now();
        auto waited_ms =
                std::chrono::duration<double, std::milli>(end_collect - start_collect).count();

        /*std::cout << "[NeuralBatchService] batch size: " << local_batch.size()
                  << ", waited: " << waited_ms << " ms before GPU call\n";*/

        // Compute on GPU/CPU
        hs.resize(local_batch.size());
        NVTX_MARK("NBS: batch ready -> GPU call");

        try {
            NVTX_RANGE("NBS: batch_fn (GPU)");

            fn(local_batch, hs);
        } catch (const std::exception &ex) {
            std::cerr << "[NeuralBatchService] batch_fn threw exception: "
                    << ex.what() << std::endl;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stop_ = true;
            }
            cv_.notify_all();
            return;
        }

        // Store results
        {
            NVTX_RANGE("NBS: store results");

            std::lock_guard<std::mutex> lock(mutex_);
            const std::size_t n = (std::min)(local_batch.size(), hs.size());
            for (std::size_t i = 0; i < n; ++i) {
                const Key key = local_keys[i];
                auto it = entries_.find(key);
                if (it == entries_.end()) {
                    continue;
                }
                it->second.ready = true;
                it->second.h_value = hs[i];
            }
        }

        cv_.notify_all();
    }
}
