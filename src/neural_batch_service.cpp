#include "neural_batch_service.h"

#include <algorithm>
#include <stdexcept>

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
                               std::chrono::milliseconds max_wait) {
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
    entries_.erase(it);
    return true;
}

// Main worker thread loop.
//
// This function:
//   * Waits for new states to be enqueued, or for shutdown.
//   * Collects up to max_batch_size_ states that have not yet been scheduled.
//   * Uses max_wait_ as a timeout to avoid waiting forever for a "perfect" batch.
//   * Calls the provided BatchComputeFn on the batch.
//   * Stores the resulting heuristic values back into the shared map,
//     marking entries as ready and notifying any waiters.
//
// The loop exits when stop_ is set to true.
void NeuralBatchService::worker_loop(BatchComputeFn fn) {
    using clock = std::chrono::steady_clock;

    while (!stop_) {
        std::vector<State> local_batch;
        std::vector<Key> local_keys;

        {
            std::unique_lock<std::mutex> lock(mutex_);

            // Wait until there is at least one pending entry, or we are stopping.
            cv_.wait(lock, [&] {
                return stop_ || !entries_.empty();
            });

            if (stop_) {
                return;
            }

            auto deadline = clock::now() + max_wait_;

            // Collect up to max_batch_size_ unscheduled entries into local_batch.
            // We loop until:
            //   * we have a non-empty batch AND
            //       - reached max_batch_size_, or
            //       - max_wait_ elapsed, OR
            //   * stop_ becomes true.
            while (!stop_) {
                for (auto& kv : entries_) {
                    Entry& e = kv.second;
                    if (!e.scheduled) {
                        e.scheduled = true;
                        local_batch.push_back(e.state);
                        local_keys.push_back(kv.first);

                        if (local_batch.size() >= max_batch_size_) {
                            break;
                        }
                    }
                }

                if (!local_batch.empty() &&
                    (local_batch.size() >= max_batch_size_ ||
                     clock::now() >= deadline)) {
                    // We have a non-empty batch to process.
                    break;
                }

                if (local_batch.empty()) {
                    // Still no unscheduled entries; wait until either:
                    //   * stop_ is set,
                    //   * enough entries accumulate to form a full batch,
                    //   * or the deadline is reached.
                    cv_.wait_until(lock, deadline, [&] {
                        return stop_ ||
                               entries_.size() >= max_batch_size_;
                    });
                } else {
                    // We already have some entries, but not enough and
                    // not past the deadline; break and process them anyway.
                    break;
                }
            }

            if (local_batch.empty()) {
                // Either stop_ is set or we woke spuriously with nothing to do.
                // Go back to the top of the loop.
                continue;
            }
        } // release mutex_ while running the GPU computation

        // Compute heuristics for the whole batch (typically on the GPU).
        std::vector<int> hs(local_batch.size());
        fn(local_batch, hs);

        // Store results back into the shared map, marking entries as ready.
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
                it->second.ready = true;
                it->second.h_value = hs[i];
            }
        }

        // Notify any threads waiting for results (try_get_h or others).
        cv_.notify_all();
    }
}
