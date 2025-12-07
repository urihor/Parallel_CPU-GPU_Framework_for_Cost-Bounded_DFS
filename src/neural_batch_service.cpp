#include "neural_batch_service.h"

#include <algorithm>
#include <stdexcept>

NeuralBatchService& NeuralBatchService::instance() {
    static NeuralBatchService inst;
    return inst;
}

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

    worker_thread_ = std::thread(&NeuralBatchService::worker_loop, this, fn);
}

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

NeuralBatchService::~NeuralBatchService() {
    shutdown();
}

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

    cv_.notify_one();
}

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

void NeuralBatchService::worker_loop(BatchComputeFn fn) {
    using clock = std::chrono::steady_clock;

    while (!stop_) {
        std::vector<State> local_batch;
        std::vector<Key> local_keys;

        {
            std::unique_lock<std::mutex> lock(mutex_);

            // Wait until there is at least one pending entry or we are stopping.
            cv_.wait(lock, [&] {
                return stop_ || !entries_.empty();
            });

            if (stop_) {
                return;
            }

            auto deadline = clock::now() + max_wait_;

            // Collect up to max_batch_size_ unscheduled entries.
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
                    break; // we have a non-empty batch to process
                }

                if (local_batch.empty()) {
                    // Still no unscheduled entries; wait a bit more.
                    cv_.wait_until(lock, deadline, [&] {
                        return stop_ ||
                               entries_.size() >= max_batch_size_;
                    });
                } else {
                    break;
                }
            }

            if (local_batch.empty()) {
                // Either stop_ is set or we spurious-woke. Loop again.
                continue;
            }
        } // release mutex_ while running the GPU computation

        // Compute heuristics for the whole batch (GPU).
        std::vector<int> hs(local_batch.size());
        fn(local_batch, hs);

        // Store results back into the shared map.
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const std::size_t n = std::min(local_batch.size(), hs.size());
            for (std::size_t i = 0; i < n; ++i) {
                const Key key = local_keys[i];
                auto it = entries_.find(key);
                if (it == entries_.end()) {
                    continue; // entry might have been removed meanwhile
                }
                it->second.ready = true;
                it->second.h_value = hs[i];
            }
        }

        cv_.notify_all();
    }
}
