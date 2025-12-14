#pragma once

#include <vector>
#include <cstdint>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <functional>
#include <unordered_map>

#include "puzzle15_state.h"

// Asynchronous batched heuristic evaluation service for the 15-puzzle.
//
// Each search thread can:
//   * enqueue(states) whenever it generates a new successor.
//   * try_get_h(state, h_out) when it wants h_M(state).
//
// A dedicated worker thread accumulates states into a batch and calls
// a user-provided "batch compute" function on the GPU (e.g. NeuralDelta15).

class NeuralBatchService {
public:
    using State = puzzle15_state;
    using BatchComputeFn =
        std::function<void(const std::vector<State>&, std::vector<int>&)>;

    static NeuralBatchService& instance();

    // Start the worker thread.
    // Must be called once before using enqueue()/try_get_h().
    //
    // 'fn' will typically wrap NeuralDelta15::h_M_batch, for example:
    //
    //   auto fn = [](const std::vector<puzzle15_state>& batch,
    //                std::vector<int>& out) {
    //       out = NeuralDelta15::instance().h_M_batch(batch);
    //   };
    //
    void start(BatchComputeFn fn,
               std::size_t max_batch_size = 512,
               std::chrono::nanoseconds max_wait =
                   std::chrono::milliseconds(1));

    // Stop the worker thread and flush all pending work.
    void shutdown();

    // Clear all cached states and their heuristic values.
    // Intended to be called between IDA* bounds, after an iteration finishes.
    // Safe to call while the worker thread is running.
    void reset_for_new_bound();


    // Non-copyable.
    NeuralBatchService(const NeuralBatchService&) = delete;
    NeuralBatchService& operator=(const NeuralBatchService&) = delete;

    // Schedule a state for batched evaluation (idempotent).
    void enqueue(const State& s);

    // Try to obtain h(s). Returns true if the value is already available.
    // If 'false' is returned, the caller should normally give up the current
    // logical stack and try another Work.
    bool try_get_h(const State& s, int& h_out);

    bool is_running() const noexcept { return running_; }

private:
    NeuralBatchService() = default;
    ~NeuralBatchService();

    void worker_loop(BatchComputeFn fn);

    struct Entry {
        State state;
        bool scheduled = false; // already picked for a batch
        bool ready = false;     // h_value has been computed
        int h_value = 0;
    };

    using Key = std::uint64_t; // puzzle15_state::pack()

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::unordered_map<Key, Entry> entries_;

    std::size_t max_batch_size_ = 8000;
    std::chrono::nanoseconds max_wait_{std::chrono::milliseconds (1)};

    std::thread worker_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_{false};
};


// Small helper to enable/disable the neural batching logic inside DoIteration.
// When disabled, DoIteration falls back to the normal (synchronous) heuristic.

namespace batch_ida {

inline bool& neural_batch_enabled_flag() {
    static bool enabled = false;
    return enabled;
}

inline void set_neural_batch_enabled(const bool enabled) {
    neural_batch_enabled_flag() = enabled;
}

inline bool neural_batch_enabled() {
    return neural_batch_enabled_flag();
}

} // namespace batch_ida
