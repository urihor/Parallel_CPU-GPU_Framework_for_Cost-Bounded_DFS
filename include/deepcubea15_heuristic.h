//
// Created by Owner on 31/12/2025.
//
#pragma once

#include "neural_batch_service.h"
#include "puzzle15_state.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

class DeepCubeA15Heuristic {
public:
    struct Options {
        std::string ts_path; // path to puzzle15_torchscript.pt
        torch::Device device = torch::kCPU; // CPU / CUDA

        // Convert model output y (float) to int heuristic:
        // h = max(0, floor(scale * (y - base)))
        float scale = 1.0f;

        // If base_override is finite => use it; otherwise compute from goal state.
        float base_override = std::numeric_limits<float>::quiet_NaN();

        // Blocking wait strategy when using NeuralBatchService
        int spin_yields = 200; // yield() this many times first
        std::chrono::microseconds sleep_after =
                std::chrono::microseconds(50); // then sleep in a loop
    };

    explicit DeepCubeA15Heuristic(const Options &opt);

    // Start/stop batched service. You should call start_service() once at program init.
    void start_service(std::size_t max_batch_size,
                       std::chrono::nanoseconds max_wait);

private:
    // Called by NeuralBatchService worker thread
    void compute_batch_fn(const std::vector<NeuralBatchService::State> &batch,
                          std::vector<int> &hs);

    // Direct synchronous inference (used if service isn't running).
    float forward_score_single(const puzzle15_state &s);

    void forward_score_batch(const std::vector<NeuralBatchService::State> &batch,
                             std::vector<float> &out_scores);

    int score_to_h(float y) const;

private:
    torch::jit::script::Module module_;
    torch::Device device_;

    float base_;
    float scale_;
    int spin_yields_;
    std::chrono::microseconds sleep_after_;

    // Only needed if you ever call forward from multiple threads (e.g., service not running).
    std::mutex forward_mutex_;
};
