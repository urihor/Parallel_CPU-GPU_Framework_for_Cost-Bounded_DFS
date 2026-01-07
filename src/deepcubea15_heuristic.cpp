//
// Created by Owner on 31/12/2025.
//
#include "../include/deepcubea15_heuristic.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

static torch::Tensor make_input_u8_cpu(const std::vector<NeuralBatchService::State>& batch) {
    const int64_t B = static_cast<int64_t>(batch.size());
    auto x = torch::empty({B, 16}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    auto* dst = x.data_ptr<uint8_t>();

    for (int64_t i = 0; i < B; ++i) {
        // puzzle15_state::tiles is std::array<uint8_t,16> in row-major, exactly what DeepCubeA expects
        std::memcpy(dst + i * 16, batch[static_cast<size_t>(i)].tiles.data(), 16);
    }
    return x;
}

DeepCubeA15Heuristic::DeepCubeA15Heuristic(const Options& opt)
    : device_(opt.device),
      base_(0.0f),
      scale_(opt.scale),
      spin_yields_(opt.spin_yields),
      sleep_after_(opt.sleep_after)
{
    if (opt.ts_path.empty()) {
        throw std::invalid_argument("DeepCubeA15Heuristic: ts_path is empty");
    }

    module_ = torch::jit::load(opt.ts_path, device_);
    module_.eval();

    // Compute base from goal (unless overridden)
    if (std::isfinite(opt.base_override)) {
        base_ = opt.base_override;
    } else {
        // Forward on goal state once
        const puzzle15_state goal = puzzle15_state::Goal();
        base_ = forward_score_single(goal);
    }
}

void DeepCubeA15Heuristic::start_service(std::size_t max_batch_size,
                                        std::chrono::nanoseconds max_wait)
{
    auto fn = [this](const std::vector<NeuralBatchService::State>& batch,
                     std::vector<int>& hs) {
        this->compute_batch_fn(batch, hs);
    };

    NeuralBatchService::instance().start(fn, max_batch_size, max_wait);
}


int DeepCubeA15Heuristic::score_to_h(float y) const {
    // h = max(0, floor(scale * (y - base)))
    float z = (y - base_) * scale_;
    if (z <= 0.0f) return 0;
    return static_cast<int>(std::floor(z + 1e-6f));
}

float DeepCubeA15Heuristic::forward_score_single(const puzzle15_state& s) {
    std::lock_guard<std::mutex> lock(forward_mutex_);
    at::InferenceMode guard(true);

    // Build [1,16] uint8 input
    std::vector<NeuralBatchService::State> tmp;
    tmp.push_back(s);

    auto x_cpu = make_input_u8_cpu(tmp);
    auto x = x_cpu.to(device_);

    auto y = module_.forward({x}).toTensor();   // [1,1]
    y = y.to(torch::kCPU).contiguous().view({1});

    return y.data_ptr<float>()[0];
}

void DeepCubeA15Heuristic::forward_score_batch(
    const std::vector<NeuralBatchService::State>& batch,
    std::vector<float>& out_scores)
{
    at::InferenceMode guard(true);

    const int64_t B = static_cast<int64_t>(batch.size());
    out_scores.resize(static_cast<size_t>(B));

    auto x_cpu = make_input_u8_cpu(batch);
    auto x = x_cpu.to(device_);

    auto y = module_.forward({x}).toTensor();   // [B,1]
    y = y.to(torch::kCPU).contiguous().view({B});

    const float* p = y.data_ptr<float>();
    for (int64_t i = 0; i < B; ++i) {
        out_scores[static_cast<size_t>(i)] = p[i];
    }
}

void DeepCubeA15Heuristic::compute_batch_fn(
    const std::vector<NeuralBatchService::State>& batch,
    std::vector<int>& hs)
{
    // This runs in the NeuralBatchService worker thread
    std::vector<float> scores;
    forward_score_batch(batch, scores);

    hs.resize(scores.size());
    for (size_t i = 0; i < scores.size(); ++i) {
        hs[i] = score_to_h(scores[i]);
        if (hs[i] > 4)
            hs[i] -= 4;
    }
}
