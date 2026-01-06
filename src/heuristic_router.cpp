//
// Created by Owner on 05/01/2026.
//
// Implementation of HeuristicRouter:
//  - Wires DeepCubeA and NeuralDeltaQuantile backends to the shared
//    NeuralBatchService.
//  - Provides both batched (async) and synchronous heuristic evaluation
//    for the 15-puzzle.
//
#include "heuristic_router.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "nvtx_helpers.h"
// Included so we can enable the neural batch path in IDA*.
// (We call batch_ida::set_neural_batch_enabled(true) in start()).
#include "do_iteration.h"

namespace {
    /// Build a CPU tensor of shape [B,16] with uint8 tiles from a batch of states.
    /// Row i contains the 16 tile values of batch[i] (including the blank as 0).
    static torch::Tensor make_u8_input_cpu(const std::vector<puzzle15_state> &batch) {
        const int64_t B = static_cast<int64_t>(batch.size());
        auto x = torch::empty({B, 16}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
        auto *dst = x.data_ptr<uint8_t>();
        for (int64_t i = 0; i < B; ++i) {
            std::memcpy(dst + i * 16, batch[static_cast<size_t>(i)].tiles.data(), 16);
        }
        return x;
    }
} // namespace

HeuristicRouter &HeuristicRouter::instance() {
    static HeuristicRouter inst;
    return inst;
}

const char *HeuristicRouter::backend_name() const noexcept {
    switch (backend_) {
        case Backend::DeepCubeA: return "DeepCubeA";
        case Backend::NeuralDeltaQuantile: return "NeuralDeltaQuantile";
        default: return "None";
    }
}

void HeuristicRouter::use_deepcubea(const DeepCubeAOptions &opt) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (started_) throw std::runtime_error("HeuristicRouter: cannot switch backend while started");

    if (opt.ts_path.empty()) throw std::invalid_argument("DeepCubeAOptions.ts_path is empty");
    backend_ = Backend::DeepCubeA;

    dc_device_ = opt.device;
    dc_norm_goal_ = opt.normalize_goal_to_zero;
    dc_add_manhattan_ = opt.add_manhattan;
    dc_rounding_ = opt.rounding;

    // Load TorchScript module and move it to the requested device.
    dc_module_ = torch::jit::load(opt.ts_path, dc_device_);
    dc_module_.eval();
    dc_loaded_ = true;

    // Optionally precompute the network output on the goal state, so that
    // we can normalize scores to make h(goal) ≈ 0.
    dc_base_ = 0.0f;
    if (dc_norm_goal_) {
        puzzle15_state goal = puzzle15_state::Goal();
        dc_base_ = deepcubea_forward_single(goal);
    }
}

void HeuristicRouter::use_neural_delta_quantile(const neural15::NeuralDelta15QuantileOptions &opt) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (started_) throw std::runtime_error("HeuristicRouter: cannot switch backend while started");
    backend_ = Backend::NeuralDeltaQuantile;

    // Construct Li-style quantile ensemble heuristic over deltas.
    ndq_ = std::make_shared<neural15::NeuralDelta15Quantile>(opt);
}

void HeuristicRouter::start(std::size_t max_batch_size,
                            std::chrono::nanoseconds max_wait) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (started_) return;
    if (backend_ == Backend::None) throw std::runtime_error("HeuristicRouter: backend not selected");

    // Register our batch callback with the global NeuralBatchService.
    auto fn = [this](const std::vector<State> &batch, std::vector<int> &hs) {
        this->compute_batch_fn(batch, hs);
    };

    NeuralBatchService::instance().start(fn, max_batch_size, max_wait);

    // Required so that DoIteration switches to the asynchronous neural
    // path instead of calling the heuristic synchronously.
    batch_ida::set_neural_batch_enabled(true);

    started_ = true;
}

void HeuristicRouter::shutdown() {
    std::lock_guard<std::mutex> lk(mtx_);
    NeuralBatchService::instance().shutdown();
    started_ = false;
}

void HeuristicRouter::reset_for_new_bound() {
    // Forward the "new IDA* bound" event to the batch service.
    // Currently this may be a no-op, but it allows future backends
    // to reset any bound-dependent caches.
    NeuralBatchService::instance().reset_for_new_bound();
}

int HeuristicRouter::manhattan_4x4(const State &s) {
    // Precompute goal positions for tiles 0..15 (0 = blank).
    static int goal_pos[16];
    static bool init = false;
    if (!init) {
        goal_pos[0] = 15;
        for (int t = 1; t <= 15; ++t) goal_pos[t] = t - 1;
        init = true;
    }

    int dist = 0;
    for (int idx = 0; idx < 16; ++idx) {
        int v = static_cast<int>(s.tiles[static_cast<size_t>(idx)]);
        if (v == 0) continue; // ignore blank
        int g = goal_pos[v];

        int r = idx / 4, c = idx % 4;
        int gr = g / 4, gc = g % 4;

        dist += std::abs(r - gr) + std::abs(c - gc);
    }
    return dist;
}

int HeuristicRouter::deepcubea_score_to_h(float y) const {
    // Apply optional normalization: subtract the goal output so that
    // the goal has score ≈ 0. Then map the float score to an integer
    // using the configured rounding mode. Negative/NaN/Inf are treated
    // as 0 to stay safe and admissible.
    float z = y;
    if (dc_norm_goal_) z -= dc_base_;
    if (!std::isfinite(z) || z <= 0.0f) return 0;

    switch (dc_rounding_) {
        case DeepCubeAOptions::Rounding::Floor:
            return static_cast<int>(std::floor(z + 1e-6f));
        case DeepCubeAOptions::Rounding::Ceil:
            return static_cast<int>(std::ceil(z - 1e-6f));
        case DeepCubeAOptions::Rounding::Round:
        default:
            return static_cast<int>(std::lround(z));
    }
}

float HeuristicRouter::deepcubea_forward_single(const State &s) {
    // Single-state forward pass through the DeepCubeA model.
    // Used for h_sync and for measuring the goal baseline.
    at::InferenceMode guard(true);

    std::vector<State> tmp{s};
    auto x_cpu = make_u8_input_cpu(tmp);
    auto x = x_cpu.to(dc_device_);

    auto y = dc_module_.forward({x}).toTensor(); // [1,1]
    y = y.to(torch::kCPU).to(torch::kFloat32).contiguous().view({1});
    return y.data_ptr<float>()[0];
}

void HeuristicRouter::deepcubea_forward_batch(const std::vector<State> &batch,
                                              std::vector<float> &out_scores) const {
    // Batched forward pass for DeepCubeA: used by NeuralBatchService's worker
    // thread when the DeepCubeA backend is selected.
    at::InferenceMode guard(true);

    const int64_t B = static_cast<int64_t>(batch.size());
    out_scores.resize(static_cast<size_t>(B));

    auto x_cpu = make_u8_input_cpu(batch);
    auto x = x_cpu.to(dc_device_);

    auto y = dc_module_.forward({x}).toTensor(); // [B,1]
    y = y.to(torch::kCPU).to(torch::kFloat32).contiguous().view({B});
    const float *p = y.data_ptr<float>();
    for (int64_t i = 0; i < B; ++i) out_scores[static_cast<size_t>(i)] = p[i];
}

void HeuristicRouter::compute_batch_fn(const std::vector<State> &batch, std::vector<int> &hs) {
    // This function is executed in the NeuralBatchService worker thread.
    // It dispatches to the currently selected backend and fills 'hs'
    // with one heuristic value per input state.

    if (backend_ == Backend::NeuralDeltaQuantile) {
        if (!ndq_) throw std::runtime_error("NeuralDeltaQuantile backend not initialized");
        ndq_->compute_batch(batch, hs);
        return;
    }

    if (backend_ == Backend::DeepCubeA) {
        if (!dc_loaded_) throw std::runtime_error("DeepCubeA backend not loaded");
        std::vector<float> scores;
        deepcubea_forward_batch(batch, scores);

        hs.resize(scores.size());
        for (size_t i = 0; i < scores.size(); ++i) {
            int h = deepcubea_score_to_h(scores[i]);
            if (dc_add_manhattan_) h += manhattan_4x4(batch[i]);
            hs[i] = h;
        }
        return;
    }

    throw std::runtime_error("HeuristicRouter: compute_batch_fn called with backend None");
}

int HeuristicRouter::h_sync(const State &s) {
    // Synchronous heuristic evaluation (no batching / background thread).
    // Useful for initial IDA* bound or for debugging the NN heuristic.
    std::lock_guard<std::mutex> lk(mtx_);

    if (backend_ == Backend::NeuralDeltaQuantile) {
        if (!ndq_) throw std::runtime_error("NeuralDeltaQuantile backend not initialized");
        // If there is no dedicated compute_one(), we just run a batch of size 1.
        std::vector<State> b{s};
        std::vector<int> hs;
        ndq_->compute_batch(b, hs);
        return hs.empty() ? 0 : hs[0];
    }

    if (backend_ == Backend::DeepCubeA) {
        float y = deepcubea_forward_single(s);
        int h = deepcubea_score_to_h(y);
        if (dc_add_manhattan_) h += manhattan_4x4(s);
        return h;
    }

    // No backend selected: caller should fall back to another heuristic.
    return 0;
}
