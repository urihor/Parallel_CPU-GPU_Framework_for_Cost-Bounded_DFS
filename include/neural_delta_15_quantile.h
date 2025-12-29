//
// Created by Owner on 22/12/2025.
//
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "puzzle15_state.h"

namespace neural15 {

    struct NeuralDelta15QuantileOptions {
        // state_dict files (*.pt)
        // 1..7 usually single model; we still allow ensemble.
        std::vector<std::string> weights_1_7;
        std::vector<std::string> weights_8_15; // can be single or ensemble (ens0..ens3)

        // quantile parameter (default q=0.3)
        double quantile_q = 0.2;

        // device
        torch::Device device = torch::kCPU;

        // optional: FP16 on CUDA
        bool use_half_on_cuda = false;

        // include Manhattan in final h?
        bool add_manhattan = true;
    };

    class NeuralDelta15Quantile {
    public:
        explicit NeuralDelta15Quantile(const NeuralDelta15QuantileOptions& opt);

        // BatchComputeFn-compatible:
        // fills hs[i] for each state in batch.
        void compute_batch(const std::vector<puzzle15_state>& batch, std::vector<int>& hs);

        // Convenience (sync) for a single state.
        int compute_one(const puzzle15_state& s);

    private:
        struct Impl;
        std::shared_ptr<Impl> impl_;
    };

} // namespace neural15
