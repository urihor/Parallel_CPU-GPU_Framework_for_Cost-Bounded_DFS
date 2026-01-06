#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "puzzle15_state.h"

namespace neural15 {

    struct NeuralDelta15QuantileOptions {
        std::vector<std::string> weights_1_7;   // usually single
        std::vector<std::string> weights_8_15;  // ensemble supported

        double quantile_q = 0.3;

        torch::Device device = torch::kCPU;
        bool use_half_on_cuda = false;

        bool add_manhattan = true;

        // Optional: correction tables built offline (rank -> over)
        // If empty -> no correction is applied for that side.
        std::string corrections_1_7_path;
        std::string corrections_8_15_path;
    };

    class NeuralDelta15Quantile {
    public:
        explicit NeuralDelta15Quantile(const NeuralDelta15QuantileOptions& opt);

        // BatchComputeFn-compatible:
        void compute_batch(const std::vector<puzzle15_state>& batch, std::vector<int>& hs);

        int compute_one(const puzzle15_state& s);

    private:
        struct Impl;
        std::shared_ptr<Impl> impl_;
    };

} // namespace neural15
