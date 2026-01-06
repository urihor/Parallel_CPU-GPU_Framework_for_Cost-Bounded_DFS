#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "puzzle15_state.h"

namespace neural15 {
    /**
     * Configuration options for NeuralDelta15Quantile:
     *
     *  - Two ensembles of TorchScript models:
     *      * weights_1_7:  networks that predict delta for tiles 1–7
     *      * weights_8_15: networks that predict delta for tiles 8–15
     *
     *  - The networks output a distribution over delta-classes (0..K).
     *    We pick a quantile (e.g. q=0.3 or q=0.5), then optionally apply
     *    offline “over-correction” tables to guarantee admissibility.
     *
     *  - The final heuristic is:
     *        h(s) = [Manhattan(s) if add_manhattan] + d_1_7 + d_8_15
     *    where d_1_7 and d_8_15 are quantile-based, ensemble-min deltas.
     */
    struct NeuralDelta15QuantileOptions {
        // File paths to TorchScript models for the 1–7 subset.
        // In practice we usually use a single model here.
        std::vector<std::string> weights_1_7; // usually single

        // File paths to TorchScript models for the 8–15 subset.
        // Multiple entries are supported (ensemble). We take the minimum
        // predicted delta across all ensemble members.
        std::vector<std::string> weights_8_15; // ensemble supported

        // Quantile q in (0, 1]. For each model and each state we take
        // the smallest class index whose CDF >= q, then ensemble-min.
        double quantile_q = 0.25;

        // Torch device on which all models will run (CPU or CUDA).
        torch::Device device = torch::kCPU;

        // If true and device.is_cuda(), convert models to half-precision
        // (FP16) to save memory and bandwidth.
        bool use_half_on_cuda = false;

        // If true, add the full 15-puzzle Manhattan distance on top
        // of the predicted deltas. If false, the heuristic is “delta only”.
        bool add_manhattan = true;

        // Optional: offline correction tables for overestimation.
        //
        // Each file encodes an array corr[d] such that we transform
        // the raw delta d_raw into:
        //      d = max(0, d_raw - corr[d_raw])
        //
        // This allows us to empirically guarantee admissibility:
        // for every abstract state, h_pred <= PDB.
        //
        // If the path is empty, no correction is applied on that side.
        std::string corrections_1_7_path; // rank -> max over-correction for tiles 1–7
        std::string corrections_8_15_path; // rank -> max over-correction for tiles 8–15
    };

    /**
     * Quantile-based neural heuristic for the 15-puzzle.
     *
     * The idea:
     *   - Represent each state by two abstract features:
     *       * positions of tiles 1–7 + blank
     *       * positions of tiles 8–15 + blank
     *   - For each side (1–7 and 8–15), an ensemble of TorchScript models
     *     predicts a discrete distribution over delta = PDB - Manhattan.
     *   - We pick a quantile index for each model, then take the minimum
     *     over the ensemble to be conservative.
     *   - Optionally apply offline over-correction tables to force
     *     admissibility.
     *   - Optionally add full Manhattan to obtain a final heuristic h(s).
     *
     * This class is thread-safe for inference when shared via std::shared_ptr
     * and used in a typical “read-only weights” pattern (no parameter updates).
     */
    class NeuralDelta15Quantile {
    public:
        /// Construct the quantile-based heuristic with the given options.
        explicit NeuralDelta15Quantile(const NeuralDelta15QuantileOptions &opt);

        /**
         * Batched heuristic evaluation.
         *
         * @param batch  Vector of puzzle states.
         * @param hs     Output vector, resized to batch.size(), where
         *               hs[i] = heuristic(batch[i]).
         *
         * This is the main entry point used by NeuralBatchService.
         */
        void compute_batch(const std::vector<puzzle15_state> &batch, std::vector<int> &hs);

        /**
         * Convenience method for a single-state evaluation.
         * Internally implemented via a tiny batch of size 1.
         */
        int compute_one(const puzzle15_state &s);

    private:
        // PIMPL: implementation details (models, buffers, weights…) are hidden
        // inside Impl to keep the public header light and stable.
        struct Impl;
        std::shared_ptr<Impl> impl_;
    };
} // namespace neural15
