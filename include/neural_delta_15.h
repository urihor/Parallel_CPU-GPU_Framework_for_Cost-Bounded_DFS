//
// Created by Owner on 07/12/2025.
//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "puzzle15_state.h"
#include "manhattan_15.h"
#include "neural_batch_service.h"

namespace neural15 {

    //
    // NeuralDelta15
    // -------------
    // Wrapper around a neural model that predicts a *delta* on top of the
    // Manhattan heuristic for the 15-puzzle.
    //
    // Conceptually:
    //   h_M(s) = Manhattan(s) + neural_delta(s)
    //
    // The class is used as a process-wide singleton and can provide both
    // single-state and batched evaluations.
    //
    class NeuralDelta15 {
    public:
        // Singleton-style access: returns the single global instance
        // of NeuralDelta15 for the whole process.
        static NeuralDelta15& instance();

        // Initialize the neural model.
        //
        // Must be called exactly once at program startup (e.g. in main()).
        //
        // Parameters:
        //   model_dir - directory containing the model files (.pt, .json, etc.)
        //               For example: "../models" or "." if you run from the bin dir.
        void initialize(const std::string& model_dir);

        // Neural delta for patterns 1–7 and 8–15.
        //
        // These functions assume that the internal model has been initialized
        // and return the delta contribution that should be added on top of
        // the base Manhattan heuristic for the given state.
        int delta_1_7(const puzzle15_state& s) const;
        int delta_8_15(const puzzle15_state& s) const;

        // Batched versions of the delta functions.
        //
        // Input:
        //   states - vector of puzzle states.
        //
        // Output:
        //   A vector of the same length where each entry is the corresponding
        //   delta value for that state (for the chosen pattern set).
        std::vector<int> delta_1_7_batch(const std::vector<puzzle15_state>& states) const;
        std::vector<int> delta_8_15_batch(const std::vector<puzzle15_state>& states) const;

        // Convenience functions that return the full h_M heuristic.
        //
        // h_M(s, manhattan_heuristic):
        //   * Use this when you already have the Manhattan value computed
        //     externally (e.g. from your existing heuristic code).
        //
        // h_M_single(s):
        //   * Computes the full h_M for a single state, including Manhattan
        //     inside the implementation.
        int h_M(const puzzle15_state& s, int manhattan_heuristic) const;
        int h_M_single(const puzzle15_state& s) const;

        // Batched version: compute h_M for a vector of states.
        // The returned vector has the same size as 'states'.
        std::vector<int> h_M_batch(const std::vector<puzzle15_state>& states) const;

    private:
        NeuralDelta15() = default;

        // Disable copying of the singleton.
        NeuralDelta15(const NeuralDelta15&) = delete;
        NeuralDelta15& operator=(const NeuralDelta15&) = delete;

        // Initialization flag: true after initialize() has been called successfully.
        bool initialized_ = false;

        // Opaque implementation details (Torch model, buffers, etc.).
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

    // Initialize a default global NeuralBatchService configuration
    // wired to NeuralDelta15, so that batched neural evaluations can
    // be used seamlessly by the search code.
    void init_default_batch_service();

} // namespace neural15
