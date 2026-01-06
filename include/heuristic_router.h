//
// Created by Owner on 05/01/2026.
//
#pragma once
#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "puzzle15_state.h"
#include "neural_batch_service.h"
#include "neural_delta_15_quantile.h"

/**
 * HeuristicRouter
 *
 * Central place for selecting and using a heuristic backend for the 15-puzzle.
 *
 * Responsibilities:
 *   - Hold configuration for different heuristic backends (DeepCubeA, NeuralDeltaQuantile).
 *   - Provide a single, global entry point (singleton) for the search code.
 *   - Integrate with NeuralBatchService by exposing a batching callback.
 *   - Provide a synchronous heuristic (h_sync) for initial bounds / debugging.
 *
 * Typical usage:
 *   1) Choose backend and initialize:
 *        HeuristicRouter::instance().use_deepcubea(opts);
 *        // or:
 *        HeuristicRouter::instance().use_neural_delta_quantile(ndq_opts);
 *
 *   2) Start the batching service:
 *        HeuristicRouter::instance().start(max_batch_size, max_wait);
 *
 *   3) In the search:
 *        - Try asynchronous evaluation via NeuralBatchService.
 *        - Fallback to h_sync(...) when needed.
 *
 *   4) On shutdown:
 *        HeuristicRouter::instance().shutdown();
 */
class HeuristicRouter {
public:
    using State = puzzle15_state;

    /**
     * Supported heuristic backends.
     *
     * None:
     *   No NN-based heuristic configured. The caller should fall back to
     *   Manhattan or PDB directly.
     *
     * DeepCubeA:
     *   TorchScript DeepCubeA-style network that predicts a heuristic value.
     *
     * NeuralDeltaQuantile:
     *   Li-style quantile ensemble over deltas (1–7, 8–15) that is wrapped
     *   by NeuralDelta15Quantile.
     */
    enum class Backend {
        None = 0,
        DeepCubeA = 1,
        NeuralDeltaQuantile = 2,
    };

    // ---- DeepCubeA backend options ----
    struct DeepCubeAOptions {
        /// Path to the TorchScript model file, e.g. "deepcubea_puzzle15_ts.pt".
        std::string ts_path; // e.g. "deepcubea_puzzle15_ts.pt"

        /// Device to run the model on (CPU or CUDA).
        torch::Device device = torch::kCPU;

        /**
         * If true, we subtract the network output on the goal state so that
         * h(goal) is approximately 0. This often makes the heuristic scale
         * more natural for search.
         */
        bool normalize_goal_to_zero = true;

        /**
         * If true, we add the 4x4 Manhattan distance on top of the network
         * prediction, treating the network output as a delta on top of
         * Manhattan.
         */
        bool add_manhattan = false;

        /// How to convert a floating-point NN score into an integer heuristic.
        enum class Rounding { Floor, Ceil, Round };

        Rounding rounding = Rounding::Round;
    };

    /// Global singleton instance used by the rest of the search code.
    static HeuristicRouter &instance();

    /**
     * Configure the router to use a DeepCubeA TorchScript backend.
     *
     * Must be called before start(). This loads the model path and stores
     * all options (device, normalization, rounding mode, etc.).
     */
    void use_deepcubea(const DeepCubeAOptions &opt);

    /**
     * Configure the router to use the NeuralDeltaQuantile backend
     * (Li-style 1–7 / 8–15 delta quantile ensemble).
     *
     * Must be called before start(). Internally constructs a
     * neural15::NeuralDelta15Quantile instance from the provided options.
     */
    void use_neural_delta_quantile(const neural15::NeuralDelta15QuantileOptions &opt);

    /// Return the currently selected backend (or Backend::None).
    Backend backend() const noexcept { return backend_; }

    /// Human-readable name of the current backend (for logging / CLI).
    const char *backend_name() const noexcept;

    /**
     * Start the shared NeuralBatchService using the selected backend.
     *
     * max_batch_size:
     *   Maximum number of states to process in a single GPU/NN batch.
     *
     * max_wait:
     *   Maximum time to wait for the batch to fill before forcing a flush.
     *
     * This registers compute_batch_fn(...) as the callback into the selected
     * backend and starts the background worker thread in NeuralBatchService.
     */
    void start(std::size_t max_batch_size,
               std::chrono::nanoseconds max_wait);

    /**
     * Stop the batching service and release associated resources.
     *
     * Safe to call multiple times; subsequent calls are no-ops.
     * After shutdown(), asynchronous NN evaluation will no longer run.
     */
    void shutdown();

    /**
     * Reset any per-IDA*-bound state.
     *
     * Intended to be called at the start of each new IDA* iteration so that
     * any bound-dependent caches can be cleared (if the backend uses them).
     * Depending on the implementation this may be a lightweight no-op.
     */
    void reset_for_new_bound();

    /**
     * Synchronous heuristic evaluation for a single state.
     *
     * This bypasses NeuralBatchService and calls the selected backend
     * directly in the current thread. Useful for:
     *   - Computing the initial IDA* bound.
     *   - Debugging / sanity checks against the asynchronous path.
     *
     * If no backend is configured (Backend::None), the implementation may
     * return a simple fallback (e.g. Manhattan) or throw, depending on how
     * you implemented it in the .cpp.
     */
    int h_sync(const State &s);

private:
    HeuristicRouter() = default;

    /**
     * Dispatch target registered with NeuralBatchService.
     *
     * The batch service calls this with a batch of states; the router:
     *   - For DeepCubeA: runs the TorchScript model on the batch and converts
     *     float outputs into integer heuristics.
     *   - For NeuralDeltaQuantile: delegates to NeuralDelta15Quantile::compute_batch.
     *
     * The resulting heuristic values are written into 'hs' (same size as batch).
     */
    void compute_batch_fn(const std::vector<State> &batch, std::vector<int> &hs);

    // ----- DeepCubeA impl -----

    /**
     * Convert a raw DeepCubeA score to an integer heuristic.
     *
     * Applies (optional) base normalization, optional Manhattan addition, and
     * the configured rounding mode (floor/ceil/round).
     */
    int deepcubea_score_to_h(float y) const;

    /**
     * Run the DeepCubeA model on a single state and return the raw
     * floating-point score (before normalization / Manhattan / rounding).
     */
    float deepcubea_forward_single(const State &s);

    /**
     * Run the DeepCubeA model on a batch of states.
     *
     * Fills 'out_scores' with one raw score per input state. The caller is
     * responsible for converting those scores to integer heuristics by calling
     * deepcubea_score_to_h().
     */
    void deepcubea_forward_batch(const std::vector<State> &batch, std::vector<float> &out_scores) const;

    /**
     * Standard Manhattan distance on a 4x4 15-puzzle board (tiles 1..15,
     * blank ignored). Used when DeepCubeAOptions::add_manhattan is true.
     */
    static int manhattan_4x4(const State &s);

private:
    /// Protects backend configuration and internal state that can be accessed concurrently.
    mutable std::mutex mtx_;

    /// Currently selected backend (None, DeepCubeA, or NeuralDeltaQuantile).
    Backend backend_ = Backend::None;

    /// Indicates whether the NeuralBatchService has been started via start().
    bool started_ = false;

    // ---- DeepCubeA state ----

    /// TorchScript module for the DeepCubeA model.
    mutable torch::jit::script::Module dc_module_;

    /// Device on which the DeepCubeA model runs (CPU or CUDA).
    torch::Device dc_device_ = torch::kCPU;

    /// True once the DeepCubeA model has been successfully loaded.
    bool dc_loaded_ = false;

    /// If true, subtract dc_base_ so that the goal state has score ~0.
    bool dc_norm_goal_ = true;

    /// If true, add Manhattan(4x4) on top of the DeepCubeA score.
    bool dc_add_manhattan_ = false;

    /// Rounding mode applied when converting float scores to integer h.
    DeepCubeAOptions::Rounding dc_rounding_ = DeepCubeAOptions::Rounding::Round;

    /// Baseline output value on the goal state (used when normalize_goal_to_zero is enabled).
    float dc_base_ = 0.0f; // output on goal

    // ---- NeuralDeltaQuantile state ----

    /**
     * Shared pointer to the Li-style quantile neural heuristic over PDB deltas
     * (1–7 and 8–15 patterns). Used when backend_ == Backend::NeuralDeltaQuantile.
     */
    std::shared_ptr<neural15::NeuralDelta15Quantile> ndq_;
};
