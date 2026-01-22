//
// Created by Owner on 05/01/2026.
//
#pragma once
#include <string>
#include <filesystem>
#include <chrono>
#include <limits>   // std::numeric_limits

namespace app {
    enum class RunMode {
        PdbOnly, // PDB only, no GPU, no NeuralBatchService
        PdbGuideNN, // PDB prune + GPU guide (NeuralBatchService running but not pruning)
        NNPrune, // GPU heuristic as prune (NeuralBatchService provides h)
        DeepCubeA // DeepCubeA backend (GPU batched heuristic)
    };

    struct AppConfig {
        RunMode mode = RunMode::PdbOnly;
        // If --board is provided, we solve ONLY that board (korf_n is ignored).
        std::string board; // empty => use Korf
        int korf_n = 100; // number of Korf instances (take first N)
        int d_init = 13;
        int work_num = 22;
        int num_threads = 0;

        // batching params (when service is on)
        std::size_t max_batch_size = 800;
        std::chrono::microseconds max_wait{200};

        // PDB location (default: current working dir)
        std::filesystem::path pdb_dir = std::filesystem::current_path();

        // if true and mode uses PDB, do preload() before timing
        bool preload_pdb = true;

        // ----------------------------
        // NN quantile config
        // ----------------------------
        float quantile_q = 0.25f;
        bool add_manhattan = true;

        std::string w1_7 = "nn_pdb_1_7_delta_full_ts.pt";
        std::string corr1_7 = "corr_1_7_0.25.bin";

        // ensemble 8-15
        std::string w8_15_0 = "nn_pdb_8_15_delta_lcg_ens0_ts.pt";
        std::string w8_15_1 = "nn_pdb_8_15_delta_lcg_ens1_ts.pt";
        std::string w8_15_2 = "nn_pdb_8_15_delta_lcg_ens2_ts.pt";
        std::string w8_15_3 = "nn_pdb_8_15_delta_lcg_ens3_ts.pt";
        std::string corr8_15 = "corr_8_15_0.25.bin";

        // ----------------------------
        // DeepCubeA params (defaults)
        // ----------------------------
        // TorchScript model path (relative to bin/working dir)
        std::string deepcubea_ts = "puzzle15_torchscript.pt";

        // Optional: override base (score of goal). If NaN -> compute base from goal at init.
        float deepcubea_base_override = std::numeric_limits<float>::quiet_NaN();
    };

    AppConfig parse_args(int argc, char **argv);

    int run(const AppConfig &cfg);
} // namespace app
