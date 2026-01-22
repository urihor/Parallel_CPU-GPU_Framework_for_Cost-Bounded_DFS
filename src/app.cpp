//
// Created by Owner on 05/01/2026.
//
// High-level:
//   This file implements the top-level application logic for the 15-puzzle
//   Batch IDA* solver. It parses command-line arguments, prepares PDBs,
//   configures which heuristic backend to use (pure PDB, NN-guided, NN-prune,
//   or DeepCubeA), starts/stops the global NeuralBatchService, and finally
//   runs BatchIDA on either Korf's benchmark boards or a single board
//   provided via --board=... .
//
#include "app.h"

#include <iostream>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <system_error>
#include <sstream>
#include <cctype>
#include <cstdlib>

#include <torch/torch.h>

#include "puzzle_env.h"
#include "pdb15.h"
#include "puzzle15_state.h"
#include "batch_ida.h"
#include "korf_examples.h"
#include "nvtx_helpers.h"
#include "solution_printer.h"
#include "manhattan_15.h"


#include "neural_batch_service.h"
#include "neural_delta_15_quantile.h"
#include "deepcubea15_heuristic.h"


// ------------------------------------------------------------
// PDB utilities
// ------------------------------------------------------------
namespace fs = std::filesystem;

// Returns the expected on-disk size (in bytes) of a PDB for pattern size k.
// Internally, pdb15::states_for_pattern(k) = P(16, k+1) states, and we pack
// either 8-bit or 4-bit entries depending on PDB_BITS.
static std::uint64_t expected_bytes_for_k(int k) {
    std::uint64_t n = pdb15::states_for_pattern(k); // P(16, k+1)
#if PDB_BITS == 8
    return n;
#else
    return (n + 1) / 2;
#endif
}

// Checks if file p exists and has exactly the size we expect for a k-PDB.
// Used to decide whether we need to rebuild the PDB or can reuse it.
static bool file_ok(const fs::path &p, int k) {
    std::error_code ec;
    if (!fs::exists(p, ec))
        return false;
    auto sz = fs::file_size(p, ec);
    if (ec)
        return false;
    return sz == expected_bytes_for_k(k);
}

// Ensure that the 1–7 and 8–15 PDB files exist in out_dir.
// If missing or with wrong size, they are built on the fly.
// Also sets the default PDB paths for heuristic_78_auto.
static void ensure_78(const fs::path &out_dir) {
    fs::create_directories(out_dir);

    fs::path p7 = out_dir / "pdb_1_7.bin";
    fs::path p8 = out_dir / "pdb_8_15.bin";

    bool ok7 = file_ok(p7, 7);
    bool ok8 = file_ok(p8, 8);

    std::cout << "[ensure_78] dir: " << fs::absolute(out_dir) << "\n";

    if (!ok7) {
        std::cout << "[ensure_78] building 7-PDB -> " << fs::absolute(p7) << "\n";
        pdb15::build_pdb_01bfs({1, 2, 3, 4, 5, 6, 7}, p7.string(), /*verbose=*/true);
    } else {
        std::cout << "[ensure_78] 7-PDB OK -> " << fs::absolute(p7) << "\n";
    }

    if (!ok8) {
        std::cout << "[ensure_78] building 8-PDB -> " << fs::absolute(p8) << "\n";
        pdb15::build_pdb_01bfs({8, 9, 10, 11, 12, 13, 14, 15}, p8.string(), /*verbose=*/true);
    } else {
        std::cout << "[ensure_78] 8-PDB OK -> " << fs::absolute(p8) << "\n";
    }

    pdb15::set_default_paths_78(p7.string(), p8.string());
}

// Touches the PDBs once (by querying the goal state) to force them into
// the OS file cache / RAM. This can reduce I/O stalls during the first run.
static void preload_pdbs_to_ram() {
    puzzle15_state goal;
    volatile int sink = 0;
    sink += pdb15::heuristic_78_auto(goal);
    (void) sink;
}

// ------------------------------------------------------------
// Heuristic adapters
// ------------------------------------------------------------

// Simple adapter from our 7/8-split PDB implementation to the StpEnv
// heuristic signature used by BatchIDA.
static int PdbHeuristic78(const StpEnv::State &s) {
    return pdb15::heuristic_78_auto(s);
}

// Manhattan fallback heuristic (always available, no files required).
static int ManhattanHeuristic(const StpEnv::State &s) {
    return manhattan_15(s);
}

// Trivial zero heuristic (for experimentation / debugging only).
static int Heuristic0(const StpEnv::State &) {
    return 0;
}

// ------------------------------------------------------------
// NN service (our models + correction tables)
// ------------------------------------------------------------
// This global shared_ptr keeps the quantile NN heuristic alive as long as the
// NeuralBatchService lambda is using it.
static std::shared_ptr<neural15::NeuralDelta15Quantile> g_nn_quantile;

// Start the NeuralBatchService with our NeuralDelta15Quantile backend.
// All configuration is taken from AppConfig (paths, quantile q, device,
// correction tables, batch size and max wait).
static void start_nn_service_quantile(const app::AppConfig &cfg) {
    neural15::NeuralDelta15QuantileOptions opt;

    // TorchScript weights for the 1–7 and 8–15 networks:
    opt.weights_1_7 = {cfg.w1_7};
    opt.weights_8_15 = {cfg.w8_15_0, cfg.w8_15_1, cfg.w8_15_2, cfg.w8_15_3};

    opt.quantile_q = cfg.quantile_q;
    opt.device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    opt.add_manhattan = cfg.add_manhattan;

    // overestimation correction tables.
    opt.corrections_1_7_path = cfg.corr1_7;
    opt.corrections_8_15_path = cfg.corr8_15;

    g_nn_quantile = std::make_shared<neural15::NeuralDelta15Quantile>(opt);

    // Start the global batching service with a lambda that calls our NN
    // heuristic on a whole batch of puzzle15_state.
    NeuralBatchService::instance().start(
        [](const std::vector<puzzle15_state> &batch, std::vector<int> &hs) {
            // capture via global shared_ptr to keep lambda copyable
            g_nn_quantile->compute_batch(batch, hs);
        },
        cfg.max_batch_size,
        cfg.max_wait
    );
}

// ------------------------------------------------------------
// NN service (DeepCubeA model)
// ------------------------------------------------------------
// Global handle to the DeepCubeA heuristic; used by its own batching wrapper.
static std::shared_ptr<DeepCubeA15Heuristic> g_deepcubea;

// Start the DeepCubeA-based heuristic service. This assumes CUDA is available
// (we only call this in such a case) and uses AppConfig for base etc.
static void start_deepcubea_service(const app::AppConfig &cfg) {
    DeepCubeA15Heuristic::Options opt;
    opt.ts_path = cfg.deepcubea_ts; // TorchScript model path
    opt.device = torch::kCUDA; // we only enter here if CUDA is available
    opt.base_override = cfg.deepcubea_base_override; // optional fixed base (NaN means "auto")

    g_deepcubea = std::make_shared<DeepCubeA15Heuristic>(opt);
    g_deepcubea->start_service(cfg.max_batch_size, cfg.max_wait);
}

// ------------------------------------------------------------
// Run loop (Batch IDA*)
// ------------------------------------------------------------
//
// Runs Batch IDA* on either:
//   * all Korf benchmark boards, or
//   * a single board provided via --board=... ,
// depending on what was passed to run_batch_ida().
//
// The synchronous heuristic pointer passed to BatchIDA is chosen as follows:
//   * In PDB modes (PdbOnly, PdbGuideNN): we pass the 7/8 PDB heuristic.
//   * In GPU modes (NNPrune, DeepCubeA): we prefer a "file-free" fallback
//     heuristic (Manhattan) unless valid PDB files already exist on disk,
//     in which case we can use the PDB heuristic as the synchronous baseline.
//
// In NN modes, DoIteration may internally use the NeuralBatchService output
// for pruning and/or guiding (depending on flags), and may avoid calling the
// synchronous heuristic on NN-pruned states. Still, BatchIDA/DoIteration can
// require a synchronous heuristic for bookkeeping / fallback paths, so we must
// ensure the passed heuristic is always safe to call even when PDB files
// were not built/loaded.
//

static void run_batch_ida(const app::AppConfig &cfg,
                          const std::vector<puzzle15_state> &boards) {
    StpEnv env;

    int solution_cost = 0;
    std::vector<StpEnv::Action> solution;

    // Choose the synchronous heuristic to pass into BatchIDA.
    //
    // - In pure PDB modes we always use the 7/8 PDB heuristic.
    // - In GPU modes (NN / DeepCubeA) we prefer Manhattan unless PDB files already exist,
    //   because GPU modes may run without building/ensuring PDB files first.
    int (*heuristic)(const StpEnv::State &) = &ManhattanHeuristic;

    const bool pdb_mode =
            (cfg.mode == app::RunMode::PdbOnly || cfg.mode == app::RunMode::PdbGuideNN);

    if (pdb_mode) {
        // PDB modes: run() should have already called ensure_78() and set_default_paths_78().
        heuristic = &PdbHeuristic78;
    } else {
        // GPU modes: use PDB only if the expected files already exist and look valid.
        // This avoids crashing / throwing when PDB files are missing.
        const auto p7 = cfg.pdb_dir / "pdb_1_7.bin";
        const auto p8 = cfg.pdb_dir / "pdb_8_15.bin";
        const bool have_pdb = file_ok(p7, 7) && file_ok(p8, 8);

        if (have_pdb) {
            pdb15::set_default_paths_78(p7.string(), p8.string());
            heuristic = &PdbHeuristic78;
        } else {
            heuristic = &ManhattanHeuristic;
        }
    }

    std::cout << "[BatchIDA] d_init=" << cfg.d_init
            << " work_num=" << cfg.work_num
            << " threads=" << cfg.num_threads
            << "\n";

    const bool svc_running = NeuralBatchService::instance().is_running();
    std::cout << "[BatchIDA] neural_batch_enabled=" << (batch_ida::neural_batch_enabled() ? 1 : 0)
            << " guide_batch_enabled=" << (batch_ida::guide_batch_enabled() ? 1 : 0)
            << " service_running=" << (svc_running ? 1 : 0)
            << "\n";

    auto t_all0 = std::chrono::high_resolution_clock::now();

    int board_num = 1;
    for (const auto &board: boards) {
        auto t0 = std::chrono::high_resolution_clock::now();

        auto start = board;
        solution.clear();

        if (svc_running) {
            NeuralBatchService::instance().reset_for_new_bound();
        }

        NVTX_RANGE("Solve one board");
        const bool found = batch_ida::BatchIDA(env,
                                               start,
                                               heuristic,
                                               cfg.d_init,
                                               cfg.work_num,
                                               solution_cost,
                                               solution,
                                               cfg.num_threads);

        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dt = t1 - t0;

        if (found) {
            std::cout << "board " << board_num
                    << " | cost=" << solution_cost
                    << " | time=" << dt.count() << " sec\n";
        } else {
            std::cout << "board " << board_num
                    << " | NO SOLUTION"
                    << " | time=" << dt.count() << " sec\n";
        }
        ++board_num;
    }

    auto t_all1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt_all = t_all1 - t_all0;
    std::cout << "[BatchIDA] total time: " << dt_all.count() << " sec\n";
}


// ------------------------------------------------------------
// Args parsing (minimal, robust enough)
// ------------------------------------------------------------
namespace app {
    // Parse the --mode=... argument into a RunMode enum.
    // Supported values: pdb, pdb-guide-nn, nn, deepcubea.
    static RunMode parse_mode(std::string s) {
        for (auto &c: s)
            c = (char) std::tolower((unsigned char) c);

        if (s == "pdb")
            return RunMode::PdbOnly;
        if (s == "pdb-guide-nn")
            return RunMode::PdbGuideNN;
        if (s == "nn")
            return RunMode::NNPrune;
        if (s == "deepcubea")
            return RunMode::DeepCubeA;

        throw std::invalid_argument("Unknown --mode. Use: pdb | pdb-guide-nn | nn | deepcubea");
    }

    // Minimal, hand-rolled parser for command line arguments.
    // Recognized flags (all in the form --key=value):
    //   --mode=...         (pdb | pdb-guide-nn | nn | deepcubea)
    //   --korf=N           (how many Korf benches to run; 0 = all, default from AppConfig)
    //   --dinit=N          (initial IDA* bound d_init)
    //   --worknum=N        (number of initial subtrees / works)
    //   --threads=N        (number of CPU search threads)
    //   --batch=N          (max GPU batch size)
    //   --wait_us=N        (max waiting time in microseconds in NeuralBatchService)
    //   --pdb_dir=PATH     (directory for 1–7 / 8–15 PDB files)
    //   --board=CSV16      (single board, 16 comma-separated integers 0..15)
    // Any unknown flag is treated as a hard error.
    AppConfig parse_args(int argc, char **argv) {
        AppConfig cfg;

        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];

            auto get_val = [&](const char *key) -> std::string {
                const std::string k = std::string(key) + "=";
                if (a.rfind(k, 0) == 0)
                    return a.substr(k.size());
                return {};
            };

            if (a == "--help") {
                std::cout
                        << "Usage:\n"
                        << "  --mode=pdb | pdb-guide-nn | nn | deepcubea\n"
                        << "  --korf=N\n"
                        << "  --dinit=N\n"
                        << "  --worknum=N\n"
                        << "  --threads=N\n"
                        << "  --batch=N\n"
                        << "  --wait_us=N\n"
                        << "  --pdb_dir=PATH\n"
                        << "  --board=CSV16 (16 numbers, comma-separated)\n"
                        << "Example:\n"
                        << "  --board=9,1,3,4,2,5,6,8,10,14,7,12,13,11,15,0\n";
                std::exit(0);
            }

            if (auto v = get_val("--mode"); !v.empty()) {
                cfg.mode = parse_mode(v);
                continue;
            }
            if (auto v = get_val("--korf"); !v.empty()) {
                cfg.korf_n = std::stoi(v);
                continue;
            }
            if (auto v = get_val("--dinit"); !v.empty()) {
                cfg.d_init = std::stoi(v);
                continue;
            }
            if (auto v = get_val("--worknum"); !v.empty()) {
                cfg.work_num = std::stoi(v);
                continue;
            }
            if (auto v = get_val("--threads"); !v.empty()) {
                cfg.num_threads = std::stoi(v);
                continue;
            }
            if (auto v = get_val("--batch"); !v.empty()) {
                cfg.max_batch_size = (std::size_t) std::stoul(v);
                continue;
            }
            if (auto v = get_val("--wait_us"); !v.empty()) {
                cfg.max_wait = std::chrono::microseconds(std::stoi(v));
                continue;
            }
            if (auto v = get_val("--pdb_dir"); !v.empty()) {
                cfg.pdb_dir = fs::path(v);
                continue;
            }
            if (auto v = get_val("--board"); !v.empty()) {
                cfg.board = v;
                continue;
            }

            // keep unknown args as hard error (less silent bugs)
            throw std::invalid_argument("Unknown argument: " + a);
        }

        return cfg;
    }

    // Parse the --board=... argument, which must be exactly 16 comma-separated
    // integers in [0,15]. Whitespace around tokens is allowed.
    // Example: --board=9,1,3,4,2,5,6,8,10,14,7,12,13,11,15,0
    static puzzle15_state parse_board_csv_16(const std::string &csv) {
        std::vector<int> vals;
        vals.reserve(16);

        std::istringstream iss(csv);
        std::string token;

        while (std::getline(iss, token, ',')) {
            if (token.empty()) {
                throw std::invalid_argument("--board: empty token (double comma?)");
            }
            // std::stoi ignores leading/trailing spaces, so " 9" is fine
            int v = std::stoi(token);
            vals.push_back(v);
        }

        if (vals.size() != 16) {
            throw std::invalid_argument("--board must have exactly 16 comma-separated numbers (0..15)");
        }

        std::vector<puzzle15_state::Tile> tiles;
        tiles.reserve(16);

        for (int v: vals) {
            if (v < 0 || v > 15) {
                throw std::invalid_argument("--board values must be in [0..15]");
            }
            tiles.push_back(static_cast<puzzle15_state::Tile>(v));
        }

        return puzzle15_state(tiles);
    }

    // Main application entry point (called from main.cpp).
    // Responsibilities:
    //   * Resolve the effective RunMode (including CUDA fallback).
    //   * Print a small banner describing the chosen mode.
    //   * Prepare the list of boards: either a single --board or Korf-100 subset.
    //   * Build/load the required PDBs (in modes that use PDB).
    //   * Start/stop NeuralBatchService according to selected mode (PdbOnly,
    //     PdbGuideNN, NNPrune, DeepCubeA).
    //   * Invoke run_batch_ida() and return 0 on success.
    //
    int run(const AppConfig &cfg) {
        RunMode mode = cfg.mode; // Effective mode (may change on fallback)
        auto require_file = [](const std::string &p, const char *what) {
            if (p.empty() || !std::filesystem::exists(p)) {
                throw std::runtime_error(std::string("Missing required file for ") + what + ": " + p);
            }
        };

        if (cfg.mode == RunMode::NNPrune || cfg.mode == RunMode::PdbGuideNN) {
            require_file(cfg.w1_7,"NN model (tiles 1-7)");
            require_file(cfg.w8_15_0, "NN model (tiles 8-15_0)");
            require_file(cfg.w8_15_1, "NN model (tiles 8-15_1)");
            require_file(cfg.w8_15_2, "NN model (tiles 8-15_2)");
            require_file(cfg.w8_15_3, "NN model (tiles 8-15_3)");
            require_file(cfg.corr1_7, "corr1_7");
            require_file(cfg.corr8_15, "corr8_15");
        } else if (mode == RunMode::DeepCubeA) {
            // require the TorchScript model file to exist.
            require_file(cfg.deepcubea_ts, "DeepCubeA TorchScript model ");
        }

        const bool cuda_ok = torch::cuda::is_available();

        // If a GPU-based mode was requested but CUDA is not available,
        // we fall back to pure PDB mode (CPU only).
        const bool requested_gpu_mode =
                (mode == RunMode::PdbGuideNN || mode == RunMode::NNPrune || mode == RunMode::DeepCubeA);

        if (requested_gpu_mode && !cuda_ok) {
            std::cout << "[WARN] CUDA not available -> falling back to MODE: PDB only (CPU)\n";
            mode = RunMode::PdbOnly;
        }

        // Banner for the effective mode.
        std::cout << "\n=============================================\n";
        switch (mode) {
            case RunMode::PdbOnly: std::cout << "MODE: PDB only (CPU)\n";
                break;
            case RunMode::PdbGuideNN: std::cout << "MODE: PDB prune + GPU guide (NN)\n";
                break;
            case RunMode::NNPrune: std::cout << "MODE: GPU heuristic prune (NN)\n";
                break;
            case RunMode::DeepCubeA: std::cout << "MODE: DeepCubeA\n";
                break;
        }
        std::cout << "=============================================\n\n";

        // Only print CUDA/device info if we are actually in a GPU-related mode.
        if (mode != RunMode::PdbOnly) {
            std::cout << "torch::cuda::is_available() = " << (cuda_ok ? "true" : "false") << "\n";
            torch::Device device = cuda_ok ? torch::kCUDA : torch::kCPU;
            std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";
        }

        // Prepare boards:
        //   * If --board=... is given, solve only that board.
        //   * Otherwise, use Korf's 100 instances and optionally truncate to korf_n.
        std::vector<puzzle15_state> boards;

        if (!cfg.board.empty()) {
            boards.clear();
            boards.push_back(parse_board_csv_16(cfg.board));
        } else {
            boards = MakeKorf100StatesForOurGoal();
            if (cfg.korf_n > 0 && (std::size_t) cfg.korf_n < boards.size()) {
                boards.resize((std::size_t) cfg.korf_n);
            }
        }

        // Build/load PDBs only in modes that actually use them for pruning.
        const bool uses_pdb = (mode == RunMode::PdbOnly || mode == RunMode::PdbGuideNN);
        if (uses_pdb) {
            ensure_78(cfg.pdb_dir);
            if (cfg.preload_pdb) {
                std::cout << "[PDB] preloading tables to RAM...\n";
                preload_pdbs_to_ram();
                std::cout << "[PDB] preload done.\n";
            }
        }

        // Make sure no previous NeuralBatchService is still running
        // (e.g., from a previous run in the same process).
        if (NeuralBatchService::instance().is_running()) {
            NeuralBatchService::instance().shutdown();
        }

        // Configure batching flags and start the appropriate service for the
        // effective mode.
        if (mode == RunMode::PdbOnly) {
            // Pure PDB: no NN, no batching.
            batch_ida::set_neural_batch_enabled(false);
            batch_ida::set_guide_batch_enabled(false);
        } else if (mode == RunMode::PdbGuideNN) {
            // NN is used only for guiding (ordering), pruning is still PDB-only.
            start_nn_service_quantile(cfg);
            batch_ida::set_neural_batch_enabled(false);
            batch_ida::set_guide_batch_enabled(true);
        } else if (mode == RunMode::NNPrune) {
            // NN heuristic is used directly for pruning, PDB is bypassed in DoIteration.
            start_nn_service_quantile(cfg);
            batch_ida::set_neural_batch_enabled(true);
            batch_ida::set_guide_batch_enabled(false);
        } else if (mode == RunMode::DeepCubeA) {
            // DeepCubeA heuristic (GPU) is used as pruning heuristic.
            start_deepcubea_service(cfg);
            batch_ida::set_neural_batch_enabled(true); // DeepCubeA as prune
            batch_ida::set_guide_batch_enabled(false);
        }

        // Run Batch IDA* on the chosen set of boards.
        run_batch_ida(cfg, boards);

        // Clean shutdown of the batching service, if enabled.
        if (NeuralBatchService::instance().is_running()) {
            NeuralBatchService::instance().shutdown();
        }

        std::cout << "[app done]\n";
        return 0;
    }
} // namespace app
