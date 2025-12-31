//
// Created by Owner on 24/11/2025.
//

#include <iostream>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <array>
#include <vector>
#include <cstdint>
#include <torch/torch.h>
#include <torch/script.h>

#include "puzzle_env.h"
#include "pdb15.h"
#include "puzzle15_state.h"
#include "batch_ida.h"
#include "solution_printer.h"
#include "korf_examples.h"
#include "neural_delta_15.h"
#include "manhattan_15.h"
#include "nvtx_helpers.h"
#include "deepcubea15_heuristic.h"



namespace fs = std::filesystem;

// Compute expected file size for k (number of tiles in the pattern)
static std::uint64_t expected_bytes_for_k(int k) {
    std::uint64_t n = pdb15::states_for_pattern(k); // P(16, k+1)
#if PDB_BITS == 8
    return n;
#else
    return (n + 1) / 2;
#endif
}

// Check that the file exists and has the exact expected size
static bool file_ok(const fs::path &p, int k) {
    std::error_code ec;
    if (!fs::exists(p, ec))
        return false;
    auto sz = fs::file_size(p, ec);
    if (ec)
        return false;
    return sz == expected_bytes_for_k(k);
}

// Build 7/8 PDBs if missing/corrupted; files will be written under out_dir
static void ensure_78(const fs::path &out_dir) {
    fs::create_directories(out_dir);
    fs::path p7 = out_dir / "pdb_1_7.bin";
    fs::path p8 = out_dir / "pdb_8_15.bin";

    bool ok7 = file_ok(p7, 7);
    bool ok8 = file_ok(p8, 8);

    std::cout << "[ensure_78] output dir: " << fs::absolute(out_dir) << "\n";
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

    // Configure the default auto-lookup paths for 7/8 PDBs
    pdb15::set_default_paths_78(p7.string(), p8.string());
}

// Build 7/4/4 PDBs if missing/corrupted; files will be written under out_dir
static void ensure_744(const fs::path &out_dir) {
    fs::create_directories(out_dir);
    fs::path p7 = out_dir / "pdb_1_7.bin";
    fs::path p4_first = out_dir / "pdb_8_11.bin";
    fs::path p4_second = out_dir / "pdb_12_15.bin";

    bool ok7 = file_ok(p7, 7);
    bool ok4_first = file_ok(p4_first, 4);
    bool ok4_second = file_ok(p4_second, 4);

    std::cout << "[ensure_744] output dir: " << fs::absolute(out_dir) << "\n";
    if (!ok7) {
        std::cout << "[ensure_744] building 7-PDB -> " << fs::absolute(p7) << "\n";
        pdb15::build_pdb_01bfs({1, 2, 3, 4, 5, 6, 7}, p7.string(), /*verbose=*/true);
    } else {
        std::cout << "[ensure_744] 7-PDB OK -> " << fs::absolute(p7) << "\n";
    }
    if (!ok4_first) {
        std::cout << "[ensure_744] building first 4-PDB -> " << fs::absolute(p4_first) << "\n";
        pdb15::build_pdb_01bfs({8, 9, 10, 11}, p4_first.string(), /*verbose=*/true);
    } else {
        std::cout << "[ensure_744] first 4-PDB OK -> " << fs::absolute(p4_first) << "\n";
    }
    if (!ok4_second) {
        std::cout << "[ensure_744] building second 4-PDB -> " << fs::absolute(p4_second) << "\n";
        pdb15::build_pdb_01bfs({12, 13, 14, 15}, p4_second.string(), /*verbose=*/true);
    } else {
        std::cout << "[ensure_744] second 4-PDB OK -> " << fs::absolute(p4_second) << "\n";
    }

    // Configure the default auto-lookup paths for 7/4/4 PDBs
    pdb15::set_default_paths_744(p7.string(), p4_first.string(), p4_second.string());
}

// Simple adapter so BatchIDA can call our PDB-based heuristic
static int PdbHeuristic78(const StpEnv::State &s) {
    return pdb15::heuristic_78_auto(s);
}

static int PdbHeuristic744(const StpEnv::State &s) {
    return pdb15::heuristic_744_auto(s);
}

static int Heuristic0(const StpEnv::State &s) {
    return 0;
}

static DeepCubeA15Heuristic* g_dc = nullptr;

static int DeepCubeAHeuristic(const StpEnv::State& s) {
    // חסימה עד שיש ערך (כי DeepCubeA15Heuristic::h הוא blocking)
    return g_dc ? g_dc->h(s) : 0;
}

void start_deepcubea_service() {
    DeepCubeA15Heuristic::Options opt;
    opt.ts_path = "puzzle15_torchscript.pt";
    opt.device  = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    opt.scale   = 1.0f;
    opt.base_override = 0.0f;

    static DeepCubeA15Heuristic dc(opt);
    g_dc = &dc;

    dc.start_service(/*max_batch_size=*/800,
                     /*max_wait=*/std::chrono::microseconds(200));
}



// Run Batch IDA* on a list of boards using the 7/8 PDB heuristic
void run_batch_ida_example(const std::vector<puzzle15_state> &boards) {
    StpEnv env;

    int d_init = 13; // initial depth bound for GenerateWork
    int work_num = 22; // number of logical stacks
    int solution_cost = 0;
    int board_num = 1;

    std::vector<StpEnv::Action> solution;

    // Decide which heuristic to pass to BatchIDA.
    int (*heuristic)(const StpEnv::State &) = &PdbHeuristic78;

    if (batch_ida::neural_batch_enabled() &&
        NeuralBatchService::instance().is_running()) {
        // In neural-batched mode, the synchronous heuristic is set to 0.
        // The actual h_M values are supplied asynchronously via the
        // NeuralBatchService (Algorithm 4 style).
        heuristic = &DeepCubeAHeuristic;;
        std::cout << "[run_batch_ida_example] Using asynchronous neural h_M via GPU\n";
    } else {
        std::cout << "[run_batch_ida_example] Using PDB heuristic (no neural batching)\n";
    }

    auto start_time_a = std::chrono::high_resolution_clock::now();

    for (const auto &board: boards) {
        auto start_time_b = std::chrono::high_resolution_clock::now();

        // Make a mutable copy of the initial state for BatchIDA
        auto start = board;

        solution.clear();
        if ((batch_ida::neural_batch_enabled() &&
        NeuralBatchService::instance().is_running())) {
                NeuralBatchService::instance().reset_for_new_bound();
        }
        NVTX_RANGE("Solve one board");
        bool found = batch_ida::BatchIDA(env,
                                        start, // non-const lvalue
                                        heuristic, // int(const StpEnv::State&)
                                        d_init,
                                        work_num,
                                        solution_cost,
                                        solution,
                                        1);

        if (found) {
            std::cout << "board number: " << board_num << std::endl;
            std::cout << "Solution cost = " << solution_cost << std::endl;
            //PrintSolution(env, start, solution);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration_s = end - start_time_b;
            std::cout << "duration time: " << duration_s.count() << " seconds";
            std::cout << std::endl << std::endl;
        } else {
            std::cout << "No solution found for board " << board_num << std::endl << std::endl;
        }
        board_num++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_s = end - start_time_a;
    std::cout << "duration time of all boards: " << duration_s.count() << " seconds" << std::endl;
    std::cout << std::endl << std::endl;

}

// Force PDB tables to be loaded into RAM before timing starts
static void preload_pdbs_to_ram() {
    puzzle15_state goal; // default goal state

    // Call the heuristics once (or a few times) to trigger lazy loading.
    // The volatile sink prevents the compiler from optimizing these calls away.
    volatile int sink = 0;
    sink += pdb15::heuristic_78_auto(goal);
    (void) sink;
}


int main() {
    try {
        // Build / verify the PDBs in the current build directory
        const fs::path out_dir = fs::current_path();
        ensure_78(out_dir);
        // Prepare the 100 Korf instances mapped to our goal
        //const std::vector<puzzle15_state> boards = MakeKorf100StatesForOurGoal();
        const std::vector<puzzle15_state> boards = MakeKorf50StatesForOurGoal();


        //preload_pdbs_to_ram();

        // --- GPU / CUDA sanity check ---
        std::cout << "torch::cuda::is_available() = "
                << (torch::cuda::is_available() ? "true" : "false")
                << std::endl;

        torch::Device device = torch::cuda::is_available()
                                   ? torch::Device(torch::kCUDA)
                                   : torch::Device(torch::kCPU);

        std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

        start_deepcubea_service();        //neural15::NeuralDelta15::instance().initialize(".");

        //neural15::init_default_batch_service();
        batch_ida::set_neural_batch_enabled(true);

        std::cout
        << "neural_batch_enabled=" << batch_ida::neural_batch_enabled()
        << "  service_running=" << NeuralBatchService::instance().is_running()
        << std::endl;

        run_batch_ida_example(boards);

        NeuralBatchService::instance().shutdown();

        std::cout << "[bootcamp_main done]\n";

        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "[error] " << ex.what() << "\n";
        return 1;
    }
}
