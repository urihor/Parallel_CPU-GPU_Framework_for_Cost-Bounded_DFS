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

#include "puzzle_env.h"
#include "pdb15.h"
#include "puzzle15_state.h"
#include "batch_ida.h"
#include "solution_printer.h"
#include "korf_examples.h"


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
    if (!fs::exists(p, ec)) return false;
    auto sz = fs::file_size(p, ec);
    if (ec) return false;
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

// Run Batch IDA* on a list of boards using the 7/8 PDB heuristic
void run_batch_ida_example(const std::vector<puzzle15_state>& boards) {

    StpEnv env;

    int d_init   = 13;   // initial depth bound for GenerateWork
    int work_num = 7;    // number of logical stacks
    int solution_cost = 0;
    int board_num = 1;
    std::vector<StpEnv::Action> solution;
        for (const auto& board : boards) {

            auto start_time = std::chrono::high_resolution_clock::now();

            // Make a mutable copy of the initial state for BatchIDA
            auto start = board;   // or: auto start = board;

            solution.clear();

            bool found = batch_ida::BatchIDA(env,
                                            start,              // non-const lvalue
                                            &PdbHeuristic744,    // int(const StpEnv::State&)
                                            d_init,
                                            work_num,
                                            solution_cost,
                                            solution);

            if (found) {
                std::cout << "board number: " << board_num << std::endl;
                std::cout << "Solution cost = " << solution_cost << std::endl;
                //PrintSolution(env, start, solution);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> duration_s = end - start_time;
                std::cout << "duration time: " << duration_s.count() <<" seconds";
                std::cout << std::endl << std::endl;

            } else {
                std::cout << "No solution found for board " << board_num << std::endl << std::endl;
            }
            ++board_num;
        }

    }







// Force PDB tables to be loaded into RAM before timing starts
static void preload_pdbs_to_ram() {
    puzzle15_state goal;// default goal state

    // Call the heuristics once (or a few times) to trigger lazy loading.
    // The volatile sink prevents the compiler from optimizing these calls away.
    volatile int sink = 0;
    sink += pdb15::heuristic_78_auto(goal);
    (void) sink;
}


int main() {
    try {
        // Build / verify the 7/8 PDBs in the current build directory
        const fs::path out_dir = fs::current_path();
        ensure_78(out_dir);
        // Prepare the 100 Korf instances mapped to our goal
        std::vector<puzzle15_state> boards = MakeKorf100StatesForOurGoal();

        preload_pdbs_to_ram();
        boards = MakeKorf100StatesForOurGoal();
        std::cout<< std::endl << std::endl;
        std::cout << "=============================================" << std::endl;
        std::cout << "using pdb 744" << std::endl;
        std::cout << "=============================================" << std::endl;
        std::cout<< std::endl << std::endl;

        run_batch_ida_example(boards);
        std::vector<puzzle15_state> boards2;
        boards2.emplace_back(puzzle15_state{0,12,9,13,15,11,10,14,3,7,2,5,4,8,6,1});
        //run_batch_ida_example(boards2);

        std::cout << "[bootcamp_main done]\n";

        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "[error] " << ex.what() << "\n";
        return 1;
    }
}
