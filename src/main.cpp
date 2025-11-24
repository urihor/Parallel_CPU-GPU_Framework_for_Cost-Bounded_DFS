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

using Board = std::array<std::uint8_t, 16>;

/**
 * Returns the 100 Korf instances, mapped so that the goal state is:
 *   1  2  3  4
 *   5  6  7  8
 *   9 10 11 12
 *  13 14 15  0
 *
 * The original Korf instances are defined w.r.t. goal:
 *   0  1  2  3
 *   4  5  6  7
 *   8  9 10 11
 *  12 13 14 15
 *
 * Mapping = 180° rotation + value permutation (keeping 0 as blank),
 * so optimal solution depths are preserved.
 */
static std::vector<Board> MakeKorf100BoardsForOurGoal()
{
    static const Board korf_raw[100] = {
        //  1
        {14,13,15, 7,11,12, 9, 5, 6, 0, 2, 1, 4, 8,10, 3},
        //  2
        {13, 5, 4,10, 9,12, 8,14, 2, 3, 7, 1, 0,15,11, 6},
        //  3
        {14, 7, 8, 2,13,11,10, 4, 9,12, 5, 0, 3, 6, 1,15},
        //  4
        { 5,12,10, 7,15,11,14, 0, 8, 2, 1,13, 3, 4, 9, 6},
        //  5
        { 4, 7,14,13,10, 3, 9,12,11, 5, 6,15, 1, 2, 8, 0},
        //  6
        {14, 7, 1, 9,12, 3, 6,15, 8,11, 2, 5,10, 0, 4,13},
        //  7
        { 2,11,15, 5,13, 4, 6, 7,12, 8,10, 1, 9, 3,14, 0},
        //  8
        {12,11,15, 3, 8, 0, 4, 2, 6,13, 9, 5,14, 1,10, 7},
        //  9
        { 3,14, 9,11, 5, 4, 8, 2,13,12, 6, 7,10, 1,15, 0},
        // 10
        {13,11, 8, 9, 0,15, 7,10, 4, 3, 6,14, 5,12, 2, 1},
        // 11
        { 5, 9,13,14, 6, 3, 7,12,10, 8, 4, 0,15, 2,11, 1},
        // 12
        {14, 1, 9, 6, 4, 8,12, 5, 7, 2, 3, 0,10,11,13,15},
        // 13
        { 3, 6, 5, 2,10, 0,15,14, 1, 4,13,12, 9, 8,11, 7},
        // 14
        { 7, 6, 8, 1,11, 5,14,10, 3, 4, 9,13,15, 2, 0,12},
        // 15
        {13,11, 4,12, 1, 8, 9,15, 6, 5,14, 2, 7, 3,10, 0},
        // 16
        { 1, 3, 2, 5,10, 9,15, 6, 8,14,13,11,12, 4, 7, 0},
        // 17
        {15,14, 0, 4,11, 1, 6,13, 7, 5, 8, 9, 3, 2,10,12},
        // 18
        { 6, 0,14,12, 1,15, 9,10,11, 4, 7, 2, 8, 3, 5,13},
        // 19
        { 7,11, 8, 3,14, 0, 6,15, 1, 4,13, 9, 5,12, 2,10},
        // 20
        { 6,12,11, 3,13, 7, 9,15, 2,14, 8,10, 4, 1, 5, 0},
        // 21
        {12, 8,14, 6,11, 4, 7, 0, 5, 1,10,15, 3,13, 9, 2},
        // 22
        {14, 3, 9, 1,15, 8, 4, 5,11, 7,10,13, 0, 2,12, 6},
        // 23
        {10, 9, 3,11, 0,13, 2,14, 5, 6, 4, 7, 8,15, 1,12},
        // 24
        { 7, 3,14,13, 4, 1,10, 8, 5,12, 9,11, 2,15, 6, 0},
        // 25
        {11, 4, 2, 7, 1, 0,10,15, 6, 9,14, 8, 3,13, 5,12},
        // 26
        { 5, 7, 3,12,15,13,14, 8, 0,10, 9, 6, 1, 4, 2,11},
        // 27
        {14, 1, 8,15, 2, 6, 0, 3, 9,12,10,13, 4, 7, 5,11},
        // 28
        {13,14, 6,12, 4, 5, 1, 0, 9, 3,10, 2,15,11, 8, 7},
        // 29
        { 9, 8, 0, 2,15, 1, 4,14, 3,10, 7, 5,11,13, 6,12},
        // 30
        {12,15, 2, 6, 1,14, 4, 8, 5, 3, 7, 0,10,13, 9,11},
        // 31
        {12, 8,15,13, 1, 0, 5, 4, 6, 3, 2,11, 9, 7,14,10},
        // 32
        {14,10, 9, 4,13, 6, 5, 8, 2,12, 7, 0, 1, 3,11,15},
        // 33
        {14, 3, 5,15,11, 6,13, 9, 0,10, 2,12, 4, 1, 7, 8},
        // 34
        { 6,11, 7, 8,13, 2, 5, 4, 1,10, 3, 9,14, 0,12,15},
        // 35
        { 1, 6,12,14, 3, 2,15, 8, 4, 5,13, 9, 0, 7,11,10},
        // 36
        {12, 6, 0, 4, 7, 3,15, 1,13, 9, 8,11, 2,14, 5,10},
        // 37
        { 8, 1, 7,12,11, 0,10, 5, 9,15, 6,13,14, 2, 3, 4},
        // 38
        { 7,15, 8, 2,13, 6, 3,12,11, 0, 4,10, 9, 5, 1,14},
        // 39
        { 9, 0, 4,10, 1,14,15, 3,12, 6, 5, 7,11,13, 8, 2},
        // 40
        {11, 5, 1,14, 4,12,10, 0, 2, 7,13, 3, 9,15, 6, 8},
        // 41
        { 8,13,10, 9,11, 3,15, 6, 0, 1, 2,14,12, 5, 4, 7},
        // 42
        { 4, 5, 7, 2, 9,14,12,13, 0, 3, 6,11, 8, 1,15,10},
        // 43
        {11,15,14,13, 1, 9,10, 4, 3, 6, 2,12, 7, 5, 8, 0},
        // 44
        {12, 9, 0, 6, 8, 3, 5,14, 2, 4,11, 7,10, 1,15,13},
        // 45
        { 3,14, 9, 7,12,15, 0, 4, 1, 8, 5, 6,11,10, 2,13},
        // 46
        { 8, 4, 6, 1,14,12, 2,15,13,10, 9, 5, 3, 7, 0,11},
        // 47
        { 6,10, 1,14,15, 8, 3, 5,13, 0, 2, 7, 4, 9,11,12},
        // 48
        { 8,11, 4, 6, 7, 3,10, 9, 2,12,15,13, 0, 1, 5,14},
        // 49
        {10, 0, 2, 4, 5, 1, 6,12,11,13, 9, 7,15, 3,14, 8},
        // 50
        {12, 5,13,11, 2,10, 0, 9, 7, 8, 4, 3,14, 6,15, 1},
        // 51
        {10, 2, 8, 4,15, 0, 1,14,11,13, 3, 6, 9, 7, 5,12},
        // 52
        {10, 8, 0,12, 3, 7, 6, 2, 1,14, 4,11,15,13, 9, 5},
        // 53
        {14, 9,12,13,15, 4, 8,10, 0, 2, 1, 7, 3,11, 5, 6},
        // 54
        {12,11, 0, 8,10, 2,13,15, 5, 4, 7, 3, 6, 9,14, 1},
        // 55
        {13, 8,14, 3, 9, 1, 0, 7,15, 5, 4,10,12, 2, 6,11},
        // 56
        { 3,15, 2, 5,11, 6, 4, 7,12, 9, 1, 0,13,14,10, 8},
        // 57
        { 5,11, 6, 9, 4,13,12, 0, 8, 2,15,10, 1, 7, 3,14},
        // 58
        { 5, 0,15, 8, 4, 6, 1,14,10,11, 3, 9, 7,12, 2,13},
        // 59
        {15,14, 6, 7,10, 1, 0,11,12, 8, 4, 9, 2, 5,13, 3},
        // 60
        {11,14,13, 1, 2, 3,12, 4,15, 7, 9, 5,10, 6, 8, 0},
        // 61
        { 6,13, 3, 2,11, 9, 5,10, 1, 7,12,14, 8, 4, 0,15},
        // 62
        { 4, 6,12, 0,14, 2, 9,13,11, 8, 3,15, 7,10, 1, 5},
        // 63
        { 8,10, 9,11,14, 1, 7,15,13, 4, 0,12, 6, 2, 5, 3},
        // 64
        { 5, 2,14, 0, 7, 8, 6, 3,11,12,13,15, 4,10, 9, 1},
        // 65
        { 7, 8, 3, 2,10,12, 4, 6,11,13, 5,15, 0, 1, 9,14},
        // 66
        {11, 6,14,12, 3, 5, 1,15, 8, 0,10,13, 9, 7, 4, 2},
        // 67
        { 7, 1, 2, 4, 8, 3, 6,11,10,15, 0, 5,14,12,13, 9},
        // 68
        { 7, 3, 1,13,12,10, 5, 2, 8, 0, 6,11,14,15, 4, 9},
        // 69
        { 6, 0, 5,15, 1,14, 4, 9, 2,13, 8,10,11,12, 7, 3},
        // 70
        {15, 1, 3,12, 4, 0, 6, 5, 2, 8,14, 9,13,10, 7,11},
        // 71
        { 5, 7, 0,11,12, 1, 9,10,15, 6, 2, 3, 8, 4,13,14},
        // 72
        {12,15,11,10, 4, 5,14, 0,13, 7, 1, 2, 9, 8, 3, 6},
        // 73
        { 6,14,10, 5,15, 8, 7, 1, 3, 4, 2, 0,12, 9,11,13},
        // 74
        {14,13, 4,11,15, 8, 6, 9, 0, 7, 3, 1, 2,10,12, 5},
        // 75
        {14, 4, 0,10, 6, 5, 1, 3, 9, 2,13,15,12, 7, 8,11},
        // 76
        {15,10, 8, 3, 0, 6, 9, 5, 1,14,13,11, 7, 2,12, 4},
        // 77
        { 0,13, 2, 4,12,14, 6, 9,15, 1,10, 3,11, 5, 8, 7},
        // 78
        { 3,14,13, 6, 4,15, 8, 9, 5,12,10, 0, 2, 7, 1,11},
        // 79
        { 0, 1, 9, 7,11,13, 5, 3,14,12, 4, 2, 8, 6,10,15},
        // 80
        {11, 0,15, 8,13,12, 3, 5,10, 1, 4, 6,14, 9, 7, 2},
        // 81
        {13, 0, 9,12,11, 6, 3, 5,15, 8, 1,10, 4,14, 2, 7},
        // 82
        {14,10, 2, 1,13, 9, 8,11, 7, 3, 6,12,15, 5, 4, 0},
        // 83
        {12, 3, 9, 1, 4, 5,10, 2, 6,11,15, 0,14, 7,13, 8},
        // 84
        {15, 8,10, 7, 0,12,14, 1, 5, 9, 6, 3,13,11, 4, 2},
        // 85
        { 4, 7,13,10, 1, 2, 9, 6,12, 8,14, 5, 3, 0,11,15},
        // 86
        { 6, 0, 5,10,11,12, 9, 2, 1, 7, 4, 3,14, 8,13,15},
        // 87
        { 9, 5,11,10,13, 0, 2, 1, 8, 6,14,12, 4, 7, 3,15},
        // 88
        {15, 2,12,11,14,13, 9, 5, 1, 3, 8, 7, 0,10, 6, 4},
        // 89
        {11, 1, 7, 4,10,13, 3, 8, 9,14, 0,15, 6, 5, 2,12},
        // 90
        { 5, 4, 7, 1,11,12,14,15,10,13, 8, 6, 2, 0, 9, 3},
        // 91
        { 9, 7, 5, 2,14,15,12,10,11, 3, 6, 1, 8,13, 0, 4},
        // 92
        { 3, 2, 7, 9, 0,15,12, 4, 6,11, 5,14, 8,13,10, 1},
        // 93
        {13, 9,14, 6,12, 8, 1, 2, 3, 4, 0, 7, 5,10,11,15},
        // 94
        { 5, 7,11, 8, 0,14, 9,13,10,12, 3,15, 6, 1, 4, 2},
        // 95
        { 4, 3, 6,13, 7,15, 9, 0,10, 5, 8,11, 2,12, 1,14},
        // 96
        { 1, 7,15,14, 2, 6, 4, 9,12,11,13, 3, 0, 8, 5,10},
        // 97
        { 9,14, 5, 7, 8,15, 1, 2,10, 4,13, 6,12, 0,11, 3},
        // 98
        { 0,11, 3,12, 5, 2, 1, 9, 8,10,14,15, 7, 4,13, 6},
        // 99
        { 7,15, 4, 0,10, 9, 2, 5,12,11,13, 6, 1, 3,14, 8},
        //100
        {11, 4, 0, 8, 6,10, 5,13,12, 7,14, 3, 1, 2, 9,15},
    };

    // value permutation: keeps 0 and reverses 1..15
    static constexpr std::uint8_t perm[16] = {
        0, 15,14,13,12,11,10, 9,
        8,  7, 6, 5, 4, 3, 2, 1
    };

    std::vector<Board> result;
    result.reserve(100);

    for (const Board& b : korf_raw) {
        Board mapped{};
        // 180° rotation: new_index = 15 - old_index
        for (int old_idx = 0; old_idx < 16; ++old_idx) {
            int new_idx = 15 - old_idx;
            std::uint8_t tile = b[old_idx];
            mapped[new_idx] = perm[tile];
        }
        result.push_back(mapped);
    }

    return result;
}

std::vector<puzzle15_state> MakeKorf100StatesForOurGoal()
{
    std::vector<Board> boards = MakeKorf100BoardsForOurGoal();
    std::vector<puzzle15_state> states;
    states.reserve(boards.size());

    for (const Board& b : boards) {
        std::vector v(b.begin(), b.end());
        states.emplace_back(v);
    }

    return states;
}

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
    auto start = std::chrono::high_resolution_clock::now();

    StpEnv env;

    int d_init   = 13;   // initial depth bound for GenerateWork
    int work_num = 8;    // number of logical stacks
    int solution_cost = 0;
    int board_num = 1;
    std::vector<StpEnv::Action> solution;

    for (const auto& board : boards) {
        // Make a mutable copy of the initial state for BatchIDA
        auto start = board;   // or: auto start = board;

        solution.clear();

        const bool found = batch_ida::BatchIDA(env,
                                         start,              // non-const lvalue
                                         &PdbHeuristic78,    // int(const StpEnv::State&)
                                         d_init,
                                         work_num,
                                         solution_cost,
                                         solution);

        if (found) {
            std::cout << "board number: " << (board_num % 100) << std::endl;
            std::cout << "Solution cost = " << solution_cost << std::endl << std::endl;
            PrintSolution(env, start, solution);
            std::cout << std::endl;
        } else {
            std::cout << "No solution found for board " << (board_num % 100) << "\n";
        }
        ++board_num;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_s = end - start;

    std::cout << "duration time: " << duration_s.count() << " seconds" << std::endl;
}


int main() {
    try {
        // Build / verify the 7/8 PDBs in the current build directory
        const fs::path out_dir = fs::current_path();
        ensure_78(out_dir);
        // Prepare the 100 Korf instances mapped to our goal
        std::vector<puzzle15_state> boards = MakeKorf100StatesForOurGoal();

        run_batch_ida_example(boards);

        std::cout << "[bootcamp_main done]\n";

        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "[error] " << ex.what() << "\n";
        return 1;
    }
}
