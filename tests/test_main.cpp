//
// Created by Owner on 05/11/2025.
//


// test_main.cpp
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include "puzzle_env.h"
#include "pdb15.h"
#include "puzzle15_state.h"
#include "generate_work.h"
#include "test_generate_work.h"
#include <unordered_set>
#include "test_do_iteration.h"
#include "test_cb-dfs.h"
#include "batch_ida.h"
#include "solution_printer.h"

void RunPuzzle15StateTests();

void RunStpEnvTests();

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

// File check: exists and is exactly the expected size
static bool file_ok(const fs::path &p, int k) {
    std::error_code ec;
    if (!fs::exists(p, ec)) return false;
    auto sz = fs::file_size(p, ec);
    if (ec) return false;
    return sz == expected_bytes_for_k(k);
}

// Build 7/8 if missing/corrupted; files will be written under out_dir
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

    // Define the paths for the auto version
    pdb15::set_default_paths_78(p7.string(), p8.string());
}

// Build 7/4/4 if missing/corrupted; files will be written under out_dir
static void ensure_744(const fs::path &out_dir) {
    fs::create_directories(out_dir);
    fs::path pA = out_dir / "pdb_1_7.bin";
    fs::path pB = out_dir / "pdb_8_11.bin";
    fs::path pC = out_dir / "pdb_12_15.bin";

    bool okA = file_ok(pA, 7);
    bool okB = file_ok(pB, 4);
    bool okC = file_ok(pC, 4);

    std::cout << "[ensure_744] output dir: " << fs::absolute(out_dir) << "\n";
    if (!okA) {
        std::cout << "[ensure_744] building A(1..7) -> " << fs::absolute(pA) << "\n";
        pdb15::build_pdb_01bfs({1, 2, 3, 4, 5, 6, 7}, pA.string(), /*verbose=*/true);
    } else {
        std::cout << "[ensure_744] A OK -> " << fs::absolute(pA) << "\n";
    }
    if (!okB) {
        std::cout << "[ensure_744] building B(8..11) -> " << fs::absolute(pB) << "\n";
        pdb15::build_pdb_01bfs({8, 9, 10, 11}, pB.string(), /*verbose=*/true);
    } else {
        std::cout << "[ensure_744] B OK -> " << fs::absolute(pB) << "\n";
    }
    if (!okC) {
        std::cout << "[ensure_744] building C(12..15) -> " << fs::absolute(pC) << "\n";
        pdb15::build_pdb_01bfs({12, 13, 14, 15}, pC.string(), /*verbose=*/true);
    } else {
        std::cout << "[ensure_744] C OK -> " << fs::absolute(pC) << "\n";
    }

    // Define the paths for the auto version
    pdb15::set_default_paths_744(pA.string(), pB.string(), pC.string());
}

static void demo_queries() {
    // Goal state
    puzzle15_state goal;
    // Single-move mode (via the environment)
    StpEnv env;
    puzzle15_state one = goal;
    auto acts = env.GetActions(goal);
    if (!acts.empty()) env.ApplyAction(one, acts[0]);

    // Manually configured mode (the constructor validates and finds blankPos)
    puzzle15_state custom({
        0, 12, 9, 13, 15, 11, 10, 14, 3, 7, 2, 5, 4, 8, 6, 1
    });

    // --- 7/8 (auto) ---
    const int h78_goal = pdb15::heuristic_78_auto(goal);
    const int h78_one = pdb15::heuristic_78_auto(one);
    const int h78_cus = pdb15::heuristic_78_auto(custom);

    std::cout << "[78] h(goal)=" << h78_goal
            << "  h(one)=" << h78_one
            << "  h(custom)=" << h78_cus << "\n";

        // --- 7/4/4 (auto) ---
        const int h744_goal = pdb15::heuristic_744_auto(goal);
        const int h744_one  = pdb15::heuristic_744_auto(one);
        const int h744_cus  = pdb15::heuristic_744_auto(custom);

        std::cout << "[744] h(goal)=" << h744_goal
                  << "  h(one)="    << h744_one
                  << "  h(custom)=" << h744_cus << "\n";
}

void build_works_example() {
    StpEnv env;
    puzzle15_state start{1,2,3,4,5,6,7,8,9,10,11,12,0,13,14,15};
    constexpr int d_init = 14;
    int best_len = d_init;

    std::vector<StpMove> hist, best_sol;
    std::vector<WorkFor<StpEnv> > works;

    // without deduplication
    GenerateWork(env, start, d_init, hist, works, best_len, best_sol);
    std::cout << works.size() << std::endl;
    if (best_len <= d_init - 1) {
        std::cout << "the best solution is " << best_len << " moves" << std::endl;
    }
    works.clear();

    //  with deduplication
    std::unordered_set<std::size_t> seen;
    best_len = d_init;
    GenerateWorkDedup(env, start, d_init, hist, works, seen,
                      [](const puzzle15_state &s) { return std::hash<puzzle15_state>{}(s); }, best_len, best_sol);
    std::cout << works.size() << std::endl;
    if (best_len <= d_init - 1) {
        std::cout << "the best solution is " << best_len << " moves" << std::endl;
    }
}

static int PdbHeuristic(const StpEnv::State& s) {
    return pdb15::heuristic_744_auto(s);
}

void run_batch_ida_example() {
    StpEnv env;
    puzzle15_state start{8,7,9,4,1,5,3,6,14,12,0,11,2,13,10,15}; // או מצב שרירותי שאתה מגדיר

    int d_init   = 4;   // עומק ל-GenerateWork
    int work_num = 8;   // מספר stack-ים לוגיים
    int solution_cost = 0;
    std::vector<StpEnv::Action> solution;


    bool found = batch_ida::BatchIDA(env,
                                     start,
                                     &PdbHeuristic,
                                     d_init,
                                     work_num,
                                     solution_cost,
                                     solution);

    if (found) {
        std::cout << "Solution cost = " << solution_cost << "\n";
        PrintSolution(env, start, solution);
    } else {
        std::cout << "No solution found.\n";
    }
}

int main() {
    /*try {
        // Write the files to the run's working directory (Debug/Release)
        const fs::path out_dir = fs::current_path();

        // Build/verify 7/8
        //ensure_78(out_dir);

        // Build/verify 7/4/4
        ensure_744(out_dir);

        // Demonstration of heuristic calculations
       //demo_queries();

        std::cout << "[done]\n";
    } catch (const std::exception &ex) {
        std::cerr << "[error] " << ex.what() << "\n";
        return 1;
    }*/
    run_batch_ida_example();
    /*GenerateWorkTests::RunAll();
    DoIterationTests::RunAll();
    CBDfsTests::RunAll();
    build_works_example();
    std::cout << "== running assert-based tests ==\n";
    RunPuzzle15StateTests();
    RunStpEnvTests();
    std::cout << "[ALL OK]\n";*/
    return 0;
}
