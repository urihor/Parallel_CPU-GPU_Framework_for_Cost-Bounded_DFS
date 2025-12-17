//
// Created by Owner on 24/11/2025.
//
#include "test_ida_star_korf_examples.h"
#include "pdb15.h"
#include "puzzle_env.h"
#include "batch_ida.h"
#include "solution_printer.h"
#include <cassert>
#include <iostream>
#include <filesystem>

std::vector<puzzle15_state> MakeKorf100StatesForOurGoal();

// Simple adapter for PDB-based heuristic
static int PdbHeuristic78(const StpEnv::State& s) {
    return pdb15::heuristic_78_auto(s);
}

namespace Korf100Tests {

void RunAll() {
    std::cout << "[Korf100Tests] Running BatchIDA on 100 Korf instances...\n";

    static const int OPTIMAL_COSTS[100] = {
        57, 55, 59, 56, 56, 52, 52, 50, 46, 59,
        57, 45, 46, 59, 62, 42, 66, 55, 46, 52,
        54, 59, 49, 54, 52, 58, 53, 52, 54, 47,
        50, 59, 60, 52, 55, 52, 58, 53, 49, 54,
        54, 42, 64, 50, 51, 49, 47, 49, 59, 53,
        56, 56, 64, 56, 41, 55, 50, 51, 57, 66,
        45, 57, 56, 51, 47, 61, 50, 51, 53, 52,
        44, 56, 49, 56, 48, 57, 54, 53, 42, 57,
        53, 62, 49, 55, 44, 45, 52, 65, 54, 50,
        57, 57, 46, 53, 50, 49, 44, 54, 57, 54

    };

    // Ensure PDBs are ready (optional, can remove if already built)
    const std::filesystem::path out_dir = std::filesystem::current_path();
    std::cout << "[Korf100Tests] Verifying PDB availability...\n";
    pdb15::set_default_paths_78(
        (out_dir / "pdb_1_7.bin").string(),
        (out_dir / "pdb_8_15.bin").string()
    );

    // Create 100 start states
    std::vector<puzzle15_state> boards = MakeKorf100StatesForOurGoal();

    StpEnv env;
    int d_init   = 13;
    int work_num = 7;
    int solution_cost = 0;
    std::vector<StpEnv::Action> solution;

    for (size_t i = 0; i < boards.size(); ++i) {
        StpEnv::State start = boards[i];
        solution.clear();

        bool found = batch_ida::BatchIDA(env,
                                         start,
                                         &PdbHeuristic78,
                                         d_init,
                                         work_num,
                                         solution_cost,
                                         solution);

        assert(found && "BatchIDA failed to find a solution");

        int optimal = OPTIMAL_COSTS[i];
        std::cout << "Board " << (i + 1)
                  << ": cost=" << solution_cost
                  << ", expected=" << optimal << "\n";

        // Assert optimality
        assert(solution_cost == optimal && "Solution cost mismatch with optimal value");
    }

    std::cout << "[Korf100Tests] All 100 boards passed.\n";
}

} // namespace Korf100Tests
