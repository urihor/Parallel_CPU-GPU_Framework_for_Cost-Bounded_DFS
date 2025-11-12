//
// Created by Owner on 05/11/2025.
//


// test_main.cpp
#include <iostream>
#include "generate_work.h"
#include "work.h"
#include "puzzle_env.h"
#include "puzzle15_state.h"
#include <unordered_set>


void RunPuzzle15StateTests();

void RunStpEnvTests();

void build_works_example() {
    StpEnv env;
    puzzle15_state start{1,2,3,4,5,6,7,8,9,10,11,12,0,13,14,15};
    constexpr int d_init = 14;
    int best_len = d_init;

    std::vector<StpMove> hist, best_sol;
    std::vector<WorkFor<StpEnv> > works;

    // without deduplication
    GenerateWork(env, start, d_init, hist, works, &best_len, &best_sol);
    std::cout << works.size() << std::endl;
    if (best_len <= d_init - 1) {
        std::cout << "the best solution is " << best_len << " moves" << std::endl;
    }
    works.clear();

    //  with deduplication
    std::unordered_set<std::size_t> seen;
    best_len = d_init;
    GenerateWorkDedup(env, start, d_init, hist, works, seen,
                      [](const puzzle15_state &s) { return std::hash<puzzle15_state>{}(s); }, &best_len, &best_sol);
    std::cout << works.size() << std::endl;
    if (best_len <= d_init - 1) {
        std::cout << "the best solution is " << best_len << " moves" << std::endl;
    }
}

int main() {
    build_works_example();
    std::cout << "== running assert-based tests ==\n";
    RunPuzzle15StateTests();
    RunStpEnvTests();
    std::cout << "[ALL OK]\n";
    return 0;
}
