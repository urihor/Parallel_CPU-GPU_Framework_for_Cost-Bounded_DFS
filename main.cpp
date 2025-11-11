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
    puzzle15_state start{0, 12, 9, 13, 15, 11, 10, 14, 3, 7, 2, 5, 4, 8, 6, 1};
    constexpr int d_init = 6;

    std::vector<StpMove> hist;
    std::vector<WorkFor<StpEnv> > works;

    // without deduplication
    GenerateWork(env, start, d_init, hist, works);

    //  with deduplication
    /*std::unordered_set<std::size_t> seen;
    GenerateWorkDedup(env, start, d_init, hist, works, seen,
                      [](const puzzle15_state &s) { return std::hash<puzzle15_state>{}(s); });*/
}

int main() {
    build_works_example();
    std::cout << "== running assert-based tests ==\n";
    RunPuzzle15StateTests();
    RunStpEnvTests();
    std::cout << "[ALL OK]\n";
    return 0;
}
