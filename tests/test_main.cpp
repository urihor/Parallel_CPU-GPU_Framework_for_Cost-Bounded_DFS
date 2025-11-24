//
// Created by Owner on 05/11/2025.
//


// test_main.cpp
#include <iostream>
#include <exception>

#include "test_generate_work.h"
#include "test_do_iteration.h"
#include "test_cb-dfs.h"

// Declared in other test source files (tests.cpp, etc.)
void RunPuzzle15StateTests();
void RunStpEnvTests();

int main() {
    try {
        std::cout << "== running assert-based tests ==\n";

        RunPuzzle15StateTests();
        RunStpEnvTests();

        GenerateWorkTests::RunAll();
        DoIterationTests::RunAll();
        CBDfsTests::RunAll();

        std::cout << "[ALL TESTS OK]\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[TEST ERROR] " << ex.what() << "\n";
        return 1;
    }
}
