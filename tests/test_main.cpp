//
// Created by Owner on 05/11/2025.
//


// test_main.cpp
#include <iostream>

// הצהרות קדמיות (ההגדרות ב-tests.cpp)
void RunPuzzle15StateTests();
void RunStpEnvTests();

int main() {
    std::cout << "== running assert-based tests ==\n";
    RunPuzzle15StateTests();
    RunStpEnvTests();
    std::cout << "[ALL OK]\n";
    return 0;
}
