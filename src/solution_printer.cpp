//
// Created by Owner on 19/11/2025.
//
#include "solution_printer.h"

#include <iostream>
#include <iomanip>

// Helper: print a single 4x4 15-puzzle board.
static void print_board( puzzle15_state& s)
{
    std::cout << std::setfill(' ');

    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            int v = static_cast<int>(s.tiles[r * 4 + c]);
            if (v == 0) {
                std::cout << std::setw(2) << '.';
            } else {
                std::cout << std::setw(2) << v;
            }
            if (c < 3) {
                std::cout << ' ';
            }
        }
        std::cout << '\n';
    }
}

static const char* action_name(StpEnv::Action a)
{
    switch (a) {
        case StpEnv::Action::Up:    return "Up";
        case StpEnv::Action::Down:  return "Down";
        case StpEnv::Action::Left:  return "Left";
        case StpEnv::Action::Right: return "Right";
        default:                    return "?";
    }
}

void PrintSolution( StpEnv& env,
                    puzzle15_state& start,
                    std::vector<StpEnv::Action>& solution)
{
    puzzle15_state current = start;

    std::cout << "Initial state:\n";
    print_board(current);

    for (std::size_t i = 0; i < solution.size(); ++i) {
        StpEnv::Action a = solution[i];
        env.ApplyAction(current, a);

        std::cout << "\nStep " << (i + 1)
                  << " (" << action_name(a) << "):\n";
        print_board(current);
    }
}
