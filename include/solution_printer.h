//
// Created by Owner on 19/11/2025.
//

#pragma once

#include <vector>
#include "puzzle15_state.h"
#include "puzzle_env.h"   // StpEnv

/// Print the sequence of boards starting from 'start' and applying all
/// actions in 'solution'. For each step, prints the step number, the move,
/// and the resulting board.
void PrintSolution(StpEnv &env,
                   const puzzle15_state &start,
                   const std::vector<StpEnv::Action> &solution);
