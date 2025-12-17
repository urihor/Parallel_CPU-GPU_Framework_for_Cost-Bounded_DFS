//
// Created by Owner on 24/11/2025.
//
#pragma once

#include <vector>
#include "puzzle15_state.h"

/// Returns the 100 Korf benchmark instances, mapped to our goal state.
/// (in Korf's boards 0 is in the upper left corner in the goal state)
std::vector<puzzle15_state> MakeKorf100StatesForOurGoal();
