//
// Created by Owner on 24/11/2025.
//

#pragma once
#include "puzzle15_state.h"
#include <vector>

/**
 * @brief Runs BatchIDA* on the 100 Korf benchmark instances.
 *        For each instance, asserts that the returned solution cost
 *        matches the known optimal value.
 *
 * The list of optimal costs should be filled in test_korf100.cpp.
 */
namespace Korf100Tests {
    void RunAll();
}