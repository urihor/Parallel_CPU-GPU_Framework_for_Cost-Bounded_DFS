//
// Created by uriel on 04/11/2025.
//

#pragma once
#include <cstdint>

enum class StpMove : std::uint8_t { Up, Down, Left, Right };

// return the inverse move of move m
inline StpMove inverse(const StpMove m) {
    switch (m) {
        case StpMove::Up: return StpMove::Down;
        case StpMove::Down: return StpMove::Up;
        case StpMove::Left: return StpMove::Right;
        case StpMove::Right: return StpMove::Left;
    }
    return m;
}

// return the name of the move - for nice printing
inline const char *to_string(const StpMove m) {
    switch (m) {
        case StpMove::Up: return "U";
        case StpMove::Down: return "D";
        case StpMove::Left: return "L";
        case StpMove::Right: return "R";
    }
    return "?";
}
