//
// Created by Owner on 07/12/2025.
//
#include "manhattan_15.h"
int manhattan_15(const puzzle15_state& s) {
    int h = 0;

    for (int idx = 0; idx < puzzle15_state::Size; ++idx) {
        puzzle15_state::Tile t = s.tiles[static_cast<std::size_t>(idx)];
        if (t == 0) {
            // ignore blank
            continue;
        }

        // goal index of tile t in the standard goal:
        // [ 1  2  3  4
        //   5  6  7  8
        //   9 10 11 12
        //  13 14 15  _ ]
        int goal_idx = static_cast<int>(t) - 1;

        int r  = puzzle15_state::row(idx);
        int c  = puzzle15_state::col(idx);
        int gr = puzzle15_state::row(goal_idx);
        int gc = puzzle15_state::col(goal_idx);

        h += std::abs(r - gr) + std::abs(c - gc);
    }

    return h;
}