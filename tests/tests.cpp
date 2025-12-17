#include <cassert>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <array>
#include <stdexcept>

#include "puzzle_env.h"
#include "puzzle15_state.h"
#include "puzzle_actions.h"

// ===== Helpers =====
static bool hasMove(const std::vector<StpMove>& v, StpMove m) {
    return std::find(v.begin(), v.end(), m) != v.end();
}

static bool isOpposite(StpMove a, StpMove b) {
    return (a == StpMove::Up    && b == StpMove::Down) ||
           (a == StpMove::Down  && b == StpMove::Up)   ||
           (a == StpMove::Left  && b == StpMove::Right)||
           (a == StpMove::Right && b == StpMove::Left);
}

// ===== Tests for puzzle15_state =====
void RunPuzzle15StateTests() {
    std::cout << "[state] start\n";
    StpEnv env;

    // 0) Valid constructor (initializer_list) – a single move returns to the goal
    {
        puzzle15_state s({ 1,  2,  3,  4,
                           5,  6,  7,  8,
                           9, 10, 11, 12,
                          13, 14,  0, 15 });
        auto acts = env.GetActions(s);
        assert(hasMove(acts, StpMove::Right));
        env.ApplyAction(s, StpMove::Right);
        assert(env.IsGoal(s));
    }

    // 1) Round-trip pack/unpack on a random path + full undo
    {
        std::mt19937 rng(12345);
        puzzle15_state s; // goal
        std::vector<StpMove> path;
        path.reserve(200);

        for (int step = 0; step < 200; ++step) {
            auto acts = env.GetActions(s);
            std::vector<StpMove> filt;
            filt.reserve(acts.size());
            if (!path.empty()) {
                for (auto a : acts) if (!isOpposite(a, path.back())) filt.push_back(a);
            }
            if (filt.empty()) filt = acts;

            std::uniform_int_distribution<int> dist(0, (int)filt.size() - 1);
            StpMove a = filt[dist(rng)];
            env.ApplyAction(s, a);
            path.push_back(a);
        }

        auto packed = s.pack();
        auto s2 = puzzle15_state::unpack(packed);
        assert(s2.pack() == s.pack());

        for (int i = (int)path.size() - 1; i >= 0; --i) {
            env.UndoAction(s2, path[i]);
        }
        assert(env.IsGoal(s2));
    }

    // 2) pack/unpack on a clean goal state
    {
        puzzle15_state g; // goal
        auto packed = g.pack();
        auto g2 = puzzle15_state::unpack(packed);
        assert(g2.pack() == g.pack());
    }

    // 3) Validation: duplicates and missing 0 (note: vector has size 16!)
    {
        bool thrown = false;
        try {
            std::vector<std::uint8_t> bad(16);
            // 1 appears twice, and 0 is missing
            for (int i = 0; i < 16; ++i)
                bad[i] = (i == 0 ? 1 : static_cast<std::uint8_t>(i));
            puzzle15_state t(bad); // expected to throw
            (void)t;
        } catch (const std::exception&) { thrown = true; }
        assert(thrown && "expected exception on duplicate / missing 0");
    }

    // 4) Validation: value out of range (16)
    {
        bool thrown = false;
        try {
            std::vector<std::uint8_t> bad(16);
            for (int i = 0; i < 16; ++i) bad[i] = static_cast<std::uint8_t>(i);
            bad[0] = 16; // out of range 0..15
            puzzle15_state t(bad); // expected to throw
            (void)t;
        } catch (const std::exception&) { thrown = true; }
        assert(thrown && "expected exception on out-of-range (>=16)");
    }

    // 5) Validation: two zeros (some other tile is missing)
    {
        bool thrown = false;
        try {
            std::vector<std::uint8_t> bad = {
                1,  2,  3,  4,
                5,  6,  7,  8,
                9, 10, 11, 12,
               13,  0,  0, 15 // two zeros, tile 14 is missing
            };
            puzzle15_state t(bad); // expected to throw
            (void)t;
        } catch (const std::exception&) { thrown = true; }
        assert(thrown && "expected exception on duplicate 0 / missing tile");
    }

    std::cout << "[state] OK\n";
}

// ===== Tests for StpEnv =====
void RunStpEnvTests() {
    std::cout << "[env] start\n";
    StpEnv env;

    // 1) Goal: move legality + Apply/Undo + IsGoal
    {
        puzzle15_state s; // goal
        auto acts = env.GetActions(s);
        assert(acts.size() == 2);
        assert(hasMove(acts, StpMove::Up));
        assert(hasMove(acts, StpMove::Left));

        auto base = s.pack();
        for (auto a : acts) {
            env.ApplyAction(s, a);
            env.UndoAction(s, a);
            assert(s.pack() == base);
        }

        assert(env.IsGoal(s));
        env.ApplyAction(s, StpMove::Up);
        assert(!env.IsGoal(s));
        env.UndoAction(s, StpMove::Up);
        assert(env.IsGoal(s));
    }

    // 2) Corner = 2 moves, edge = 3 moves, center = 4 moves
    {
        puzzle15_state s; // goal — move blank to corner (0,0)
        env.ApplyAction(s, StpMove::Left);
        env.ApplyAction(s, StpMove::Left);
        env.ApplyAction(s, StpMove::Left);
        env.ApplyAction(s, StpMove::Up);
        env.ApplyAction(s, StpMove::Up);
        env.ApplyAction(s, StpMove::Up);

        auto at_corner = env.GetActions(s);
        assert(at_corner.size() == 2);

        env.ApplyAction(s, StpMove::Right);
        auto at_edge = env.GetActions(s);
        assert(at_edge.size() == 3);

        env.ApplyAction(s, StpMove::Down);
        auto at_center = env.GetActions(s);
        assert(at_center.size() == 4);
    }

    // 3) Check move legality indirectly via GetActions (without touching private IsLegal)
    {
        puzzle15_state s; // goal — move blank to corner again
        env.ApplyAction(s, StpMove::Left);
        env.ApplyAction(s, StpMove::Left);
        env.ApplyAction(s, StpMove::Left);
        env.ApplyAction(s, StpMove::Up);
        env.ApplyAction(s, StpMove::Up);
        env.ApplyAction(s, StpMove::Up);

        auto acts = env.GetActions(s);
        assert(!hasMove(acts, StpMove::Up));
        assert(!hasMove(acts, StpMove::Left));
        assert( hasMove(acts, StpMove::Right));
        assert( hasMove(acts, StpMove::Down));
    }

    // 4) Random walk + full undo
    {
        std::mt19937 rng(999);
        puzzle15_state s; // goal
        std::vector<StpMove> path;
        path.reserve(300);

        for (int step = 0; step < 300; ++step) {
            auto acts = env.GetActions(s);
            std::vector<StpMove> filt;
            filt.reserve(acts.size());
            if (!path.empty()) {
                for (auto a : acts) if (!isOpposite(a, path.back())) filt.push_back(a);
            }
            if (filt.empty()) filt = acts;

            std::uniform_int_distribution<int> dist(0, (int)filt.size() - 1);
            StpMove a = filt[dist(rng)];
            env.ApplyAction(s, a);
            path.push_back(a);
        }

        auto s2 = s;
        for (int i = (int)path.size() - 1; i >= 0; --i) {
            env.UndoAction(s2, path[i]);
        }
        assert(env.IsGoal(s2));
    }

    std::cout << "[env] OK\n";
}
