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

// ===== עזרים =====
static bool hasMove(const std::vector<StpMove>& v, StpMove m) {
    return std::find(v.begin(), v.end(), m) != v.end();
}
static bool isOpposite(StpMove a, StpMove b) {
    return (a == StpMove::Up    && b == StpMove::Down) ||
           (a == StpMove::Down  && b == StpMove::Up)   ||
           (a == StpMove::Left  && b == StpMove::Right)||
           (a == StpMove::Right && b == StpMove::Left);
}

// ===== בדיקות עבור puzzle15_state =====
void RunPuzzle15StateTests() {
    std::cout << "[state] start\n";
    StpEnv env;

    // 0) בנאי תקין (initializer_list) – צעד יחיד מחזיר ליעד
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

    // 1) round-trip pack/unpack על מסלול רנדומי + Undo מלא
    {
        std::mt19937 rng(12345);
        puzzle15_state s; // יעד
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

    // 2) pack/unpack על יעד נקי
    {
        puzzle15_state g;
        auto packed = g.pack();
        auto g2 = puzzle15_state::unpack(packed);
        assert(g2.pack() == g.pack());
    }

    // 3) ולידציה: כפילויות וללא 0  (שימו לב: vector עם גודל 16!)
    {
        bool thrown = false;
        try {
            std::vector<std::uint8_t> bad(16);
            // 1 מופיע פעמיים, ואין 0
            for (int i = 0; i < 16; ++i)
                bad[i] = (i == 0 ? 1 : static_cast<std::uint8_t>(i));
            puzzle15_state t(bad); // מצופה לזרוק
            (void)t;
        } catch (const std::exception&) { thrown = true; }
        assert(thrown && "expected exception on duplicate / missing 0");
    }

    // 4) ולידציה: ערך מחוץ לטווח (16)
    {
        bool thrown = false;
        try {
            std::vector<std::uint8_t> bad(16);
            for (int i = 0; i < 16; ++i) bad[i] = static_cast<std::uint8_t>(i);
            bad[0] = 16; // מחוץ לטווח 0..15
            puzzle15_state t(bad); // מצופה לזרוק
            (void)t;
        } catch (const std::exception&) { thrown = true; }
        assert(thrown && "expected exception on out-of-range (>=16)");
    }

    // 5) ולידציה: שתי אפסים (חסר ערך אחר)
    {
        bool thrown = false;
        try {
            std::vector<std::uint8_t> bad = {
                1,  2,  3,  4,
                5,  6,  7,  8,
                9, 10, 11, 12,
               13,  0,  0, 15 // שני אפסים, חסר 14
            };
            puzzle15_state t(bad);
            (void)t;
        } catch (const std::exception&) { thrown = true; }
        assert(thrown && "expected exception on duplicate 0 / missing tile");
    }

    std::cout << "[state] OK\n";
}

// ===== בדיקות עבור StpEnv =====
void RunStpEnvTests() {
    std::cout << "[env] start\n";
    StpEnv env;

    // 1) יעד: חוקיות מהלכים + Apply/Undo + IsGoal
    {
        puzzle15_state s; // יעד
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

    // 2) פינה=2, שפה=3, מרכז=4
    {
        puzzle15_state s; // יעד — נעביר לפינה (0,0)
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

    // 3) “חוקיות” דרך GetActions (לא נוגעים ב-IsLegal הפרטית)
    {
        puzzle15_state s;
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

    // 4) מסלול רנדומי + Undo מלא
    {
        std::mt19937 rng(999);
        puzzle15_state s; // יעד
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
