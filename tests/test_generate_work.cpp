#include "test_generate_work.h"

#include <cassert>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <cstdint>

#include "work.h"
#include "generate_work.h"
#include "puzzle_env.h"
#include "puzzle15_state.h"
#include "puzzle_actions.h" // inverse(StpMove)



static void test_dinit0_single_work() {
    // Arrange
    StpEnv env;
    puzzle15_state start; // default is the goal for 15-puzzle

    const int d_init = 0;
    std::vector<StpMove> history;
    std::vector<WorkFor<StpEnv>> works;

    int best_len = d_init;
    std::vector<StpMove> best_sol;

    // Act
    GenerateWork(env, start, d_init, history, works, best_len, best_sol);

    // Assert
    assert(works.size() == 1 && "With d_init=0 we must have exactly one Work");
    assert(works[0].init.empty() && "init must be empty at depth 0");
    assert(works[0].root.pack() == start.pack() && "root must equal start at depth 0");

    // No shallow solution should be found (we started at goal but d_init=0 causes boundary hit first)
    assert(best_len == d_init && "best_len should remain d_init when no shallow solution is recorded");
    assert(best_sol.empty() && "best_sol should remain empty when no shallow solution is recorded");
}

static void test_dinit1_branching_matches() {
    // Arrange
    StpEnv env;
    puzzle15_state start; // goal state; blank in bottom-right
    const auto actions = env.GetActions(start);
    const std::size_t branching = actions.size();

    const int d_init = 1;
    std::vector<StpMove> history;

    // no-dedup
    std::vector<WorkFor<StpEnv>> works_no_dedup;
    int best_len = d_init;
    std::vector<StpMove> best_sol;

    // dedup
    std::vector<WorkFor<StpEnv>> works_dedup;
    std::unordered_set<std::size_t> seen;

    // Act
    GenerateWork(env, start, d_init, history, works_no_dedup, best_len, best_sol);

    best_len = d_init; best_sol.clear();
    GenerateWorkDedup(env, start, d_init, history, works_dedup, seen,
        [](const puzzle15_state& s){ return std::hash<puzzle15_state>{}(s); },
        best_len, best_sol);

    // Assert
    assert(works_no_dedup.size() == branching && "d_init=1 must match legal branching factor");
    assert(works_dedup.size()    == branching && "dedup should be identical at depth 1");

    for (const auto& w : works_no_dedup) {
        assert(w.init.size() == 1 && "init length must be exactly 1 at depth 1");
        assert(w.root.pack() != start.pack() && "root must differ from start for any legal move");
    }
}

static void test_best_len_one_move_from_goal() {
    // Arrange: create a start that is exactly one move away from the goal
    StpEnv env;
    puzzle15_state start;               // goal
    {
        auto acts = env.GetActions(start);
        assert(!acts.empty() && "goal state must have at least one legal move");
        env.ApplyAction(start, acts[0]); // now distance(start, goal) == 1
    }

    const int d_init = 3;
    std::vector<StpMove> history;
    std::vector<WorkFor<StpEnv>> works;
    std::unordered_set<std::size_t> seen;

    int best_len = d_init;
    std::vector<StpMove> best_sol;

    // Act
    GenerateWorkDedup(env, start, d_init, history, works, seen,
        [](const puzzle15_state& s){ return std::hash<puzzle15_state>{}(s); },
        best_len, best_sol);

    // Assert
    assert(best_len == 1 && "best_len must be 1 when start is one move from the goal");
    assert(best_sol.size() == 1 && "best_sol must contain exactly one move");
}

static void test_inverse_pruning_no_immediate_inverse() {
    // Arrange
    StpEnv env;
    puzzle15_state start;
    const int d_init = 4;

    std::vector<StpMove> history;
    std::vector<WorkFor<StpEnv>> works;

    int best_len = d_init;
    std::vector<StpMove> best_sol;

    // Act
    GenerateWork(env, start, d_init, history, works, best_len, best_sol);

    // Assert: no adjacent (a, inverse(a)) inside init sequences
    for (const auto& w : works) {
        for (std::size_t i = 1; i < w.init.size(); ++i) {
            assert(inverse(w.init[i-1]) != w.init[i] && "immediate inverse moves must be pruned");
        }
    }
}

static void test_dedup_counts_and_uniqueness() {
    // Arrange: move one step from goal to increase branching variety
    StpEnv env;
    puzzle15_state start;
    {
        auto acts = env.GetActions(start);
        assert(!acts.empty());
        env.ApplyAction(start, acts[0]);
    }

    const int d_init = 2;
    std::vector<StpMove> history;

    // no-dedup
    std::vector<WorkFor<StpEnv>> works_no_dedup;
    int best_len = d_init;
    std::vector<StpMove> best_sol;
    GenerateWork(env, start, d_init, history, works_no_dedup, best_len, best_sol);

    // dedup
    best_len = d_init; best_sol.clear();
    std::vector<WorkFor<StpEnv>> works_dedup;
    std::unordered_set<std::size_t> seen;
    GenerateWorkDedup(env, start, d_init, history, works_dedup, seen,
        [](const puzzle15_state& s){ return std::hash<puzzle15_state>{}(s); },
        best_len, best_sol);

    // Assert: dedup should never create more works than no-dedup
    assert(works_dedup.size() <= works_no_dedup.size() && "dedup must not exceed no-dedup count");

    // Assert: all roots in dedup are unique
    std::unordered_set<std::uint64_t> uniq;
    for (const auto& w : works_dedup) {
        const bool inserted = uniq.insert(w.root.pack()).second;
        assert(inserted && "dedup must ensure unique boundary states");
    }
}

void GenerateWorkTests::RunAll() {
    test_dinit0_single_work();
    test_dinit1_branching_matches();
    test_best_len_one_move_from_goal();
    test_inverse_pruning_no_immediate_inverse();
    test_dedup_counts_and_uniqueness();

    std::cout << "[OK] All GenerateWork tests passed.\n";
}
