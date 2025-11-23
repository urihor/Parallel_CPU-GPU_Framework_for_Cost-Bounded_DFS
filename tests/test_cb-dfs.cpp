//
// Created by uriel on 17/11/2025.
//
#include "test_cb-dfs.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <limits>

#include "work.h"
#include "cb-dfs.h"
#include "puzzle_env.h"
#include "puzzle15_state.h"
#include "pdb15.h"

// ---------------------------------------------------------------------
// Helper heuristic functions
// ---------------------------------------------------------------------

// Real PDB-based heuristic.
static int PdbHeuristic(const StpEnv::State& s) {
    return pdb15::heuristic_744_auto(s);
}

// Constant heuristic used only for testing threshold updates.
// Not admissible, but fine for unit tests.
static int OneHeuristic(const StpEnv::State& s) {
    (void)s;
    return 1;
}

static constexpr int INF_INT = std::numeric_limits<int>::max();

// ---------------------------------------------------------------------
// Helper to create a non-goal state one move away from the goal.
// ---------------------------------------------------------------------

static puzzle15_state make_single_move_from_goal(StpEnv& env) {
    puzzle15_state goal; // default ctor = goal state
    auto actions = env.GetActions(goal);
    assert(!actions.empty());

    puzzle15_state start = goal;
    env.ApplyAction(start, actions[0]);
    assert(!env.IsGoal(start));

    return start;
}

// ---------------------------------------------------------------------
// Test 1: Goal at the root.
//
// Setup:
//   - Single Work whose root state is already the goal.
//   - g(root) = 0, h(root) = 0 (PDB).
//   - bound = 0.
//   - work_num >= 1, but there is only one Work.
//
// Expected:
//   - CB_DFS immediately finds the goal.
//   - found == true.
//   - next_bound remains INF_INT (no pruned nodes).
// ---------------------------------------------------------------------
static void test_goal_found_at_root() {
    StpEnv env;
    puzzle15_state start; // goal state
    std::vector<StpEnv::Action> empty_path;

    WorkFor<StpEnv> w(start, empty_path);
    std::vector<WorkFor<StpEnv>> works;
    works.push_back(w);

    constexpr int bound = 0;
    int next_bound = 0;

    constexpr int work_num = 4; // more "slots" than works is fine

    bool found = batch_ida::CB_DFS(env, works, work_num, bound, &PdbHeuristic, next_bound);

    assert(found);
    assert(next_bound == INF_INT);

    std::cout << "[OK] test_goal_found_at_root\n";
}

// ---------------------------------------------------------------------
// Test 2: Non-goal start, one move from the goal, PDB heuristic.
//
// Setup:
//   - start is one legal move away from the goal.
//   - Single Work whose root is 'start' (g = 0).
//   - Real PDB heuristic: h(start) > 0, h(goal) = 0.
//   - We use a large bound so that no pruning occurs (e.g., bound = 100).
//
// Expected behavior:
//   - CB_DFS explores the subtree of this single Work.
//   - It eventually expands the child that is the goal.
//   - found == true.
//   - next_bound remains INF_INT (no node with f > bound was pruned).
// ---------------------------------------------------------------------
static void test_find_goal_one_move_from_goal() {
    StpEnv env;
    puzzle15_state start = make_single_move_from_goal(env);
    std::vector<StpEnv::Action> empty_path;

    WorkFor<StpEnv> w(start, empty_path);
    std::vector<WorkFor<StpEnv>> works;
    works.push_back(w);

    const int h_start = PdbHeuristic(start);
    assert(h_start > 0);

    constexpr int bound = 100; // large enough, no pruning expected
    int next_bound = 0;

    constexpr int work_num = 2;

    const bool found = batch_ida::CB_DFS(env, works, work_num, bound, &PdbHeuristic, next_bound);

    assert(found);

    std::cout << "[OK] test_find_goal_one_move_from_goal\n";
}

// ---------------------------------------------------------------------
// Test 3: Threshold update across multiple Works.
//
// Setup:
//   - Several Works, each with the goal state as root.
//   - Heuristic: OneHeuristic(s) = 1 for all states.
//   - bound = 0, so f(root) = 1 > bound for every Work.
//
// We set:
//   - works.size() >= 2
//   - work_num < works.size() to exercise the scheduler and refill logic.
//
// Expected behavior:
//   - No solution is reported (found == false), because f > bound everywhere.
//   - Every root node is pruned (f > bound).
//   - next_bound is updated to the minimal f > bound, which is 1.
// ---------------------------------------------------------------------
static void test_threshold_aggregated_across_works() {
    StpEnv env;
    puzzle15_state start; // goal state
    std::vector<StpEnv::Action> empty_path;

    std::vector<WorkFor<StpEnv>> works;

    // Create several identical Works.
    constexpr int num_works = 3;
    for (int i = 0; i < num_works; ++i) {
        works.emplace_back(start, empty_path);
    }

    constexpr int bound = 0; // f(root) = 1 > bound for all roots under OneHeuristic
    int next_bound = 0;

    constexpr int work_num = 2; // fewer slots than Works to test refill behavior

    bool found = batch_ida::CB_DFS(env, works, work_num, bound, &OneHeuristic, next_bound);

    assert(!found);
    assert(next_bound == 1);

    std::cout << "[OK] test_threshold_aggregated_across_works\n";
}

// ---------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------

void CBDfsTests::RunAll() {
    test_goal_found_at_root();
    test_find_goal_one_move_from_goal();
    test_threshold_aggregated_across_works();

    std::cout << "[OK] All CB_DFS tests passed.\n";
}
