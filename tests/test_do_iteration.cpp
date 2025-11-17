//
// Created by uriel on 17/11/2025.
//
#include "test_do_iteration.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <limits>

#include "work.h"
#include "do_iteration.h"
#include "puzzle_env.h"
#include "puzzle15_state.h"
#include "puzzle_actions.h"

using batch_ida::DoIteration;
using TestWork = WorkFor<StpEnv>;

static int ZeroHeuristic(const StpEnv::State& s) {
    (void)s;
    return 0;
}

// Deliberately non-admissible constant heuristic, used only to test
// threshold-updating behavior (f > bound → next_bound updated).
static int OneHeuristic(const StpEnv::State& s) {
    (void)s;
    return 1;
}

static constexpr int INF_INT = std::numeric_limits<int>::max();

/**
 * Test 1: Goal detection within the current threshold.
 *
 * Setup:
 *   - start state is the goal (default-constructed puzzle15_state).
 *   - g(start) = 0
 *   - h(start) = 0  (ZeroHeuristic)
 *   - bound = 0     → f(start) = 0 <= bound
 *
 * Expected behavior:
 *   - DoIteration should recognize the goal and return true.
 *   - The subtree should be finished (no nodes left on the stack).
 *   - next_bound should remain unchanged (INF_INT), since no node with f > bound was pruned.
 */
static void test_goal_found_within_bound() {
    StpEnv env;
    puzzle15_state start; // goal state
    std::vector<StpEnv::Action> empty_path;

    TestWork work(start, empty_path);

    constexpr int bound = 0;
    int next_bound = INF_INT;

    const bool found = DoIteration(env, work, bound, &ZeroHeuristic, next_bound);

    assert(found);
    assert(work.is_done());
    assert(work.stack_size() == 0);
    assert(next_bound == INF_INT);

    std::cout << "[OK] test_goal_found_within_bound\n";
}

/**
 * Test 2: Expanding a non-goal root and pushing its children.
 *
 * Setup:
 *   - start is one move away from the goal (apply a single legal move from the goal).
 *   - start is not a goal state.
 *   - bound is large enough so that f(start) = 0 < bound (ZeroHeuristic).
 *
 * Expected behavior:
 *   - DoIteration returns false (no goal found).
 *   - The root node is expanded once.
 *   - All legal successors of 'start' are pushed onto the DFS stack.
 *   - stack_size() equals the branching factor from 'start'.
 *   - The top node on the stack is one of these successors, with g = 1.
 */
static void test_expand_non_goal_root_push_children() {
    StpEnv env;
    puzzle15_state goal; // goal state

    auto moves_from_goal = env.GetActions(goal);
    assert(!moves_from_goal.empty());

    // Create a start state that is one move away from the goal.
    puzzle15_state start = goal;
    env.ApplyAction(start, moves_from_goal[0]);
    assert(!env.IsGoal(start));

    std::vector<StpEnv::Action> empty_path;
    TestWork work(start, empty_path);

    constexpr int bound = 10;
    int next_bound = INF_INT;

    auto moves_from_start = env.GetActions(start);
    assert(!moves_from_start.empty());

    bool found = DoIteration(env, work, bound, &ZeroHeuristic, next_bound);

    assert(!found);
    assert(!work.is_done());
    assert(work.stack_size() == moves_from_start.size());

    const auto& top = work.top_node();
    assert(top.g == 1);

    bool top_is_child = false;
    for (StpEnv::Action a : moves_from_start) {
        puzzle15_state child = start;
        env.ApplyAction(child, a);
        if (child.pack() == top.state.pack()) {
            top_is_child = true;
            break;
        }
    }
    assert(top_is_child);
    assert(next_bound == INF_INT);

    std::cout << "[OK] test_expand_non_goal_root_push_children\n";
}

/**
 * Test 3: Calling DoIteration on a finished Work is a no-op.
 *
 * Setup:
 *   - First call to DoIteration is on a goal state within bound,
 *     so it returns true and empties the stack.
 *   - Second call is on the same Work, which is already done.
 *
 * Expected behavior:
 *   - The first call returns true and finishes the subtree.
 *   - The second call returns false and does not change the Work.
 *   - next_bound must remain unchanged across the second call.
 */
static void test_do_iteration_on_finished_work_is_noop() {
    StpEnv env;
    puzzle15_state start; // goal
    std::vector<StpEnv::Action> empty_path;

    TestWork work(start, empty_path);
    constexpr int bound = 0;
    int next_bound = INF_INT;

    bool found1 = DoIteration(env, work, bound, &ZeroHeuristic, next_bound);
    assert(found1);
    assert(work.is_done());
    assert(work.stack_size() == 0);

    const int next_before = next_bound;

    bool found2 = DoIteration(env, work, bound, &ZeroHeuristic, next_bound);
    assert(!found2);
    assert(work.is_done());
    assert(work.stack_size() == 0);
    assert(next_bound == next_before);

    std::cout << "[OK] test_do_iteration_on_finished_work_is_noop\n";
}

/**
 * Test 4: Threshold update on pruned nodes.
 *
 * Setup:
 *   - start is the goal state.
 *   - heuristic(s) = 1 (OneHeuristic) for all s.
 *   - bound = 0 → f(start) = g + h = 0 + 1 = 1 > bound.
 *
 * Expected behavior:
 *   - start is pruned because f > bound.
 *   - DoIteration returns false (the goal is not considered a solution
 *     within this bound because it violates the f <= bound condition).
 *   - next_bound is updated to f(start) = 1.
 *
 * Note: This test uses a non-admissible heuristic on purpose to validate
 *       the threshold-update behavior. In real use, h(goal) should be 0.
 */
static void test_threshold_updated_on_pruned_node() {
    StpEnv env;
    puzzle15_state start; // goal state
    std::vector<StpEnv::Action> empty_path;

    TestWork work(start, empty_path);
    constexpr int bound = 0;
    int next_bound = INF_INT;

    bool found = DoIteration(env, work, bound, &OneHeuristic, next_bound);

    assert(!found);
    assert(work.is_done());
    assert(work.stack_size() == 0);
    assert(next_bound == 1);

    std::cout << "[OK] test_threshold_updated_on_pruned_node\n";
}

/**
 * Test suite runner.
 */
void DoIterationTests::RunAll() {
    test_goal_found_within_bound();
    test_expand_non_goal_root_push_children();
    test_do_iteration_on_finished_work_is_noop();
    test_threshold_updated_on_pruned_node();
    std::cout << "[OK] All DoIteration tests passed.\n";
}
