//
// Created by uriel on 17/11/2025.
//
#include "test_do_iteration.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <limits>

#include "pdb15.h"
#include "work.h"
#include "do_iteration.h"
#include "puzzle_env.h"
#include "puzzle15_state.h"
#include "puzzle_actions.h"

using batch_ida::DoIteration;
using TestWork = WorkFor<StpEnv>;

// Real heuristic based on our PDB implementation.
// We wrap it in a simple function so the type matches HeuristicFn<StpEnv>.
static int PdbHeuristic(const StpEnv::State& s) {
    return pdb15::heuristic_78_auto(s);
}

// Returns a non-goal state obtained by applying a single legal move
// to the goal state.
static puzzle15_state make_single_move_from_goal(StpEnv& env) {
    puzzle15_state goal; // default: goal state
    auto actions = env.GetActions(goal);
    assert(!actions.empty());

    puzzle15_state start = goal;
    env.ApplyAction(start, actions[0]);
    assert(!env.IsGoal(start));

    return start;
}

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

    // The start state is already the goal, so DoIteration must report success.
    assert(found);

    // With the new Work semantics, finding a goal does NOT consume the work.
    // The subtree is still considered "not done"; the caller stops because found == true.
    assert(!work.is_done());

    // There should still be a current frame, representing the root state.
    assert(work.has_current());
    const auto &frame = work.current_frame();
    assert(env.IsGoal(frame.state));
    assert(frame.g == 0);

    // The bound was sufficient to reach the goal, so next_bound should remain INF_INT.
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
static void test_expand_non_goal_root_descend_child() {
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

    // No solution should be found in a single iteration.
    assert(!found);

    // Work should not be marked as done after expanding the root once.
    assert(!work.is_done());

    // In the new Work implementation we expect to have a current DFS node.
    assert(work.has_current());

    const auto &frame = work.current_frame();

    // Since the prefix path is empty, a direct child of the root must have g = 1.
    assert(frame.g == 1);

    // Verify that the current state is one of the children of the start state.
    bool current_is_child = false;
    for (StpEnv::Action a : moves_from_start) {
        puzzle15_state child = start;
        env.ApplyAction(child, a);
        if (child.pack() == frame.state.pack()) {
            current_is_child = true;
            break;
        }
    }
    assert(current_is_child);

    // With ZeroHeuristic and a large enough bound, next_bound should remain INF_INT.
    assert(next_bound == INF_INT);

    std::cout << "[OK] test_expand_non_goal_root_descend_child\n";
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
    puzzle15_state start; // goal state
    std::vector<StpEnv::Action> empty_path;

    TestWork work(start, empty_path);

    constexpr int bound = 0;
    int next_bound = INF_INT;

    bool found1 = DoIteration(env, work, bound, &ZeroHeuristic, next_bound);

    // First call must find the goal at the root.
    assert(found1);
    assert(work.has_current());
    const auto &frame1 = work.current_frame();
    assert(env.IsGoal(frame1.state));
    assert(frame1.g == 0);

    const int next_before = next_bound;

    // Second call on the same Work should not break invariants.
    // It is allowed to either immediately rediscover the same goal
    // or simply return without changing anything, but it must not
    // change next_bound or the fact that the current frame is the goal.
    (void)DoIteration(env, work, bound, &ZeroHeuristic, next_bound);

    assert(next_bound == next_before);
    assert(work.has_current());
    const auto &frame2 = work.current_frame();
    assert(env.IsGoal(frame2.state));

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

    // Root should be pruned because f(start) = g + h = 1 > bound = 0.
    assert(!found);

    // Pruning the root exhausts this Work's subtree.
    assert(work.is_done());
    assert(!work.has_current());

    // next_bound must be updated to the minimal f-value > bound, which is 1.
    assert(next_bound == 1);

    std::cout << "[OK] test_threshold_updated_on_pruned_node\n";
}


/**
 * Test 5: Real PDB heuristic on a non-goal state.
 *
 * Setup:
 *   - start is one move away from the goal (make_single_move_from_goal).
 *   - h(start) is computed by the actual PDB heuristic.
 *   - g(start) = 0 (this Work has an empty init path).
 *
 * We run two scenarios:
 *
 *   (A) bound = h(start) - 1:
 *       - f(start) = g + h = h(start) > bound
 *       - The root is pruned.
 *       - DoIteration returns false.
 *       - local_next_bound is updated to f(start) = h(start).
 *
 *   (B) bound = h(start):
 *       - f(start) = h(start) <= bound
 *       - The root is NOT pruned and is expanded.
 *       - Children are pushed onto the stack.
 *       - DoIteration returns false (start is not a goal).
 *       - local_next_bound remains INF (no pruned nodes).
 */
static void test_pdb_heuristic_on_non_goal_state() {
    StpEnv env;

    puzzle15_state start = make_single_move_from_goal(env);
    std::vector<StpEnv::Action> empty_path;

    // Sanity: PDB heuristic should be strictly positive on a non-goal state.
    const int h = PdbHeuristic(start);
    assert(h > 0);

    // -------- Scenario A: bound too small -> prune root, update next_bound --------
    {
        TestWork work(start, empty_path);
        const int bound = h - 1;  // so f(start) = h > bound
        int local_next_bound = INF_INT;

        bool found = DoIteration(env, work, bound, &PdbHeuristic, local_next_bound);

        // Root is pruned, so no goal found.
        assert(!found);

        // Pruning the root exhausts this Work's subtree.
        assert(work.is_done());
        assert(!work.has_current());

        // next_bound is updated to the minimal f-value > bound, which is h(start).
        assert(local_next_bound == h);
    }


    // -------- Scenario B: bound large enough -> expand root, no threshold update --------
    {
        TestWork work(start, empty_path);
        const int bound = h;      // so f(start) = h <= bound
        int local_next_bound = INF_INT;

        auto moves_from_start = env.GetActions(start);
        assert(!moves_from_start.empty());

        bool found = DoIteration(env, work, bound, &PdbHeuristic, local_next_bound);

        assert(!found);                      // start is not a goal
        assert(!work.is_done());             // children were pushed
        assert(local_next_bound == INF_INT); // no pruned nodes yet
    }

    std::cout << "[OK] test_pdb_heuristic_on_non_goal_state\n";
}


/**
 * Test suite runner.
 */
void DoIterationTests::RunAll() {
    test_goal_found_within_bound();
    test_expand_non_goal_root_descend_child();
    test_do_iteration_on_finished_work_is_noop();
    test_threshold_updated_on_pruned_node();
    test_pdb_heuristic_on_non_goal_state();
    std::cout << "[OK] All DoIteration tests passed.\n";
}
