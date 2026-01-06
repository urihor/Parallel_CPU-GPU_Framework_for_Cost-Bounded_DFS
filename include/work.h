#pragma once

#include <vector>
#include <cassert>
#include <cstddef>
#include <cstdint>

template<class State, class Action>
class Work {
public:
    /**
     * A single frame on the DFS path from this Work's root to the
     * current node.
     */
    struct Frame {
        State state{}; // state at this depth
        int g = 0; // depth from global start

        // Children of this node (actions). Filled once when the node
        // is expanded for the first time.
        std::vector<Action> actions;
        std::size_t next_child_index = 0; // which child to generate next

        bool expanded = false; // has this node been "expanded" already?

        // For reconstructing the solution path:
        Action action_from_parent{}; // action taken from parent to reach this node
        bool has_parent = false; // false for the root frame
        bool h_cached_ready = false; // was a NN value computed & cached?
        int h_cached_value = 0; // cached heuristic value (from NN)
        std::uint64_t h_cached_key = 0; // packed state key that this value corresponds to
    };

    // These are used by GenerateWork / BatchIDA externally.
    State root{}; // root state of this subtree
    std::vector<Action> init; // prefix of actions from global start to root

    Work() = default;

    Work(const State &root_state, const std::vector<Action> &prefix)
        : root(root_state)
          , init(prefix) {
    }

    /// Reset all per-iteration search state (called at the start of each IDA* iteration).
    void reset_for_new_iteration() noexcept {
        path_.clear();
        goal_found_ = false;
        solution_suffix_.clear();
        initialized_ = false;
        expanded_nodes_ = 0;
    }

    /// A Work is considered "done" when its DFS path is empty
    /// (and it has been initialized at least once).
    [[nodiscard]] bool is_done() const noexcept {
        return initialized_ && path_.empty();
    }


    /// Was a goal found somewhere in this subtree?
    [[nodiscard]] bool goal_found() const noexcept {
        return goal_found_;
    }

    /// Solution depth (g) for this Work (including init prefix), or -1 if no goal.
    [[nodiscard]] int goal_solution_depth() const noexcept {
        if (!goal_found_) {
            return -1;
        }
        return static_cast<int>(init.size() + solution_suffix_.size());
    }

    /// Reconstruct the full action sequence from global start to this Work's goal.
    template<class ActionContainer>
    void reconstruct_full_path(ActionContainer &out) const {
        assert(goal_found_);
        out.clear();

        // 1) prefix from the global root to this Work's root
        out.insert(out.end(), init.begin(), init.end());

        // 2) suffix from this Work's root to the goal (collected when goal was found)
        out.insert(out.end(), solution_suffix_.begin(), solution_suffix_.end());
    }

    /// Number of nodes that were expanded in this Work (across all DoIteration calls).
    [[nodiscard]] std::uint64_t expanded_nodes() const noexcept {
        return expanded_nodes_;
    }

    // ----------------- Internal operations used by DoIteration -----------------

    /// Ensure the DFS path is initialized with the subtree root.
    void ensure_initialized() {
        if (initialized_) {
            return;
        }

        path_.clear();

        Frame root_frame;
        root_frame.state = root;
        root_frame.g = static_cast<int>(init.size());
        root_frame.has_parent = false;
        root_frame.expanded = false;
        root_frame.next_child_index = 0;
        // actions is initially empty; it will be filled when this node is expanded.

        path_.push_back(std::move(root_frame));
        initialized_ = true;
    }

    /// Is there a current node on the DFS path?
    [[nodiscard]] bool has_current() const noexcept {
        return initialized_ && !path_.empty();
    }

    /// Access the current (deepest) frame on the DFS path.
    Frame &current_frame() {
        assert(initialized_ && !path_.empty());
        return path_.back();
    }

    const Frame &current_frame() const {
        assert(initialized_ && !path_.empty());
        return path_.back();
    }

    /// Pop the current frame (backtrack one level).
    void pop_frame() {
        assert(initialized_ && !path_.empty());
        path_.pop_back();
    }

    /// Push a child frame corresponding to successor state reached via action 'a'.
    void push_child(const State &child_state, int child_g, const Action &a) {
        Frame child;
        child.state = child_state;
        child.g = child_g;
        child.action_from_parent = a;
        child.has_parent = true;
        child.expanded = false;
        child.next_child_index = 0;
        // 'actions' will be filled when we expand this child.
        path_.push_back(std::move(child));
    }

    /// Mark the current frame as a goal and record the solution suffix.
    void mark_goal_current() {
        assert(initialized_ && !path_.empty());
        goal_found_ = true;

        solution_suffix_.clear();
        // Collect actions along the path, skipping the root (has_parent == false).
        for (const Frame &f: path_) {
            if (f.has_parent) {
                solution_suffix_.push_back(f.action_from_parent);
            }
        }
    }

    /// Increment the counter of expanded nodes (called by DoIteration).
    void increment_expanded() noexcept {
        ++expanded_nodes_;
    }

private:
    // Current DFS path from this Work's root to the node being processed.
    std::vector<Frame> path_;

    bool initialized_ = false;
    bool goal_found_ = false;

    // Actions from this Work's root to the goal, filled when mark_goal_current() is called.
    std::vector<Action> solution_suffix_;

    // Number of nodes expanded in this Work across all calls.
    std::uint64_t expanded_nodes_ = 0;
};

// Convenience alias:
template<class Env>
using WorkFor = Work<typename Env::State, typename Env::Action>;
