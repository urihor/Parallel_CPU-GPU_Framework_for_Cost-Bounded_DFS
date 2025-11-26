//
// Work descriptor and DFS stack for Batch IDA* / CB-DFS.
// Updated to support Algorithm 4 (DoIteration).
//
#pragma once

#include <vector>
#include <cassert>
#include <algorithm>

template<class State, class Action>
class Work {
public:
    struct Node {
        State state{};
        int g = 0;                // depth from global start
        int parent = -1;          // index in nodes_, -1 for the root
        Action action_from_parent{}; // valid if parent != -1
    };

    // Public for compatibility with existing code (GenerateWork, etc.)
    State root{};
    std::vector<Action> init; // prefix from global start to this subtree root

    Work() = default;

    Work(const State& root_state, const std::vector<Action>& init_actions)
        : root(root_state),
          init(init_actions)
    {}

    /// True if this Work has been initialized and its DFS stack is empty.
    bool is_done() const noexcept {
        return initialized_ && stack_.empty();
    }

    /// Initialize the DFS stack for this Work (if not already initialized).
    /// The root node's g-value is set to init.size(), so g is the depth from
    /// the global start state.
    void ensure_initialized() {
        if (initialized_) {
            return;
        }
        nodes_.clear();
        stack_.clear();

        Node root_node;
        root_node.state  = root;
        root_node.g      = static_cast<int>(init.size());
        root_node.parent = -1; // no parent
        nodes_.push_back(root_node);

        const int root_index = 0;
        stack_.push_back(root_index);
        current_node_index_ = root_index;

        initialized_ = true;
    }

    /// Reset this Work so it can be reused in a new IDA* iteration.
    /// Keeps the logical root (root, init) but clears all search state.
    void reset_for_new_iteration() noexcept {
        // Next call to ensure_initialized() will rebuild the root node & stack.
        initialized_ = false;

        nodes_.clear();
        stack_.clear();

        current_node_index_ = -1;
        goal_found_ = false;
        goal_node_index_ = -1;
    }


    std::size_t stack_size() const noexcept {
        return initialized_ ? stack_.size() : 0;
    }

    /// Return a const reference to the node at the top of the DFS stack.
    /// This is mainly useful for tests and debugging.
    const Node& top_node() const {
        assert(initialized_);
        assert(!stack_.empty());
        int idx = stack_.back();
        return nodes_[idx];
    }


    /// Pop the top node index from the DFS stack, remember it as the current
    /// node, and return a copy of that Node.
    Node pop_node() {
        assert(initialized_ && !stack_.empty());
        const int idx = stack_.back();
        stack_.pop_back();

        current_node_index_ = idx;
        return nodes_[idx];
    }

    /// Push a child node onto this Work's DFS stack.
    ///
    /// parent_index must be a valid index in nodes_ (the index of the node
    /// from which this child was generated), and 'a' is the Action applied
    /// from the parent to get to 's'.
    void push_node(const State& s, int g, int parent_index, const Action& a) {
        assert(initialized_);
        assert(parent_index >= 0 && parent_index < static_cast<int>(nodes_.size()));

        Node child;
        child.state = s;
        child.g = g;
        child.parent = parent_index;
        child.action_from_parent = a;

        nodes_.push_back(child);
        const int idx = static_cast<int>(nodes_.size()) - 1;
        stack_.push_back(idx);
    }

    /// Convenience overload: use the last popped node as parent.
    void push_node_from_current(const State& s, int g, const Action& a) {
        assert(current_node_index_ >= 0);
        push_node(s, g, current_node_index_, a);
    }

    /// Mark the current node (the one last returned by pop_node()) as a goal.
    void mark_goal_current() {
        assert(current_node_index_ >= 0);
        goal_found_ = true;
        goal_node_index_ = current_node_index_;
    }

    /// Has this Work found a goal node?
    bool goal_found() const noexcept {
        return goal_found_;
    }

    /// Return the solution depth (g-value) of the goal node in this Work,
    /// or -1 if no goal was recorded.
    int goal_solution_depth() const noexcept {
        if (!goal_found_ || goal_node_index_ < 0) {
            return -1;
        }
        return nodes_[goal_node_index_].g;
    }

    /// Reconstruct the full action sequence from the global start state
    /// to this Work's goal node:
    ///   - prefix = init
    ///   - then actions from the root of this Work down to the goal node.
    template<class ActionContainer>
    void reconstruct_full_path(ActionContainer& out) const {
        assert(goal_found_);
        assert(goal_node_index_ >= 0);

        out.clear();

        // 1) prefix: actions from the global root to this Work's root.
        out.insert(out.end(), init.begin(), init.end());

        // 2) actions from this Work's root to the goal, recovered via parents.
        std::vector<Action> tail;
        int idx = goal_node_index_;
        while (idx >= 0) {
            const Node& n = nodes_[idx];
            if (n.parent < 0) {
                // Reached the root of this Work.
                break;
            }
            tail.push_back(n.action_from_parent);
            idx = n.parent;
        }

        // Now 'tail' holds actions from goal back to Work root; reverse it.
        std::reverse(tail.begin(), tail.end());
        out.insert(out.end(), tail.begin(), tail.end());
    }

private:
    std::vector<Node> nodes_;  // all nodes in this Work's search tree
    std::vector<int>  stack_;  // DFS stack: indices into nodes_
    bool initialized_ = false;
    int  current_node_index_ = -1;
    bool goal_found_      = false;
    int  goal_node_index_ = -1;
};

template<class Env>
using WorkFor = Work<typename Env::State, typename Env::Action>;
