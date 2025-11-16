//
// Work descriptor and DFS stack for Batch IDA* / CB-DFS.
// Updated to support Algorithm 4 (DoIteration).
//
#pragma once
#include <vector>
#include <cstddef>
#include <cassert>

// Work represents a single subtree for CB-DFS / Batch IDA*.
//
// Template parameters:
//  - State  : problem-specific state type (e.g., puzzle15_state)
//  - Action : problem-specific action type (e.g., StpMove)
template<class State, class Action>
struct Work {
    // A single node kept on the DFS stack for this subtree.
    struct Node {
        State state{}; // copy of the concrete state
        int g = 0; // cost from the global start state to 'state'
    };

    // Fixed descriptor of the subtree:
    //  - 'root' is the root state of the subtree.
    //  - 'init' are the actions from the *problem start* to 'root'.
    State root; // subtree root state
    std::vector<Action> init; // path (actions) from global start to root

    Work() = default;

    Work(const State &root_state, const std::vector<Action> &init_actions)
        : root(root_state),
          init(init_actions),
          stack_(),
          initialized_(false),
          done_(false) {
    }

    // Has this subtree finished exploring all of its nodes?
    [[nodiscard]] bool is_done() const noexcept {
        return done_ || (initialized_ && stack_.empty());
    }

    // Ensure that the DFS stack contains at least the root node.
    // This is called lazily from the search code so GenerateWork
    // does not need to worry about search-specific initialization.
    void ensure_initialized() {
        if (!initialized_) {
            stack_.clear();
            Node root_node;
            root_node.state = root;
            root_node.g = static_cast<int>(init.size());
            stack_.push_back(root_node);
            initialized_ = true;
            done_ = false;
        }
    }

    // Number of nodes currently stored in the internal DFS stack.
    [[nodiscard]] std::size_t stack_size() const noexcept {
        return initialized_ ? stack_.size() : 0;
    }

    // Peek at the node currently on top of the DFS stack (without popping).
    [[nodiscard]] const Node &top_node() const {
        assert(initialized_ && !stack_.empty());
        return stack_.back();
    }

    // Pop the node currently on top of the DFS stack and return it by value.
    Node pop_node() {
        assert(initialized_ && !stack_.empty());
        Node node = stack_.back();
        stack_.pop_back();
        if (stack_.empty()) {
            done_ = true;
        }
        return node;
    }

    // Push a new node (state + g-cost) onto the DFS stack.
    void push_node(const State &s, int g) {
        ensure_initialized();
        Node child;
        child.state = s;
        child.g = g;
        stack_.push_back(child);
    }

private:
    std::vector<Node> stack_;
    bool initialized_ = false;
    bool done_ = false;
};

// Convenience alias: a Work for a concrete Env.
template<class Env>
using WorkFor = Work<typename Env::State, typename Env::Action>;
