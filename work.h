//
// Created by uriel on 11/11/2025.
//
#pragma once
#include <vector>


template<class State, class Action>
struct Work {
    State root; // the root of the subtree
    std::vector<Action> init; // the actions from the start state to root
};

template<class Env>
using WorkFor = Work<typename Env::State, typename Env::Action>;
