//
// Created by uriel on 11/11/2025.
//

#pragma once
#include <vector>
#include <unordered_set>
#include <cstdint>
#include <type_traits>
#include "work.h"

// Basic version – without deduplication.
template<class Env>
void GenerateWork(Env &env,
                  typename Env::State &s,
                  int d_init,
                  std::vector<typename Env::Action> &history,
                  std::vector<WorkFor<Env> > &works) {
    using State = Env::State;
    using Action = Env::Action;

    if (static_cast<int>(history.size()) == d_init) {
        works.push_back(Work<State, Action>{s, history});
        return;
    }

    auto actions = env.GetActions(s);
    for (Action a: actions) {
        history.push_back(a);
        env.ApplyAction(s, a);

        GenerateWork(env, s, d_init, history, works);

        env.UndoAction(s, a);
        history.pop_back();
    }

}

// Version with deduplication on the boundary layer.
// KeyFn: a function/lambda that takes (const State&) and returns a hashable key (std::size_t).
template<class Env, class KeyFn>
void GenerateWorkDedup(Env &env,
                       typename Env::State &s,
                       int d_init,
                       std::vector<typename Env::Action> &history,
                       std::vector<WorkFor<Env> > &works,
                       std::unordered_set<std::size_t> &seen_at_boundary,
                       KeyFn key_of) {
    using State = Env::State;
    using Action = Env::Action;

    if (static_cast<int>(history.size()) == d_init) {
        std::size_t key = key_of(s);
        if (seen_at_boundary.insert(key).second) {
            works.push_back(Work<State, Action>{s, history});
        }
        return;
    }

    auto actions = env.GetActions(s);
    for (Action a: actions) {
        history.push_back(a);
        env.ApplyAction(s, a);

        GenerateWorkDedup(env, s, d_init, history, works, seen_at_boundary, key_of);

        env.UndoAction(s, a);
        history.pop_back();
    }
}
