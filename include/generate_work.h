//
// Created by uriel on 11/11/2025.
//

#pragma once
#include <vector>
#include <unordered_set>
#include <cstddef>
#include "action_concepts.h"
#include "work.h"

// Basic version – without deduplication.
template<class Env>
void GenerateWork(Env &env, typename Env::State &s, int d_init,
                  std::vector<typename Env::Action> &history, std::vector<WorkFor<Env>> &works,
                  int& best_len, std::vector<typename Env::Action>& best_sol) {


    if (static_cast<int>(history.size()) == d_init) {
        works.push_back(WorkFor<Env>{s, history});
        return;
    }
    if (static_cast<int>(history.size()) >= best_len)
        return;

    auto actions = env.GetActions(s);
    for (typename Env::Action a: actions) {
        if constexpr (ActionHasInverse<typename Env::Action>) {
            if (!history.empty() && inverse(history.back()) == a)
                continue;
        }
        history.push_back(a);
        env.ApplyAction(s, a);

        if (env.IsGoal(s)) {
            int cur = static_cast<int>(history.size());
            if (cur < best_len) {
                best_len = cur;
                best_sol = history;
            }
        }

        GenerateWork(env, s, d_init, history, works, best_len, best_sol);

        env.UndoAction(s, a);
        history.pop_back();
    }
}

// Version with deduplication on the boundary layer.
// KeyFn: a function/lambda that takes (const State&) and returns a hashable key (std::size_t).
template<class Env, class KeyFn>
void GenerateWorkDedup(Env &env, typename Env::State &s, int d_init,
                       std::vector<typename Env::Action> &history, std::vector<WorkFor<Env> > &works,
                       std::unordered_set<std::size_t> &seen_at_boundary, KeyFn key_of,
                       int& best_len, std::vector<typename Env::Action>& best_sol) {
    if (static_cast<int>(history.size()) == d_init) {
        std::size_t key = key_of(s);
        if (seen_at_boundary.insert(key).second) {
            works.push_back(WorkFor<Env>{s, history});
        }
        return;
    }
    if (static_cast<int>(history.size()) >= best_len)
        return;

    auto actions = env.GetActions(s);
    for (typename Env::Action a: actions) {
        if constexpr (ActionHasInverse<typename Env::Action>) {
            if (!history.empty() && inverse(history.back()) == a) {
                continue;
            }
        }

        history.push_back(a);
        env.ApplyAction(s, a);

        if (env.IsGoal(s)) {
            int cur = static_cast<int>(history.size());
            if (cur < best_len) {
                best_len = cur;
                best_sol = history;
            }
        }
        GenerateWorkDedup(env, s, d_init, history, works, seen_at_boundary, key_of, best_len, best_sol);

        env.UndoAction(s, a);
        history.pop_back();

    }
}