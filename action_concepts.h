//
// Created by Owner on 12/11/2025.
//
#pragma once
#include <concepts>

// חוסם/מאפשר קומפילציה אם קיימת פונקציה inverse(Action) שמחזירה Action
template<class Action>
concept ActionHasInverse = requires(Action a) {
    { inverse(a) } -> std::same_as<Action>;
};
