//
// Created by Owner on 12/11/2025.
//

#pragma once
#include <concepts>

// check if inverse function exist
template<class Action>
concept ActionHasInverse = requires(Action a)
{
    { inverse(a) } -> std::same_as<Action>;
};
