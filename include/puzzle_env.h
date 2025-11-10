//
// Created by uriel on 04/11/2025.
//

#pragma once
#include <vector>
#include "i_env.h"
#include "puzzle15_state.h"
#include "puzzle_actions.h"

struct StpEnv final : i_env<puzzle15_state, StpMove> {
    using State  = puzzle15_state;
    using Action = StpMove;

    // GetActions מחזיר את המהלכים החוקיים לפי מיקום החור
    std::vector<Action> GetActions(const State& s) override;

    // מפעיל מהלך אחד: מזיז את החור לשכן הרלוונטי
    void ApplyAction(State& s, Action a) override;

    // מבטל מהלך: מפעיל את המהלך ההופכי
    void UndoAction(State& s, Action a) override;

    // בדיקת מטרה מהירה ללא הקצאות
    [[nodiscard]] bool IsGoal(const State& s) const override;


private:

    // אינדקס השכן שמחליפים איתו (מניח שהמהלך חוקי)
    [[nodiscard]] static puzzle15_state::Index NeighborIndex(const State& s, Action a);
};
