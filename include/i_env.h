//
// Created by uriel on 04/11/2025.
//

#pragma once
#include <vector>

template<class S, class A>
struct i_env {
    // כל הסרצ'ר עובד מול הממשק הזה בלבד:
    virtual std::vector<A> GetActions(const S& s) = 0; // מהלכים חוקיים ממצב s
    virtual void ApplyAction(S& s, A a) = 0;           // מבצע מהלך אחד (in-place)
    virtual void UndoAction(S& s, A a) = 0;            // מבטל מהלך
    virtual bool IsGoal(const S& s) const = 0;         // האם s הוא מטרה
    virtual bool IsSolvable(const S& s) const = 0;

    virtual ~i_env() = default;
};