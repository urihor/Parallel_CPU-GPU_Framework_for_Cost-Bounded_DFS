//
// Created by Owner on 07/12/2025.
//
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "puzzle15_state.h"
#include "manhattan_15.h"
#include "neural_batch_service.h"

namespace neural15 {

    class NeuralDelta15 {
    public:
        // Singleton-style גישה כללית אחת בכל התהליך
        static NeuralDelta15& instance();

        // אתחול – לקרוא פעם אחת בתחילת התוכנית (main)
        // model_dir: תיקייה שבה נמצאים קבצי ה-pt וה-json
        // לדוגמה: "../models" או פשוט "." אם אתה מריץ מה-bin
        void initialize(const std::string& model_dir);

        // דלתא לפטרן 1-7 ו-8-15
        int delta_1_7(const puzzle15_state& s) const;
        int delta_8_15(const puzzle15_state& s) const;

        // גרסאות batch – מקבל וקטור של מצבים ומחזיר וקטור דלתאות
        std::vector<int> delta_1_7_batch(const std::vector<puzzle15_state>& states) const;
        std::vector<int> delta_8_15_batch(const std::vector<puzzle15_state>& states) const;

        // פונקציה נוחה שמחזירה h_M אם יש לך כבר Manhattan
        // (תוכל לחבר לפונקציית המנהטן הקיימת שלך מחוץ למחלקה)
        int h_M(const puzzle15_state& s, int manhattan_heuristic) const;

        std::vector<int> h_M_batch(const std::vector<puzzle15_state>& states) const;

    private:
        NeuralDelta15() = default;

        // לא מעתיקים את הסינגלטון
        NeuralDelta15(const NeuralDelta15&) = delete;
        NeuralDelta15& operator=(const NeuralDelta15&) = delete;

        // flags
        bool initialized_ = false;

        // Torch
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
    void init_default_batch_service();

} // namespace neural15
