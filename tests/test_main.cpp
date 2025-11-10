//
// Created by Owner on 05/11/2025.
//


// test_main.cpp
#include <iostream>
#include <filesystem>
#include "pdb15.h"
#include "puzzle15_state.h"
#include "puzzle_env.h"   // בשביל לייצר מצב מהלך-אחד בנוחות

// הצהרות קדמיות (ההגדרות ב-tests.cpp)
void RunPuzzle15StateTests();
void RunStpEnvTests();



namespace fs = std::filesystem;

// קבצי ה-7/8
static const char* PDB7_PATH = "pdb_1_7.bin";
static const char* PDB8_PATH = "pdb_8_15.bin";

// תבניות 7/8 זרות (דיסג'וינט)
static const pdb15::Pattern PAT7 = {1,2,3,4,5,6,7};
static const pdb15::Pattern PAT8 = {8,9,10,11,12,13,14,15};

static void build_78_if_needed() {
    const bool have7 = fs::exists(PDB7_PATH);
    const bool have8 = fs::exists(PDB8_PATH);

    if (have7 && have8) {
        std::cout << "[build] 7/8 PDBs already exist, skipping build.\n";
        return;
    }

    std::cout << "[build] Building 7/8 PDBs (default 8-bit entries)...\n";
    if (!have7) {
        pdb15::build_pdb_01bfs(PAT7, PDB7_PATH, /*verbose=*/true);
        std::cout << "[build] 7-PDB done -> " << PDB7_PATH << "\n";
    } else {
        std::cout << "[build] 7-PDB exists -> " << PDB7_PATH << "\n";
    }

    if (!have8) {
        // שים לב: זה גדול מאוד (≈3.87 GiB ב-8bit, ≈1.93 GiB ב-4bit)
        pdb15::build_pdb_01bfs(PAT8, PDB8_PATH, /*verbose=*/true);
        std::cout << "[build] 8-PDB done -> " << PDB8_PATH << "\n";
    } else {
        std::cout << "[build] 8-PDB exists -> " << PDB8_PATH << "\n";
    }
}

static void demo_query_78() {
    // טוענים את הטבלאות מהדיסק
    auto pdb7 = pdb15::load_pdb_from_file(PDB7_PATH, /*k=*/static_cast<int>(PAT7.size())); // k=7
    auto pdb8 = pdb15::load_pdb_from_file(PDB8_PATH, /*k=*/static_cast<int>(PAT8.size())); // k=8

    // מכינים שני מצבים: יעד, ומהלך-אחד מהיעד
    puzzle15_state goal;
    // מלא ידנית את יעד למקרה שברירת־מחדל איננה יעד (אפשר להשאיר אם כבר יעד):
    goal.tiles = {  1,  2,  3,  4,
                    5,  6,  7,  8,
                    9, 10, 11, 12,
                   13, 14, 15,  0 };
    goal.blankPos = 15;

    // מייצרים מצב מהלך-אחד חוקי ע"י שימוש בסביבה
    StpEnv env;
    auto acts = env.GetActions(goal);
    puzzle15_state one = goal;
    if (!acts.empty()) {
        env.ApplyAction(one, acts[0]); // יישום המהלך הראשון (כל מהלך חוקי יספיק להדגמה)
    }
    // 3) מצב שנבנה עם הבנאי (initializer-list)
    puzzle15_state custom({
    0,12,10,13,15,11,14,9,3,7,2,5,4,8,6,1  // שים לב: ה-0 כאן, הבנאי יזהה את מיקום החור לבד
    });

    // חישוב היוריסטיקה האדיטיבית: h = h7 + h8
    const int h_goal = pdb15::additive_heuristic(goal, {{&pdb7, PAT7}, {&pdb8, PAT8}});
    const int h_one  = pdb15::additive_heuristic(one,  {{&pdb7, PAT7}, {&pdb8, PAT8}});
    const int h_custom = pdb15::additive_heuristic(custom, {{&pdb7, PAT7}, {&pdb8, PAT8}});

    std::cout << "[query] h(goal) = " << h_goal << "\n";
    std::cout << "[query] h(one-move) = " << h_one  << "\n";
    std::cout << "[query] h(custom) = " << h_custom << "\n";
}

int main() {
    try {
        build_78_if_needed();
        demo_query_78();
    } catch (const std::exception& ex) {
        std::cerr << "[error] " << ex.what() << "\n";
        return 1;
    }
    std::cout << "== running assert-based tests ==\n";
    RunPuzzle15StateTests();
    RunStpEnvTests();
    std::cout << "[ALL OK]\n";
    return 0;
}
