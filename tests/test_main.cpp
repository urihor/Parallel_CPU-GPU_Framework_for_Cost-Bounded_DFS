//
// Created by Owner on 05/11/2025.
//


// test_main.cpp
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include "puzzle_env.h" // בשביל לייצר מצב "מהלך-אחד" בנוחות
#include "pdb15.h"
#include "puzzle15_state.h"

// הצהרות קדמיות (ההגדרות ב-tests.cpp)
void RunPuzzle15StateTests();
void RunStpEnvTests();

namespace fs = std::filesystem;

// חשב גודל-קובץ צפוי עבור k (מס' אריחים בתבנית)
static std::uint64_t expected_bytes_for_k(int k) {
    std::uint64_t n = pdb15::states_for_pattern(k); // P(16, k+1)
#if PDB_BITS == 8
    return n;
#else
    return (n + 1) / 2;
#endif
}

// בדיקת קובץ: קיים ובדיוק בגודל הצפוי
static bool file_ok(const fs::path& p, int k) {
    std::error_code ec;
    if (!fs::exists(p, ec)) return false;
    auto sz = fs::file_size(p, ec);
    if (ec) return false;
    return sz == expected_bytes_for_k(k);
}

// בנה 7/8 אם חסר/פגום; קבצים יכתבו תחת out_dir
static void ensure_78(const fs::path& out_dir) {
    fs::create_directories(out_dir);
    fs::path p7 = out_dir / "pdb_1_7.bin";
    fs::path p8 = out_dir / "pdb_8_15.bin";

    bool ok7 = file_ok(p7, 7);
    bool ok8 = file_ok(p8, 8);

    std::cout << "[ensure_78] output dir: " << fs::absolute(out_dir) << "\n";
    if (!ok7) {
        std::cout << "[ensure_78] building 7-PDB -> " << fs::absolute(p7) << "\n";
        pdb15::build_pdb_01bfs({1,2,3,4,5,6,7}, p7.string(), /*verbose=*/true);
    } else {
        std::cout << "[ensure_78] 7-PDB OK -> " << fs::absolute(p7) << "\n";
    }
    if (!ok8) {
        std::cout << "[ensure_78] building 8-PDB -> " << fs::absolute(p8) << "\n";
        pdb15::build_pdb_01bfs({8,9,10,11,12,13,14,15}, p8.string(), /*verbose=*/true);
    } else {
        std::cout << "[ensure_78] 8-PDB OK -> " << fs::absolute(p8) << "\n";
    }

    // נגדיר את הנתיבים לגרסת ה-auto
    pdb15::set_default_paths_78(p7.string(), p8.string());
}

// בנה 7/4/4 אם חסר/פגום; קבצים יכתבו תחת out_dir
static void ensure_744(const fs::path& out_dir) {
    fs::create_directories(out_dir);
    fs::path pA = out_dir / "pdb_1_7.bin";
    fs::path pB = out_dir / "pdb_8_11.bin";
    fs::path pC = out_dir / "pdb_12_15.bin";

    bool okA = file_ok(pA, 7);
    bool okB = file_ok(pB, 4);
    bool okC = file_ok(pC, 4);

    std::cout << "[ensure_744] output dir: " << fs::absolute(out_dir) << "\n";
    if (!okA) {
        std::cout << "[ensure_744] building A(1..7) -> " << fs::absolute(pA) << "\n";
        pdb15::build_pdb_01bfs({1,2,3,4,5,6,7}, pA.string(), /*verbose=*/true);
    } else {
        std::cout << "[ensure_744] A OK -> " << fs::absolute(pA) << "\n";
    }
    if (!okB) {
        std::cout << "[ensure_744] building B(8..11) -> " << fs::absolute(pB) << "\n";
        pdb15::build_pdb_01bfs({8,9,10,11}, pB.string(), /*verbose=*/true);
    } else {
        std::cout << "[ensure_744] B OK -> " << fs::absolute(pB) << "\n";
    }
    if (!okC) {
        std::cout << "[ensure_744] building C(12..15) -> " << fs::absolute(pC) << "\n";
        pdb15::build_pdb_01bfs({12,13,14,15}, pC.string(), /*verbose=*/true);
    } else {
        std::cout << "[ensure_744] C OK -> " << fs::absolute(pC) << "\n";
    }

    // נגדיר את הנתיבים לגרסת ה-auto
    pdb15::set_default_paths_744(pA.string(), pB.string(), pC.string());
}

static void demo_queries() {
    // מצב יעד
    puzzle15_state goal; // בנאי ברירת-מחדל = יעד
    // מצב מהלך-אחד (באמצעות הסביבה)
    StpEnv env;
    puzzle15_state one = goal;
    auto acts = env.GetActions(goal);
    if (!acts.empty()) env.ApplyAction(one, acts[0]);

    // מצב מותאם ידנית (הבנאי עושה ולידציה ומוצא blankPos)
    puzzle15_state custom({
        0,12,9,13,15,11,10,14,3,7,2,5,4,8,6,1
    });

    // --- 7/8 (auto) ---
    const int h78_goal = pdb15::heuristic_78_auto(goal);
    const int h78_one  = pdb15::heuristic_78_auto(one);
    const int h78_cus  = pdb15::heuristic_78_auto(custom);

    std::cout << "[78] h(goal)=" << h78_goal
              << "  h(one)="   << h78_one
              << "  h(custom)="<< h78_cus << "\n";

    // --- 7/4/4 (auto) ---
    const int h744_goal = pdb15::heuristic_744_auto(goal);
    const int h744_one  = pdb15::heuristic_744_auto(one);
    const int h744_cus  = pdb15::heuristic_744_auto(custom);

    std::cout << "[744] h(goal)=" << h744_goal
              << "  h(one)="    << h744_one
              << "  h(custom)=" << h744_cus << "\n";
}

int main() {
    try {
        // נכתוב לקבצים בתיקיית ה-working של הריצה (Debug/Release)
        const fs::path out_dir = fs::current_path();

        // בנה/בדוק 7/8
        ensure_78(out_dir);

        // בנה/בדוק 7/4/4
        ensure_744(out_dir);

        // הדגמת חישובי היוריסטיקה
        demo_queries();

        std::cout << "[done]\n";
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
