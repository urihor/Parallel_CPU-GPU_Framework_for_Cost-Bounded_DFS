//
// Created by Owner on 05/11/2025.
//


// test_main.cpp
#include <iostream>
#include <exception>
#include <filesystem>

#include "pdb15.h"
#include "test_generate_work.h"
#include "test_do_iteration.h"
#include "test_cb-dfs.h"
#include "test_ida_star_korf_examples.h"

// Declared in other test source files (tests.cpp, etc.)
void RunPuzzle15StateTests();
void RunStpEnvTests();
namespace fs = std::filesystem;
// Compute expected file size for k (number of tiles in the pattern)
static std::uint64_t expected_bytes_for_k(int k) {
    std::uint64_t n = pdb15::states_for_pattern(k); // P(16, k+1)
#if PDB_BITS == 8
    return n;
#else
    return (n + 1) / 2;
#endif
}

// Check that the file exists and has the exact expected size
static bool file_ok(const fs::path &p, int k) {
    std::error_code ec;
    if (!fs::exists(p, ec)) return false;
    auto sz = fs::file_size(p, ec);
    if (ec) return false;
    return sz == expected_bytes_for_k(k);
}

// Build 7/8 PDBs if missing/corrupted; files will be written under out_dir
static void ensure_78(const fs::path &out_dir) {
    fs::create_directories(out_dir);
    fs::path p7 = out_dir / "pdb_1_7.bin";
    fs::path p8 = out_dir / "pdb_8_15.bin";

    bool ok7 = file_ok(p7, 7);
    bool ok8 = file_ok(p8, 8);

    std::cout << "[ensure_78] output dir: " << fs::absolute(out_dir) << "\n";
    if (!ok7) {
        std::cout << "[ensure_78] building 7-PDB -> " << fs::absolute(p7) << "\n";
        pdb15::build_pdb_01bfs({1, 2, 3, 4, 5, 6, 7}, p7.string(), /*verbose=*/true);
    } else {
        std::cout << "[ensure_78] 7-PDB OK -> " << fs::absolute(p7) << "\n";
    }
    if (!ok8) {
        std::cout << "[ensure_78] building 8-PDB -> " << fs::absolute(p8) << "\n";
        pdb15::build_pdb_01bfs({8, 9, 10, 11, 12, 13, 14, 15}, p8.string(), /*verbose=*/true);
    } else {
        std::cout << "[ensure_78] 8-PDB OK -> " << fs::absolute(p8) << "\n";
    }

    // Configure the default auto-lookup paths for 7/8 PDBs
    pdb15::set_default_paths_78(p7.string(), p8.string());
}



int main() {
    try {
        // Build / verify the 7/8 PDBs in the current build directory
        const fs::path out_dir = fs::current_path();
        ensure_78(out_dir);
        std::cout << "== running assert-based tests ==\n";

        RunPuzzle15StateTests();
        RunStpEnvTests();

        GenerateWorkTests::RunAll();
        DoIterationTests::RunAll();
        CBDfsTests::RunAll();
        Korf100Tests::RunAll();

        std::cout << "[ALL TESTS OK]\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[TEST ERROR] " << ex.what() << "\n";
        return 1;
    }
}
