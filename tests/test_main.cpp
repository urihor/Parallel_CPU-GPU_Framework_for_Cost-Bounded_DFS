#include <gtest/gtest.h>
#include <filesystem>
#include <cstdlib>

#include "pdb15.h"
#include "test_generate_work.h"
#include "test_do_iteration.h"
#include "test_cb-dfs.h"
#include "test_ida_star_korf_examples.h" // Korf's 100 benchmark instances

void RunPuzzle15StateTests();
void RunStpEnvTests();

namespace fs = std::filesystem;

static bool try_configure_pdb78_from_env_or_cwd() {
    const char* p7 = std::getenv("PDB_1_7");
    const char* p8 = std::getenv("PDB_8_15");

    fs::path f7 = p7 ? fs::path(p7) : (fs::current_path() / "pdb_1_7.bin");
    fs::path f8 = p8 ? fs::path(p8) : (fs::current_path() / "pdb_8_15.bin");

    std::error_code ec;
    if (!fs::exists(f7, ec) || !fs::exists(f8, ec)) return false;

    pdb15::set_default_paths_78(f7.string(), f8.string());
    return true;
}

static bool have_pdb78() {
    static bool ok = try_configure_pdb78_from_env_or_cwd();
    return ok;
}

TEST(Puzzle15State, LegacySuite) { RunPuzzle15StateTests(); }
TEST(StpEnv,        LegacySuite) { RunStpEnvTests(); }
TEST(GenerateWork,  LegacySuite) { GenerateWorkTests::RunAll(); }

// PDB-dependent tests -> we do not build PDBs in CI, only run if files already exist
TEST(DoIteration, RequiresPdb78) {
    if (!have_pdb78()) GTEST_SKIP() << "Missing PDB_1_7/PDB_8_15 (no PDB generation in tests).";
    DoIterationTests::RunAll();
}

TEST(CBDfs, RequiresPdb78) {
    if (!have_pdb78()) GTEST_SKIP() << "Missing PDB_1_7/PDB_8_15 (no PDB generation in tests).";
    CBDfsTests::RunAll();
}

// Also PDB-dependent and very slow -> disabled by default
TEST(Korf100, DISABLED_RequiresPdb78AndSlow) {
    if (!have_pdb78()) GTEST_SKIP() << "Missing PDB_1_7/PDB_8_15.";
    Korf100Tests::RunAll();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
