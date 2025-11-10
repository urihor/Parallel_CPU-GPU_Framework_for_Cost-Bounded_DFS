// ===============================
// File: pdb15.h
// ===============================
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <cassert>
#include <cstddef>
#include <filesystem>
#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "puzzle15_state.h"

// Public API for building & querying disjoint PDBs for the 15‑puzzle.
// - No command-line; call functions directly from your code.
// - Accepts/returns puzzle15_state for lookups.
// - Supports nibble-packed (4-bit) or byte (8-bit) storage via PDB_BITS.
//
// Notes:
// * Pattern is a list of tile numbers (1..15). The blank (0) is implicit.
// * Pattern size must be 1..8 (the builder supports up to 8 tiles).
// * Storage grows with P(16, k+1). For k=8, this is huge; consider 7/4/4.

#ifndef PDB_BITS
#define PDB_BITS 8  // 4 for nibble-packed (values saturate at 15), or 8 for 1 byte/entry
#endif

namespace pdb15 {
    using Pattern = std::vector<int>; // e.g., {1,2,3,4,5,6,7}

    class PackedPDB {
    public:
        explicit PackedPDB(std::uint64_t n_states);

        // Construct from raw file bytes (previously saved)
        static PackedPDB from_file_bytes(std::uint64_t n_states, const std::vector<std::uint8_t> &bytes);

        inline std::uint8_t get(std::uint64_t idx) const;

        inline void set(std::uint64_t idx, std::uint8_t val);

        inline std::uint64_t size() const { return size_; }

        bool save(const std::string &path) const; // raw dump (packed)

    private:
        std::uint64_t size_;
#if PDB_BITS == 8
        std::vector<std::uint8_t> data8_;
#else
        std::vector<std::uint8_t> data4_; // 2 entries per byte
#endif
        friend PackedPDB load_pdb_from_file(const std::string& path, int k);

    };

    // ---- Building / Loading -------------------------------------------------------

    // Build a single disjoint PDB using 0–1 BFS (pattern-tile moves cost 1; others cost 0),
    // then save it to 'out_path'. Returns the constructed PDB in-memory as well.
    PackedPDB build_pdb_01bfs(const Pattern &pattern_tiles,
                              const std::string &out_path,
                              bool verbose = true);

    // Number of abstract states for a pattern of size k (P(16, k+1)).
    std::uint64_t states_for_pattern(int k);

    // Load a PDB from disk (created by build_pdb_01bfs). k = pattern size
    PackedPDB load_pdb_from_file(const std::string &path, int k);

    // ---- Lookup / Heuristic -------------------------------------------------------

    // Lookup a single PDB value for a concrete 15‑puzzle state.
    std::uint8_t lookup(const PackedPDB &pdb, const Pattern &pattern_tiles, const puzzle15_state &s);

    // Sum of multiple disjoint PDBs (additive heuristic). The patterns must be disjoint.
    int additive_heuristic(const puzzle15_state &s,
                           const std::vector<std::pair<const PackedPDB *, Pattern> > &dbs);

    // Convenience: build the common 7/4/4 split and save to three files. Returns nothing.
    void build_744(const std::string &outA, const std::string &outB, const std::string &outC,
                   bool verbose = true);

    // Convenience: additive lookup for 7/4/4 when you have the three file paths.
    int heuristic_744_from_files(const puzzle15_state &s,
                                 const std::string &pathA,
                                 const std::string &pathB,
                                 const std::string &pathC);

    // Convenience: build the common 7/8 split and save to two files.
    void build_78(const std::string &out7, const std::string &out8,
                  bool verbose = true);

    // Heuristic for 7/8 split using file paths (loads on the fly).
    int heuristic_78_from_files(const puzzle15_state &s,
                                const std::string &path7,
                                const std::string &path8);

    // Heuristic using already-loaded PDBs (no file I/O).
    int heuristic_78(const puzzle15_state &s, const PackedPDB &pdb7, const PackedPDB &pdb8);

    // Heuristic for 7/4/4 using already-loaded PDBs (no file I/O).
    int heuristic_744(const puzzle15_state &s,
                      const PackedPDB &pdbA, const PackedPDB &pdbB, const PackedPDB &pdbC);

    // Autoloading (state-only) API — defaults & setters
    // --- Heuristic-by-state only (auto-load & cache PDBs) ---
    void set_default_paths_78(const std::string& path7, const std::string& path8);
    void set_default_paths_744(const std::string& pathA, const std::string& pathB, const std::string& pathC);

    // Loads on first use (and caches). Just give a state:
    int heuristic_78_auto(const puzzle15_state& s);
    int heuristic_744_auto(const puzzle15_state& s);
} // namespace pdb15




