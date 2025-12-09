// ===============================
// File: pdb15.cpp
// ===============================


#include "../include/pdb15.h"

static_assert(PDB_BITS == 4 || PDB_BITS == 8, "PDB_BITS must be 4 or 8");

namespace {
    // Board constants (fixed for 15‑puzzle 4x4)
    constexpr int N = 16; // cells 0..15
    constexpr int BLANK_TILE = 0; // blank is tile 0

    int goal_pos_of_tile(int t) {
        return (t == BLANK_TILE) ? 15 : (t - 1);
    }

    // Precompute 4-neighbors for each cell 0..15
    std::array<std::array<int, 4>, N> build_neighbors(int &max_deg) {
        std::array<std::array<int, 4>, N> nb{};
        max_deg = 0;
        for (int idx = 0; idx < N; ++idx) {
            int r = idx / 4, c = idx % 4;
            std::array<int, 4> v{-1, -1, -1, -1};
            int d = 0;
            if (r > 0)
                v[d++] = idx - 4;
            if (r < 3)
                v[d++] = idx + 4;
            if (c > 0)
                v[d++] = idx - 1;
            if (c < 3)
                v[d++] = idx + 1;
            nb[idx] = v;
            if (d > max_deg)
                max_deg = d;
        }
        return nb;
    }

    // P(16, m) = 16*15*...*(16-m+1)
    inline std::uint64_t perm_count(const int m) {
        std::uint64_t r = 1;
        for (int i = 0; i < m; ++i)
            r *= (N - i);
        return r;
    }

    // weight[i] = P(16-i-1, m-i-1). For i==m-1, weight=1.
    std::vector<std::uint64_t> build_radix_weights(const int m) {
        std::vector<std::uint64_t> w(m, 1);
        for (int i = 0; i < m - 1; ++i) {
            std::uint64_t prod = 1;
            for (int a = 0; a < (m - i - 1); ++a)
                prod *= (N - i - 1 - a);
            w[i] = prod;
        }
        w[m - 1] = 1ULL;
        return w;
    }

    // Rank a length-m sequence (no repetition) drawn from 0..15 using the above weights.
    std::uint64_t rank_partial(const std::vector<int> &seq, const std::vector<std::uint64_t> &w) {
        const int m = static_cast<int>(seq.size());
        std::array<int, N> avail{};
        for (int i = 0; i < N; ++i)
            avail[i] = i;
        int avail_len = N;

        std::uint64_t rank = 0;
        for (int i = 0; i < m; ++i) {
            int x = seq[i];
            int idx = 0;
            while (idx < avail_len && avail[idx] != x)
                ++idx;
            if (idx >= avail_len)
                throw std::runtime_error("rank_partial: invalid seq");
            rank += static_cast<std::uint64_t>(idx) * w[i];
            for (int j = idx; j + 1 < avail_len; ++j)
                avail[j] = avail[j + 1];
            --avail_len;
        }
        return rank;
    }

    // Node in the abstract space for a given pattern: positions of pattern tiles + blank position.
    struct Node {
        std::array<std::uint8_t, 8> pos{}; // up to 8 tiles (indices 0..15)
        std::uint8_t blank{}; // 0..15
    };

    // Build a Node for a concrete board state and a given pattern.
    Node node_from_state_for_pattern(const puzzle15_state &s, const pdb15::Pattern &pattern_tiles) {
        Node n{};
        for (int cell = 0; cell < 16; ++cell) {
            int t = static_cast<int>(s.tiles[cell]);
            if (t == 0)
                n.blank = static_cast<std::uint8_t>(cell);
            else {
                for (int i = 0; i < static_cast<int>(pattern_tiles.size()); ++i) {
                    if (pattern_tiles[i] == t) {
                        n.pos[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(cell);
                        break;
                    }
                }
            }
        }
        return n;
    }
}

namespace pdb15 {
    // PackedPDB
    // ----------
    // Compact in-memory representation of a pattern database (PDB) table.
    //
    // Depending on the compile-time flag PDB_BITS, the PDB values are stored as:
    //
    //   * PDB_BITS == 8:
    //       - One byte per entry (0..255).
    //       - data8_ has exactly n_states entries.
    //
    //   * PDB_BITS == 4:
    //       - Two 4-bit values (nibbles) packed into a single byte.
    //       - data4_.size() == ceil(n_states / 2).
    //       - Each value is saturated to 0..15 when stored.
    //
    // The class supports:
    //   * Construction with a given number of states (initialized to 0xFF).
    //   * Loading from a raw byte vector (from_file_bytes).
    //   * Getting and setting individual entries by index.
    //   * Saving the packed table to disk, optionally with progress reporting.
    //

    // ---- PackedPDB implementation -------------------------------------------------

    PackedPDB::PackedPDB(const std::uint64_t n_states)
        : size_(n_states)
#if PDB_BITS == 8
          , data8_(static_cast<std::size_t>(n_states), 0xFF)
#else
    , data4_((static_cast<std::size_t>(n_states) + 1) / 2, 0xFF)
#endif
    {
    }

    // Construct a PackedPDB from an in-memory byte buffer that was read from disk.
    // The caller provides the expected number of logical states (n_states) and
    // the raw bytes. We verify that the buffer size matches the expected layout
    // for the current PDB_BITS mode and then copy the data in.
    //
    // Throws std::runtime_error if the byte size does not match the expected size.
    PackedPDB PackedPDB::from_file_bytes(std::uint64_t n_states,
                                         const std::vector<std::uint8_t> &bytes) {
        PackedPDB p(n_states);
#if PDB_BITS == 8
        // In 8-bit mode, there is exactly one byte per PDB entry.
        if (bytes.size() != static_cast<std::size_t>(n_states))
            throw std::runtime_error("PDB file size mismatch (8-bit)");
        p.data8_ = bytes;
#else
        // In 4-bit mode, two entries are packed into each byte.
        std::size_t need = static_cast<std::size_t>((n_states + 1) / 2);
        if (bytes.size() != need)
            throw std::runtime_error("PDB file size mismatch (4-bit)");
        p.data4_ = bytes;
#endif
        return p;
    }

    // Return the PDB value stored at logical index idx.
    //
    // For PDB_BITS == 8:
    //   - Directly read the byte from data8_[idx].
    //
    // For PDB_BITS == 4:
    //   - Two 4-bit values are packed into one byte:
    //       * even idx  → low nibble (bits 0..3)
    //       * odd  idx  → high nibble (bits 4..7)
    std::uint8_t PackedPDB::get(const std::uint64_t idx) const {
#if PDB_BITS == 8
        return data8_[static_cast<std::size_t>(idx)];
#else
        std::size_t byte = static_cast<std::size_t>(idx >> 1);
        bool high = (idx & 1ULL) != 0ULL;
        std::uint8_t v = data4_[byte];
        return high
                   ? static_cast<std::uint8_t>(v >> 4)
                   : static_cast<std::uint8_t>(v & 0x0F);
#endif
    }

    // Store a PDB value at logical index idx.
    //
    // For PDB_BITS == 8:
    //   - The value is written directly into data8_[idx].
    //
    // For PDB_BITS == 4:
    //   - The value is clamped to [0..15] (4-bit saturation).
    //   - We update only the relevant nibble (high or low) of the target byte,
    //     preserving the other nibble.
    void PackedPDB::set(std::uint64_t idx, std::uint8_t val) {
#if PDB_BITS == 8
        data8_[static_cast<std::size_t>(idx)] = val;
#else
        if (val > 15) val = 15; // saturate to 4-bit range
        std::size_t byte = static_cast<std::size_t>(idx >> 1);
        bool high = (idx & 1ULL) != 0ULL;
        std::uint8_t &b = data4_[byte];
        if (high)
            b = static_cast<std::uint8_t>((b & 0x0F) | (val << 4));
        else
            b = static_cast<std::uint8_t>((b & 0xF0) | (val & 0x0F));
#endif
    }


    // Save the packed PDB contents to a binary file at 'path'.
    // The file format is just the raw bytes of the underlying storage
    // (either data8_ or data4_, depending on PDB_BITS).
    //
    // If with_progress == true, the function prints periodic progress reports
    // (every 256 MiB written) including total MiB written and write speed.
    //
    // Returns true on success, false on I/O failure.
    bool PackedPDB::save(const std::string &path, bool with_progress) const {
        using clock = std::chrono::steady_clock;
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out)
            return false;

#if PDB_BITS == 8
        const std::uint8_t *ptr = data8_.data();
        auto bytes = static_cast<std::uint64_t>(data8_.size());
#else
        const std::uint8_t *ptr = data4_.data();
        std::uint64_t bytes = static_cast<std::uint64_t>(data4_.size());
#endif

        constexpr std::size_t CHUNK = 32 * 1024 * 1024; // write in 32 MiB chunks
        std::uint64_t written = 0;
        auto t0 = clock::now();

        while (written < bytes) {
            std::size_t to_write = static_cast<std::size_t>(std::min<std::uint64_t>(CHUNK, bytes - written));
            out.write(reinterpret_cast<const char *>(ptr + written), to_write);
            if (!out)
                return false;
            written += to_write;

            if (with_progress) {
                // Report progress every 256 MiB written, and at the very end.
                static constexpr std::uint64_t REPORT = 256ull * 1024 * 1024;
                if (written >= REPORT && ((written % REPORT) == 0 || written == bytes)) {
                    double sec = std::chrono::duration<double>(clock::now() - t0).count();
                    double mib_done = written / (1024.0 * 1024.0);
                    double mib_all = bytes / (1024.0 * 1024.0);
                    double rate = sec > 0 ? (mib_done / sec) : 0.0;

                    std::cout.setf(std::ios::unitbuf);
                    std::cout << "[build] wrote "
                            << static_cast<std::uint64_t>(mib_done)
                            << " / " << static_cast<std::uint64_t>(mib_all)
                            << " MiB  ("
                            << static_cast<int>(100.0 * mib_done / mib_all)
                            << "%)  "
                            << rate << " MiB/s\n";
                }
            }
        }
        out.flush();
        return static_cast<bool>(out);
    }

    // Save the PDB to disk in an atomic way.
    //
    // Strategy:
    //   1. Ensure the parent directory of `path` exists (create it if needed).
    //   2. Write the data to a temporary file "<filename>.tmp" in the same directory.
    //   3. Rename the temporary file to the final `path`.
    //
    // If the rename fails, the temporary file is removed and the function returns false.
    // This guarantees that we never leave a partially-written file at `path`.
    bool PackedPDB::save_atomic(const std::string &path, bool with_progress) const {
        std::error_code ec;
        std::filesystem::path p(path);

        // Create parent directories if needed (ignore errors via ec).
        if (!p.parent_path().empty())
            std::filesystem::create_directories(p.parent_path(), ec);

        // Temporary file in the same directory: "<name>.tmp".
        auto tmp = (p.parent_path() / (p.filename().string() + ".tmp")).string();

        // First write to the temporary file.
        if (!save(tmp, with_progress))
            return false;

        // Try to rename temp → final atomically (on most filesystems).
        std::filesystem::rename(tmp, path, ec);
        if (ec) {
            // If rename failed, try to clean up the temp file.
            std::filesystem::remove(tmp, ec);
            return false;
        }
        return true;
    }

    // ---- IO helpers ---------------------------------------------------------------

    // Return the number of distinct states in the pattern database for pattern size k.
    // This calls perm_count(k + 1), which computes P(16, k+1) = 16! / (16 - (k+1))!.
    // In other words: the number of ways to choose and arrange (k+1) tiles out of 16
    // (including the blank).
    std::uint64_t states_for_pattern(int k) {
        return ::perm_count(k + 1);
    }

    // Compute how many bytes we expect to need on disk for a pattern of size k,
    // given the current PDB_BITS mode.
    //
    // For PDB_BITS == 8:
    //   * One byte per logical state → expected_bytes = n_states.
    // For PDB_BITS == 4:
    //   * Two 4-bit entries per byte → expected_bytes = ceil(n_states / 2).

    static inline std::uint64_t expected_bytes_for_k(int k) {
        const std::uint64_t n_states = states_for_pattern(k); // P(16, k+1)
#if PDB_BITS == 8
        return n_states;
#else
        return (n_states + 1) / 2;
#endif
    }


    // Load a PackedPDB from a binary file on disk.
    //
    // Parameters:
    //   path - filesystem path to the PDB file.
    //   k    - pattern size parameter; used to compute the expected number of
    //          logical states and therefore the expected file size.
    //
    // The function:
    //   1. Computes the expected number of bytes for the given k (expected_bytes_for_k).
    //   2. Opens the file and checks that its size matches the expectation.
    //   3. Constructs a PackedPDB with the correct number of logical states.
    //   4. Reads the raw bytes from disk into the internal storage (data8_ or data4_)
    //      in chunks, to support very large PDBs.
    //   5. Returns the fully populated PackedPDB.
    //
    // Throws std::runtime_error on any I/O error or size mismatch.
    PackedPDB load_pdb_from_file(const std::string &path, int k) {
        const std::uint64_t need_bytes = expected_bytes_for_k(k);

        std::ifstream in(path, std::ios::binary);
        if (!in)
            throw std::runtime_error("cannot open file: " + path);

        // Check that the file size matches exactly the expected number of bytes.
        in.seekg(0, std::ios::end);
        const std::uint64_t len = static_cast<std::uint64_t>(in.tellg());
        if (len != need_bytes) {
            throw std::runtime_error("corrupt size: expected " + std::to_string(need_bytes) +
                                     " got " + std::to_string(len) + " for " + path);
        }
        in.seekg(0, std::ios::beg);

        // Create a PDB with the correct number of logical states.
        PackedPDB pdb(states_for_pattern(k));

#if PDB_BITS == 8
        // In 8-bit mode, the file has one byte per logical entry.
        std::uint8_t *dst = pdb.data8_.data();
#else
        // In 4-bit mode, each byte stores two logical entries.
        std::uint8_t *dst = pdb.data4_.data();
#endif

        // Read the file in fixed-size chunks via the underlying file buffer.
        std::filebuf *fb = in.rdbuf();
        constexpr std::size_t CHUNK = 8 * 1024 * 1024; // 8 MiB
        std::uint64_t pos = 0;

        while (pos < need_bytes) {
            std::size_t want = static_cast<std::size_t>(std::min<std::uint64_t>(CHUNK, need_bytes - pos));

            std::streamsize got = fb->sgetn(reinterpret_cast<char *>(dst + pos),
                                            static_cast<std::streamsize>(want));
            if (got <= 0)
                throw std::runtime_error("read failed: " + path);

            pos += static_cast<std::uint64_t>(got);
        }

        return pdb;
    }


    // ---- Build (0–1 BFS) ----------------------------------------------------------
    //
    // Build a pattern database (PDB) for the given set of tiles using a 0–1 BFS
    // over the abstract state space of pattern tiles + blank.
    //
    // High-level idea:
    //   * We work backwards from the goal configuration (pattern tiles and blank in
    //     their goal locations).
    //   * Each abstract state (Node) stores:
    //        - pos[i]  : board cell index of pattern_tiles[i]
    //        - blank   : board cell index of the blank
    //   * Edges correspond to sliding the blank to a neighboring cell.
    //       - If the blank swaps with a tile that is in the pattern, the cost is 1.
    //       - If the blank swaps with a tile that is *not* in the pattern, the cost is 0.
    //   * 0–1 BFS is used to compute the minimum number of “pattern moves” (cost 1)
    //     from every abstract state to the goal state.
    //
    // The resulting distances are stored in a PackedPDB and written to disk.
    //
    PackedPDB build_pdb_01bfs(const Pattern &pattern_tiles,
                              const std::string &out_path,
                              bool verbose) {
        const int k = static_cast<int>(pattern_tiles.size());
        if (k <= 0 || k > 8)
            throw std::runtime_error("pattern size must be 1..8");

        int max_deg = 0;
        // Precomputed neighbors table: for each cell, which cells are reachable
        // by a single blank move (up/down/left/right).
        const auto neighbors = ::build_neighbors(max_deg);

        const int m = k + 1; // pattern tiles + blank
        // Ranking weights for mapping Node → integer index [0..n_states).
        const auto weights = ::build_radix_weights(m);
        const std::uint64_t n_states = ::perm_count(m); // number of abstract states: P(16, m)

        if (verbose) {
            std::cout << "Building PDB for tiles {";
            for (int i = 0; i < k; ++i) {
                std::cout << pattern_tiles[i] << (i + 1 == k ? "" : ", ");
            }
            std::cout << "} -> states = " << n_states;
#if PDB_BITS == 8
            auto bytes = static_cast<double>(n_states);
#else
            double bytes = static_cast<double>((n_states + 1) / 2);
#endif
            double gib = bytes / (1024.0 * 1024.0 * 1024.0);
            std::cout << ", storage ~ " << gib << " GiB (PDB_BITS=" << PDB_BITS << ")\n";
        }

        // Distance table (PDB). Initially all entries are 0xFF (unknown / “infinite”).
        PackedPDB dist(n_states);

        // Goal node: pattern tiles and blank at their goal positions.
        Node goal{};
        for (int i = 0; i < k && i < 8; ++i) {
            int tile = pattern_tiles[i];
            goal.pos[static_cast<std::size_t>(i)] =
                    static_cast<std::uint8_t>(::goal_pos_of_tile(tile));
        }
        goal.blank = static_cast<std::uint8_t>(::goal_pos_of_tile(BLANK_TILE));

        // Convert a Node to a unique integer index using the ranking scheme.
        auto rank_node = [&](const Node &s) -> std::uint64_t {
            std::vector<int> seq;
            seq.reserve(m);
            for (int i = 0; i < k; ++i)
                seq.push_back(static_cast<int>(s.pos[static_cast<std::size_t>(i)]));
            seq.push_back(static_cast<int>(s.blank));
            return ::rank_partial(seq, weights);
        };

        // Deque for 0–1 BFS: cost-0 edges push_front, cost-1 edges push_back.
        std::deque<Node> dq;
        {
            const std::uint64_t ridx = rank_node(goal);
            dist.set(ridx, 0); // distance(goal) = 0
            dq.push_back(goal);
        }

        // Utility: check if any pattern tile occupies the given board cell.
        // Returns index in pattern_tiles[] if found, or -1 if cell has no pattern tile.
        auto contains_tile_at = [&](const Node &s, int cell) -> int {
            for (int i = 0; i < k; ++i)
                if (s.pos[static_cast<std::size_t>(i)] == cell)
                    return i;
            return -1;
        };

        using clock = std::chrono::steady_clock;
        auto t0 = clock::now();
        constexpr std::uint64_t PROGRESS_EVERY = 50'000'000ULL;
        std::uint64_t next_report = PROGRESS_EVERY;
        std::uint64_t expanded = 0, relaxed = 0;
        if (verbose)
            std::cout.setf(std::ios::unitbuf);

        // 0–1 BFS main loop.
        // We repeatedly take a state from the front of the deque, relax all
        // neighbors, and push them either to the front (cost 0) or back (cost 1).
        while (!dq.empty()) {
            Node cur = dq.front();
            dq.pop_front();
            ++expanded;

            const std::uint64_t cur_idx = rank_node(cur);
            const std::uint8_t cur_d = dist.get(cur_idx);

            // Try moving the blank in each of the 4 directions.
            for (int di = 0; di < 4; ++di) {
                int nb = neighbors[cur.blank][di];
                if (nb < 0)
                    continue; // invalid move from this cell

                Node nxt = cur;
                int j = contains_tile_at(cur, nb);

                if (j >= 0) {
                    // cost 1: the blank swaps with a tile that is part of the pattern.
                    nxt.pos[static_cast<std::size_t>(j)] = cur.blank;
                    nxt.blank = static_cast<std::uint8_t>(nb);
                    const std::uint64_t nidx = rank_node(nxt);
                    auto nd = static_cast<std::uint8_t>(cur_d + 1);
                    if (dist.get(nidx) > nd) {
                        dist.set(nidx, nd);
                        dq.push_back(nxt); // cost-1 edge → push_back
                        ++relaxed;
                    }
                } else {
                    // cost 0: the blank moves over a non-pattern tile.
                    nxt.blank = static_cast<std::uint8_t>(nb);
                    const std::uint64_t nidx = rank_node(nxt);
                    if (dist.get(nidx) > cur_d) {
                        dist.set(nidx, cur_d);
                        dq.push_front(nxt); // cost-0 edge → push_front
                        ++relaxed;
                    }
                }
            }

            // Periodic progress logging.
            if (verbose && expanded >= next_report) {
                double sec = std::chrono::duration<double>(clock::now() - t0).count();
                double rate = (sec > 0.0) ? (expanded / sec) : 0.0;
                std::cout << "[build] exp=" << expanded
                        << "  relax=" << relaxed
                        << "  q=" << dq.size()
                        << "  rate=" << rate << " states/s\n";
                next_report += PROGRESS_EVERY;
            }
        }

        // Final summary.
        if (verbose) {
            double sec = std::chrono::duration<double>(clock::now() - t0).count();
            double rate = (sec > 0.0) ? (expanded / sec) : 0.0;
            std::cout << "[build] BFS done: exp=" << expanded
                    << "  relax=" << relaxed
                    << "  time=" << sec << "s"
                    << "  avg=" << rate << " states/s\n";
        }

#if PDB_BITS == 8
        std::uint64_t bytes_to_write = n_states;
#else
        std::uint64_t bytes_to_write = (n_states + 1) / 2;
#endif

        if (verbose) {
            std::cout << "[build] writing file (" << bytes_to_write << " bytes) -> "
                    << std::filesystem::absolute(out_path) << "\n";
        }

        // Save the PDB to disk atomically (write to temp file, then rename).
        if (!dist.save_atomic(out_path, /*with_progress=*/true)) {
            throw std::runtime_error("failed to write PDB file: " + out_path);
        }
        if (verbose)
            std::cout << "[build] done.\n";

        return dist;
    }


    // ---- Lookup & Additive --------------------------------------------------------
    //
    // This section provides:
    //   * lookup(...) – query a single PDB for a given pattern and state.
    //   * additive_heuristic(...) – sum of several disjoint PDBs (additive heuristic).
    //   * helper builders (build_744, build_78) for standard tile splits.
    //   * convenience functions that load PDBs from files and evaluate a heuristic.
    //

    // Look up the PDB value for a given state and pattern.
    //
    // Steps:
    //   1. Build an abstract Node that contains the positions of the pattern tiles
    //      and the blank for the given global state s.
    //   2. Convert that Node into a unique integer index using rank_partial(...).
    //   3. Return pdb.get(index).
    //
    // The returned value is the precomputed distance (in moves) from this abstract
    // pattern state to the goal, according to the chosen pattern and PDB.
    std::uint8_t lookup(const PackedPDB &pdb,
                        const Pattern &pattern_tiles,
                        const puzzle15_state &s) {
        const int k = static_cast<int>(pattern_tiles.size());
        const int m = k + 1;
        const auto weights = ::build_radix_weights(m);

        Node n = ::node_from_state_for_pattern(s, pattern_tiles);
        std::vector<int> seq;
        seq.reserve(m);
        for (int i = 0; i < k; ++i)
            seq.push_back(static_cast<int>(n.pos[static_cast<std::size_t>(i)]));
        seq.push_back(static_cast<int>(n.blank));

        std::uint64_t idx = ::rank_partial(seq, weights);
        return pdb.get(idx);
    }

    // Compute an additive heuristic as the sum of multiple disjoint PDBs.
    //
    // Parameters:
    //   s   - full 15-puzzle state.
    //   dbs - vector of (PDB pointer, Pattern) pairs. Each Pattern defines which
    //         tiles the corresponding PDB covers. The patterns should be mutually
    //         disjoint for the heuristic to be strictly admissible.
    //
    // For each pair (pdb, pat), we call lookup(*pdb, pat, s) and sum the results.
    // If any PDB pointer is null, an exception is thrown.
    int additive_heuristic(
        const puzzle15_state &s,
        const std::vector<std::pair<const PackedPDB *, Pattern> > &dbs) {
        int sum = 0;
        for (const auto &pr: dbs) {
            const PackedPDB *pdb = pr.first;
            const Pattern &pat = pr.second;
            if (!pdb)
                throw std::invalid_argument("additive_heuristic: null PDB pointer");
            sum += static_cast<int>(lookup(*pdb, pat, s));
        }
        return sum;
    }

    // Build a 7-4-4 pattern database set and write them to three files.
    //
    // Patterns:
    //   A: { 1,  2,  3,  4,  5,  6,  7}
    //   B: { 8,  9, 10, 11}
    //   C: {12, 13, 14, 15}
    //
    // Each PDB is built using the 0–1 BFS builder (build_pdb_01bfs),
    // and stored to outA / outB / outC.
    void build_744(const std::string &outA,
                   const std::string &outB,
                   const std::string &outC,
                   bool verbose) {
        Pattern A{1, 2, 3, 4, 5, 6, 7};
        Pattern B{8, 9, 10, 11};
        Pattern C{12, 13, 14, 15};
        (void) build_pdb_01bfs(A, outA, verbose);
        (void) build_pdb_01bfs(B, outB, verbose);
        (void) build_pdb_01bfs(C, outC, verbose);
    }

    // Convenience function: load the 7-4-4 PDBs from files and return the sum
    // of the three PDB lookups for the given state.
    //
    // This is equivalent to:
    //   * load PDB for A, B, C from pathA, pathB, pathC
    //   * return additive_heuristic(s, { (A,PDB_A), (B,PDB_B), (C,PDB_C) })
    int heuristic_744_from_files(const puzzle15_state &s,
                                 const std::string &pathA,
                                 const std::string &pathB,
                                 const std::string &pathC) {
        Pattern A{1, 2, 3, 4, 5, 6, 7};
        Pattern B{8, 9, 10, 11};
        Pattern C{12, 13, 14, 15};
        PackedPDB pdbA = load_pdb_from_file(pathA, static_cast<int>(A.size()));
        PackedPDB pdbB = load_pdb_from_file(pathB, static_cast<int>(B.size()));
        PackedPDB pdbC = load_pdb_from_file(pathC, static_cast<int>(C.size()));
        return additive_heuristic(s, {{&pdbA, A}, {&pdbB, B}, {&pdbC, C}});
    }

    // Build a 7-8 pattern database set and write to two files.
    //
    // Patterns:
    //   P7: { 1,  2,  3,  4,  5,  6,  7}
    //   P8: { 8,  9, 10, 11, 12, 13, 14, 15}
    //
    // Each PDB is built using 0–1 BFS and stored to out7 / out8.
    void build_78(const std::string &out7,
                  const std::string &out8,
                  bool verbose) {
        Pattern P7{1, 2, 3, 4, 5, 6, 7};
        Pattern P8{8, 9, 10, 11, 12, 13, 14, 15};
        (void) build_pdb_01bfs(P7, out7, verbose);
        (void) build_pdb_01bfs(P8, out8, verbose);
    }

    // Evaluate the 7-8 additive heuristic given two already-loaded PDBs.
    //
    // This uses the fixed partition:
    //   P7: tiles 1..7
    //   P8: tiles 8..15
    // and returns lookup(pdb7, P7, s) + lookup(pdb8, P8, s).
    int heuristic_78(const puzzle15_state &s,
                     const PackedPDB &pdb7,
                     const PackedPDB &pdb8) {
        Pattern P7{1, 2, 3, 4, 5, 6, 7};
        Pattern P8{8, 9, 10, 11, 12, 13, 14, 15};
        return static_cast<int>(lookup(pdb7, P7, s)) +
               static_cast<int>(lookup(pdb8, P8, s));
    }

    // Evaluate the 7-4-4 additive heuristic given three already-loaded PDBs.
    //
    // Fixed partition:
    //   A: tiles 1..7
    //   B: tiles  8..11
    //   C: tiles 12..15
    int heuristic_744(const puzzle15_state &s,
                      const PackedPDB &pdbA,
                      const PackedPDB &pdbB,
                      const PackedPDB &pdbC) {
        Pattern A{1, 2, 3, 4, 5, 6, 7};
        Pattern B{8, 9, 10, 11};
        Pattern C{12, 13, 14, 15};
        return static_cast<int>(lookup(pdbA, A, s)) +
               static_cast<int>(lookup(pdbB, B, s)) +
               static_cast<int>(lookup(pdbC, C, s));
    }

    // Convenience function: load the 7-8 PDBs from files and evaluate the heuristic.
    //
    // This:
    //
    //   1. Loads the 7-tile PDB from path7.
    //   2. Loads the 8-tile PDB from path8.
    //   3. Returns heuristic_78(s, pdb7, pdb8).
    int heuristic_78_from_files(const puzzle15_state &s,
                                const std::string &path7,
                                const std::string &path8) {
        Pattern P7{1, 2, 3, 4, 5, 6, 7};
        Pattern P8{8, 9, 10, 11, 12, 13, 14, 15};
        PackedPDB pdb7 = load_pdb_from_file(path7, static_cast<int>(P7.size()));
        PackedPDB pdb8 = load_pdb_from_file(path8, static_cast<int>(P8.size()));
        return heuristic_78(s, pdb7, pdb8);
    }

    // ---- Auto-loading, state-only API (with simple caching) ----------------------
    //
    // This section provides "easy" APIs that:
    //   * Remember default file paths for the PDBs.
    //   * Lazily load the PDBs on first use.
    //   * Cache them in static std::unique_ptrs for reuse.
    //
    // The idea:
    //   - Client code can call heuristic_78_auto(s) or heuristic_744_auto(s)
    //     without worrying about loading files or keeping PDB objects around.
    //   - Paths can be overridden via set_default_paths_78 / set_default_paths_744.

    namespace {
        // Default file paths (can be overridden by set_default_paths_*).
        std::string g_p7 = "pdb_1_7.bin";
        std::string g_p8 = "pdb_8_15.bin";
        std::string g_pA = "pdb_1_7.bin";
        std::string g_pB = "pdb_8_11.bin";
        std::string g_pC = "pdb_12_15.bin";

        // Cached PDB instances, loaded on first access.
        std::unique_ptr<pdb15::PackedPDB> g_7, g_8, g_A, g_B, g_C;
    }

    // Set default file paths for the 7-8 PDBs.
    // Also clears any existing cached PDB objects so they will be reloaded
    // on the next call to heuristic_78_auto.
    void set_default_paths_78(const std::string &path7,
                              const std::string &path8) {
        g_p7 = path7;
        g_p8 = path8;
        g_7.reset();
        g_8.reset();
    }

    // Set default file paths for the 7-4-4 PDBs.
    // Also clears any existing cached PDB objects so they will be reloaded
    // on the next call to heuristic_744_auto.
    void set_default_paths_744(const std::string &pathA,
                               const std::string &pathB,
                               const std::string &pathC) {
        g_pA = pathA;
        g_pB = pathB;
        g_pC = pathC;
        g_A.reset();
        g_B.reset();
        g_C.reset();
    }

    // Autoloading version of the 7-8 heuristic:
    //
    //   * Uses the global default paths g_p7 and g_p8.
    //   * On first call, loads the PDBs from disk into g_7 and g_8.
    //   * On subsequent calls, reuses the cached PDBs.
    //   * Returns heuristic_78(s, *g_7, *g_8).
    int heuristic_78_auto(const puzzle15_state &s) {
        if (!g_7)
            g_7 = std::make_unique<PackedPDB>(load_pdb_from_file(g_p7, 7));
        if (!g_8)
            g_8 = std::make_unique<PackedPDB>(load_pdb_from_file(g_p8, 8));
        return heuristic_78(s, *g_7, *g_8);
    }

    // Autoloading version of the 7-4-4 heuristic:
    //
    //   * Uses default paths g_pA, g_pB, g_pC.
    //   * On first call, loads each PDB from disk into g_A, g_B, g_C.
    //   * Reuses the cached PDBs on subsequent calls.
    //   * Returns heuristic_744(s, *g_A, *g_B, *g_C).
    int heuristic_744_auto(const puzzle15_state &s) {
        if (!g_A)
            g_A = std::make_unique<PackedPDB>(load_pdb_from_file(g_pA, 7));
        if (!g_B)
            g_B = std::make_unique<PackedPDB>(load_pdb_from_file(g_pB, 4));
        if (!g_C)
            g_C = std::make_unique<PackedPDB>(load_pdb_from_file(g_pC, 4));
        return heuristic_744(s, *g_A, *g_B, *g_C);
    }
} // namespace pdb15
