// ===============================
// File: pdb15.cpp
// ===============================


#include "pdb15.h"

static_assert(PDB_BITS == 4 || PDB_BITS == 8, "PDB_BITS must be 4 or 8");

namespace {
    // Board constants (fixed for 15‑puzzle 4x4)
    constexpr int N = 16; // cells 0..15
    constexpr int BLANK_TILE = 0; // blank is tile 0

    inline int goal_pos_of_tile(int t) {
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
            if (r > 0) v[d++] = idx - 4;
            if (r < 3) v[d++] = idx + 4;
            if (c > 0) v[d++] = idx - 1;
            if (c < 3) v[d++] = idx + 1;
            nb[idx] = v;
            if (d > max_deg) max_deg = d;
        }
        return nb;
    }

    // P(16, m) = 16*15*...*(16-m+1)
    inline std::uint64_t perm_count(int m) {
        std::uint64_t r = 1;
        for (int i = 0; i < m; ++i) r *= (N - i);
        return r;
    }

    // weight[i] = P(16-i-1, m-i-1). For i==m-1, weight=1.
    std::vector<std::uint64_t> build_radix_weights(int m) {
        std::vector<std::uint64_t> w(m, 1);
        for (int i = 0; i < m - 1; ++i) {
            std::uint64_t prod = 1;
            for (int a = 0; a < (m - i - 1); ++a) prod *= (N - i - 1 - a);
            w[i] = prod;
        }
        w[m - 1] = 1ULL;
        return w;
    }

    // Rank a length-m sequence (no repetition) drawn from 0..15 using the above weights.
    std::uint64_t rank_partial(const std::vector<int> &seq, const std::vector<std::uint64_t> &w) {
        const int m = static_cast<int>(seq.size());
        std::array<int, N> avail{};
        for (int i = 0; i < N; ++i) avail[i] = i;
        int avail_len = N;

        std::uint64_t rank = 0;
        for (int i = 0; i < m; ++i) {
            int x = seq[i];
            int idx = 0;
            while (idx < avail_len && avail[idx] != x) ++idx;
            if (idx >= avail_len) throw std::runtime_error("rank_partial: invalid seq");
            rank += static_cast<std::uint64_t>(idx) * w[i];
            for (int j = idx; j + 1 < avail_len; ++j) avail[j] = avail[j + 1];
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
            if (t == 0) n.blank = static_cast<std::uint8_t>(cell);
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
    // ---- PackedPDB implementation -------------------------------------------------

    PackedPDB::PackedPDB(std::uint64_t n_states)
        : size_(n_states)
#if PDB_BITS == 8
          , data8_(static_cast<std::size_t>(n_states), 0xFF)
#else
    , data4_((static_cast<std::size_t>(n_states) + 1) / 2, 0xFF)
#endif
    {
    }

    PackedPDB PackedPDB::from_file_bytes(std::uint64_t n_states, const std::vector<std::uint8_t> &bytes) {
        PackedPDB p(n_states);
#if PDB_BITS == 8
        if (bytes.size() != static_cast<std::size_t>(n_states))
            throw std::runtime_error("PDB file size mismatch (8-bit)");
        p.data8_ = bytes;
#else
        std::size_t need = static_cast<std::size_t>((n_states + 1) / 2);
        if (bytes.size() != need)
            throw std::runtime_error("PDB file size mismatch (4-bit)");
        p.data4_ = bytes;
#endif
        return p;
    }

    std::uint8_t PackedPDB::get(std::uint64_t idx) const {
#if PDB_BITS == 8
        return data8_[static_cast<std::size_t>(idx)];
#else
        std::size_t byte = static_cast<std::size_t>(idx >> 1);
        bool high = (idx & 1ULL) != 0ULL;
        std::uint8_t v = data4_[byte];
        return high ? static_cast<std::uint8_t>(v >> 4) : static_cast<std::uint8_t>(v & 0x0F);
#endif
    }

    void PackedPDB::set(std::uint64_t idx, std::uint8_t val) {
#if PDB_BITS == 8
        data8_[static_cast<std::size_t>(idx)] = val;
#else
        if (val > 15) val = 15; // saturate for 4-bit
        std::size_t byte = static_cast<std::size_t>(idx >> 1);
        bool high = (idx & 1ULL) != 0ULL;
        std::uint8_t &b = data4_[byte];
        if (high) b = static_cast<std::uint8_t>((b & 0x0F) | (val << 4));
        else b = static_cast<std::uint8_t>((b & 0xF0) | (val & 0x0F));
#endif
    }

    bool PackedPDB::save(const std::string& path, bool with_progress) const {
    using clock = std::chrono::steady_clock;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;

#if PDB_BITS == 8
    const std::uint8_t* ptr = data8_.data();
    auto bytes = static_cast<std::uint64_t>(data8_.size());
#else
    const std::uint8_t* ptr = data4_.data();
    std::uint64_t bytes = static_cast<std::uint64_t>(data4_.size());
#endif

    const std::size_t CHUNK = 32 * 1024 * 1024; // 32 MiB
    std::uint64_t written = 0;
    auto t0 = clock::now();

    while (written < bytes) {
        std::size_t to_write = static_cast<std::size_t>(std::min<std::uint64_t>(CHUNK, bytes - written));
        out.write(reinterpret_cast<const char*>(ptr + written), to_write);
        if (!out) return false;
        written += to_write;

        if (with_progress) {
            static const std::uint64_t REPORT = 256ull * 1024 * 1024; // דו"ח כל 256 MiB
            if (written >= REPORT && ((written % REPORT) == 0 || written == bytes)) {
                double sec = std::chrono::duration<double>(clock::now() - t0).count();
                double mib_done = written / (1024.0 * 1024.0);
                double mib_all  = bytes   / (1024.0 * 1024.0);
                double rate     = sec > 0 ? (mib_done / sec) : 0.0;
                std::cout.setf(std::ios::unitbuf);
                std::cout << "[build] wrote " << static_cast<std::uint64_t>(mib_done)
                          << " / " << static_cast<std::uint64_t>(mib_all) << " MiB  ("
                          << static_cast<int>(100.0 * mib_done / mib_all) << "%)  "
                          << rate << " MiB/s\n";
            }
        }
    }
    out.flush();
    return static_cast<bool>(out);
}

bool PackedPDB::save_atomic(const std::string& path, bool with_progress) const {
    std::error_code ec;
    std::filesystem::path p(path);
    if (!p.parent_path().empty()) std::filesystem::create_directories(p.parent_path(), ec);
    auto tmp = (p.parent_path() / (p.filename().string() + ".tmp")).string();
    if (!save(tmp, with_progress)) return false;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        return false;
    }
    return true;
}

    // ---- IO helpers ---------------------------------------------------------------


    std::uint64_t states_for_pattern(int k) { return ::perm_count(k + 1); }

    static inline std::uint64_t expected_bytes_for_k(int k) {
        const std::uint64_t n_states = states_for_pattern(k); // P(16,k+1)
#if PDB_BITS == 8
        return n_states;
#else
        return (n_states + 1) / 2;
#endif
    }

    PackedPDB load_pdb_from_file(const std::string& path, int k) {
        const std::uint64_t need_bytes = expected_bytes_for_k(k);

        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open file: " + path);

        in.seekg(0, std::ios::end);
        const std::uint64_t len = static_cast<std::uint64_t>(in.tellg());
        if (len != need_bytes) {
            throw std::runtime_error("corrupt size: expected " + std::to_string(need_bytes) +
                                     " got " + std::to_string(len) + " for " + path);
        }
        in.seekg(0, std::ios::beg);

        PackedPDB pdb(states_for_pattern(k));

#if PDB_BITS == 8
        std::uint8_t* dst = pdb.data8_.data();
#else
        std::uint8_t* dst = pdb.data4_.data();
#endif

        std::filebuf* fb = in.rdbuf();
        const std::size_t CHUNK = 8 * 1024 * 1024; // 8 MiB
        std::uint64_t pos = 0;

        while (pos < need_bytes) {
            std::size_t want = static_cast<std::size_t>(std::min<std::uint64_t>(CHUNK, need_bytes - pos));
            std::streamsize got = fb->sgetn(reinterpret_cast<char*>(dst + pos),
                                            static_cast<std::streamsize>(want));
            if (got <= 0) throw std::runtime_error("read failed: " + path);
            pos += static_cast<std::uint64_t>(got);
        }
        return pdb;
    }


    // ---- Build (0–1 BFS) ----------------------------------------------------------

    PackedPDB build_pdb_01bfs(const Pattern &pattern_tiles,
                              const std::string &out_path,
                              bool verbose) {
        const int k = static_cast<int>(pattern_tiles.size());
        if (k <= 0 || k > 8) throw std::runtime_error("pattern size must be 1..8");

        int max_deg = 0;
        const auto neighbors = ::build_neighbors(max_deg);

        const int m = k + 1; // tiles + blank
        const auto weights = ::build_radix_weights(m);
        const std::uint64_t n_states = ::perm_count(m); // P(16, m)

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

        PackedPDB dist(n_states);

        // Goal node
        Node goal{};
        for (int i = 0; i < k && i < 8; ++i) {
            int tile = pattern_tiles[i];
            goal.pos[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(::goal_pos_of_tile(tile));
        }
        goal.blank = static_cast<std::uint8_t>(::goal_pos_of_tile(BLANK_TILE));

        auto rank_node = [&](const Node &s) -> std::uint64_t {
            std::vector<int> seq;
            seq.reserve(m);
            for (int i = 0; i < k; ++i) seq.push_back(static_cast<int>(s.pos[static_cast<std::size_t>(i)]));
            seq.push_back(static_cast<int>(s.blank));
            return ::rank_partial(seq, weights);
        };

        std::deque<Node> dq;
        {
            const std::uint64_t ridx = rank_node(goal);
            dist.set(ridx, 0);
            dq.push_back(goal);
        }

        auto contains_tile_at = [&](const Node &s, int cell) -> int {
            for (int i = 0; i < k; ++i) if (s.pos[static_cast<std::size_t>(i)] == cell) return i;
            return -1;
        };
        using clock = std::chrono::steady_clock;
        auto t0 = clock::now();
        constexpr std::uint64_t PROGRESS_EVERY = 50'000'000ULL; // ← כל 50M הרחבות
        std::uint64_t next_report = PROGRESS_EVERY;
        std::uint64_t expanded = 0, relaxed = 0;
        if (verbose) std::cout.setf(std::ios::unitbuf); // הדפסה מיידית

        // 0–1 BFS
        while (!dq.empty()) {
            Node cur = dq.front(); dq.pop_front();
            ++expanded;

            const std::uint64_t cur_idx = rank_node(cur);
            const std::uint8_t  cur_d   = dist.get(cur_idx);

            for (int di = 0; di < 4; ++di) {
                int nb = neighbors[cur.blank][di];
                if (nb < 0) continue;

                Node nxt = cur;
                int j = contains_tile_at(cur, nb);
                if (j >= 0) {
                    // עלות 1 – אריח בתבנית
                    nxt.pos[static_cast<std::size_t>(j)] = cur.blank;
                    nxt.blank = static_cast<std::uint8_t>(nb);
                    const std::uint64_t nidx = rank_node(nxt);
                    auto nd = static_cast<std::uint8_t>(cur_d + 1);
                    if (dist.get(nidx) > nd) {
                        dist.set(nidx, nd);
                        dq.push_back(nxt);
                        ++relaxed;
                    }
                } else {
                    // עלות 0 – אריח לא בתבנית
                    nxt.blank = static_cast<std::uint8_t>(nb);
                    const std::uint64_t nidx = rank_node(nxt);
                    if (dist.get(nidx) > cur_d) {
                        dist.set(nidx, cur_d);
                        dq.push_front(nxt);
                        ++relaxed;
                    }
                }
            }

            // --- לוג כל 50M הרחבות ---
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
        if (!dist.save_atomic(out_path, /*with_progress=*/true)) {
            throw std::runtime_error("failed to write PDB file: " + out_path);
        }
        if (verbose) std::cout << "[build] done.\n";

        return dist;
    }

    // ---- Lookup & Additive --------------------------------------------------------

    std::uint8_t lookup(const PackedPDB &pdb, const Pattern &pattern_tiles, const puzzle15_state &s) {
        const int k = static_cast<int>(pattern_tiles.size());
        const int m = k + 1;
        const auto weights = ::build_radix_weights(m);

        Node n = ::node_from_state_for_pattern(s, pattern_tiles);
        std::vector<int> seq;
        seq.reserve(m);
        for (int i = 0; i < k; ++i) seq.push_back(static_cast<int>(n.pos[static_cast<std::size_t>(i)]));
        seq.push_back(static_cast<int>(n.blank));
        std::uint64_t idx = ::rank_partial(seq, weights);
        return pdb.get(idx);
    }

    int additive_heuristic(const puzzle15_state &s,
                           const std::vector<std::pair<const PackedPDB *, Pattern> > &dbs) {
        int sum = 0;
        for (const auto &pr: dbs) {
            const PackedPDB *pdb = pr.first;
            const Pattern &pat = pr.second;
            if (!pdb) throw std::invalid_argument("additive_heuristic: null PDB pointer");
            sum += static_cast<int>(lookup(*pdb, pat, s));
        }
        return sum;
    }

    void build_744(const std::string &outA, const std::string &outB, const std::string &outC,
                   bool verbose) {
        Pattern A{1, 2, 3, 4, 5, 6, 7};
        Pattern B{8, 9, 10, 11};
        Pattern C{12, 13, 14, 15};
        (void) build_pdb_01bfs(A, outA, verbose);
        (void) build_pdb_01bfs(B, outB, verbose);
        (void) build_pdb_01bfs(C, outC, verbose);
    }

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

    void build_78(const std::string &out7, const std::string &out8, bool verbose) {
        Pattern P7{1, 2, 3, 4, 5, 6, 7};
        Pattern P8{8, 9, 10, 11, 12, 13, 14, 15};
        (void) build_pdb_01bfs(P7, out7, verbose);
        (void) build_pdb_01bfs(P8, out8, verbose);
    }

    int heuristic_78(const puzzle15_state &s, const PackedPDB &pdb7, const PackedPDB &pdb8) {
        Pattern P7{1, 2, 3, 4, 5, 6, 7};
        Pattern P8{8, 9, 10, 11, 12, 13, 14, 15};
        return static_cast<int>(lookup(pdb7, P7, s)) + static_cast<int>(lookup(pdb8, P8, s));
    }

    int heuristic_744(const puzzle15_state &s,
                      const PackedPDB &pdbA, const PackedPDB &pdbB, const PackedPDB &pdbC) {
        Pattern A{1, 2, 3, 4, 5, 6, 7};
        Pattern B{8, 9, 10, 11};
        Pattern C{12, 13, 14, 15};
        return static_cast<int>(lookup(pdbA, A, s)) +
               static_cast<int>(lookup(pdbB, B, s)) +
               static_cast<int>(lookup(pdbC, C, s));
    }

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
    namespace {
        // קאש + נתיבי ברירת־מחדל
        std::string g_p7 = "pdb_1_7.bin";
        std::string g_p8 = "pdb_8_15.bin";
        std::string g_pA = "pdb_1_7.bin";
        std::string g_pB = "pdb_8_11.bin";
        std::string g_pC = "pdb_12_15.bin";

        std::unique_ptr<pdb15::PackedPDB> g_7, g_8, g_A, g_B, g_C;
    }

    void set_default_paths_78(const std::string &path7, const std::string &path8) {
        g_p7 = path7;
        g_p8 = path8;
        g_7.reset();
        g_8.reset(); // לנקות קאש כדי שייטען מחדש בפעם הבאה
    }

    void set_default_paths_744(const std::string &pathA, const std::string &pathB, const std::string &pathC) {
        g_pA = pathA;
        g_pB = pathB;
        g_pC = pathC;
        g_A.reset();
        g_B.reset();
        g_C.reset();
    }

    int heuristic_78_auto(const puzzle15_state &s) {
        if (!g_7) g_7 = std::make_unique<PackedPDB>(load_pdb_from_file(g_p7, 7));
        if (!g_8) g_8 = std::make_unique<PackedPDB>(load_pdb_from_file(g_p8, 8));
        return heuristic_78(s, *g_7, *g_8);
    }

    int heuristic_744_auto(const puzzle15_state &s) {
        if (!g_A) g_A = std::make_unique<PackedPDB>(load_pdb_from_file(g_pA, 7));
        if (!g_B) g_B = std::make_unique<PackedPDB>(load_pdb_from_file(g_pB, 4));
        if (!g_C) g_C = std::make_unique<PackedPDB>(load_pdb_from_file(g_pC, 4));
        return heuristic_744(s, *g_A, *g_B, *g_C);
    }
} // namespace pdb15
