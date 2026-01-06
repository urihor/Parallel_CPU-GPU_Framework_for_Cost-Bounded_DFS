//
// Created by Owner on 29/12/2025.
// build_nn_over_corrections.cpp  (TorchScript version)
//
// This tool scans a PDB pattern file (1–7 or 8–15), evaluates a TorchScript
// ensemble heuristic on every abstract state (in rank order), and records
// all *overestimations* of the neural delta:
//
//    delta_true  = PDB(pattern) - Manhattan(pattern)
//    delta_pred  = argmin_q_ensemble(softmax(logits))   // quantile index
//
// Whenever delta_pred > delta_true, we write a record:
//    (rank : uint32, over : uint8),  where over = min(delta_pred - delta_true, 255)
//
// The output file format is:
//
//   Header (packed):
//     magic[8]  = "NNOVR001"
//     version   = 1
//     pattern   = 17 or 815  (for patterns 1–7 or 8–15)
//     m         = 8 or 9     (length of abstract seq)
//     q         = quantile used for the classifier
//     count     = number of (rank, over) records
//
//   Then `count` records, each:
//     rank : uint32
//     over : uint8
//
// This file is later loaded by NnOverCorrections, which merges duplicates
// and builds a fast lookup structure (rank -> max over) so that the neural
// delta can be corrected to remain admissible.
//

#include <torch/torch.h>
#include <torch/script.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
#pragma pack(push, 1)
    /**
     * On-disk header for the corrections file.
     *
     * magic   : "NNOVR001"
     * version : currently 1
     * pattern : 17 for tiles 1–7, 815 for tiles 8–15
     * m       : 8 or 9 (abstract sequence length including blank)
     * q       : quantile used when generating delta_pred
     * count   : number of (rank, over) records written after the header
     */
    struct Header {
        char magic[8]; // "NNOVR001"
        std::uint32_t version; // 1
        std::uint32_t pattern; // 17 or 815
        std::uint32_t m; // 8 or 9
        float q; // quantile used
        std::uint64_t count; // records count
    };
#pragma pack(pop)

    // Expected magic bytes for sanity checking an existing file
    static constexpr char MAGIC[8] = {'N', 'N', 'O', 'V', 'R', '0', '0', '1'};

    /**
     * Compute permutation count P(n, k) = n * (n-1) * ... * (n-k+1).
     * Used to build mixed-radix weights for ranking/unranking partial permutations.
     */
    static std::uint64_t perm_count(int n, int k) {
        if (k <= 0) return 1;
        std::uint64_t r = 1;
        for (int i = 0; i < k; ++i) r *= static_cast<std::uint64_t>(n - i);
        return r;
    }

    /**
     * Build the mixed-radix weights for the partial permutation rank.
     *
     * For m = 8 (pattern 1–7) or m = 9 (pattern 8–15), we use:
     *
     *   w[i] = P(16 - i - 1, m - i - 1)
     *
     * so that each abstract state (seq of m distinct cells in [0,15]) can
     * be represented by an integer rank in [0, P(16, m)).
     */
    static std::array<std::uint64_t, 9> make_weights(int m) {
        std::array<std::uint64_t, 9> w{};
        for (int i = 0; i < m; ++i) {
            w[i] = perm_count(16 - i - 1, m - i - 1);
        }
        return w;
    }

    /**
     * Inverse mapping from rank -> partial permutation (length m) over {0..15}.
     *
     * Given:
     *   rank in [0, P(16, m)),
     *   weights w[i] as from make_weights(),
     * this reconstructs the unique sequence out_seq[0..m-1] of distinct
     * cell indices (0..15) corresponding to this rank.
     */
    static void unrank_partial_perm(std::uint32_t rank,
                                    int m,
                                    const std::array<std::uint64_t, 9> &w,
                                    int out_seq[9]) {
        int avail[16];
        int avail_len = 16;
        for (int i = 0; i < 16; ++i) avail[i] = i;

        std::uint64_t r = rank;
        for (int i = 0; i < m; ++i) {
            const std::uint64_t wi = w[i];
            const int idx = static_cast<int>(r / wi);
            r = r % wi;

            out_seq[i] = avail[idx];

            // remove avail[idx]
            for (int j = idx; j + 1 < avail_len; ++j) {
                avail[j] = avail[j + 1];
            }
            --avail_len;
        }
    }

    /**
     * Goal index mapping for 4x4 puzzle:
     *   tiles 1..15 -> goal positions 0..14
     *   blank (0)   -> position 15
     */
    static int goal_index(int tile) { return (tile == 0) ? 15 : (tile - 1); }

    /**
     * Compute Manhattan distance for tiles 1..7 only, given an abstract
     * sequence of positions seq8[0..7]:
     *
     *   seq8[0..6] = positions of tiles 1..7
     *   seq8[7]    = position of blank (ignored for the distance)
     */
    static int manhattan_from_seq_1_7(const int seq8[8]) {
        int sum = 0;
        for (int i = 0; i < 7; ++i) {
            const int tile = i + 1;
            const int p = seq8[i];
            const int g = goal_index(tile);
            sum += std::abs(p / 4 - g / 4) + std::abs(p % 4 - g % 4);
        }
        return sum;
    }

    /**
     * Compute Manhattan distance for tiles 8..15 only, given an abstract
     * sequence seq9[0..8]:
     *
     *   seq9[0..7] = positions of tiles 8..15
     *   seq9[8]    = position of blank (ignored for the distance)
     */
    static int manhattan_from_seq_8_15(const int seq9[9]) {
        int sum = 0;
        for (int i = 0; i < 8; ++i) {
            const int tile = 8 + i;
            const int p = seq9[i];
            const int g = goal_index(tile);
            sum += std::abs(p / 4 - g / 4) + std::abs(p % 4 - g % 4);
        }
        return sum;
    }

    /**
     * Given logits [B, C], compute a quantile-based class index for each
     * sample:
     *
     *   probs = softmax(logits)
     *   cdf   = cumulative sum of probs along classes
     *   idx   = first index where CDF >= q
     *
     * This returns a tensor of shape [B] with the chosen class per sample.
     */
    static torch::Tensor quantile_index_from_logits(const torch::Tensor &logits, double q) {
        auto probs = torch::softmax(logits, /*dim=*/1);
        auto cdf = torch::cumsum(probs, /*dim=*/1);
        auto mask = cdf.ge(q).to(torch::kInt64); // [B,C] of 0/1
        return mask.argmax(/*dim=*/1); // [B]
    }

    // ----------------- TorchScript loading -----------------

    /**
     * Load a single TorchScript model from disk, move it to the target device,
     * and switch to eval() mode.
     */
    static torch::jit::script::Module load_ts_model(const std::string &path, const torch::Device &device) {
        torch::jit::script::Module m = torch::jit::load(path);
        m.eval();
        m.to(device);
        return m;
    }

    /**
     * Load an ensemble (vector) of TorchScript models from multiple paths.
     * All models are moved to the same device.
     */
    static std::vector<torch::jit::script::Module>
    load_ensemble(const std::vector<std::string> &paths, const torch::Device &device) {
        std::vector<torch::jit::script::Module> models;
        models.reserve(paths.size());
        for (const auto &p: paths) {
            models.emplace_back(load_ts_model(p, device));
        }
        return models;
    }

    /**
     * Simple helper: return the value following `key` in argv, or a default.
     * Example: --pdb path.bin  -> arg1(..., "--pdb") == "path.bin".
     */
    static std::string arg1(int argc, char **argv, const std::string &key, const std::string &def = "") {
        for (int i = 1; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == key) return std::string(argv[i + 1]);
        }
        return def;
    }

    /**
     * Helper: check if a flag (key) appears anywhere in argv.
     * Example: "--cpu" or "--append".
     */
    static bool has_flag(int argc, char **argv, const std::string &key) {
        for (int i = 1; i < argc; ++i) if (std::string(argv[i]) == key) return true;
        return false;
    }

    /**
     * Helper: collect all values that follow a repeated key.
     * Example:
     *   --w m0.pt --w m1.pt --w m2.pt
     * gives vector{"m0.pt","m1.pt","m2.pt"}.
     */
    static std::vector<std::string> args_multi(int argc, char **argv, const std::string &key) {
        std::vector<std::string> out;
        for (int i = 1; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == key) out.push_back(std::string(argv[i + 1]));
        }
        return out;
    }
} // namespace

int main(int argc, char **argv) {
    try {
        // ----------- Parse command-line arguments -----------
        const std::string pattern = arg1(argc, argv, "--pattern"); // "1_7" or "8_15"
        const std::string pdb_path = arg1(argc, argv, "--pdb");
        const std::string out_path = arg1(argc, argv, "--out");
        const auto weights = args_multi(argc, argv, "--w"); // TorchScript model paths
        const double q = std::stod(arg1(argc, argv, "--q", "0.3"));
        const std::size_t batch = static_cast<std::size_t>(std::stoull(arg1(argc, argv, "--batch", "4096")));
        const std::uint64_t start = static_cast<std::uint64_t>(std::stoull(arg1(argc, argv, "--start", "0")));
        const std::uint64_t count = static_cast<std::uint64_t>(std::stoull(arg1(argc, argv, "--count", "0")));
        const bool use_cpu = has_flag(argc, argv, "--cpu");
        const bool append = has_flag(argc, argv, "--append");

        if (pattern.empty() || pdb_path.empty() || out_path.empty() || weights.empty()) {
            std::cerr <<
                    "Usage:\n"
                    "  build_nn_over_corrections --pattern 1_7|8_15 --pdb <pdb.bin> --out <corr.bin>\n"
                    "                          --w <model_ts.pt> [--w <model2_ts.pt> ...]\n"
                    "                          [--q 0.3] [--batch 4096] [--start 0] [--count 0]\n"
                    "                          [--cpu] [--append]\n";
            return 1;
        }

        // pattern_id and m define the abstract encoding:
        //   1_7  -> pattern_id=17,  m=8  (tiles 1..7 + blank)
        //   8_15 -> pattern_id=815, m=9  (tiles 8..15 + blank)
        const int m = (pattern == "1_7") ? 8 : 9;
        const std::uint32_t pattern_id = (pattern == "1_7") ? 17u : 815u;
        const auto w = make_weights(m);

        // ----------- Open PDB file and determine N -----------

        std::ifstream pdb(pdb_path, std::ios::binary);
        if (!pdb) throw std::runtime_error("Failed to open PDB file: " + pdb_path);

        pdb.seekg(0, std::ios::end);
        const std::uint64_t N = static_cast<std::uint64_t>(pdb.tellg());
        pdb.seekg(0, std::ios::beg);

        // We iterate over ranks in [start, end), where end can be clipped by `count`.
        const std::uint64_t end = (count == 0) ? N : std::min<std::uint64_t>(N, start + count);
        if (start >= end) throw std::runtime_error("Bad range: start>=end");

        // ----------- Device selection (CPU vs CUDA) -----------

        torch::Device device = torch::kCPU;
        if (!use_cpu && torch::cuda::is_available()) device = torch::kCUDA;

        // ----------- Load TorchScript ensemble -----------

        auto models = load_ensemble(weights, device);
        if (models.empty()) throw std::runtime_error("No models loaded.");

        // Infer maximum delta class from the model output shape
        int max_delta = 0;
        {
            torch::InferenceMode guard;
            auto dummy = torch::zeros({1, m}, torch::TensorOptions().dtype(torch::kInt64).device(device));
            auto logits = models[0].forward({dummy}).toTensor();
            if (logits.dim() != 2) throw std::runtime_error("Model output must be [B,C] logits.");
            max_delta = static_cast<int>(logits.size(1)) - 1;
        }

        // ----------- Open / create output file and header -----------

        std::fstream out;
        Header header{};
        if (append) {
            // Append mode: read existing header, then continue writing records at the end.
            out.open(out_path, std::ios::in | std::ios::out | std::ios::binary);
            if (!out) throw std::runtime_error("Failed to open for append: " + out_path);

            out.read(reinterpret_cast<char *>(&header), sizeof(header));
            if (!out) throw std::runtime_error("Failed to read existing header: " + out_path);

            for (int i = 0; i < 8; ++i)
                if (header.magic[i] != MAGIC[i]) throw std::runtime_error("Bad magic in existing file");
            if (header.version != 1) throw std::runtime_error("Bad version in existing file");
            if (header.m != static_cast<std::uint32_t>(m)) throw
                    std::runtime_error("Pattern mismatch in existing file");

            out.seekp(0, std::ios::end);
        } else {
            // Fresh file: write a new header with count=0 (we patch it later).
            out.open(out_path, std::ios::out | std::ios::binary | std::ios::trunc);
            if (!out) throw std::runtime_error("Failed to open output: " + out_path);

            for (int i = 0; i < 8; ++i) header.magic[i] = MAGIC[i];
            header.version = 1;
            header.pattern = pattern_id;
            header.m = static_cast<std::uint32_t>(m);
            header.q = static_cast<float>(q);
            header.count = 0;
            out.write(reinterpret_cast<const char *>(&header), sizeof(header));
        }

        std::uint64_t written = header.count;

        // Seek PDB file to the starting byte (rank index)
        pdb.seekg(static_cast<std::streamoff>(start), std::ios::beg);

        // Buffers reused per batch
        std::vector<std::uint8_t> pdb_vals;
        pdb_vals.reserve(batch);

        std::vector<int> true_delta;
        true_delta.reserve(batch);

        std::uint64_t cur = start;

        // ----------- Main scanning loop over ranks -----------

        while (cur < end) {
            const std::size_t B = static_cast<std::size_t>(std::min<std::uint64_t>(end - cur, batch));

            // 1) Read B PDB entries (one byte per rank) from file
            pdb_vals.assign(B, 0);
            pdb.read(reinterpret_cast<char *>(pdb_vals.data()), static_cast<std::streamsize>(B));
            if (!pdb) throw std::runtime_error("Failed reading PDB bytes");

            // 2) Build Torch input [B, m] and compute true deltas for this batch
            auto x_cpu = torch::empty({static_cast<int64_t>(B), m},
                                      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
            auto xa = x_cpu.accessor<int64_t, 2>();

            true_delta.assign(B, 0);

            for (std::size_t i = 0; i < B; ++i) {
                const std::uint64_t r64 = cur + i;
                if (r64 > 0xFFFFFFFFULL) {
                    throw std::runtime_error("Rank exceeds uint32 range (unexpected for these patterns).");
                }
                const std::uint32_t rank = static_cast<std::uint32_t>(r64);

                int seq[9] = {0};
                unrank_partial_perm(rank, m, w, seq);

                // Fill Torch input row with abstract positions
                for (int j = 0; j < m; ++j) {
                    xa[static_cast<int64_t>(i)][j] = seq[j];
                }

                // Compute delta_true = PDB - Manhattan(pattern)
                const int manh = (m == 8)
                                     ? manhattan_from_seq_1_7(seq)
                                     : manhattan_from_seq_8_15(seq);
                int dt = static_cast<int>(pdb_vals[i]) - manh;
                if (dt < 0) dt = 0;
                if (dt > max_delta) dt = max_delta;
                true_delta[i] = dt;
            }

            auto x = x_cpu.to(device);

            // 3) Ensemble quantile prediction: delta_pred = min over models (quantile index)
            torch::Tensor best; // [B]
            {
                torch::InferenceMode guard;

                bool init = false;
                for (auto &model: models) {
                    auto logits = model.forward({x}).toTensor(); // [B,C]
                    auto d = quantile_index_from_logits(logits, q); // [B]
                    if (!init) {
                        best = d;
                        init = true;
                    } else { best = torch::min(best, d); }
                }
            }

            auto best_cpu = best.to(torch::kCPU);
            auto ba = best_cpu.accessor<int64_t, 1>();

            // 4) For each sample, if delta_pred > delta_true, write (rank, over) record.
            for (std::size_t i = 0; i < B; ++i) {
                const int pred = static_cast<int>(ba[i]);
                const int tru = true_delta[i];
                if (pred > tru) {
                    const std::uint32_t rank = static_cast<std::uint32_t>(cur + i);
                    const int diff = pred - tru;
                    const std::uint8_t over = static_cast<std::uint8_t>(std::min(diff, 255));
                    out.write(reinterpret_cast<const char *>(&rank), sizeof(rank));
                    out.write(reinterpret_cast<const char *>(&over), sizeof(over));
                    ++written;
                }
            }

            cur += B;

            // Periodic progress log (approximately every ~4M ranks)
            if (((cur - start) & ((1ULL << 22) - 1ULL)) == 0) {
                std::cout << "progress rank=" << cur << "/" << end
                        << " written=" << written << "\n";
            }
        }

        // ----------- Patch header.count with final number of records -----------

        out.seekp(0, std::ios::beg);
        for (int i = 0; i < 8; ++i) header.magic[i] = MAGIC[i];
        header.version = 1;
        header.pattern = pattern_id;
        header.m = static_cast<std::uint32_t>(m);
        header.q = static_cast<float>(q);
        header.count = written;
        out.write(reinterpret_cast<const char *>(&header), sizeof(header));
        out.flush();

        std::cout << "DONE. wrote " << written << " records to " << out_path << "\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
