#include "neural_delta_15_quantile.h"
#include "nn_over_corrections.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>
#include <torch/script.h>

namespace neural15 {

// -------------------------
// Helpers
// -------------------------

static int goal_index(int tile) {
    return (tile == 0) ? 15 : (tile - 1); // goal: [1..15,0]
}

static int manhattan_full(const puzzle15_state& s) {
    std::array<int, 16> pos{};
    for (int i = 0; i < 16; ++i) pos[s.tiles[i]] = i;

    int sum = 0;
    for (int t = 1; t <= 15; ++t) {
        const int p = pos[t];
        const int g = goal_index(t);
        sum += std::abs(p / 4 - g / 4) + std::abs(p % 4 - g % 4);
    }
    return sum;
}

static torch::Tensor quantile_index_from_logits(const torch::Tensor& logits, double q) {
    auto probs = torch::softmax(logits, /*dim=*/1);  // [B,C]
    auto cdf   = torch::cumsum(probs, /*dim=*/1);    // [B,C]
    auto mask  = cdf.ge(q).to(torch::kInt64);        // [B,C] 0/1
    return mask.argmax(/*dim=*/1);                   // [B]
}

static std::uint64_t perm_count(int n, int k) {
    if (k <= 0) return 1;
    std::uint64_t r = 1;
    for (int i = 0; i < k; ++i) r *= static_cast<std::uint64_t>(n - i);
    return r;
}

static std::array<std::uint64_t, 9> make_weights(int m) {
    std::array<std::uint64_t, 9> w{};
    for (int i = 0; i < m; ++i) {
        const int n_rem = 16 - i - 1;
        const int k_rem = m - i - 1;
        w[i] = perm_count(n_rem, k_rem);
    }
    return w;
}

static std::uint32_t rank_partial_perm(const int* seq, int m, const std::array<std::uint64_t, 9>& w) {
    bool used[16] = {false};
    std::uint64_t rank = 0;

    for (int i = 0; i < m; ++i) {
        const int x = seq[i];
        int smaller_unused = 0;
        for (int v = 0; v < x; ++v) {
            if (!used[v]) ++smaller_unused;
        }
        rank += static_cast<std::uint64_t>(smaller_unused) * w[i];
        used[x] = true;
    }
    return static_cast<std::uint32_t>(rank);
}

    static torch::jit::Module load_ts(const std::string& path,
                                      const torch::Device& device,
                                      bool use_half_on_cuda)
{
    auto m = torch::jit::load(path);
    m.eval();
    m.to(device);
    if (device.is_cuda() && use_half_on_cuda) {
        m.to(torch::kHalf);
    }
    return m;
}


// -------------------------
// Impl
// -------------------------

struct NeuralDelta15Quantile::Impl {
    NeuralDelta15QuantileOptions opt;

    std::vector<torch::jit::script::Module> nets_1_7;
    std::vector<torch::jit::script::Module> nets_8_15;

    bool have_corr_1_7 = false;
    bool have_corr_8_15 = false;
    NnOverCorrections corr_1_7;
    NnOverCorrections corr_8_15;

    std::array<std::uint64_t, 9> w8{};
    std::array<std::uint64_t, 9> w9{};

    explicit Impl(NeuralDelta15QuantileOptions o) : opt(std::move(o)) {
        if (opt.weights_1_7.empty()) throw std::runtime_error("weights_1_7 is empty");
        if (opt.weights_8_15.empty()) throw std::runtime_error("weights_8_15 is empty");
        if (!(opt.quantile_q > 0.0 && opt.quantile_q <= 1.0)) throw std::runtime_error("quantile_q must be in (0,1]");

        w8 = make_weights(8);
        w9 = make_weights(9);

        // corrections (optional)
        if (!opt.corrections_1_7_path.empty()) {
            corr_1_7.load(opt.corrections_1_7_path);
            have_corr_1_7 = true;
        }
        if (!opt.corrections_8_15_path.empty()) {
            corr_8_15.load(opt.corrections_8_15_path);
            have_corr_8_15 = true;
        }

        // load TorchScript ensembles
        nets_1_7.reserve(opt.weights_1_7.size());
        for (const auto& p : opt.weights_1_7) {
            nets_1_7.emplace_back(load_ts(p, opt.device, opt.use_half_on_cuda));
        }

        nets_8_15.reserve(opt.weights_8_15.size());
        for (const auto& p : opt.weights_8_15) {
            nets_8_15.emplace_back(load_ts(p, opt.device, opt.use_half_on_cuda));
        }

        // warmup
        {
            torch::InferenceMode guard;
            auto x1 = torch::zeros({1, 8}, torch::TensorOptions().dtype(torch::kInt64).device(opt.device));
            auto x2 = torch::zeros({1, 9}, torch::TensorOptions().dtype(torch::kInt64).device(opt.device));
            (void)nets_1_7[0].forward({x1}).toTensor();
            (void)nets_8_15[0].forward({x2}).toTensor();
        }
        if (opt.device.is_cuda()) torch::cuda::synchronize();
    }

    void compute_batch(const std::vector<puzzle15_state>& batch, std::vector<int>& hs) {
        hs.resize(batch.size());
        if (batch.empty()) return;

        const int64_t B = static_cast<int64_t>(batch.size());

        // Build inputs on CPU + ranks for corrections
        auto x1_cpu = torch::empty({B, 8}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        auto x2_cpu = torch::empty({B, 9}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        auto a1 = x1_cpu.accessor<int64_t, 2>();
        auto a2 = x2_cpu.accessor<int64_t, 2>();

        std::vector<std::uint32_t> r1, r2;
        if (have_corr_1_7) r1.resize(batch.size());
        if (have_corr_8_15) r2.resize(batch.size());

        for (int64_t i = 0; i < B; ++i) {
            std::array<int, 16> pos{};
            const auto& tiles = batch[i].tiles;
            for (int j = 0; j < 16; ++j) pos[tiles[j]] = j;

            // 1..7: [pos(1)..pos(7), pos(0)]
            int seq8[8];
            for (int t = 1; t <= 7; ++t) {
                const int p = pos[t];
                a1[i][t - 1] = p;
                seq8[t - 1] = p;
            }
            a1[i][7] = pos[0];
            seq8[7] = pos[0];

            // 8..15: [pos(8)..pos(15), pos(0)]
            int seq9[9];
            int k = 0;
            for (int t = 8; t <= 15; ++t) {
                const int p = pos[t];
                a2[i][k] = p;
                seq9[k] = p;
                ++k;
            }
            a2[i][8] = pos[0];
            seq9[8] = pos[0];

            if (have_corr_1_7) r1[static_cast<std::size_t>(i)] = rank_partial_perm(seq8, 8, w8);
            if (have_corr_8_15) r2[static_cast<std::size_t>(i)] = rank_partial_perm(seq9, 9, w9);
        }

        auto x1 = x1_cpu.to(opt.device);
        auto x2 = x2_cpu.to(opt.device);

        torch::Tensor best_d1;
        torch::Tensor best_d2;

        {
            torch::InferenceMode guard;

            // 1..7 ensemble min
            bool init1 = false;
            for (auto& m : nets_1_7) {
                auto logits = m.forward({x1}).toTensor();
                auto d = quantile_index_from_logits(logits, opt.quantile_q);
                if (!init1) { best_d1 = d; init1 = true; }
                else { best_d1 = torch::min(best_d1, d); }
            }

            // 8..15 ensemble min
            bool init2 = false;
            for (auto& m : nets_8_15) {
                auto logits = m.forward({x2}).toTensor();
                auto d = quantile_index_from_logits(logits, opt.quantile_q);
                if (!init2) { best_d2 = d; init2 = true; }
                else { best_d2 = torch::min(best_d2, d); }
            }
        }

        auto d1_cpu = best_d1.to(torch::kCPU);
        auto d2_cpu = best_d2.to(torch::kCPU);
        auto p1 = d1_cpu.accessor<int64_t, 1>();
        auto p2 = d2_cpu.accessor<int64_t, 1>();

        for (std::size_t i = 0; i < batch.size(); ++i) {
            int base = opt.add_manhattan ? manhattan_full(batch[i]) : 0;

            int d1 = static_cast<int>(p1[i]);
            int d2 = static_cast<int>(p2[i]);

            if (have_corr_1_7) {
                const int over = static_cast<int>(corr_1_7.get(r1[i]));
                d1 -= over;
                if (d1 < 0) d1 = 0;
            }
            if (have_corr_8_15) {
                const int over = static_cast<int>(corr_8_15.get(r2[i]));
                d2 -= over;
                if (d2 < 0) d2 = 0;
            }

            hs[i] = base + d1 + d2;
        }
    }

    int compute_one(const puzzle15_state& s) {
        std::vector<puzzle15_state> b{ s };
        std::vector<int> out;
        compute_batch(b, out);
        return out[0];
    }
};

NeuralDelta15Quantile::NeuralDelta15Quantile(const NeuralDelta15QuantileOptions& opt)
    : impl_(std::make_shared<Impl>(opt)) {}

void NeuralDelta15Quantile::compute_batch(const std::vector<puzzle15_state>& batch, std::vector<int>& hs) {
    impl_->compute_batch(batch, hs);
}

int NeuralDelta15Quantile::compute_one(const puzzle15_state& s) {
    return impl_->compute_one(s);
}

} // namespace neural15
