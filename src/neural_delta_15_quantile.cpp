#include "neural_delta_15_quantile.h"

#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <torch/script.h>   // torch::jit::load

namespace neural15 {

// -------------------------
// Helpers
// -------------------------

static int goal_index(int tile) {
    return (tile == 0) ? 15 : (tile - 1);
}

static int manhattan_full(const puzzle15_state& s) {
    std::array<int, 16> pos{};
    for (int i = 0; i < 16; ++i) pos[s.tiles[i]] = i;

    int sum = 0;
    for (int t = 1; t <= 15; ++t) {
        const int p = pos[t];
        const int g = goal_index(t);
        const int pr = p / 4, pc = p % 4;
        const int gr = g / 4, gc = g % 4;
        sum += std::abs(pr - gr) + std::abs(pc - gc);
    }
    return sum;
}

static torch::Tensor quantile_index_from_logits(const torch::Tensor& logits, double q) {
    auto probs = torch::softmax(logits, /*dim=*/1);
    auto cdf   = torch::cumsum(probs, /*dim=*/1);
    auto mask  = cdf.ge(q).to(torch::kInt64);
    return mask.argmax(/*dim=*/1); // [B]
}

static torch::Tensor build_x1_cpu(const std::vector<puzzle15_state>& batch) {
    // [B,8] = [pos(1)..pos(7), pos(0)]  (blank LAST)
    const int64_t B = (int64_t)batch.size();
    auto x = torch::empty({B, 8}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto a = x.accessor<int64_t, 2>();

    for (int64_t i = 0; i < B; ++i) {
        std::array<int64_t, 16> pos{};
        const auto& tiles = batch[i].tiles;
        for (int j = 0; j < 16; ++j) pos[tiles[j]] = j;

        for (int t = 1; t <= 7; ++t) a[i][t - 1] = pos[t];
        a[i][7] = pos[0];
    }
    return x;
}

static torch::Tensor build_x2_cpu(const std::vector<puzzle15_state>& batch) {
    // [B,9] = [pos(8)..pos(15), pos(0)]  (blank LAST)
    const int64_t B = (int64_t)batch.size();
    auto x = torch::empty({B, 9}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto a = x.accessor<int64_t, 2>();

    for (int64_t i = 0; i < B; ++i) {
        std::array<int64_t, 16> pos{};
        const auto& tiles = batch[i].tiles;
        for (int j = 0; j < 16; ++j) pos[tiles[j]] = j;

        int k = 0;
        for (int t = 8; t <= 15; ++t) a[i][k++] = pos[t];
        a[i][8] = pos[0];
    }
    return x;
}

// -------------------------
// Impl
// -------------------------

struct NeuralDelta15Quantile::Impl {
    NeuralDelta15QuantileOptions opt;

    std::vector<torch::jit::script::Module> mods_1_7;
    std::vector<torch::jit::script::Module> mods_8_15;

    explicit Impl(NeuralDelta15QuantileOptions o) : opt(std::move(o)) {
        if (opt.weights_1_7.empty()) throw std::runtime_error("weights_1_7 is empty");
        if (opt.weights_8_15.empty()) throw std::runtime_error("weights_8_15 is empty");
        if (!(opt.quantile_q > 0.0 && opt.quantile_q <= 1.0)) throw std::runtime_error("quantile_q must be in (0,1]");

        mods_1_7.reserve(opt.weights_1_7.size());
        for (const auto& p : opt.weights_1_7) {
            auto m = torch::jit::load(p);
            m.to(opt.device);
            m.eval();
            mods_1_7.push_back(std::move(m));
        }

        mods_8_15.reserve(opt.weights_8_15.size());
        for (const auto& p : opt.weights_8_15) {
            auto m = torch::jit::load(p);
            m.to(opt.device);
            m.eval();
            mods_8_15.push_back(std::move(m));
        }

        // warmup (optional)
        std::vector<puzzle15_state> dummy(1);
        std::vector<int> tmp;
        compute_batch(dummy, tmp);
        if (opt.device.is_cuda()) torch::cuda::synchronize();
    }

    void compute_batch(const std::vector<puzzle15_state>& batch, std::vector<int>& hs) {
        hs.resize(batch.size());
        if (batch.empty()) return;

        torch::InferenceMode guard(true);

        auto x1 = build_x1_cpu(batch).to(opt.device);
        auto x2 = build_x2_cpu(batch).to(opt.device);

        // 1..7: min over models of quantile index
        torch::Tensor best_d1;
        for (size_t i = 0; i < mods_1_7.size(); ++i) {
            auto logits = mods_1_7[i].forward({x1}).toTensor();   // [B,C]
            auto d = quantile_index_from_logits(logits, opt.quantile_q);
            best_d1 = (i == 0) ? d : torch::min(best_d1, d);
        }

        // 8..15: ensemble min
        torch::Tensor best_d2;
        for (size_t i = 0; i < mods_8_15.size(); ++i) {
            auto logits = mods_8_15[i].forward({x2}).toTensor();   // [B,C]
            auto d = quantile_index_from_logits(logits, opt.quantile_q);
            best_d2 = (i == 0) ? d : torch::min(best_d2, d);
        }

        auto d1_cpu = best_d1.to(torch::kCPU);
        auto d2_cpu = best_d2.to(torch::kCPU);

        auto a1 = d1_cpu.accessor<int64_t, 1>();
        auto a2 = d2_cpu.accessor<int64_t, 1>();

        for (size_t i = 0; i < batch.size(); ++i) {
            const int base = opt.add_manhattan ? manhattan_full(batch[i]) : 0;
            hs[i] = base + (int)a1[i] + (int)a2[i];
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
