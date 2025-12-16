//
// Created by Owner on 07/12/2025.
//
#include "neural_delta_15.h"
#include "nvtx_helpers.h"


#include <torch/script.h>
#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace neural15 {
    struct NeuralDelta15::Impl {
        torch::jit::script::Module model_1_7;
        torch::jit::script::Module model_8_15;
        std::vector<int> delta_vals_1_7;
        std::vector<int> delta_vals_8_15;
        torch::Device device;

        Impl() : device(torch::kCPU) {
        }
    };

    // ===== Utilities =====

    static std::vector<int> load_delta_values_json(const std::string &path) {
        std::ifstream in(path);
        if (!in) {
            throw std::runtime_error("Failed to open delta values file: " + path);
        }
        std::stringstream buffer;
        buffer << in.rdbuf();
        std::string s = buffer.str();

        std::vector<int> result;
        std::string num;
        bool in_number = false;
        for (char c: s) {
            if ((c >= '0' && c <= '9') || (c == '-')) {
                num.push_back(c);
                in_number = true;
            } else {
                if (in_number) {
                    result.push_back(std::stoi(num));
                    num.clear();
                    in_number = false;
                }
            }
        }
        if (in_number && !num.empty()) {
            result.push_back(std::stoi(num));
        }
        return result;
    }

    // פונקציה שעושה טנזור one-hot לפטרן 1-7
    static torch::Tensor make_input_1_7(const puzzle15_state &s, torch::Device device) {
        const auto &tiles = s.tiles; // std::array<puzzle15_state::Tile, puzzle15_state::Size>

        const int B = 1;
        const int C = 7;
        const int H = 4;
        const int W = 4;

        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        torch::Tensor input = torch::zeros({B, C, H, W}, options);

        for (int idx = 0; idx < puzzle15_state::Size; ++idx) {
            std::uint8_t tile = tiles[idx];
            if (tile >= 1 && tile <= 7) {
                int ch = static_cast<int>(tile) - 1; // 1..7 -> 0..6
                int r = idx / 4;
                int c = idx % 4;
                input[0][ch][r][c] = 1.0f;
            }
        }

        return input;
    }

    // אותו דבר לפטרן 8-15
    static torch::Tensor make_input_8_15(const puzzle15_state &s, torch::Device device) {
        const auto &tiles = s.tiles;

        const int B = 1;
        const int C = 8;
        const int H = 4;
        const int W = 4;

        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        torch::Tensor input = torch::zeros({B, C, H, W}, options);

        for (int idx = 0; idx < puzzle15_state::Size; ++idx) {
            std::uint8_t tile = tiles[idx];
            if (tile >= 8 && tile <= 15) {
                int ch = static_cast<int>(tile) - 8; // 8..15 -> 0..7
                int r = idx / 4;
                int c = idx % 4;
                input[0][ch][r][c] = 1.0f;
            }
        }

        return input;
    }

    // גרסת batch ל-1-7
    static torch::Tensor make_input_1_7_batch(const std::vector<puzzle15_state> &states,
                                              torch::Device device) {
        const int B = static_cast<int>(states.size());
        const int C = 7;
        const int H = 4;
        const int W = 4;

        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        torch::Tensor input = torch::zeros({B, C, H, W}, options);

        for (int b = 0; b < B; ++b) {
            const auto &tiles = states[b].tiles;
            for (int idx = 0; idx < puzzle15_state::Size; ++idx) {
                std::uint8_t tile = tiles[idx];
                if (tile >= 1 && tile <= 7) {
                    int ch = static_cast<int>(tile) - 1;
                    int r = idx / 4;
                    int c = idx % 4;
                    input[b][ch][r][c] = 1.0f;
                }
            }
        }

        return input;
    }


    // גרסת batch ל-8-15
    static torch::Tensor make_input_8_15_batch(const std::vector<puzzle15_state> &states,
                                               torch::Device device) {
        const int B = static_cast<int>(states.size());
        const int C = 8;
        const int H = 4;
        const int W = 4;

        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        torch::Tensor input = torch::zeros({B, C, H, W}, options);

        for (int b = 0; b < B; ++b) {
            const auto &tiles = states[b].tiles;
            for (int idx = 0; idx < puzzle15_state::Size; ++idx) {
                std::uint8_t tile = tiles[idx];
                if (tile >= 8 && tile <= 15) {
                    int ch = static_cast<int>(tile) - 8;
                    int r = idx / 4;
                    int c = idx % 4;
                    input[b][ch][r][c] = 1.0f;
                }
            }
        }

        return input;
    }


    // ===== NeuralDelta15 methods =====

    NeuralDelta15 &NeuralDelta15::instance() {
        static NeuralDelta15 inst;
        return inst;
    }

    void NeuralDelta15::initialize(const std::string &model_dir) {
        if (initialized_) {
            return;
        }

        impl_ = std::make_unique<Impl>();

        // בוחרים device – אם CUDA זמין נשתמש בו
        if (torch::cuda::is_available()) {
            impl_->device = torch::Device(torch::kCUDA);
        } else {
            impl_->device = torch::Device(torch::kCPU);
        }

        const std::string model_1_7_path = model_dir + "/pdb_1_7_ens2.pt";
        const std::string model_8_15_path = model_dir + "/pdb_8_15_ens3.pt";
        const std::string delta_1_7_path = model_dir + "/delta_values_1_7.json";
        const std::string delta_8_15_path = model_dir + "/delta_values_8_15.json";

        try {
            impl_->model_1_7 = torch::jit::load(model_1_7_path, impl_->device);
            impl_->model_8_15 = torch::jit::load(model_8_15_path, impl_->device);
        } catch (const c10::Error &e) {
            throw std::runtime_error(std::string("Failed to load TorchScript models: ") + e.what());
        }

        impl_->delta_vals_1_7 = load_delta_values_json(delta_1_7_path);
        impl_->delta_vals_8_15 = load_delta_values_json(delta_8_15_path);

        initialized_ = true;
    }

    int NeuralDelta15::delta_1_7(const puzzle15_state &s) const {
        if (!initialized_) {
            throw std::runtime_error("NeuralDelta15::initialize() must be called before use.");
        }

        torch::Tensor input = make_input_1_7(s, impl_->device);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        NVTX_RANGE("NeuralDelta15::delta_1_7_batch");

        torch::Tensor logits = impl_->model_1_7.forward(inputs).toTensor();
        // logits shape: [1, num_classes]
        auto pred = logits.argmax(1).item<int64_t>();
        if (pred < 0 || static_cast<size_t>(pred) >= impl_->delta_vals_1_7.size()) {
            throw std::runtime_error("delta_1_7: predicted class out of range");
        }
        return impl_->delta_vals_1_7[static_cast<size_t>(pred)];
    }

    int NeuralDelta15::delta_8_15(const puzzle15_state &s) const {
        if (!initialized_) {
            throw std::runtime_error("NeuralDelta15::initialize() must be called before use.");
        }

        torch::Tensor input = make_input_8_15(s, impl_->device);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);

        torch::Tensor logits = impl_->model_8_15.forward(inputs).toTensor();
        auto pred = logits.argmax(1).item<int64_t>();
        if (pred < 0 || static_cast<size_t>(pred) >= impl_->delta_vals_8_15.size()) {
            throw std::runtime_error("delta_8_15: predicted class out of range");
        }
        return impl_->delta_vals_8_15[static_cast<size_t>(pred)];
    }

    std::vector<int> NeuralDelta15::delta_1_7_batch(const std::vector<puzzle15_state> &states) const {
        if (!initialized_) {
            throw std::runtime_error("NeuralDelta15::initialize() must be called before use.");
        }
        if (states.empty()) {
            return {};
        }

        torch::Tensor input = make_input_1_7_batch(states, impl_->device);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        NVTX_RANGE("NeuralDelta15::delta_1_7_batch");

        torch::Tensor logits = impl_->model_1_7.forward(inputs).toTensor();
        // logits shape: [B, num_classes]
        torch::Tensor preds = logits.argmax(1); // (B)
        preds = preds.to(torch::kCPU).contiguous();

        std::vector<int> result;
        result.reserve(states.size());

        auto preds_acc = preds.accessor<int64_t, 1>();

        for (int64_t i = 0; i < preds.size(0); ++i) {
            int64_t cls = preds_acc[i];
            if (cls < 0 || static_cast<size_t>(cls) >= impl_->delta_vals_1_7.size()) {
                throw std::runtime_error("delta_1_7_batch: predicted class out of range");
            }
            result.push_back(impl_->delta_vals_1_7[static_cast<size_t>(cls)]);
        }
        return result;
    }

    std::vector<int> NeuralDelta15::delta_8_15_batch(const std::vector<puzzle15_state> &states) const {
        if (!initialized_) {
            throw std::runtime_error("NeuralDelta15::initialize() must be called before use.");
        }
        if (states.empty()) {
            return {};
        }

        torch::Tensor input = make_input_8_15_batch(states, impl_->device);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        NVTX_RANGE("NeuralDelta15::delta_8_15_batch");

        torch::Tensor logits = impl_->model_8_15.forward(inputs).toTensor();
        torch::Tensor preds = logits.argmax(1);
        preds = preds.to(torch::kCPU).contiguous();

        std::vector<int> result;
        result.reserve(states.size());
        auto preds_acc = preds.accessor<int64_t, 1>();
        for (int i = 0; i < preds.size(0); ++i) {
            int64_t cls = preds_acc[i];
            if (cls < 0 || static_cast<size_t>(cls) >= impl_->delta_vals_8_15.size()) {
                throw std::runtime_error("delta_8_15_batch: predicted class out of range");
            }
            result.push_back(impl_->delta_vals_8_15[static_cast<size_t>(cls)]);
        }
        return result;
    }

    int NeuralDelta15::h_M(const puzzle15_state &s, int manhattan_heuristic) const {
        int d1 = delta_1_7(s);
        int d2 = delta_8_15(s);
        return manhattan_heuristic + d1 + d2;
    }

    int NeuralDelta15::h_M_single(const puzzle15_state &s) const {
        if (!initialized_) {
            throw std::runtime_error("NeuralDelta15::initialize() must be called before use.");
        }
        /*int md = manhattan_15(s);*/
        int md = 0;
        return h_M(s, md);
    }

    std::vector<int> NeuralDelta15::h_M_batch(const std::vector<puzzle15_state> &states) const {
        if (!initialized_) {
            throw std::runtime_error("NeuralDelta15::initialize() must be called before use.");
        }
        if (states.empty()) {
            return {};
        }

        // 1. דלתא לפטרן 1–7 ו-8–15 ב-batch, על ה-GPU
        NVTX_RANGE("NN: h_M_batch");
        std::vector<int> d1 = delta_1_7_batch(states);
        std::vector<int> d2 = delta_8_15_batch(states);

        if (d1.size() != states.size() || d2.size() != states.size()) {
            throw std::runtime_error("h_M_batch: delta batch size mismatch");
        }

        // 2. מנהטן לכל מצב (CPU – זול יחסית)
        std::vector<int> result;
        result.reserve(states.size());
        for (std::size_t i = 0; i < states.size(); ++i) {
            /*int md = manhattan_15(states[i]);*/
            int md = 0;
            int h = md + d1[i] + d2[i];
            result.push_back(h);
        }

        return result;
    }

    void init_default_batch_service() {
        auto batch_fn = [](const std::vector<puzzle15_state> &batch,
                           std::vector<int> &out) {
            out = NeuralDelta15::instance().h_M_batch(batch);
        };

        NeuralBatchService::instance().start(
            batch_fn,
            /*max_batch_size=*/8000,
            std::chrono::microseconds(100)
        );

        batch_ida::set_neural_batch_enabled(true);
    }
} // namespace neural15
