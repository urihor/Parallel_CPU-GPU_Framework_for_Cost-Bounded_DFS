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
    // Implementation details for NeuralDelta15 (PIMPL idiom).
    //
    // Holds:
    //   * Two TorchScript models:
    //       - model_1_7  : network for tiles 1–7
    //       - model_8_15 : network for tiles 8–15
    //   * delta_vals_*   : mapping from predicted class index to delta value.
    //   * device         : CPU or CUDA device on which the models run.
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

    // Load an array of integer delta values from a JSON file.
    //
    // The file format is assumed to be a simple JSON array of integers,
    // for example: [0, 1, -1, 2, ...].
    //
    // We parse it manually in a very permissive way:
    //   * scan the file as text
    //   * extract sequences of characters that look like integers
    //   * convert them via std::stoi
    //
    // On failure to open the file, throws std::runtime_error.
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

    // Build a one-hot tensor input for the 1–7 pattern, single state.
    //
    // Layout:
    //   * input shape: [B=1, C=7, H=4, W=4]
    //   * channel c (0..6) corresponds to tile (c+1)
    //   * input[0, c, r, col] = 1.0 if tile (c+1) is at position (r, col)
    //   * all other entries are 0
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

    // Same idea as make_input_1_7, but for the 8–15 pattern.
    //
    // Layout:
    //   * input shape: [B=1, C=8, H=4, W=4]
    //   * channel c (0..7) corresponds to tile (8 + c)
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

    // Batched version of the 1–7 pattern input builder.
    //
    // Layout:
    //   * input shape: [B, C=7, H=4, W=4]
    //   * Batch is built on CPU memory; if device is CUDA, we use pinned
    //     memory and then perform a single async transfer to the GPU.
    static torch::Tensor make_input_1_7_batch(const std::vector<puzzle15_state> &states,
                                              torch::Device device) {
        const int B = static_cast<int>(states.size());
        const int C = 7;
        const int H = 4;
        const int W = 4;

        // Build on CPU (optionally pinned if we will copy to CUDA).
        auto cpu_opts = torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(torch::kCPU)
                .pinned_memory(device.is_cuda());

        torch::Tensor input_cpu = torch::zeros({B, C, H, W}, cpu_opts);

        float *p = input_cpu.data_ptr<float>();
        const int strideC = H * W;
        const int strideB = C * strideC;

        for (int b = 0; b < B; ++b) {
            const auto &tiles = states[b].tiles;
            float *base = p + b * strideB;

            for (int idx = 0; idx < puzzle15_state::Size; ++idx) {
                std::uint8_t tile = tiles[idx];
                if (tile >= 1 && tile <= 7) {
                    int ch = static_cast<int>(tile) - 1;
                    int r = idx / 4;
                    int c = idx % 4;
                    base[ch * strideC + r * W + c] = 1.0f;
                }
            }
        }

        // Single transfer to GPU (if needed).
        if (device.is_cuda()) {
            return input_cpu.to(device, /*non_blocking=*/true);
        }
        return input_cpu;
    }

    // Batched version of the 8–15 pattern input builder.
    //
    // Layout:
    //   * input shape: [B, C=8, H=4, W=4]
    //   * Batch is built on CPU memory; if device is CUDA, we use pinned
    //     memory and then perform a single async transfer to the GPU.
    static torch::Tensor make_input_8_15_batch(const std::vector<puzzle15_state> &states,
                                               torch::Device device) {
        const int B = static_cast<int>(states.size());
        const int C = 8;
        const int H = 4;
        const int W = 4;

        // Build on CPU (optionally pinned if we will copy to CUDA).
        auto cpu_opts = torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(torch::kCPU)
                .pinned_memory(device.is_cuda());

        torch::Tensor input_cpu = torch::zeros({B, C, H, W}, cpu_opts);

        float *p = input_cpu.data_ptr<float>();
        const int strideC = H * W;
        const int strideB = C * strideC;

        for (int b = 0; b < B; ++b) {
            const auto &tiles = states[b].tiles;
            float *base = p + b * strideB;

            for (int idx = 0; idx < puzzle15_state::Size; ++idx) {
                std::uint8_t tile = tiles[idx];
                if (tile >= 8 && tile <= 15) {
                    int ch = static_cast<int>(tile) - 8;
                    int r = idx / 4;
                    int c = idx % 4;
                    base[ch * strideC + r * W + c] = 1.0f;
                }
            }
        }

        // Single transfer to GPU (if needed).
        if (device.is_cuda()) {
            return input_cpu.to(device, /*non_blocking=*/true);
        }
        return input_cpu;
    }

    // ===== NeuralDelta15 methods =====

    // Singleton accessor for NeuralDelta15.
    NeuralDelta15 &NeuralDelta15::instance() {
        static NeuralDelta15 inst;
        return inst;
    }

    // Initialize the neural models and delta tables.
    //
    // - If already initialized, this is a no-op.
    // - Chooses a device: CUDA if available, otherwise CPU.
    // - Loads:
    //     * TorchScript models:
    //         model_dir + "/pdb_1_7_ens2.pt"
    //         model_dir + "/pdb_8_15_ens3.pt"
    //     * Delta value lookup tables from JSON:
    //         model_dir + "/delta_values_1_7.json"
    //         model_dir + "/delta_values_8_15.json"
    //
    // On any load failure, throws std::runtime_error.
    void NeuralDelta15::initialize(const std::string &model_dir) {
        if (initialized_) {
            return;
        }

        impl_ = std::make_unique<Impl>();

        // Choose device: use CUDA if available, otherwise CPU.
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

    // Compute the neural delta for the 1–7 pattern for a single state.
    //
    // Steps:
    //   1. Build input tensor using make_input_1_7.
    //   2. Run the 1–7 model to obtain logits.
    //   3. Take argmax over classes to get a class index.
    //   4. Map class index to an integer delta via delta_vals_1_7.
    //
    // Throws std::runtime_error if not initialized or if the class index
    // is out of range for the delta table.
    int NeuralDelta15::delta_1_7(const puzzle15_state &s) const {
        if (!initialized_) {
            throw std::runtime_error("NeuralDelta15::initialize() must be called before use.");
        }

        // Build a batch of size 1 on CPU (pinned if CUDA) and perform a single
        // host->device transfer, instead of writing element-by-element on the GPU.
        std::vector<puzzle15_state> states;
        states.reserve(1);
        states.push_back(s);

        torch::Tensor input = make_input_1_7_batch(states, impl_->device);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);

        NVTX_RANGE("NeuralDelta15::delta_1_7_single");

        // logits shape: [1, num_classes]
        torch::Tensor logits = impl_->model_1_7.forward(inputs).toTensor();
        auto pred = logits.argmax(1).item<int64_t>();

        if (pred < 0 || static_cast<size_t>(pred) >= impl_->delta_vals_1_7.size()) {
            throw std::runtime_error("delta_1_7: predicted class out of range");
        }
        return impl_->delta_vals_1_7[static_cast<size_t>(pred)];
    }


    // Same as delta_1_7, but for the 8–15 pattern.
    int NeuralDelta15::delta_8_15(const puzzle15_state &s) const {
        if (!initialized_) {
            throw std::runtime_error("NeuralDelta15::initialize() must be called before use.");
        }

        // Build a batch of size 1 on CPU (pinned if CUDA) and perform a single
        // host->device transfer, instead of writing element-by-element on the GPU.
        std::vector<puzzle15_state> states;
        states.reserve(1);
        states.push_back(s);

        torch::Tensor input = make_input_8_15_batch(states, impl_->device);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);

        NVTX_RANGE("NeuralDelta15::delta_8_15_single");

        // logits shape: [1, num_classes]
        torch::Tensor logits = impl_->model_8_15.forward(inputs).toTensor();
        auto pred = logits.argmax(1).item<int64_t>();

        if (pred < 0 || static_cast<size_t>(pred) >= impl_->delta_vals_8_15.size()) {
            throw std::runtime_error("delta_8_15: predicted class out of range");
        }
        return impl_->delta_vals_8_15[static_cast<size_t>(pred)];
    }


    // Batched version of delta_1_7.
    //
    // Input:
    //   states - vector of puzzle states.
    //
    // Output:
    //   vector<int> of size states.size(), where each entry is the delta
    //   for the corresponding state, according to the 1–7 model.
    //
    // Throws std::runtime_error if not initialized or if any predicted
    // class index is out of range.
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

    // Batched version of delta_8_15.
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

    // Compute h_M(s) given an already known Manhattan heuristic value.
    //
    // h_M(s) = manhattan_heuristic + delta_1_7(s) + delta_8_15(s)
    int NeuralDelta15::h_M(const puzzle15_state &s, int manhattan_heuristic) const {
        int d1 = delta_1_7(s);
        int d2 = delta_8_15(s);
        return manhattan_heuristic + d1 + d2;
    }

    // Compute h_M(s) for a single state, including the base heuristic.
    //
    // Currently the Manhattan part is set to 0 here (placeholder).
    // Replace md = 0 with a call to manhattan_15(s) (or similar) to use
    // the true Manhattan heuristic.
    int NeuralDelta15::h_M_single(const puzzle15_state &s) const {
        if (!initialized_) {
            throw std::runtime_error("NeuralDelta15::initialize() must be called before use.");
        }
        /*int md = manhattan_15(s);*/
        int md = 0;
        return h_M(s, md);
    }

    // Batched version of h_M(s).
    //
    // Steps:
    //   1. Compute batched deltas for 1–7 and 8–15 on the GPU.
    //   2. For each state, add the base heuristic (currently md = 0) and deltas.
    //
    // Returns a vector<int> of size states.size(), with h_M for each state.
    std::vector<int> NeuralDelta15::h_M_batch(const std::vector<puzzle15_state> &states) const {
        if (!initialized_) {
            throw std::runtime_error("NeuralDelta15::initialize() must be called before use.");
        }
        if (states.empty()) {
            return {};
        }

        // 1. Batched deltas for patterns 1–7 and 8–15 on the GPU.
        NVTX_RANGE("NN: h_M_batch");
        std::vector<int> d1 = delta_1_7_batch(states);
        std::vector<int> d2 = delta_8_15_batch(states);

        if (d1.size() != states.size() || d2.size() != states.size()) {
            throw std::runtime_error("h_M_batch: delta batch size mismatch");
        }

        // 2. Manhattan for each state (CPU – relatively cheap).
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

    // Initialize a default NeuralBatchService that uses NeuralDelta15::h_M_batch
    // to evaluate batches of states asynchronously (e.g. in the GPU worker).
    //
    // This:
    //   * Sets up a BatchComputeFn that calls NeuralDelta15::instance().h_M_batch(...)
    //   * Starts the global NeuralBatchService with:
    //       max_batch_size = 800
    //       max_wait       = 200 microseconds
    //   * Enables neural batching in the batch IDA* code via set_neural_batch_enabled(true).
    void init_default_batch_service() {
        auto batch_fn = [](const std::vector<puzzle15_state> &batch,
                           std::vector<int> &out) {
            out = NeuralDelta15::instance().h_M_batch(batch);
        };

        NeuralBatchService::instance().start(
            batch_fn,
            /*max_batch_size=*/800,
            std::chrono::microseconds(200)
        );

        batch_ida::set_neural_batch_enabled(true);
    }
} // namespace neural15
