#include "nn_over_corrections.h"

#include <algorithm>
#include <fstream>
#include <stdexcept>

namespace {
#pragma pack(push, 1)
struct Header {
    char magic[8];          // "NNOVR001"
    std::uint32_t version;  // 1
    std::uint32_t pattern;  // 17 or 815 (info)
    std::uint32_t m;        // 8 or 9
    float q;                // quantile used when building
    std::uint64_t count;    // number of records
};
#pragma pack(pop)

static constexpr char MAGIC[8] = {'N','N','O','V','R','0','0','1'};
}

void NnOverCorrections::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open corrections file: " + path);
    }

    Header h{};
    in.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!in) {
        throw std::runtime_error("Failed to read corrections header: " + path);
    }

    for (int i = 0; i < 8; ++i) {
        if (h.magic[i] != MAGIC[i]) {
            throw std::runtime_error("Bad corrections magic (wrong file?): " + path);
        }
    }
    if (h.version != 1) {
        throw std::runtime_error("Unsupported corrections version: " + std::to_string(h.version));
    }

    recs_.clear();
    recs_.resize(static_cast<std::size_t>(h.count));

    if (!recs_.empty()) {
        in.read(reinterpret_cast<char*>(recs_.data()),
                static_cast<std::streamsize>(recs_.size() * sizeof(Rec)));
        if (!in) {
            throw std::runtime_error("Truncated corrections file: " + path);
        }

        // Ensure sorted (in case you appended chunks out of order)
        bool sorted = true;
        for (std::size_t i = 1; i < recs_.size(); ++i) {
            if (recs_[i-1].rank > recs_[i].rank) { sorted = false; break; }
        }
        if (!sorted) {
            std::sort(recs_.begin(), recs_.end(),
                      [](const Rec& a, const Rec& b){ return a.rank < b.rank; });
        }

        // Optional: if duplicates exist (overlapping chunks), keep the MAX over per rank
        std::size_t w = 0;
        for (std::size_t i = 0; i < recs_.size(); ) {
            std::uint32_t r = recs_[i].rank;
            std::uint8_t best = recs_[i].over;
            std::size_t j = i + 1;
            while (j < recs_.size() && recs_[j].rank == r) {
                if (recs_[j].over > best) best = recs_[j].over;
                ++j;
            }
            recs_[w++] = Rec{r, best};
            i = j;
        }
        recs_.resize(w);
    }
}

std::uint8_t NnOverCorrections::get(std::uint32_t rank) const {
    auto it = std::lower_bound(
        recs_.begin(), recs_.end(), rank,
        [](const Rec& a, std::uint32_t key){ return a.rank < key; }
    );
    if (it == recs_.end() || it->rank != rank) return 0;
    return it->over;
}
