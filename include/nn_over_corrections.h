#pragma once

#include <cstdint>
#include <string>
#include <vector>

class NnOverCorrections {
public:
    // Loads a sparse corrections file: rank -> over (uint8).
    // Supports large files efficiently (no unordered_map).
    void load(const std::string& path);

    // Returns over(rank) if present, else 0.
    [[nodiscard]] std::uint8_t get(std::uint32_t rank) const;

    [[nodiscard]] std::size_t size() const { return recs_.size(); }

private:
#pragma pack(push, 1)
    struct Rec {
        std::uint32_t rank;
        std::uint8_t over;
    };
#pragma pack(pop)

    std::vector<Rec> recs_; // sorted by rank after load
};
