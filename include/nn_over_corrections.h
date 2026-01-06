#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

class NnOverCorrections {
public:
    NnOverCorrections() = default;

    // Loads the binary corrections file (Header + (rank:uint32, over:uint8)*count).
    // The file must be sorted by rank (non-decreasing). Adjacent duplicates are merged (max over).
    void load(const std::string& path);

    void clear();

    bool empty() const { return ranks_.empty(); }

    // Returns "over" for this rank (0 if no correction exists).
    std::uint8_t get(std::uint32_t rank) const;

    std::uint64_t record_count() const { return record_count_; }
    std::uint32_t pattern_id()   const { return pattern_id_; } // 17 or 815 (from header)
    std::uint32_t m()            const { return m_; }          // 8 or 9  (from header)
    float         q()            const { return q_; }          // quantile (from header)

private:
#pragma pack(push, 1)
    struct Header {
        char magic[8];          // "NNOVR001"
        std::uint32_t version;  // 1
        std::uint32_t pattern;  // 17 or 815
        std::uint32_t m;        // 8 or 9
        float q;                // quantile used
        std::uint64_t count;    // records count
    };
#pragma pack(pop)

    static constexpr std::array<char, 8> MAGIC_ = {'N','N','O','V','R','0','0','1'};

    // Sorted by rank
    std::vector<std::uint32_t> ranks_;
    std::vector<std::uint8_t>  overs_;

    // offsets_[p]..offsets_[p+1] is the range in ranks_ whose (rank >> 16) == p
    // size 65537 so that offsets_[65536] == ranks_.size()
    std::array<std::uint32_t, 65537> offsets_{};
    bool offsets_ready_ = false;

    // Header fields (for info / validation)
    std::uint64_t record_count_ = 0;
    std::uint32_t pattern_id_   = 0;
    std::uint32_t m_            = 0;
    float q_                    = 0.0f;

    void build_bucket_offsets_();
};
