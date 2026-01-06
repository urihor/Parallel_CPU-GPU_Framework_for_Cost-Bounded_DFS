#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

/**
 * NnOverCorrections
 *
 * Stores a sparse correction table for neural heuristics, indexed by a
 * “rank” of an abstract state (partial permutation encoding).
 *
 * For each abstract state rank r we can store an integer "over" value
 * such that:
 *
 *      d_corrected = max(0, d_raw - over(r))
 *
 * where d_raw is the raw (possibly inadmissible) neural delta.  By
 * subtracting these worst-case overestimates we can guarantee that
 * the corrected heuristic never exceeds the PDB value.
 *
 * The data is stored in a compact binary format:
 *
 *   Header (packed):
 *     magic[8]  = "NNOVR001"
 *     version   = 1
 *     pattern   = 17 or 815   (meaning pattern 1–7 or 8–15)
 *     m         = 8 or 9      (# of positions used in the rank)
 *     q         = quantile used during training/evaluation
 *     count     = number of (rank, over) records
 *
 *   Followed by `count` records:
 *     rank : uint32_t
 *     over : uint8_t   (max overestimation observed for this rank)
 *
 * The file must be sorted by rank (non-decreasing). If there are
 * repeated ranks in the file, we merge them by taking the maximum
 * over value.
 *
 * At runtime, we:
 *   - Load all (rank, over) pairs into two arrays.
 *   - Build a bucketed index (offsets_) over the top 16 bits of rank
 *     to accelerate lookups.
 *   - Provide `get(rank)` to retrieve the correction value in O(1)
 *     average time with a very small memory footprint.
 */
class NnOverCorrections {
public:
    NnOverCorrections() = default;

    /**
     * Load corrections from a binary file.
     *
     * The file must follow the Header + (rank,over)*count format described
     * above. The ranks must be sorted in non-decreasing order; if any
     * duplicates appear, we keep the maximum over value for that rank.
     *
     * On success:
     *   - `ranks_` and `overs_` are filled.
     *   - `offsets_` is built for fast lookup.
     *   - header fields (`record_count_`, `pattern_id_`, `m_`, `q_`) are set.
     *
     * On failure (e.g. bad magic/version or I/O error) this throws.
     */
    void load(const std::string &path);

    /// Clear all data and reset to an empty state.
    void clear();

    /// True if no records are loaded.
    bool empty() const { return ranks_.empty(); }

    /**
     * Lookup the correction value for a given abstract rank.
     *
     * @param rank  Abstract state rank (e.g. rank_partial_perm(...)).
     * @return      Over-correction value in [0..255]. Returns 0 if this
     *              rank has no explicit correction (i.e. treated as "no over").
     *
     * This uses the bucketed index in `offsets_` to restrict the search
     * to the subset of entries whose high 16 bits match those of `rank`,
     * then performs a small binary search in that bucket.
     */
    std::uint8_t get(std::uint32_t rank) const;

    // Number of raw records reported in the file header (before merging).
    std::uint64_t record_count() const { return record_count_; }

    // Pattern identifier from header: 17 for tiles 1–7, 815 for tiles 8–15.
    std::uint32_t pattern_id() const { return pattern_id_; }

    // m from header: 8 or 9, the length of the partial permutation.
    std::uint32_t m() const { return m_; }

    // Quantile q used when these corrections were generated.
    float q() const { return q_; }

private:
#pragma pack(push, 1)
    /**
     * On-disk file header (packed, no padding).
     *
     * magic  = "NNOVR001"
     * version = 1
     * pattern = 17 or 815
     * m       = 8 or 9
     * q       = quantile
     * count   = number of (rank, over) records that follow
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

    // Expected magic bytes for verification.
    static constexpr std::array<char, 8> MAGIC_ = {'N', 'N', 'O', 'V', 'R', '0', '0', '1'};

    // Sorted arrays (by rank). For each i:
    //   ranks_[i] = abstract rank
    //   overs_[i] = max over-correction observed for that rank
    std::vector<std::uint32_t> ranks_;
    std::vector<std::uint8_t> overs_;

    /**
     * Bucketed index over the high 16 bits of rank.
     *
     * For a rank r, let p = r >> 16 in [0,65535].
     * Then all entries with that prefix lie in:
     *
     *   i ∈ [ offsets_[p], offsets_[p+1] )
     *
     * offsets_.size() == 65537 so that offsets_[65536] equals ranks_.size().
     *
     * This allows us to drastically reduce the search range before doing
     * a binary search in that small bucket.
     */
    std::array<std::uint32_t, 65537> offsets_{};
    bool offsets_ready_ = false;

    // Header fields preserved for information/validation.
    std::uint64_t record_count_ = 0;
    std::uint32_t pattern_id_ = 0;
    std::uint32_t m_ = 0;
    float q_ = 0.0f;

    /// Build the bucket offsets array from the current ranks_ vector.
    void build_bucket_offsets_();
};
