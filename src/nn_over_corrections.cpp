#include "nn_over_corrections.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {
    /// Read exactly sizeof(T) bytes from the stream into `out`.
    /// Throws if EOF or any read error occurs.
    template<typename T>
    static void read_exact(std::ifstream &in, T &out) {
        in.read(reinterpret_cast<char *>(&out), static_cast<std::streamsize>(sizeof(T)));
        if (!in) throw std::runtime_error("Failed reading from file (unexpected EOF)");
    }

    /// Return total size (in bytes) of the file associated with this stream,
    /// without changing the current read position.
    static std::uint64_t file_size(std::ifstream &in) {
        auto cur = in.tellg();
        in.seekg(0, std::ios::end);
        auto end = in.tellg();
        in.seekg(cur, std::ios::beg);
        return static_cast<std::uint64_t>(end);
    }

    /// Decode a little-endian uint32 from 4 bytes, in a portable way
    /// (independent of host endianness).
    static std::uint32_t u32_from_le_bytes(const unsigned char *p) {
        return (std::uint32_t(p[0])) |
               (std::uint32_t(p[1]) << 8) |
               (std::uint32_t(p[2]) << 16) |
               (std::uint32_t(p[3]) << 24);
    }
} // namespace

/// Reset all internal data structures and header fields to the empty state.
void NnOverCorrections::clear() {
    ranks_.clear();
    overs_.clear();
    offsets_.fill(0);
    offsets_ready_ = false;

    record_count_ = 0;
    pattern_id_ = 0;
    m_ = 0;
    q_ = 0.0f;
}

/// Load corrections from a binary file produced by `build_nn_over_corrections`.
/// The file format is:
///   Header (fixed size), then `count` records of:
///     rank : uint32 (little-endian)
///     over : uint8
///
/// Requirements:
///   * Records must be sorted by rank (non-decreasing).
///   * Adjacent duplicates for the same rank are merged on load
///     by taking max(over).
void NnOverCorrections::load(const std::string &path) {
    clear();

    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("NnOverCorrections: failed to open: " + path);

    // Read header from disk
    Header h{};
    read_exact(in, h);

    // Validate header fields (magic, version, m)
    if (std::memcmp(h.magic, MAGIC_.data(), 8) != 0) {
        throw std::runtime_error("NnOverCorrections: bad magic in: " + path);
    }
    if (h.version != 1) {
        throw std::runtime_error("NnOverCorrections: unsupported version (expected 1) in: " + path);
    }
    if (!(h.m == 8 || h.m == 9)) {
        throw std::runtime_error("NnOverCorrections: bad m (expected 8 or 9) in: " + path);
    }

    record_count_ = h.count;
    pattern_id_ = h.pattern;
    m_ = h.m;
    q_ = h.q;

    // Sanity check: file size must match header.count and record size.
    // Header is sizeof(Header), each record is 5 bytes (u32 + u8).
    const std::uint64_t sz = file_size(in);
    const std::uint64_t expected = sizeof(Header) + 5ULL * record_count_;
    if (sz != expected) {
        throw std::runtime_error(
            "NnOverCorrections: file size mismatch (expected " + std::to_string(expected) +
            ", got " + std::to_string(sz) + ") in: " + path
        );
    }

    // Reserve upper bound on storage (final number may be smaller after merging duplicates).
    if (record_count_ > 0) {
        ranks_.reserve(static_cast<std::size_t>(record_count_));
        overs_.reserve(static_cast<std::size_t>(record_count_));
    }

    // Read records in chunks for better I/O throughput.
    // Each record is:
    //   rank (4 bytes, LE) + over (1 byte) = 5 bytes total.
    static constexpr std::size_t RECORD_BYTES = 5;
    static constexpr std::size_t RECORDS_PER_CHUNK = 1u << 20; // 1,048,576 records
    static constexpr std::size_t CHUNK_BYTES = RECORDS_PER_CHUNK * RECORD_BYTES; // ~5MB

    std::vector<unsigned char> buf;
    buf.resize(CHUNK_BYTES);

    std::uint32_t last_rank = 0;
    bool have_last = false;

    std::uint64_t remaining = record_count_;
    while (remaining > 0) {
        const std::size_t take = static_cast<std::size_t>(
            std::min<std::uint64_t>(remaining, RECORDS_PER_CHUNK)
        );
        const std::size_t bytes = take * RECORD_BYTES;

        in.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(bytes));
        if (!in) throw std::runtime_error("NnOverCorrections: failed reading records from: " + path);

        const unsigned char *p = buf.data();
        for (std::size_t i = 0; i < take; ++i) {
            const std::uint32_t rank = u32_from_le_bytes(p);
            const std::uint8_t over = static_cast<std::uint8_t>(p[4]);
            p += RECORD_BYTES;

            if (have_last) {
                // Ensure non-decreasing order of ranks (required by our format).
                if (rank < last_rank) {
                    throw std::runtime_error(
                        "NnOverCorrections: ranks are not sorted (rank decreased). "
                        "Rebuild file without out-of-order appends. File: " + path
                    );
                }
                if (rank == last_rank) {
                    // Duplicate rank: merge by keeping the maximum 'over' value.
                    if (!overs_.empty() && over > overs_.back()) {
                        overs_.back() = over;
                    }
                    continue; // skip pushing a new entry
                }
            }

            // New rank: append to both vectors.
            ranks_.push_back(rank);
            overs_.push_back(over);
            last_rank = rank;
            have_last = true;
        }

        remaining -= take;
    }

    // After loading and deduplicating, build the bucket index for fast lookup.
    build_bucket_offsets_();
}

/// Build a bucketed index over `ranks_` to speed up lookups:
///
///   * For each rank, we consider the "prefix" p = rank >> 16 (top 16 bits).
///   * offsets_[p]..offsets_[p+1] gives the range in ranks_ that has
///     exactly that prefix p.
///   * So `offsets_` has size 65537, and offsets_[65536] == ranks_.size().
///
/// This allows us to restrict the binary search for a given rank to its
/// corresponding bucket, instead of searching over all entries.
void NnOverCorrections::build_bucket_offsets_() {
    offsets_.fill(0);

    // First pass: count how many ranks fall into each bucket p = (rank >> 16).
    std::array<std::uint32_t, 65536> counts{};
    counts.fill(0);

    for (std::uint32_t r: ranks_) {
        const std::uint32_t p = (r >> 16);
        ++counts[p];
    }

    // Prefix sum over counts -> offsets.
    std::uint32_t sum = 0;
    offsets_[0] = 0;
    for (std::uint32_t p = 0; p < 65536; ++p) {
        sum += counts[p];
        offsets_[p + 1] = sum;
    }

    offsets_ready_ = true;
}

/// Return the correction value "over" for this rank (0 if none).
///
/// The lookup procedure:
///   1. If no data loaded, return 0.
///   2. Compute bucket prefix p = rank >> 16.
///   3. Restrict search to ranks_[ offsets_[p] .. offsets_[p+1] ).
///   4. Binary search (lower_bound) for `rank` in that slice.
///   5. If found, return overs_[index], else 0.
std::uint8_t NnOverCorrections::get(std::uint32_t rank) const {
    if (!offsets_ready_ || ranks_.empty()) return 0;

    const std::uint32_t p = (rank >> 16);
    const std::uint32_t begin = offsets_[p];
    const std::uint32_t end = offsets_[p + 1];
    if (begin >= end) return 0; // no entries in this bucket

    const std::uint32_t *base = ranks_.data();
    const std::uint32_t *it = std::lower_bound(base + begin, base + end, rank);
    if (it == base + end || *it != rank) return 0;

    const std::size_t idx = static_cast<std::size_t>(it - base);
    return overs_[idx];
}
