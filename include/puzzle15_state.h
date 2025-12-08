//
// Created by uriel on 03/11/2025.
//

/// Representation of a 15-puzzle board (4x4 grid with tiles 1..15 and one blank ).
/// The blank tile is represented by 0 and its position is cached in blankPos.
///

#pragma once
#include <array>
#include <vector>
#include <cstdint>
#include <cstddef>   // size_t
#include <stdexcept>
#include <functional>
#include <utility>   // std::swap
#include <initializer_list>
#include <algorithm>

struct puzzle15_state {
    static constexpr int Rows = 4, Cols = 4, Size = Rows * Cols;
    using Tile = std::uint8_t; // 0..15 (0 = blank)
    using Index = std::uint8_t; // index of the position on the board

    std::array<Tile, Size> tiles{};
    Index blankPos{Size - 1}; // the position of the blank

    // default constructor (goal state)
    puzzle15_state();

    // constructor from a vector with 16 Tiles
    explicit puzzle15_state(const std::vector<Tile> &v);

    // constructor from an initializer list
    puzzle15_state(std::initializer_list<Tile> init);

    // getters
    static constexpr int size() {
        return Size;
    }

    static constexpr int rows() {
        return Rows;
    }

    static constexpr int cols() {
        return Cols;
    }

    Tile operator[](const int i) const {
        return tiles[i];
    }

    Tile &operator[](const int i) {
        return tiles[i];
    }

    bool operator==(const puzzle15_state &) const = default;

    static puzzle15_state Goal();

    // return the index on the board according to number of row and column
    static constexpr int idx(const int r, const int c) {
        return r * Cols + c;
    }

    // return the number of row according to index
    static constexpr int row(const int i) {
        return i / Cols;
    }

    // return the number of column according to index
    static constexpr int col(const int i) {
        return i % Cols;
    }

    // swap the blank index to index p, and index p to the index of the blank
    void swapBlankWith(Index p);

    // pack the board into a 64-bit integer.
    [[nodiscard]] std::uint64_t pack() const;

    // unpack a 64-bit integer into a puzzle15_state.
    static puzzle15_state unpack(std::uint64_t x);

private:
    // update blankPos to the index where tile == 0.
    void computeBlank();

    // validate that tiles[] contains a legal 15-puzzle configuration.
    void validate() const;
};

// custom hash specialization for puzzle15_state so it can be used
// as a key in unordered_map, unordered_set, etc.
template<>
struct std::hash<puzzle15_state> {
    // compute a hash value for a puzzle15_state.
    // We first pack the entire board into a 64-bit integer using s.pack(),
    // and then reuse std::hash<std::uint64_t> on that packed value.
    size_t operator()(const puzzle15_state &s) const noexcept {
        return std::hash<std::uint64_t>{}(s.pack());
    }
};
