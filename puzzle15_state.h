//
// Created by uriel on 03/11/2025.
//

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
    using Tile = std::uint8_t; // 0..15 (0 = חור)
    using Index = std::uint8_t;

    std::array<Tile, Size> tiles{};      // אחסון יעיל, קבוע-גודל
    Index blankPos { Size - 1 };  // מיקום החור

    // קונסטרקטור ברירת מחדל (מטרה)
    puzzle15_state();

    // *** קונסטרקטור פשוט מוקטור בגודל 16 ***
    explicit puzzle15_state(const std::vector<Tile>& v);

    // אופציונלי: גם מ-initializer_list לנוחות
    explicit  puzzle15_state(std::initializer_list<Tile> init);

    // API מינימלי
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
    Tile& operator[](const int i) {
        return tiles[i];
    }
    bool operator==(const puzzle15_state&) const = default;

    static puzzle15_state Goal();

    static constexpr int idx(const int r, const int c) {
        return r * Cols + c;
    }
    static constexpr int row(const int i) {
        return i / Cols;
    }
    static constexpr int col(const int i) {
        return i % Cols;
    }

    void swapBlankWith(Index p);

    [[nodiscard]] std::uint64_t pack() const;
    static puzzle15_state unpack(std::uint64_t x);

private:
    void computeBlank();
    void validate() const; // בודק: טווח 0..15, בלי כפילויות
};

// hash ל-unordered_*
template<> struct std::hash<puzzle15_state> {
    size_t operator()(const puzzle15_state& s) const noexcept {
        return std::hash<std::uint64_t>{}(s.pack());
    }
};
