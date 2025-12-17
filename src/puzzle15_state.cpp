//
// Created by uriel on 03/11/2025.
//

/// Representation of a 4x4 15-puzzle board.
/// Tiles are stored in a fixed-size array in row-major order.
/// Valid tile values are 0..15, where 0 represents the blank.
/// blankPos stores the current index of the blank tile.
///

#include "puzzle15_state.h"

// Default constructor: build the canonical goal state.
// Goal layout: [1, 2, 3, ..., 15, 0], with the blank (0) in the last cell.
puzzle15_state::puzzle15_state() {
    for (int i = 0; i < Size - 1; i++) {
        tiles[i] = static_cast<Tile>(i + 1);
    }
    tiles[Size - 1] = 0;
    blankPos = static_cast<Index>(Size - 1);
}

// Construct a state from a std::vector<Tile>.
// Requirements:
//  * v.size() must be exactly Size (16).
//  * Contents must be a permutation of 0..15.
// On failure, throws std::invalid_argument.
puzzle15_state::puzzle15_state(const std::vector<Tile> &v) {
    if (static_cast<int>(v.size()) != Size) {
        throw std::invalid_argument("vector size must be 16");
    }
    Index i = 0;
    for (const auto t : v) {
        tiles[i++] = t;
    }
    // validate() checks range / duplicates / exact permutation 0..15
    validate();
    computeBlank();
}

// Construct a state from an initializer_list, e.g. {1,2,3,...,0}.
// Requirements:
//  * init.size() must be exactly Size (16).
//  * Contents must be a permutation of 0..15.
// On failure, throws std::invalid_argument.
puzzle15_state::puzzle15_state(const std::initializer_list<Tile> init) {
    if (static_cast<int>(init.size()) != Size) {
        throw std::invalid_argument("Puzzle15State: initializer_list must have 16 items");
    }
    int i = 0;
    for (const auto t : init) {
        tiles[i++] = t;
    }
    // validate() checks range / duplicates / exact permutation 0..15
    validate();
    computeBlank();
}

// Factory method that returns the canonical goal state.
puzzle15_state puzzle15_state::Goal() {
    // Default constructor already builds the goal state.
    return puzzle15_state();
}

// Swap the blank tile with the tile at board index p.
// This updates both the tiles array and the cached blankPos.
void puzzle15_state::swapBlankWith(const Index p) {
    std::swap(tiles[blankPos], tiles[p]);
    blankPos = p;
}

// Pack the board into a 64-bit integer.
// Each tile is stored in 4 bits (a nibble), so 16 tiles * 4 bits = 64 bits.
// Tile at position i, is stored at bits [4*i ... 4*i+3].
std::uint64_t puzzle15_state::pack() const {
    std::uint64_t x = 0;
    for (int i = 0; i < Size; i++) {
        x |= (static_cast<std::uint64_t>(tiles[i] & 0xF) << (i * 4));
    }
    return x;
}

// Unpack a 64-bit integer into a puzzle15_state.
// Inverse of pack(): reads each 4-bit nibble and reconstructs tiles[].
// Also recomputes the blank position while unpacking.
puzzle15_state puzzle15_state::unpack(const std::uint64_t x) {
    puzzle15_state s;
    for (int i = 0; i < Size; i++) {
        s.tiles[i] = static_cast<Tile>((x >> (i * 4)) & 0xF);
        if (s.tiles[i] == 0) {
            s.blankPos = static_cast<Index>(i);
        }
    }
    return s;
}

// Scan the tiles array and update blankPos to the index of tile 0.
// Must be called after any change to tiles[] that may move the blank.
void puzzle15_state::computeBlank() {
    for (int i = 0; i < Size; i++) {
        if (tiles[i] == 0) {
            blankPos = static_cast<Index>(i);
            return;
        }
    }
}

// Validate that the current tiles[] describe a legal 15-puzzle configuration.
//
// Conditions enforced:
//  * All values are in the range [0..15].
//  * Exactly one of each value 0..15 (i.e., a permutation).
//
// On violation, throws std::invalid_argument.
void puzzle15_state::validate() const {
    // Make a copy and sort it to check the multiset of values.
    auto tmp = tiles;
    std::ranges::sort(tmp);

    for (int i = 0; i < Size; i++) {
        const int val = tmp[i];
        // val < 0 cannot actually happen if Tile is uint8_t,
        // but the check keeps the intent explicit and readable.
        if (val < 0 || val > 15)
            throw std::invalid_argument("Puzzle15State: tile out of range (must be 0..15)");
        if (i + 1 < Size && tmp[i] == tmp[i + 1])
            throw std::invalid_argument("Puzzle15State: duplicate tile value");
    }
}
