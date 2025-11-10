//
// Created by uriel on 03/11/2025.
//

#include "puzzle15_state.h"

puzzle15_state::puzzle15_state() {
    // מצב מטרה
    for (int i = 0; i < Size - 1; i++) {
        tiles[i] = static_cast<Tile>(i + 1);
    }
    tiles[Size - 1] = 0;
    blankPos = static_cast<Index>(Size - 1);
}

puzzle15_state::puzzle15_state(const std::vector<Tile>& v) {
    if (static_cast<int>(v.size()) != Size) {
        throw std::invalid_argument("vector size must be 16");
    }
    Index i = 0;
    for (const auto t : v) {
        tiles[i++] = t;
    }
    validate();       // בודק טווח/כפילויות/פרמוטציה מדויקת 0..15
    computeBlank();
}

puzzle15_state::puzzle15_state(const std::initializer_list<Tile> init) {
    if (static_cast<int>(init.size()) != Size) {
        throw std::invalid_argument("Puzzle15State: initializer_list must have 16 items");
    }
    int i = 0;
    for (const auto t : init) {
        tiles[i++] = t;
    }
    validate();       // בודק טווח/כפילויות/פרמוטציה מדויקת 0..15
    computeBlank();
}

puzzle15_state puzzle15_state::Goal() {
    return puzzle15_state(); // כבר בונה מטרה
}

void puzzle15_state::swapBlankWith(const Index p) {
    std::swap(tiles[blankPos], tiles[p]);
    blankPos = p;
}

std::uint64_t puzzle15_state::pack() const {
    std::uint64_t x = 0;
    for (int i = 0; i < Size; i++) {
        x |= (static_cast<std::uint64_t>(tiles[i] & 0xF) << (i * 4));
    }
    return x;
}

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

void puzzle15_state::computeBlank() {
    for (int i = 0; i < Size; i++) {
        if (tiles[i] == 0) {
            blankPos = static_cast<Index>(i);
            return;
        }
    }
}

void puzzle15_state::validate() const {
    // ממיינים העתק
    auto tmp = tiles;
    std::ranges::sort(tmp);

    for (int i = 0; i < Size; i++) {
        const int val = tmp[i];
        if (val < 0 || val > 15) // val<0 לא יקרה עם uint8_t, נשאר לקריאות
            throw std::invalid_argument("Puzzle15State: tile out of range (must be 0..15)");
        if (i + 1 < Size && tmp[i] == tmp[i + 1])
            throw std::invalid_argument("Puzzle15State: duplicate tile value");
    }
}

