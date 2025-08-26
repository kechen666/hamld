// sparse_state.cpp
#include "sparse_state.h"
#include <algorithm>
#include <stdexcept> 

SparseState::SparseState() : total_len(0) {}

SparseState::SparseState(int len) : total_len(len) {}

void SparseState::flip(int index) {
    auto it = std::lower_bound(ones.begin(), ones.end(), index);
    if (it != ones.end() && *it == index) {
        ones.erase(it);
    } else {
        ones.insert(it, index);
    }
    cached_hash.reset();
}

bool SparseState::has(int index) const {
    return std::binary_search(ones.begin(), ones.end(), index);
}

bool SparseState::get_bit(int idx) const {
    return has(idx);
}

int SparseState::hamming_distance(const SparseState& other) const {
    int dist = 0;
    auto it1 = ones.begin(), it2 = other.ones.begin();
    while (it1 != ones.end() && it2 != other.ones.end()) {
        if (*it1 == *it2) {
            ++it1; ++it2;
        } else if (*it1 < *it2) {
            ++dist; ++it1;
        } else {
            ++dist; ++it2;
        }
    }
    dist += std::distance(it1, ones.end()) + std::distance(it2, other.ones.end());
    return dist;
}

std::string SparseState::to_string() const {
    std::string s(total_len, '0');
    for (int i : ones) {
        if (i >= 0 && i < total_len) {
            s[i] = '1';
        } else {
            throw std::out_of_range("index out of bounds in SparseState::to_string");
        }
    }
    return s;
}

bool SparseState::operator==(const SparseState& other) const {
    return total_len == other.total_len && ones == other.ones;
}

bool SparseState::operator<(const SparseState& other) const {
    if (total_len != other.total_len) return total_len < other.total_len;
    return ones < other.ones;
}

size_t SparseState::compute_hash() const {
    if (cached_hash.has_value()) return cached_hash.value();

    size_t seed = std::hash<int>()(total_len);
    for (int i : ones) {
        seed ^= std::hash<int>()(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    cached_hash = seed;
    return seed;
}
