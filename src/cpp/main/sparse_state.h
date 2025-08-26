#pragma once

#include <vector>
#include <optional>
#include <string>
#include <functional>

class SparseState {
public:
    SparseState();
    explicit SparseState(int len);

    void flip(int index);
    bool has(int index) const;
    bool get_bit(int idx) const;
    int hamming_distance(const SparseState& other) const;
    std::string to_string() const;

    bool operator==(const SparseState& other) const;
    bool operator<(const SparseState& other) const;

    // 供 std::hash 使用
    size_t compute_hash() const;

private:
    int total_len;
    std::vector<int> ones;
    mutable std::optional<size_t> cached_hash;  // 用于缓存哈希值
};

namespace std {
template<>
struct hash<SparseState> {
    size_t operator()(const SparseState& s) const noexcept {
        return s.compute_hash();
    }
};
}
