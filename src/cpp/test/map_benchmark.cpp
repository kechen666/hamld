#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <random>
#include <optional>
#include <algorithm>

// Boost
#include <boost/dynamic_bitset.hpp>
#include <boost/range/algorithm.hpp>

// Abseil
#include "absl/container/flat_hash_map.h"

// robin-map
#include "tsl/robin_map.h"

// robin-hood
#include "robin_hood.h"

using Clock = std::chrono::high_resolution_clock;
using ms = std::chrono::milliseconds;

constexpr int BIT_LENGTH = 2000;
constexpr int NUM_INSERTS = 50000;

// ----- SparseState -----
class SparseState {
public:
    int total_len = 0;
    std::vector<int> ones;
    mutable std::optional<size_t> cached_hash;

    SparseState() = default;
    SparseState(int len) : total_len(len) {}

    void flip(int index) {
        auto it = std::lower_bound(ones.begin(), ones.end(), index);
        if (it != ones.end() && *it == index)
            ones.erase(it);
        else
            ones.insert(it, index);
        cached_hash.reset();
    }

    bool operator==(const SparseState& other) const {
        return total_len == other.total_len && ones == other.ones;
    }

    size_t compute_hash() const {
        if (cached_hash) return *cached_hash;
        size_t seed = std::hash<int>()(total_len);
        for (int i : ones) {
            seed ^= std::hash<int>()(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        cached_hash = seed;
        return seed;
    }
};

namespace std {
template <>
struct hash<SparseState> {
    size_t operator()(const SparseState& s) const noexcept {
        return s.compute_hash();
    }
};
}

// ----- 其他哈希函数 -----
struct VectorBoolHash {
    size_t operator()(const std::vector<bool>& v) const {
        size_t seed = 0, block = 0;
        int count = 0;
        for (bool bit : v) {
            block = (block << 1) | bit;
            count++;
            if (count == 32) {
                seed ^= std::hash<size_t>()(block) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                block = 0;
                count = 0;
            }
        }
        if (count > 0) {
            seed ^= std::hash<size_t>()(block) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

struct BoostBitsetHash {
    size_t operator()(const boost::dynamic_bitset<>& b) const {
        using block_type = boost::dynamic_bitset<>::block_type;
        std::vector<block_type> blocks;
        boost::to_block_range(b, std::back_inserter(blocks));
        size_t seed = 0;
        for (block_type blk : blocks) {
            seed ^= std::hash<block_type>()(blk) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// ----- key 生成函数 -----
std::string gen_string(std::mt19937& rng) {
    std::string s(BIT_LENGTH, '0');
    for (int i = 0; i < BIT_LENGTH; ++i)
        s[i] = rng() % 2 ? '1' : '0';
    return s;
}

std::vector<bool> gen_vector_bool(std::mt19937& rng) {
    std::vector<bool> v(BIT_LENGTH);
    for (int i = 0; i < BIT_LENGTH; ++i)
        v[i] = rng() % 2;
    return v;
}

boost::dynamic_bitset<> gen_bitset(std::mt19937& rng) {
    boost::dynamic_bitset<> b(BIT_LENGTH);
    for (int i = 0; i < BIT_LENGTH; ++i)
        b[i] = rng() % 2;
    return b;
}

SparseState gen_sparse(std::mt19937& rng) {
    SparseState s(BIT_LENGTH);
    for (int i = 0; i < BIT_LENGTH; ++i)
        if (rng() % 2)
            s.flip(i);
    return s;
}

// ----- Benchmark -----
template <typename MapType, typename KeyGenFunc>
void benchmark_map(const std::string& name, KeyGenFunc&& gen) {
    using Key = typename MapType::key_type;
    std::vector<Key> keys;
    keys.reserve(NUM_INSERTS);
    std::mt19937 rng(42);
    for (int i = 0; i < NUM_INSERTS; ++i)
        keys.push_back(gen(rng));

    MapType map;
    auto t1 = Clock::now();
    for (const auto& k : keys)
        map[k]++;
    auto t2 = Clock::now();

    int sum = 0;
    auto t3 = Clock::now();
    for (const auto& k : keys)
        sum += map[k];
    auto t4 = Clock::now();

    std::cout << name << " 插入耗时: "
              << std::chrono::duration_cast<ms>(t2 - t1).count()
              << " ms, 查询耗时: "
              << std::chrono::duration_cast<ms>(t4 - t3).count()
              << " ms\n";
    (void)sum;
}

// ----- 主函数 -----
int main() {
    benchmark_map<std::unordered_map<std::string, int>>(
        "std::unordered_map", [](std::mt19937& rng) { return gen_string(rng); });

    benchmark_map<absl::flat_hash_map<std::string, int>>(
        "absl::flat_hash_map", [](std::mt19937& rng) { return gen_string(rng); });

    benchmark_map<tsl::robin_map<std::string, int>>(
        "tsl::robin_map", [](std::mt19937& rng) { return gen_string(rng); });

    benchmark_map<robin_hood::unordered_flat_map<std::string, int>>(
        "robin_hood::unordered_flat_map", [](std::mt19937& rng) { return gen_string(rng); });

    benchmark_map<std::unordered_map<std::vector<bool>, int, VectorBoolHash>>(
        "std::vector<bool>", [](std::mt19937& rng) { return gen_vector_bool(rng); });

    benchmark_map<std::unordered_map<boost::dynamic_bitset<>, int, BoostBitsetHash>>(
        "boost::bitset", [](std::mt19937& rng) { return gen_bitset(rng); });

    benchmark_map<std::unordered_map<SparseState, int>>(
        "SparseState", [](std::mt19937& rng) { return gen_sparse(rng); });

    return 0;
}
