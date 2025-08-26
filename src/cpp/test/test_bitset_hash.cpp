#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <optional>

#include <boost/dynamic_bitset.hpp>
#include <boost/range/algorithm.hpp>  // for boost::to_block_range

using Clock = std::chrono::high_resolution_clock;
using ms = std::chrono::milliseconds;

constexpr int BIT_LENGTH = 2000;
constexpr int NUM_INSERTS = 50000;

class SparseState {
public:
    int total_len = 0;
    std::vector<int> ones;
    mutable std::optional<size_t> cached_hash;

    SparseState() = default;
    SparseState(int len) : total_len(len) {}

    void flip(int index) {
        auto it = std::lower_bound(ones.begin(), ones.end(), index);
        if (it != ones.end() && *it == index) {
            ones.erase(it);
        } else {
            ones.insert(it, index);
        }
        cached_hash.reset();
    }

    bool has(int index) const {
        return std::binary_search(ones.begin(), ones.end(), index);
    }

    std::string to_string() const {
        std::string s(total_len, '0');
        for (int i : ones) {
            s[i] = '1';
        }
        return s;
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

// 正确版本：使用 boost::to_block_range 提取 block 内容哈希
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

int main() {
    using namespace std;

    mt19937 rng(42);
    uniform_int_distribution<int> bit_dist(0, 1);

    vector<SparseState> sparse_states;
    vector<vector<bool>> vector_bools;
    vector<string> strings;
    vector<boost::dynamic_bitset<>> boost_bits;

    sparse_states.reserve(NUM_INSERTS);
    vector_bools.reserve(NUM_INSERTS);
    strings.reserve(NUM_INSERTS);
    boost_bits.reserve(NUM_INSERTS);

    for (int i = 0; i < NUM_INSERTS; ++i) {
        SparseState ss(BIT_LENGTH);
        for (int j = 0; j < BIT_LENGTH; ++j) {
            if (bit_dist(rng)) ss.ones.push_back(j);
        }
        sparse_states.push_back(std::move(ss));

        vector<bool> vb(BIT_LENGTH);
        for (int j = 0; j < BIT_LENGTH; ++j) {
            vb[j] = bit_dist(rng);
        }
        vector_bools.push_back(std::move(vb));

        string s(BIT_LENGTH, '0');
        for (int j = 0; j < BIT_LENGTH; ++j) {
            s[j] = bit_dist(rng) ? '1' : '0';
        }
        strings.push_back(std::move(s));

        boost::dynamic_bitset<> b(BIT_LENGTH);
        for (int j = 0; j < BIT_LENGTH; ++j) {
            b[j] = bit_dist(rng);
        }
        boost_bits.push_back(std::move(b));
    }

    cout << "测试 SparseState 插入...\n";
    auto start = Clock::now();
    unordered_map<SparseState, int> map_sparse;
    for (auto& ss : sparse_states) {
        map_sparse[ss]++;
    }
    auto end = Clock::now();
    cout << "SparseState 插入耗时: " << chrono::duration_cast<ms>(end - start).count() << " ms\n";

    cout << "测试 vector<bool> 插入...\n";
    start = Clock::now();
    unordered_map<vector<bool>, int, VectorBoolHash> map_vector_bool;
    for (auto& vb : vector_bools) {
        map_vector_bool[vb]++;
    }
    end = Clock::now();
    cout << "vector<bool> 插入耗时: " << chrono::duration_cast<ms>(end - start).count() << " ms\n";

    cout << "测试 string 插入...\n";
    start = Clock::now();
    unordered_map<string, int> map_string;
    for (auto& s : strings) {
        map_string[s]++;
    }
    end = Clock::now();
    cout << "string 插入耗时: " << chrono::duration_cast<ms>(end - start).count() << " ms\n";

    cout << "测试 boost::dynamic_bitset 插入...\n";
    start = Clock::now();
    unordered_map<boost::dynamic_bitset<>, int, BoostBitsetHash> map_boost;
    for (auto& b : boost_bits) {
        map_boost[b]++;
    }
    end = Clock::now();
    cout << "boost::dynamic_bitset 插入耗时: " << chrono::duration_cast<ms>(end - start).count() << " ms\n";

    constexpr int NUM_LOOKUPS = 5000;  // 查找次数
    uniform_int_distribution<int> index_dist(0, NUM_INSERTS - 1);  // 随机查找索引

    cout << "\n==== 查找测试 ====\n";

    // SparseState 查找
    cout << "测试 SparseState 查找...\n";
    start = Clock::now();
    volatile int sum_sparse = 0;  // volatile 避免优化
    for (int i = 0; i < NUM_LOOKUPS; ++i) {
        const auto& key = sparse_states[index_dist(rng)];
        sum_sparse += map_sparse.at(key);
    }
    end = Clock::now();
    cout << "SparseState 查找耗时: " << chrono::duration_cast<ms>(end - start).count() << " ms\n";

    // vector<bool> 查找
    cout << "测试 vector<bool> 查找...\n";
    start = Clock::now();
    volatile int sum_vb = 0;
    for (int i = 0; i < NUM_LOOKUPS; ++i) {
        const auto& key = vector_bools[index_dist(rng)];
        sum_vb += map_vector_bool.at(key);
    }
    end = Clock::now();
    cout << "vector<bool> 查找耗时: " << chrono::duration_cast<ms>(end - start).count() << " ms\n";

    // string 查找
    cout << "测试 string 查找...\n";
    start = Clock::now();
    volatile int sum_str = 0;
    for (int i = 0; i < NUM_LOOKUPS; ++i) {
        const auto& key = strings[index_dist(rng)];
        sum_str += map_string.at(key);
    }
    end = Clock::now();
    cout << "string 查找耗时: " << chrono::duration_cast<ms>(end - start).count() << " ms\n";

    // boost::dynamic_bitset 查找
    cout << "测试 boost::dynamic_bitset 查找...\n";
    start = Clock::now();
    volatile int sum_boost = 0;
    for (int i = 0; i < NUM_LOOKUPS; ++i) {
        const auto& key = boost_bits[index_dist(rng)];
        sum_boost += map_boost.at(key);
    }
    end = Clock::now();
    cout << "boost::dynamic_bitset 查找耗时: " << chrono::duration_cast<ms>(end - start).count() << " ms\n";

    return 0;
}
