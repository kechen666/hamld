#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <unordered_map>
#include <random>  // 这里补上
#include <boost/dynamic_bitset.hpp>

// tsl::robin_map
#include "tsl/robin_map.h"

// 检查 key 是否存在的统一函数
template <typename MapType, typename KeyType>
bool map_contains(const MapType& map, const KeyType& key) {
    return map.find(key) != map.end();
}

template <typename MapType, typename KeyType>
void benchmark_find_update(const std::string& name, const std::vector<KeyType>& keys) {
    MapType map;

    // 先插入所有 key，初始化值为0
    for (const auto& k : keys) {
        map[k] = 0;
    }

    // 查找并更新
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& k : keys) {
        if (map_contains(map, k)) {
            map[k] += 1;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << name << " 查找并更新耗时: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";
}

int main() {
    constexpr int NUM_INSERTS = 50000;
    constexpr int KEY_LENGTH = 2000;

    std::mt19937 rng(42);

    // 生成随机 dynamic_bitset keys
    std::vector<boost::dynamic_bitset<>> keys;
    keys.reserve(NUM_INSERTS);
    for (int i = 0; i < NUM_INSERTS; ++i) {
        boost::dynamic_bitset<> b(KEY_LENGTH);
        for (int j = 0; j < KEY_LENGTH; ++j) {
            b[j] = rng() % 2;
        }
        keys.push_back(b);
    }

    benchmark_find_update<std::unordered_map<boost::dynamic_bitset<>, int>>("std::unordered_map<boost::dynamic_bitset>", keys);
    benchmark_find_update<tsl::robin_map<boost::dynamic_bitset<>, int>>("tsl::robin_map<boost::dynamic_bitset>", keys);

    return 0;
}
