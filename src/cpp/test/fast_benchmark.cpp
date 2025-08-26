// #include <iostream>
// #include <string>
// #include <vector>
// #include <random>
// #include <chrono>
// #include <unordered_map>
// #include <boost/dynamic_bitset.hpp>
// #include <boost/range/algorithm.hpp>
// #include <tsl/robin_map.h>

// // 自定义 boost::dynamic_bitset 哈希函数
// struct BoostBitsetHash {
//     size_t operator()(const boost::dynamic_bitset<>& b) const {
//         using block_type = boost::dynamic_bitset<>::block_type;
//         std::vector<block_type> blocks;
//         boost::to_block_range(b, std::back_inserter(blocks));
//         size_t seed = 0;
//         for (block_type blk : blocks) {
//             seed ^= std::hash<block_type>()(blk) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
//         }
//         return seed;
//     }
// };

// constexpr int NUM_INSERTS = 50000;
// constexpr int KEY_LENGTH = 2000;

// // 生成随机 dynamic_bitset
// boost::dynamic_bitset<> random_bitset(std::mt19937& rng) {
//     boost::dynamic_bitset<> b(KEY_LENGTH);
//     for (int i = 0; i < KEY_LENGTH; ++i) {
//         b[i] = rng() % 2;
//     }
//     return b;
// }

// // dynamic_bitset 转字符串
// std::string bitset_to_string(const boost::dynamic_bitset<>& b) {
//     std::string s(KEY_LENGTH, '0');
//     for (size_t i = 0; i < b.size(); ++i) {
//         s[i] = b[i] ? '1' : '0';
//     }
//     return s;
// }

// // 基准测试模板
// template <typename MapType, typename KeyType>
// void benchmark_map(const std::string& name, const std::vector<KeyType>& keys) {
//     MapType map;

//     auto insert_start = std::chrono::high_resolution_clock::now();
//     for (const auto& k : keys) {
//         map[k]++;
//     }
//     auto insert_end = std::chrono::high_resolution_clock::now();

//     int sum = 0;
//     auto query_start = std::chrono::high_resolution_clock::now();
//     for (const auto& k : keys) {
//         sum += map[k];
//     }
//     auto query_end = std::chrono::high_resolution_clock::now();

//     std::cout << name << " 插入耗时: "
//               << std::chrono::duration_cast<std::chrono::milliseconds>(insert_end - insert_start).count()
//               << " ms, 查询耗时: "
//               << std::chrono::duration_cast<std::chrono::milliseconds>(query_end - query_start).count()
//               << " ms\n";
//     (void)sum;
// }

// int main() {
//     std::mt19937 rng(42);

//     std::vector<boost::dynamic_bitset<>> bitset_keys;
//     bitset_keys.reserve(NUM_INSERTS);
//     for (int i = 0; i < NUM_INSERTS; ++i) {
//         bitset_keys.push_back(random_bitset(rng));
//     }

//     std::vector<std::string> string_keys;
//     string_keys.reserve(NUM_INSERTS);
//     for (const auto& b : bitset_keys) {
//         string_keys.push_back(bitset_to_string(b));
//     }

//     benchmark_map<std::unordered_map<boost::dynamic_bitset<>, int, BoostBitsetHash>>(
//         "std::unordered_map<boost::dynamic_bitset>", bitset_keys);

//     benchmark_map<tsl::robin_map<boost::dynamic_bitset<>, int, BoostBitsetHash>>(
//         "tsl::robin_map<boost::dynamic_bitset>", bitset_keys);

//     benchmark_map<tsl::robin_map<std::string, int>>(
//         "tsl::robin_map<std::string>", string_keys);

//     return 0;
// }

// // #include <iostream>
// // #include <vector>
// // #include <unordered_map>
// // #include <chrono>
// // #include <boost/dynamic_bitset.hpp>
// // #include <random>

// // // 自定义哈希函数，假设你已经定义好了
// // struct BoostBitsetHash {
// //     size_t operator()(const boost::dynamic_bitset<>& b) const {
// //         using block_type = boost::dynamic_bitset<>::block_type;
// //         std::vector<block_type> blocks;
// //         boost::to_block_range(b, std::back_inserter(blocks));
// //         size_t seed = 0;
// //         for (block_type blk : blocks) {
// //             seed ^= std::hash<block_type>()(blk) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
// //         }
// //         return seed;
// //     }
// // };

// // using Bitset = boost::dynamic_bitset<>;
// // using ProbDist = std::unordered_map<Bitset, double, BoostBitsetHash>;

// // // 简单flip_bits示例：翻转bitset中固定的某个位（模拟contracted_hyperedge）
// // Bitset flip_bits(const Bitset& b, size_t flip_index) {
// //     Bitset res = b;
// //     res.flip(flip_index);
// //     return res;
// // }

// // template <typename MapType, typename KeyType>
// // void benchmark_map(const std::string& name, const std::vector<KeyType>& keys, size_t flip_index, double contracted_prob, double non_flip_prob) {
// //     MapType map;
// //     map.reserve(keys.size() * 2);

// //     // 插入测试：模拟更新概率分布
// //     auto insert_start = std::chrono::high_resolution_clock::now();

// //     for (const auto& key : keys) {
// //         Bitset flipped_key = flip_bits(key, flip_index);

// //         double prob = 1.0; // 假设初始概率为1
// //         map[flipped_key] += prob * contracted_prob;
// //         map[key] += prob * non_flip_prob;
// //     }

// //     auto insert_end = std::chrono::high_resolution_clock::now();

// //     // 查询测试：访问全部key
// //     double sum = 0;
// //     auto query_start = std::chrono::high_resolution_clock::now();

// //     for (const auto& key : keys) {
// //         sum += map[key];
// //     }
// //     auto query_end = std::chrono::high_resolution_clock::now();

// //     std::cout << name << " 插入耗时: "
// //               << std::chrono::duration_cast<std::chrono::milliseconds>(insert_end - insert_start).count()
// //               << " ms, 查询耗时: "
// //               << std::chrono::duration_cast<std::chrono::milliseconds>(query_end - query_start).count()
// //               << " ms\n";
// //     (void)sum; // 防止优化掉
// // }

// // int main() {
// //     const size_t bitset_size = 2000;
// //     const size_t num_keys = 100000;

// //     // 生成随机bitset序列作为keys
// //     std::vector<Bitset> keys;
// //     keys.reserve(num_keys);
// //     std::mt19937 rng(42);
// //     std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

// //     for (size_t i = 0; i < num_keys; ++i) {
// //         Bitset b(bitset_size);
// //         for (size_t block_idx = 0; block_idx < b.num_blocks(); ++block_idx) {
// //             uint64_t val = dist(rng);
// //             for (size_t bit = 0; bit < 64; ++bit) {
// //                 if (val & (1ULL << bit)) {
// //                     size_t pos = block_idx * 64 + bit;
// //                     if (pos < bitset_size)
// //                         b.set(pos);
// //                 }
// //             }
// //         }
// //         keys.push_back(std::move(b));
// //     }

// //     benchmark_map<ProbDist, Bitset>("ProbDist with boost::dynamic_bitset", keys, 5, 0.6, 0.4);

// //     return 0;
// // }

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <unordered_map>
#include <boost/dynamic_bitset.hpp>
#include <boost/range/algorithm.hpp>
#include <tsl/robin_map.h>

// 自定义 boost::dynamic_bitset 哈希函数
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

constexpr int NUM_INSERTS = 10000;
constexpr int KEY_LENGTH = 1000;
constexpr int NUM_ITERATIONS = 1000; // 循环执行次数

// 生成随机 dynamic_bitset
boost::dynamic_bitset<> random_bitset(std::mt19937& rng) {
    boost::dynamic_bitset<> b(KEY_LENGTH);
    for (int i = 0; i < KEY_LENGTH; ++i) {
        b[i] = rng() % 2;
    }
    return b;
}

// dynamic_bitset 转字符串
std::string bitset_to_string(const boost::dynamic_bitset<>& b) {
    std::string s(KEY_LENGTH, '0');
    for (size_t i = 0; i < b.size(); ++i) {
        s[i] = b[i] ? '1' : '0';
    }
    return s;
}

// 翻转字符串中指定位置的位
std::string flip_bits(const std::string& binary_str, const std::vector<size_t>& positions) {
    std::string result = binary_str;
    for (size_t pos : positions) {
        if (pos < result.size()) {
            result[pos] = (result[pos] == '0') ? '1' : '0';
        }
    }
    return result;
}

// 翻转 bitset 中指定位置的位
boost::dynamic_bitset<> flip_bits(const boost::dynamic_bitset<>& bs, const std::vector<size_t>& positions) {
    boost::dynamic_bitset<> result = bs;
    for (size_t pos : positions) {
        if (pos < result.size()) {
            result.flip(pos);
        }
    }
    return result;
}

// 基准测试模板 - 处理字符串键的映射
template <typename MapType>
void benchmark_map_string(const std::string& name, const MapType& prob_dist, 
                         const std::vector<size_t>& contracted_hyperedge,
                         double contracted_prob, double non_flip_prob) {
    MapType updated_dist;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        updated_dist.clear();
        updated_dist.reserve(prob_dist.size() * 2);
        
        for (const auto& [binary_str, prob] : prob_dist) {
            std::string flipped_str = flip_bits(binary_str, contracted_hyperedge);
            
            double flipped_prob = prob * contracted_prob;
            double non_flipped_prob = prob * non_flip_prob;
            
            updated_dist[flipped_str] += flipped_prob;
            updated_dist[binary_str] += non_flipped_prob;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << name << " 执行 " << NUM_ITERATIONS << " 次循环耗时: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";
    std::cout << "最终 map 大小: " << updated_dist.size() << "\n";
}

// 基准测试模板 - 处理 dynamic_bitset 键的映射
template <typename MapType>
void benchmark_map_bitset(const std::string& name, const MapType& prob_dist, 
                         const std::vector<size_t>& contracted_hyperedge,
                         double contracted_prob, double non_flip_prob) {
    MapType updated_dist;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        updated_dist.clear();
        updated_dist.reserve(prob_dist.size() * 2);
        
        for (const auto& [bs, prob] : prob_dist) {
            boost::dynamic_bitset<> flipped_bs = flip_bits(bs, contracted_hyperedge);
            
            double flipped_prob = prob * contracted_prob;
            double non_flipped_prob = prob * non_flip_prob;
            
            updated_dist[flipped_bs] += flipped_prob;
            updated_dist[bs] += non_flipped_prob;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout << name << " 执行 " << NUM_ITERATIONS << " 次循环耗时: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";
    std::cout << "最终 map 大小: " << updated_dist.size() << "\n";
}

int main() {
    std::mt19937 rng(42);

    // 生成随机超边（用于翻转位）
    std::vector<size_t> contracted_hyperedge;
    for (int i = 0; i < 10; ++i) { // 随机选择10个位进行翻转
        contracted_hyperedge.push_back(rng() % KEY_LENGTH);
    }
    
    // 概率参数
    double contracted_prob = 0.001;
    double non_flip_prob = 0.999;

    // 准备字符串键的 prob_dist 数据
    tsl::robin_map<std::string, double> prob_dist_string;
    for (int i = 0; i < NUM_INSERTS; ++i) {
        boost::dynamic_bitset<> bs = random_bitset(rng);
        std::string key = bitset_to_string(bs);
        prob_dist_string[key] = (double)rng() / rng.max(); // 随机概率值
    }

    // 准备 dynamic_bitset 键的 prob_dist 数据
    tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash> prob_dist_bitset;
    for (int i = 0; i < NUM_INSERTS; ++i) {
        boost::dynamic_bitset<> bs = random_bitset(rng);
        prob_dist_bitset[bs] = (double)rng() / rng.max(); // 随机概率值
    }

    // 创建与函数参数类型匹配的对象
    std::unordered_map<std::string, double> prob_dist_string_unordered(prob_dist_string.begin(), prob_dist_string.end());
    std::unordered_map<boost::dynamic_bitset<>, double, BoostBitsetHash> prob_dist_bitset_unordered(prob_dist_bitset.begin(), prob_dist_bitset.end());

    // 使用不同的 map 类型进行测试
    benchmark_map_string<tsl::robin_map<std::string, double>>(
        "tsl::robin_map<std::string, double>", 
        prob_dist_string, 
        contracted_hyperedge,
        contracted_prob, 
        non_flip_prob
    );
    
    benchmark_map_string<std::unordered_map<std::string, double>>(
        "std::unordered_map<std::string, double>", 
        prob_dist_string_unordered, 
        contracted_hyperedge,
        contracted_prob, 
        non_flip_prob
    );
    
    benchmark_map_bitset<tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash>>(
        "tsl::robin_map<boost::dynamic_bitset, double>", 
        prob_dist_bitset, 
        contracted_hyperedge,
        contracted_prob, 
        non_flip_prob
    );
    
    benchmark_map_bitset<std::unordered_map<boost::dynamic_bitset<>, double, BoostBitsetHash>>(
        "std::unordered_map<boost::dynamic_bitset, double>", 
        prob_dist_bitset_unordered, 
        contracted_hyperedge,
        contracted_prob, 
        non_flip_prob
    );

    return 0;
}