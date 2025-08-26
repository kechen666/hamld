#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <tuple>
#include <boost/dynamic_bitset.hpp>
#include <tsl/robin_map.h>
#include <unordered_set>

using Bitset = boost::dynamic_bitset<>;
using Clock = std::chrono::high_resolution_clock;

int hamming_distance_prefix(const Bitset& bs1, const Bitset& bs2, size_t n) {
    if (bs2.size() != n) {
        throw std::invalid_argument("bs2 size must be equal to n");
    }
    if (bs1.size() < n) {
        throw std::invalid_argument("bs1 size must be at least n");
    }

    int distance = 0;
    std::vector<Bitset::block_type> blocks1, blocks2;
    boost::to_block_range(bs1, std::back_inserter(blocks1));
    boost::to_block_range(bs2, std::back_inserter(blocks2));

    const size_t bits_per_block = bs1.bits_per_block;
    const size_t complete_blocks = n / bits_per_block;
    const size_t remaining_bits = n % bits_per_block;

    for (size_t i = 0; i < complete_blocks; ++i) {
        auto diff = blocks1[i] ^ blocks2[i];
        distance += __builtin_popcountll(diff);
    }

    if (remaining_bits > 0) {
        auto mask = (1ULL << remaining_bits) - 1;
        auto diff = (blocks1[complete_blocks] ^ blocks2[complete_blocks]) & mask;
        distance += __builtin_popcountll(diff);
    }

    return distance;
}

void generate_data(std::vector<std::pair<Bitset, double>>& data, size_t n, size_t syndrome_len) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint64_t> bit_dist(0, 1);
    std::uniform_real_distribution<double> cost_dist(0.0, 10.0);

    for (size_t i = 0; i < n; ++i) {
        Bitset bs(syndrome_len);
        for (size_t j = 0; j < syndrome_len; ++j) {
            bs[j] = bit_dist(rng);
        }
        double cost = cost_dist(rng);
        data.emplace_back(std::move(bs), cost);
    }
}

struct BoostBitsetHash {
    std::size_t operator()(const Bitset& bs) const {
        return boost::hash_value(bs);
    }
};

double heuristic_score(const double cost,
                       const Bitset& key,
                       const Bitset& target_syndrome,
                       size_t detector_number,
                       double alpha) {
    double distance = hamming_distance_prefix(key, target_syndrome, detector_number);
    return cost + alpha * distance;
}

// 方法 A
void method_A(std::vector<std::pair<Bitset, double>>& data, size_t k,
              const Bitset& target_syndrome, size_t detector_number, double alpha) {
    auto start = Clock::now();
    size_t n = data.size();
    std::vector<double> scores(n);
    
    for (size_t i = 0; i < n; ++i) {
        scores[i] = heuristic_score(data[i].second, data[i].first, target_syndrome, detector_number, alpha);
    }

    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; ++i) indices[i] = i;

    if (k < n) {
        std::nth_element(indices.begin(), indices.begin() + k, indices.end(),
                         [&](size_t a, size_t b) { return scores[a] < scores[b]; });
    }

    tsl::robin_map<Bitset, double, BoostBitsetHash> result;
    result.reserve(k);

    for (size_t i = 0; i < std::min(k, n); ++i) {
        const auto& [key, cost] = data[indices[i]];
        auto it = result.find(key);
        if (it == result.end() || cost < it->second) {
            result[key] = cost;  // 覆盖概率更小的项
        }
    }

    data.clear();
    data.reserve(result.size());
    for (const auto& [key, cost] : result) {
        data.emplace_back(key, cost);
    }

    auto end = Clock::now();
    std::chrono::duration<double> dur = end - start;
    std::cout << "[Method A] kept " << result.size() << " entries, elapsed: " << dur.count() << " s\n";
}

// 方法 B
void method_B(std::vector<std::pair<Bitset, double>>& data, size_t k,
              const Bitset& target_syndrome, size_t detector_number, double alpha) {
    auto start = Clock::now();

    size_t n = data.size();
    std::vector<double> scores(n);
    for (size_t i = 0; i < n; ++i) {
        scores[i] = heuristic_score(data[i].second, data[i].first, target_syndrome, detector_number, alpha);
    }

    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; ++i) indices[i] = i;

    std::nth_element(indices.begin(), indices.begin() + k, indices.end(),
                     [&](size_t a, size_t b) { return scores[a] < scores[b]; });

    std::vector<std::pair<Bitset, double>> topk_data;
    topk_data.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        topk_data.push_back(data[indices[i]]);
    }

    std::sort(topk_data.begin(), topk_data.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    });

    std::vector<std::pair<Bitset, double>> result;
    result.reserve(k);

    for (const auto& item : topk_data) {
        if (result.empty() || result.back().first != item.first) {
            result.push_back(item);
        }
    }

    auto end = Clock::now();
    std::chrono::duration<double> dur = end - start;
    std::cout << "[Method B] kept " << result.size() << " entries, elapsed: " << dur.count() << " s\n";
}

// 方法 C
void method_C(std::vector<std::pair<Bitset, double>>& data, size_t k,
              const Bitset& target_syndrome, size_t detector_number, double alpha) {
    auto start = Clock::now();

    size_t n = data.size();
    std::vector<double> scores(n);
    for (size_t i = 0; i < n; ++i) {
        scores[i] = heuristic_score(data[i].second, data[i].first, target_syndrome, detector_number, alpha);
    }

    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; ++i) indices[i] = i;

    std::nth_element(indices.begin(), indices.begin() + k, indices.end(),
                     [&](size_t a, size_t b) { return scores[a] < scores[b]; });

    std::vector<std::pair<Bitset, double>> topk_data;
    topk_data.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        topk_data.push_back(data[indices[i]]);
    }

    std::sort(topk_data.begin(), topk_data.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    auto last = std::unique(topk_data.begin(), topk_data.end(), [](const auto& a, const auto& b) {
        return a.first == b.first;
    });
    topk_data.erase(last, topk_data.end());

    auto end = Clock::now();
    std::chrono::duration<double> dur = end - start;
    std::cout << "[Method C] kept " << topk_data.size() << " entries, elapsed: " << dur.count() << " s\n";
}

// 通用去重函数
void deduplicate_distribution(std::vector<std::pair<Bitset, double>>& prob_dist) {
    tsl::robin_map<Bitset, double, BoostBitsetHash> dedup_map;
    for (const auto& [bs, cost] : prob_dist) {
        auto it = dedup_map.find(bs);
        if (it == dedup_map.end() || cost < it->second) {
            dedup_map[bs] = cost;
        }
    }
    prob_dist.clear();
    for (auto& [bs, cost] : dedup_map) {
        prob_dist.emplace_back(std::move(bs), cost);
    }
}

// 方法 D
void method_D(std::vector<std::pair<Bitset, double>>& data, size_t k,
              const Bitset& target_syndrome, size_t detector_number, double alpha) {
    auto start = Clock::now();

    std::sort(data.begin(), data.end(), [&](const auto& a, const auto& b) {
        return heuristic_score(a.second, a.first, target_syndrome, detector_number, alpha) <
               heuristic_score(b.second, b.first, target_syndrome, detector_number, alpha);
    });

    if (data.size() > k) {
        data.resize(k);
    }

    deduplicate_distribution(data);

    auto end = Clock::now();
    std::chrono::duration<double> dur = end - start;
    std::cout << "[Method D] kept " << data.size() << " entries, elapsed: " << dur.count() << " s\n";
}

void method_E(std::vector<std::pair<Bitset, double>>& data, size_t k,
              const Bitset& target_syndrome, size_t detector_number, double alpha) {
    auto start = Clock::now();

    std::vector<size_t> indices;
    std::vector<double> scores;
    indices.reserve(data.size());
    scores.reserve(data.size());

    for (size_t i = 0; i < data.size(); ++i) {
        const auto& [key, cost] = data[i];
        double distance = hamming_distance_prefix(key, target_syndrome, detector_number);
        double score = cost + alpha * distance;

        indices.push_back(i);
        scores.push_back(score);
    }

    if (indices.size() <= k) return;

    std::nth_element(
        indices.begin(), indices.begin() + k, indices.end(),
        [&](size_t a, size_t b) {
            return scores[a] < scores[b];
        });

    tsl::robin_map<Bitset, double, BoostBitsetHash> result;
    result.reserve(k);

    // for (size_t i = 0; i < k; ++i) {
    //     const auto& [key, cost] = data[indices[i]];
    //     result[key] = cost;  // 重复项直接覆盖（保留最后一次）
    // }
    for (size_t i = 0; i < k; ++i) {
        const auto& [key, cost] = data[indices[i]];
        auto it = result.find(key);
        if (it == result.end() || cost < it->second) {
            result[key] = cost;  // 保留更小的 cost
        }
    }

    data.clear();
    data.reserve(result.size());
    for (const auto& [key, cost] : result) {
        data.emplace_back(key, cost);
    }

    auto end = Clock::now();
    std::chrono::duration<double> dur = end - start;
    std::cout << "[Method E] kept " << data.size() << " entries, elapsed: " << dur.count() << " s\n";
}

void method_F(std::vector<std::pair<Bitset, double>>& data, size_t k,
              const Bitset& target_syndrome, size_t detector_number, double alpha) {
    auto start = Clock::now();

    // Step 1: 使用 hash map 先聚合出每个 key 的最佳 score 和 cost
    tsl::robin_map<Bitset, std::pair<double, double>, BoostBitsetHash> best_map;
    for (const auto& [key, cost] : data) {
        double distance = hamming_distance_prefix(key, target_syndrome, detector_number);
        double score = cost + alpha * distance;

        auto it = best_map.find(key);
        if (it == best_map.end() || score < it->second.first) {
            best_map[key] = {score, cost};
        }
    }

    // Step 2: 将 hash map 转为 vector，并执行 top-k 挑选
    std::vector<std::pair<Bitset, double>> temp;
    temp.reserve(best_map.size());
    for (auto& [key, pair] : best_map) {
        temp.emplace_back(std::move(key), pair.second);  // 只保留最小得分对应的 cost
    }

    if (temp.size() > k) {
        std::nth_element(temp.begin(), temp.begin() + k, temp.end(), [&](const auto& a, const auto& b) {
            double score_a = a.second + alpha * hamming_distance_prefix(a.first, target_syndrome, detector_number);
            double score_b = b.second + alpha * hamming_distance_prefix(b.first, target_syndrome, detector_number);
            return score_a < score_b;
        });
        temp.resize(k);
    }

    data = std::move(temp);

    auto end = Clock::now();
    std::chrono::duration<double> dur = end - start;
    std::cout << "[Method F] kept " << data.size() << " entries, elapsed: " << dur.count() << " s\n";
}


// int main() {
//     // const size_t n = 2000;
//     // const size_t n = 1000;
//     const size_t n = 100;
//     const size_t syndrome_len = 64;
//     const size_t k = n / 2;
//     const size_t detector_number = 48;
//     const double alpha = 0.5;

//     std::vector<std::pair<Bitset, double>> data;
//     generate_data(data, n, syndrome_len);

//     Bitset target_syndrome(detector_number);
//     for (size_t i = 0; i < detector_number; ++i) {
//         target_syndrome[i] = i % 2;
//     }

//     method_A(data, k, target_syndrome, detector_number, alpha);
//     method_B(data, k, target_syndrome, detector_number, alpha);
//     method_C(data, k, target_syndrome, detector_number, alpha);
//     method_D(data, k, target_syndrome, detector_number, alpha);
//     method_E(data, k, target_syndrome, detector_number, alpha);
//     method_F(data, k, target_syndrome, detector_number, alpha);

//     return 0;
// }
int main() {
    const size_t n = 1000;
    const size_t syndrome_len = 64;
    const size_t k = n / 2;
    const size_t detector_number = 48;
    const double alpha = 0.5;
    const int num_repeats = 100;

    Bitset target_syndrome(detector_number);
    for (size_t i = 0; i < detector_number; ++i) {
        target_syndrome[i] = i % 2;
    }

    using MethodFunc = void(*)(std::vector<std::pair<Bitset, double>>&, size_t, const Bitset&, size_t, double);
    const std::vector<std::pair<std::string, MethodFunc>> methods = {
        {"A", static_cast<MethodFunc>(method_A)},
        {"B", static_cast<MethodFunc>(method_B)},
        {"C", static_cast<MethodFunc>(method_C)},
        {"D", static_cast<MethodFunc>(method_D)},
        {"E", static_cast<MethodFunc>(method_E)},
        {"F", static_cast<MethodFunc>(method_F)}
    };

    std::vector<double> total_times(methods.size(), 0.0);

    for (int rep = 0; rep < num_repeats; ++rep) {
        std::cout << "Repeat #" << rep + 1 << ":\n";

        for (size_t i = 0; i < methods.size(); ++i) {
            std::vector<std::pair<Bitset, double>> data;
            generate_data(data, n, syndrome_len);

            auto start = Clock::now();
            methods[i].second(data, k, target_syndrome, detector_number, alpha);
            auto end = Clock::now();

            std::chrono::duration<double> dur = end - start;
            total_times[i] += dur.count();
        }

        std::cout << "-----------------------------\n";
    }

    std::cout << "\nAverage elapsed time over " << num_repeats << " runs:\n";
    for (size_t i = 0; i < methods.size(); ++i) {
        std::cout << "[Method " << methods[i].first << "] Average Time: "
                  << (total_times[i] / num_repeats) << " s\n";
    }

    return 0;
}

