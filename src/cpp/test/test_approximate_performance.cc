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

double heuristic_score(const double cost, const Bitset& /*bs*/, const Bitset& /*target*/, size_t /*detector_num*/) {
    return cost;
}

// 方法 A：nth_element 选 top-k + robin_map 去重插入保留更小 cost
void method_A(const std::vector<std::pair<Bitset, double>>& data, size_t k) {
    auto start = Clock::now();

    size_t n = data.size();
    std::vector<double> scores(n);
    for (size_t i = 0; i < n; ++i) {
        scores[i] = heuristic_score(data[i].second, data[i].first, Bitset(), 0);
    }

    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; ++i) indices[i] = i;

    std::nth_element(indices.begin(), indices.begin() + k, indices.end(),
                     [&](size_t a, size_t b) { return scores[a] < scores[b]; });

    tsl::robin_map<Bitset, double, BoostBitsetHash> result;
    result.reserve(k);

    for (size_t i = 0; i < k; ++i) {
        const auto& [key, cost] = data[indices[i]];
        auto it = result.find(key);
        if (it == result.end()) {
            result.emplace(key, cost);
        } else {
            if (cost < it->second) {
                result.erase(it);
                result.emplace(key, cost);
            }
        }
    }

    auto end = Clock::now();
    std::chrono::duration<double> dur = end - start;
    std::cout << "[Method A] kept " << result.size() << " entries, elapsed: " << dur.count() << " s\n";
}


// 方法 B：nth_element 选 top-k + 先排序（key、cost）+ 线性去重保留更小 cost
void method_B(std::vector<std::pair<Bitset, double>> data, size_t k) {
    auto start = Clock::now();

    size_t n = data.size();
    std::vector<double> scores(n);
    for (size_t i = 0; i < n; ++i) {
        scores[i] = heuristic_score(data[i].second, data[i].first, Bitset(), 0);
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

    // 排序 key 升序，cost 升序
    std::sort(topk_data.begin(), topk_data.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    });

    std::vector<std::pair<Bitset, double>> result;
    result.reserve(k);

    for (const auto& item : topk_data) {
        if (result.empty() || result.back().first != item.first) {
            result.push_back(item);
        } else {
            // cost小的已在前面，跳过更大cost
        }
    }

    auto end = Clock::now();
    std::chrono::duration<double> dur = end - start;
    std::cout << "[Method B] kept " << result.size() << " entries, elapsed: " << dur.count() << " s\n";
}

// 方法 C：nth_element 选 top-k + 先排序（key）+ unique 去重（保留第一次出现）
void method_C(std::vector<std::pair<Bitset, double>> data, size_t k) {
    auto start = Clock::now();

    size_t n = data.size();
    std::vector<double> scores(n);
    for (size_t i = 0; i < n; ++i) {
        scores[i] = heuristic_score(data[i].second, data[i].first, Bitset(), 0);
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

    // 仅按 key 升序排序
    std::sort(topk_data.begin(), topk_data.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // unique 去重，只保留第一个出现
    auto last = std::unique(topk_data.begin(), topk_data.end(), [](const auto& a, const auto& b) {
        return a.first == b.first;
    });
    topk_data.erase(last, topk_data.end());

    auto end = Clock::now();
    std::chrono::duration<double> dur = end - start;
    std::cout << "[Method C] kept " << topk_data.size() << " entries, elapsed: " << dur.count() << " s\n";
}

// 去重函数（保留cost最小）
void deduplicate_distribution(std::vector<std::pair<Bitset, double>>& prob_dist) {
    std::unordered_set<Bitset, BoostBitsetHash> unique_check;
    unique_check.reserve(prob_dist.size());
    bool has_duplicates = false;

    for (const auto& [bs, cost] : prob_dist) {
        if (!unique_check.insert(bs).second) {
            has_duplicates = true;
            break;
        }
    }

    if (!has_duplicates) return;

    tsl::robin_map<Bitset, double, BoostBitsetHash> dedup_map;
    dedup_map.reserve(prob_dist.size());

    for (const auto& [bs, cost] : prob_dist) {
        auto it = dedup_map.find(bs);
        if (it == dedup_map.end()) {
            dedup_map.emplace(bs, cost);
        } else if (cost < it->second) {
            dedup_map[bs] = cost;  // Modify through the map's non-const interface
        }
    }

    prob_dist.clear();
    prob_dist.reserve(dedup_map.size());
    for (auto& [bs, cost] : dedup_map) {
        prob_dist.emplace_back(std::move(bs), cost);
    }
}

void method_D(std::vector<std::pair<Bitset, double>> data, size_t k) {
    auto start = Clock::now();

    auto cmp = [](const auto& a, const auto& b) {
        return a.second < b.second;  // 升序，保留概率大的在前
    };

    std::nth_element(data.begin(), data.begin() + k, data.end(), cmp);
    data.erase(data.begin() + k, data.end());

    deduplicate_distribution(data);

    auto end = Clock::now();
    std::chrono::duration<double> dur = end - start;
    std::cout << "[Method D] kept " << data.size() << " entries, elapsed: " << dur.count() << " s\n";
}

// int main() {
//     // const size_t n = 2000;
//     const size_t n = 10;
//     const size_t syndrome_len = 64;
//     const size_t k = n / 2;

//     std::vector<std::pair<Bitset, double>> data;
//     generate_data(data, n, syndrome_len);

//     method_A(data, k);
//     method_B(data, k);
//     method_C(data, k);
//     method_D(data, k);

//     return 0;
// }

void method_A(const std::vector<std::pair<Bitset, double>>& data, size_t k);
void method_B(std::vector<std::pair<Bitset, double>> data, size_t k);
void method_C(std::vector<std::pair<Bitset, double>> data, size_t k);
void method_D(std::vector<std::pair<Bitset, double>> data, size_t k);

int main() {
    const size_t n = 100;       // 测试规模，可以调节
    const size_t syndrome_len = 64;
    const size_t k = n / 2;
    const int num_repeats = 100;   // 重复次数

    // 预先生成一份数据，重复使用
    std::vector<std::pair<Bitset, double>> original_data;
    generate_data(original_data, n, syndrome_len);

    using MethodFunc = void(*)(std::vector<std::pair<Bitset, double>>, size_t);
    // 注意method_A参数是const引用，需要包一层lambda转换
    std::vector<std::pair<std::string, std::function<void()>>> methods = {
        {"A", [&](){ method_A(original_data, k); }},
        {"B", [&](){ method_B(original_data, k); }},
        {"C", [&](){ method_C(original_data, k); }},
        {"D", [&](){ method_D(original_data, k); }},
    };

    std::vector<double> total_times(methods.size(), 0.0);

    for (int rep = 0; rep < num_repeats; ++rep) {
        std::cout << "Repeat #" << (rep + 1) << ":\n";
        for (size_t i = 0; i < methods.size(); ++i) {
            auto start = Clock::now();
            methods[i].second();  // 调用对应方法
            auto end = Clock::now();

            std::chrono::duration<double> dur = end - start;
            total_times[i] += dur.count();
        }
        std::cout << "---------------------------\n";
    }

    std::cout << "Average elapsed time over " << num_repeats << " runs:\n";
    for (size_t i = 0; i < methods.size(); ++i) {
        std::cout << "[Method " << methods[i].first << "] Average Time: " << total_times[i] / num_repeats << " s\n";
    }

    return 0;
}