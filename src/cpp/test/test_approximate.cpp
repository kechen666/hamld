#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <tuple>

#include <tsl/robin_map.h>
#include <boost/dynamic_bitset.hpp>
#include <boost/functional/hash.hpp>

using namespace std;
using Clock = chrono::high_resolution_clock;

using Bitset = boost::dynamic_bitset<>;

// 使用 boost::hash_value 实现哈希函数
struct BoostBitsetHash {
    std::size_t operator()(const Bitset& bs) const {
        return boost::hash_value(bs);
    }
};

// 计算汉明距离
int hamming_distance(const Bitset& a, const Bitset& b) {
    Bitset tmp = a ^ b;
    return tmp.count();
}

// 模拟参数
const bool use_heuristic_ = true;
const double alpha_ = 0.5;
// 目标 syndrome 随机初始化
Bitset target_syndrome_(64);

// 计算score函数
double calc_score(const Bitset& key, double prob) {
    if (prob <= 0) return -1e9;
    if (use_heuristic_) {
        return std::log(prob) - alpha_ * hamming_distance(key, target_syndrome_);
    } else {
        return prob;
    }
}

// 生成随机 bitset 和概率
void generate_data(tsl::robin_map<Bitset, double, BoostBitsetHash>& data, size_t n) {
    mt19937 rng(12345);
    uniform_real_distribution<double> dist_prob(0.000001, 1.0);
    uniform_int_distribution<uint64_t> dist_bits(0, numeric_limits<uint64_t>::max());

    for (size_t i = 0; i < n; ++i) {
        uint64_t val = dist_bits(rng);
        Bitset bs(1000, val);
        double prob = dist_prob(rng);
        data.emplace(bs, prob);
    }
}

// top-k 用 nth_element + sort
void topk_nth_element(
    const tsl::robin_map<Bitset, double, BoostBitsetHash>& data,
    size_t k,
    tsl::robin_map<Bitset, double, BoostBitsetHash>& result)
{
    using Entry = tuple<Bitset, double, double>;
    vector<Entry> entries;
    entries.reserve(data.size());

    for (const auto& [key, prob] : data) {
        double score = calc_score(key, prob);
        entries.emplace_back(key, prob, score);
    }

    // 降序，score高的排前面
    auto cmp = [](const Entry& a, const Entry& b) {
        return std::get<2>(a) > std::get<2>(b);
    };

    // 选出top k最大score
    nth_element(entries.begin(), entries.begin() + k, entries.end(), cmp);
    // sort(entries.begin(), entries.begin() + k, cmp);

    result.clear();
    result.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        result[std::get<0>(entries[i])] = std::get<1>(entries[i]);
    }
}

// top-k 用优先队列维护
void topk_priority_queue(
    const tsl::robin_map<Bitset, double, BoostBitsetHash>& data,
    size_t k,
    tsl::robin_map<Bitset, double, BoostBitsetHash>& result)
{
    using Pair = pair<Bitset, double>;

    // 小顶堆，堆顶最小score
    auto cmp = [](const Pair& a, const Pair& b) { return a.second > b.second; };
    priority_queue<Pair, vector<Pair>, decltype(cmp)> pq(cmp);

    for (const auto& [key, prob] : data) {
        if (prob <= 0) continue;
        double score = calc_score(key, prob);

        if (pq.size() < k) {
            pq.emplace(key, score);
        } else if (score > pq.top().second) {
            pq.pop();
            pq.emplace(key, score);
        }
    }

    // 取出结果并排序
    vector<pair<Bitset, double>> tmp;
    tmp.reserve(pq.size());
    while (!pq.empty()) {
        tmp.emplace_back(pq.top().first, pq.top().second);
        pq.pop();
    }
    // 按score降序排序
    sort(tmp.begin(), tmp.end(),
         [](const auto& a, const auto& b) { return a.second > b.second; });

    result.clear();
    result.reserve(k);
    for (const auto& [key, score] : tmp) {
        // 注意：这里score是calc_score的结果，转换回prob仅作示范
        result[key] = std::exp(score + alpha_ * hamming_distance(key, target_syndrome_));
    }
}

// top-k 用 partial_sort
void topk_partial_sort(
    const tsl::robin_map<Bitset, double, BoostBitsetHash>& data,
    size_t k,
    tsl::robin_map<Bitset, double, BoostBitsetHash>& result)
{
    using Entry = tuple<Bitset, double, double>;
    vector<Entry> entries;
    entries.reserve(data.size());

    for (const auto& [key, prob] : data) {
        double score = calc_score(key, prob);
        entries.emplace_back(key, prob, score);
    }

    // 降序，score高的排前面
    auto cmp = [](const Entry& a, const Entry& b) {
        return std::get<2>(a) > std::get<2>(b);
    };

    // 选出top k最大score，用 nth_element+partial_sort
    // nth_element(entries.begin(), entries.begin() + k, entries.end(), cmp);
    std::partial_sort(entries.begin(), entries.begin() + k, entries.end(), cmp);

    result.clear();
    result.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        result[std::get<0>(entries[i])] = std::get<1>(entries[i]);
    }
}
// top-k 用线性替换，适合 n=2k 场景
void topk_linear_replace(
    const tsl::robin_map<Bitset, double, BoostBitsetHash>& data,
    size_t k,
    tsl::robin_map<Bitset, double, BoostBitsetHash>& result)
{
    using Entry = pair<Bitset, double>;
    vector<Entry> top_k;
    top_k.reserve(k);

    auto it = data.begin();
    for (size_t i = 0; i < k && it != data.end(); ++i, ++it) {
        double score = calc_score(it->first, it->second);
        top_k.emplace_back(it->first, score);
    }

    // 找到当前最小的 entry
    auto find_min = [&]() {
        return min_element(top_k.begin(), top_k.end(),
            [](const Entry& a, const Entry& b) {
                return a.second < b.second;
            });
    };

    auto min_it = find_min();

    // 剩余的 n-k 项逐一判断是否替换
    for (; it != data.end(); ++it) {
        double score = calc_score(it->first, it->second);
        if (score > min_it->second) {
            *min_it = make_pair(it->first, score);
            min_it = find_min();
        }
    }

    result.clear();
    result.reserve(k);
    for (const auto& [key, score] : top_k) {
        // 还原出原 prob（可选）
        result[key] = std::exp(score + alpha_ * hamming_distance(key, target_syndrome_));
    }
}


int main() {
    size_t n = 2000;  // 测试规模
    size_t k = 1000;   // top k
    int repeat = 10;    // 重复次数

    target_syndrome_ = Bitset(1000, 0x123456789abcdef0ULL);

    cout << "Generating data with n=" << n << "...\n";
    tsl::robin_map<Bitset, double, BoostBitsetHash> data;
    data.reserve(n);
    generate_data(data, n);

    tsl::robin_map<Bitset, double, BoostBitsetHash> result1, result2;

    // 统计 nth_element 平均时间
    double total_time_nth = 0.0;
    for (int i = 0; i < repeat; ++i) {
        auto start = Clock::now();
        topk_nth_element(data, k, result1);
        auto end = Clock::now();
        total_time_nth += chrono::duration<double>(end - start).count();
    }
    cout << "Testing top-k nth_element...\n";
    cout << "Average Time: " << total_time_nth / repeat << " s\n";

    // 统计 priority_queue 平均时间
    double total_time_pq = 0.0;
    for (int i = 0; i < repeat; ++i) {
        auto start = Clock::now();
        topk_priority_queue(data, k, result2);
        auto end = Clock::now();
        total_time_pq += chrono::duration<double>(end - start).count();
    }
    cout << "Testing top-k priority queue...\n";
    cout << "Average Time: " << total_time_pq / repeat << " s\n";

    cout << "Result sizes: nth_element=" << result1.size()
         << ", priority_queue=" << result2.size() << "\n";

    // 简单比较结果key集合是否相同（不考虑顺序）
    size_t common = 0;
    for (const auto& [key, prob] : result1) {
        if (result2.find(key) != result2.end()) ++common;
    }
    cout << "Common keys in both results: " << common << endl;


    tsl::robin_map<Bitset, double, BoostBitsetHash> result3;

    double total_time_partial_sort = 0.0;
    for (int i = 0; i < repeat; ++i) {
        auto start = Clock::now();
        topk_partial_sort(data, k, result3);
        auto end = Clock::now();
        total_time_partial_sort += chrono::duration<double>(end - start).count();
    }

    cout << "Testing top-k partial_sort...\n";
    cout << "Average Time: " << total_time_partial_sort / repeat << " s\n";

    cout << "Result size partial_sort: " << result3.size() << "\n";

    // 比较 partial_sort 结果和 nth_element 结果相似度
    size_t common_partial = 0;
    for (const auto& [key, prob] : result1) {
        if (result3.find(key) != result3.end()) ++common_partial;
    }
    cout << "Common keys between nth_element and partial_sort: " << common_partial << endl;

    tsl::robin_map<Bitset, double, BoostBitsetHash> result4;
    double total_time_linear = 0.0;
    for (int i = 0; i < repeat; ++i) {
        auto start = Clock::now();
        topk_linear_replace(data, k, result4);
        auto end = Clock::now();
        total_time_linear += chrono::duration<double>(end - start).count();
    }
    cout << "Testing top-k linear replace (suitable for n ≈ 2k)...\n";
    cout << "Average Time: " << total_time_linear / repeat << " s\n";

    size_t common_linear = 0;
    for (const auto& [key, prob] : result1) {
        if (result4.find(key) != result4.end()) ++common_linear;
    }
    cout << "Common keys between nth_element and linear_replace: " << common_linear << endl;

    return 0;
}
