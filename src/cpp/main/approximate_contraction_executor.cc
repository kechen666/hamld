#include "approximate_contraction_executor.h"
#include <algorithm>
#include <queue>
#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
// #include <omp.h> 
// #include <boost/dynamic_bitset.hpp>
#include <vector>

int hamming_distance_prefix(const boost::dynamic_bitset<>& bs1, 
                          const boost::dynamic_bitset<>& bs2, 
                          size_t n)
{
    // 参数检查
    if (bs2.size() != n) {
        throw std::invalid_argument("bs2 size must be equal to n");
    }
    if (bs1.size() < n) {
        throw std::invalid_argument("bs1 size must be at least n");
    }
    
    int distance = 0;
    
    // 使用to_block_range将位集转换为块数组
    std::vector<boost::dynamic_bitset<>::block_type> blocks1, blocks2;
    boost::to_block_range(bs1, std::back_inserter(blocks1));
    boost::to_block_range(bs2, std::back_inserter(blocks2));
    
    const size_t bits_per_block = bs1.bits_per_block;
    const size_t complete_blocks = n / bits_per_block;
    const size_t remaining_bits = n % bits_per_block;
    
    // 比较完整的块
    for (size_t i = 0; i < complete_blocks; ++i) {
        auto diff = blocks1[i] ^ blocks2[i];
        distance += __builtin_popcount(diff);
    }
    
    // 比较剩余的不完整块
    if (remaining_bits > 0) {
        auto mask = (1 << remaining_bits) - 1;
        auto diff = (blocks1[complete_blocks] ^ blocks2[complete_blocks]) & mask;
        distance += __builtin_popcount(diff);
    }
    
    return distance;
}

namespace {
    const std::unordered_map<ApproximateStrategy, double> strategy_params = {
        {ApproximateStrategy::NODE_TOPK, 100.0},
        {ApproximateStrategy::HYPEREDGE_TOPK, 1000.0},
        {ApproximateStrategy::NODE_THRESHOLD, 1e-10},
        {ApproximateStrategy::HYPEREDGE_THRESHOLD, 1e-10},
        {ApproximateStrategy::NO_NO, -1.0}
    };
}

ApproximateContractionExecutor::ApproximateContractionExecutor(
    int detector_number,
    int logical_number,
    const std::string& approximatestrategy,
    double approximate_param,
    std::shared_ptr<DetectorErrorModelHypergraph> hypergraph,
    int priority,
    int priority_topk,
    bool use_heuristic,
    double alpha,
    bool openmp,
    int openmp_num_threads,
    bool contract_logical_hyperedges
) : detector_number_(detector_number),
    logical_number_(logical_number),
    total_length_(detector_number + logical_number),
    approximatestrategy_(approximatestrategy),
    approximate_param_(approximate_param),
    priority_(priority),
    priority_topk_(priority_topk),
    hypergraph_(hypergraph),
    use_heuristic_(use_heuristic),
    alpha_(alpha),
    openmp_(openmp),
    openmp_num_threads_(openmp_num_threads),
    contract_logical_hyperedges_(contract_logical_hyperedges),
    order_({}),
    relevant_hyperedge_cache_({}),
    execution_contraction_time_(0.0),
    execution_hypergraph_approximate_time_(0.0),
    execution_init_time_(0.0),
    execution_order_finder_time_(0.0),
    execution_max_distribution_size_(0) {
    
    // Parse strategy
    size_t underscore_pos = approximatestrategy.find('_');
    if (underscore_pos == std::string::npos) {
        throw std::invalid_argument("Invalid approximate strategy format");
    }
    
    approximate_position_ = approximatestrategy.substr(0, underscore_pos);
    approximatestrategy_method_ = approximatestrategy.substr(underscore_pos + 1);
    is_topk_ = (approximatestrategy_method_ == "topk");
    is_threshold_ = (approximatestrategy_method_ == "threshold");
    
    // Set default parameter if not provided
    ApproximateStrategy strategy_enum;
    if (approximatestrategy == "node_topk") strategy_enum = ApproximateStrategy::NODE_TOPK;
    else if (approximatestrategy == "hyperedge_topk") strategy_enum = ApproximateStrategy::HYPEREDGE_TOPK;
    else if (approximatestrategy == "node_threshold") strategy_enum = ApproximateStrategy::NODE_THRESHOLD;
    else if (approximatestrategy == "hyperedge_threshold") strategy_enum = ApproximateStrategy::HYPEREDGE_THRESHOLD;
    else if (approximatestrategy == "no_no") strategy_enum = ApproximateStrategy::NO_NO;
    else throw std::invalid_argument("Invalid approximate strategy");
    
    if (approximate_param < 0) {
        approximate_param_ = strategy_params.at(strategy_enum);
    } else {
        approximate_param_ = approximate_param;
    }
}

tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash> 
ApproximateContractionExecutor::mld_contraction_no_slicing(const std::vector<bool>& syndrome) {
    // Convert syndrome to flipped detector set
    flipped_detector_set_.clear();
    
    
    for (size_t i = 0; i < syndrome.size(); ++i) {
        if (syndrome[i]) {
            flipped_detector_set_.push_back("D" + std::to_string(i));
        }
    }
    
    if (flipped_detector_set_.empty()) {
        tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash> init_prob_dist;
        init_prob_dist[boost::dynamic_bitset<>(total_length_, 0)] = const_1;
        return init_prob_dist;
    }
    
    auto [sub_weights, sub_order] = priority_sub_hypergraph_contraction_strategy("greedy");
    order_ = std::move(sub_order);

    auto start_init_time = std::chrono::high_resolution_clock::now();
    build_hyperedge_contraction_caches(sub_weights, order_);
    
    execution_init_time_ = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start_init_time).count();
    auto init_prob_dist = get_task_initial_input();
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto prob_dist = single_node_online_mld_contraction(init_prob_dist, sub_weights);
    execution_contraction_time_ = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - start_time).count();

    return prob_dist;
}

std::pair<std::vector<bool>, double> ApproximateContractionExecutor::validate_logical_operator(
    const tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash>& prob_dist) const{
    
    if (prob_dist.size() == 1) {
        const auto& key = prob_dist.begin()->first;
        std::vector<bool> logical_error;
        for (int i = 0; i < logical_number_; ++i) {
            logical_error.push_back(key[total_length_ - logical_number_ + i]);
        }
        return {logical_error, 1.0};
    } else if (prob_dist.empty()) {
        return {std::vector<bool>(logical_number_, false), 0.5};
    } else {
        // Find max probability key
        auto max_it = std::max_element(prob_dist.begin(), prob_dist.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        
        std::vector<bool> logical_error;
        for (int i = 0; i < logical_number_; ++i) {
            logical_error.push_back(max_it->first[total_length_ - logical_number_ + i]);
        }
        
        double total_prob = 0.0;
        for (const auto& [_, prob] : prob_dist) {
            total_prob += prob;
        }
        double prob_correct = max_it->second / total_prob;
        
        return {logical_error, prob_correct};
    }
}

// Private method implementations...

std::pair<std::unordered_map<std::string, double>, std::vector<std::string>> 
ApproximateContractionExecutor::priority_sub_hypergraph_contraction_strategy(std:: string order_method) {
    auto start_hypergraph_time = std::chrono::high_resolution_clock::now();
    DetectorErrorModelHypergraph sub_hypergraph;

    if (openmp_) {
        sub_hypergraph = hypergraph_->get_new_priority_sub_hypergraph_parallel(
            flipped_detector_set_,
            priority_,
            priority_topk_,
            openmp_num_threads_
        );
    } else {
        sub_hypergraph = hypergraph_->get_new_priority_sub_hypergraph(
            flipped_detector_set_,
            priority_,
            priority_topk_
        );
    }
    
    // 使用稀疏表示，删除无关的节点，对于无关的节点，不进行表示。
    // 核心在于，输入的参数，需要映射为子超图的节点序列，重新排序。同时检测器的数量参数需要更新。
    detector_number_ = sub_hypergraph.detector_number;
    total_length_ = detector_number_ + logical_number_;
    // 更新target_syndrome_
    target_syndrome_ = boost::dynamic_bitset<>(detector_number_, 0);
    // std::cout << "flipped_detector_set_: " ;
    for (const auto& detector : flipped_detector_set_) {
        if (detector[0] == 'D') {
            int index = std::stoi(detector.substr(1));
            if (index < detector_number_) {
                target_syndrome_[index].flip();
            }
            // std::cout << detector<< " ";
        }
    }
    // std::cout << std::endl;
    
    execution_hypergraph_approximate_time_ = std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now() - start_hypergraph_time).count();

    std::unordered_map<std::string, double> sub_weights;
    sub_weights = sub_hypergraph.get_hyperedges_weights_dict();

    ConnectivityGraph connectivity;
    connectivity.hypergraph_to_connectivity_graph(sub_hypergraph, false);

    auto start_order_finder_time = std::chrono::high_resolution_clock::now();
    
    std::vector<std::string> order;

    if (order_method == "random_bfs"){
        order = connectivity.bfs_random_start();
    }else{
        GreedyMLDOrderFinder order_finder(connectivity);
        order = order_finder.find_order();
        // 优先处理连通性小的节点。
        // std::cout << "connectivity graph: " << std::endl;
        // auto order = connectivity.sorted_nodes_by_degree(true);  // 按照度数升序排序
        // auto order = connectivity.sorted_nodes_by_degree(false);  // 按照度数降序排序
    }
    // for (const auto& node : order) {
    //     std::cout << node << " degree: " << connectivity.degree(node) << " ";
    // }
    // std::cout << std::endl;

    execution_order_finder_time_ = std::chrono::duration<double>(
    std::chrono::high_resolution_clock::now() - start_order_finder_time).count();
    return {sub_weights, order};
}

void ApproximateContractionExecutor::build_hyperedge_contraction_caches(
    const std::unordered_map<std::string, double>& contractable_hyperedges_weights_dict,
    const std::vector<std::string>& order) {

    relevant_hyperedge_cache_.clear();

    // 使用string_view避免临时字符串构造
    std::unordered_map<std::string_view, int> detector_to_index;
    detector_to_index.reserve(order.size());

    // 预先计算平均每个检测器的超边数
    const size_t avg_edges_per_detector = contractable_hyperedges_weights_dict.size() / std::max<size_t>(1, order.size());

    relevant_hyperedge_cache_.reserve(order.size());
    
    // 先填充索引和预分配空间
    for (const auto& det : order) {
        detector_to_index[det] = static_cast<int>(relevant_hyperedge_cache_.size());
        relevant_hyperedge_cache_[det] = {};
        relevant_hyperedge_cache_[det].reserve(avg_edges_per_detector);
    }

    // 使用string_view处理超边字符串
    for (const auto& [hyperedge_str, weight] : contractable_hyperedges_weights_dict) {
        std::string_view edge_sv(hyperedge_str);

        int min_index = std::numeric_limits<int>::max();
        std::string_view min_detector;

        size_t start = 0;
        while (start < edge_sv.size()) {
            size_t end = edge_sv.find(',', start);
            if (end == std::string_view::npos) {
                end = edge_sv.size();
            }

            std::string_view detector = edge_sv.substr(start, end - start);
            
            if (auto it = detector_to_index.find(detector); 
                it != detector_to_index.end() && it->second < min_index) {
                min_index = it->second;
                min_detector = it->first;
            }

            start = end + 1;
        }

        if (min_index != std::numeric_limits<int>::max()) {
            relevant_hyperedge_cache_[std::string(min_detector)].emplace_back(hyperedge_str);
        }
    }
}


void ApproximateContractionExecutor::approximate_distribution(
    tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash>& updated_prob_dist) {
    
    if (is_topk_) {
        size_t k = static_cast<size_t>(approximate_param_);
        if (updated_prob_dist.size() <= k) return;

        bool is_hyperedge = (approximate_position_ == "hyperedge");

        if (is_hyperedge && updated_prob_dist.size() <= 2 * k) {
            if (use_heuristic_) {
                // 启发式情况下，计算score + tuple
                using Entry = std::tuple<boost::dynamic_bitset<>, double, double>;
                std::vector<Entry> entries;
                entries.reserve(updated_prob_dist.size());
                for (const auto& [key, prob] : updated_prob_dist) {
                    if (prob == 0.0) {
                        continue;
                    }
                    double score = -std::log(prob) * hamming_distance_prefix(key, target_syndrome_, detector_number_);
                    entries.emplace_back(key, prob, score);
                }

                auto cmp = [](const Entry& a, const Entry& b) {
                    return std::get<2>(a) < std::get<2>(b);  // score升序
                };

                std::nth_element(entries.begin(), entries.begin() + k, entries.end(), cmp);

                tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash> result;
                result.reserve(k);
                for (size_t i = 0; i < k; ++i) {
                    result[std::get<0>(entries[i])] = std::get<1>(entries[i]);
                }

                updated_prob_dist = std::move(result);
                return;
            } else {
                // 非启发式，score == prob，避免额外字段
                using Entry = std::pair<boost::dynamic_bitset<>, double>;
                std::vector<Entry> entries;
                entries.reserve(updated_prob_dist.size());

                for (const auto& [key, prob] : updated_prob_dist) {
                    entries.emplace_back(key, prob);
                }

                auto cmp = [](const Entry& a, const Entry& b) {
                    return a.second > b.second;  // prob降序
                };

                std::nth_element(entries.begin(), entries.begin() + k, entries.end(), cmp);

                tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash> result;
                result.reserve(k);
                for (size_t i = 0; i < k; ++i) {
                    result[entries[i].first] = entries[i].second;
                }

                updated_prob_dist = std::move(result);
                return;
            }
        }

        auto calc_score = [&](const boost::dynamic_bitset<>& key, double prob) -> double {
            return use_heuristic_
                ? std::log(prob) - alpha_ * hamming_distance_prefix(key, target_syndrome_, detector_number_)
                : prob;
        };

        // 优化后堆方法
        using Entry = std::tuple<boost::dynamic_bitset<>, double, double>;
        auto cmp = [](const Entry& a, const Entry& b) {
            return std::get<2>(a) > std::get<2>(b);
        };
        std::priority_queue<Entry, std::vector<Entry>, decltype(cmp)> pq(cmp);

        for (const auto& [key, prob] : updated_prob_dist) {
            double score = calc_score(key, prob);
            if (pq.size() < k) {
                pq.emplace(key, prob, score);
            } else if (score > std::get<2>(pq.top())) {
                pq.pop();
                pq.emplace(key, prob, score);
            }
        }

        tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash> result;
        result.reserve(k);
        while (!pq.empty()) {
            auto&& [key, prob, _] = pq.top();
            result.emplace(std::move(key), prob);
            pq.pop();
        }

        updated_prob_dist = std::move(result);
        return;
    } else if (is_threshold_) {
        for (auto it = updated_prob_dist.begin(); it != updated_prob_dist.end(); ) {
            if (it->second <= approximate_param_) {
                it = updated_prob_dist.erase(it);
            } else {
                ++it;
            }
        }
    } else {
        throw std::runtime_error("Invalid approximate strategy");
    }
}

boost::dynamic_bitset<> ApproximateContractionExecutor::flip_bits(
    const boost::dynamic_bitset<>& bs, 
    const std::string& hyperedge) const{
    
    boost::dynamic_bitset<> result = bs;
    const char* ptr = hyperedge.c_str();
    const char* end = ptr + hyperedge.size();

    while (ptr < end) {
        char type = *ptr++;  // 'D' or 'L'
        int index = 0;

        // 手动解析数字（避免 std::stoi）
        while (ptr < end && isdigit(*ptr)) {
            index = index * 10 + (*ptr++ - '0');
        }

        // 判断是否为逻辑位
        if (type == 'L') {
            index += detector_number_;
        }

        result.flip(index);

        // 跳过逗号
        if (ptr < end && *ptr == ',') {
            ++ptr;
        }
    }

    return result;
}


void ApproximateContractionExecutor::contract_hyperedge(
    tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash>& prob_dist,
    std::unordered_map<std::string, double>& contractable_hyperedges_weights_dict,
    const std::string& contracted_hyperedge) {
    
    double contracted_prob = contractable_hyperedges_weights_dict.at(contracted_hyperedge);
    contractable_hyperedges_weights_dict.erase(contracted_hyperedge);

    double non_flip_prob = const_1 - contracted_prob;
    
    tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash> updated_dist;
    updated_dist.reserve(prob_dist.size() * 2);  // conservative estimate
    
    for (const auto& [bs, prob] : prob_dist) {
        boost::dynamic_bitset<> flipped_bs = flip_bits(bs, contracted_hyperedge);
        
        double flipped_prob = prob * contracted_prob;
        double non_flipped_prob = prob * non_flip_prob;
        
        updated_dist[flipped_bs] += flipped_prob;
        updated_dist[bs] += non_flipped_prob;
    }
    
    if (approximate_position_ == "hyperedge") {
        approximate_distribution(updated_dist);
    }

    prob_dist = std::move(updated_dist);
}

tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash> 
ApproximateContractionExecutor::get_task_initial_input() const{
    tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash> init_prob_dist;
    init_prob_dist[boost::dynamic_bitset<>(total_length_, 0)] = const_1;

    return init_prob_dist;
}

tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash> 
ApproximateContractionExecutor::single_node_online_mld_contraction(
    const tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash>& init_prob_dist,
    const std::unordered_map<std::string, double>& init_contractable_hyperedges_weights) {
    
    tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash> prob_dist = init_prob_dist;
    std::unordered_map<std::string, double> contractable_weights = init_contractable_hyperedges_weights;

    for (size_t contraction_step = 0; contraction_step < order_.size(); ++contraction_step) {
        const auto& detector = order_[contraction_step];
        int detector_index = std::stoi(detector.substr(1));
        bool observed_syndrome = target_syndrome_[detector_index];

        const auto& relevant_hyperedges = relevant_hyperedge_cache_.at(detector);
        for (const auto& hyperedge : relevant_hyperedges) {
            contract_hyperedge(prob_dist, contractable_weights, hyperedge);
        }

        tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash> filtered_dist;
        filtered_dist.reserve(prob_dist.size());  // conservative estimate

        for (const auto& [candidate, prob] : prob_dist) {
            if (check_str_value(candidate, detector_index, observed_syndrome)) {
                filtered_dist[candidate] = prob;
            }
        }
        prob_dist = std::move(filtered_dist);

        // 检查是否为空，为空则直接放回，执行更高范围的解码。
        if (prob_dist.empty()) {
            // prob_dist.clear();
            return {};
        }

        if (approximate_position_ == "node") {
            approximate_distribution(prob_dist);
        }
    }

    if (contract_logical_hyperedges_ && !contractable_weights.empty()) {
        std::vector<std::string> relevant_logical_hyperedges;
        for (const auto& [hyperedge, weight] : contractable_weights) {
            relevant_logical_hyperedges.push_back(hyperedge);
        }
        // std::cout << "logical hyperedges number: " << relevant_logical_hyperedges.size() << std::endl;
        for (const auto& hyperedge : relevant_logical_hyperedges) {
            contract_hyperedge(prob_dist, contractable_weights, hyperedge);
        }
    }
    
    execution_max_distribution_size_ = std::max(execution_max_distribution_size_, static_cast<int>(prob_dist.size()));
    return prob_dist;
}

bool ApproximateContractionExecutor::check_str_value(
    const boost::dynamic_bitset<>& bs,
    int position,
    bool expected_value) const{
    
    return bs[position] == expected_value;
}