#ifndef APPROXIMATE_CONTRACTION_EXECUTOR_H
#define APPROXIMATE_CONTRACTION_EXECUTOR_H

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <chrono>
#include "detector_hypergraph.h"
#include "connectivity_graph.h"
#include "greedy_mld_order.h"
#include <tsl/robin_map.h>
#include <boost/dynamic_bitset.hpp>

// 自定义 boost::dynamic_bitset 哈希函数
// #include <functional>

// struct BoostBitsetHash {
//     std::size_t operator()(const boost::dynamic_bitset<>& bs) const noexcept {
//         std::size_t seed = 0;
//         for (std::size_t i = 0; i < bs.num_blocks(); ++i) {
//             auto block = bs.Block(i);
//             seed ^= std::hash<decltype(block)>{}(block) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
//         }
//         return seed;
//     }
// };
#include <boost/functional/hash.hpp>

struct BoostBitsetHash {
    std::size_t operator()(const boost::dynamic_bitset<>& bs) const {
        return boost::hash_value(bs);
    }
};

// 在 approximate_contraction_executor.h 中修改 BoostBitsetHash 结构体
// struct BoostBitsetHash {
//     size_t operator()(const boost::dynamic_bitset<>& b) const {
//         using block_type = boost::dynamic_bitset<>::block_type;
//         size_t seed = 0;
        
//         // 直接遍历块而不创建中间向量
//         for (size_t i = 0; i < b.num_blocks(); ++i) {
//             seed ^= b[i] * 1099511628211ULL; // 质数
//             seed = (seed << 17) | (seed >> 47); // 旋转哈希
//         }
        
//         return seed;
//     }
// };

enum class ApproximateStrategy {
    NODE_TOPK,
    HYPEREDGE_TOPK,
    NODE_THRESHOLD,
    HYPEREDGE_THRESHOLD,
    NO_NO
};

class ApproximateContractionExecutor {
public:
    ApproximateContractionExecutor(
        int detector_number,
        int logical_number,
        const std::string& approximatestrategy = "hyperedge_threshold",
        double approximate_param = -1.0,
        std::shared_ptr<DetectorErrorModelHypergraph> hypergraph = nullptr,
        int priority = 1,
        int priority_topk = 10,
        bool use_heuristic = true,
        double alpha = 0.05,
        bool openmp = false,
        int openmp_num_threads = 1, 
        bool contract_logical_hyperedges = true
    );

    // 使用 bitset 版本的概率分布
    tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash> 
    mld_contraction_no_slicing(const std::vector<bool>& syndrome);
    
    std::pair<std::vector<bool>, double> validate_logical_operator(
        const tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash>& prob_dist) const;

    double get_execution_contraction_time() const { return execution_contraction_time_; }
    double get_execution_init_time() const { return execution_init_time_; }
    double get_execution_hypergraph_approximate_time() const { return execution_hypergraph_approximate_time_; }
    double get_execution_order_finder_time() const { return execution_order_finder_time_; }
    int get_execution_max_distribution_size() const { return execution_max_distribution_size_; }

private:
    int detector_number_;
    int logical_number_;
    int total_length_;

    std::string approximatestrategy_;
    double approximate_param_;
    
    int priority_;
    int priority_topk_;
    std::vector<std::string> flipped_detector_set_;
    std::shared_ptr<DetectorErrorModelHypergraph> hypergraph_;

    const bool use_heuristic_;  // 启用启发式排序
    const double alpha_;        // 启发式强度
    const bool openmp_;
    const int openmp_num_threads_;

    const bool contract_logical_hyperedges_;

    std::vector<std::string> order_;
    boost::dynamic_bitset<> target_syndrome_; // 使用 bitset 存储目标错误 syndrome
    
    const double const_1 = 1.0;

    bool is_topk_;
    bool is_threshold_;
    std::string approximate_position_;
    std::string approximatestrategy_method_;
    
    std::unordered_map<std::string, std::vector<std::string>> relevant_hyperedge_cache_;

    double execution_contraction_time_;
    double execution_hypergraph_approximate_time_;
    double execution_init_time_;
    double execution_order_finder_time_;
    
    int execution_max_distribution_size_;

    std::pair<std::unordered_map<std::string, double>, std::vector<std::string>> priority_sub_hypergraph_contraction_strategy(std:: string order_method);
    
    void approximate_distribution(
        tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash>& updated_prob_dist);
    
    boost::dynamic_bitset<> flip_bits(
        const boost::dynamic_bitset<>& bs, 
        const std::string& hyperedge) const;
    
    void contract_hyperedge(
        tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash>& prob_dist,
        std::unordered_map<std::string, double>& contractable_hyperedges_weights_dict,
        const std::string& contracted_hyperedge);
    
    tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash> get_task_initial_input() const;
    
    tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash> single_node_online_mld_contraction(
        const tsl::robin_map<boost::dynamic_bitset<>, double, BoostBitsetHash>& init_prob_dist,
        const std::unordered_map<std::string, double>& init_contractable_hyperedges_weights);
    
    void build_hyperedge_contraction_caches(
        const std::unordered_map<std::string, double>& contractable_hyperedges_weights_dict,
        const std::vector<std::string>& order);
    
    bool check_str_value(const boost::dynamic_bitset<>& bs, int position, bool expected_value) const;
};

// 计算两个 bitset 之间的汉明距离
int hamming_distance_prefix(const boost::dynamic_bitset<>& bs1, const boost::dynamic_bitset<>& bs2, size_t n);

#endif // APPROXIMATE_CONTRACTION_EXECUTOR_H