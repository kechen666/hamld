#pragma once

#include "stim.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <boost/dynamic_bitset.hpp>
#include "approximate_contraction_executor.h"  // 你实现的近似收缩执行器
#include "detector_hypergraph.h"               // 你使用的超图类

class HAMLDCpp {
public:
    // 构造函数
    HAMLDCpp(
        const stim::DetectorErrorModel& detector_error_model,
        const std::string& approximatestrategy,
        double approximate_param,
        int priority,
        int priority_topk,
        bool use_heuristic = false,
        double alpha = 0.05,
        bool openmp = false,
        int openmp_num_threads = 1
    );

    // 单次解码，返回逻辑错误预测，状态概率映射和概率
    std::tuple<
        std::vector<bool>, 
        std::unordered_map<std::string, double>, 
        double
    > decode(const std::vector<bool>& syndrome);

    // 批量解码（串行）
    std::vector<std::vector<bool>> decode_batch(
        const std::vector<std::vector<bool>>& syndromes,
        bool output_prob = false,
        std::vector<std::unordered_map<std::string, double>>* prob_dists = nullptr
    );

    // 批量解码（多线程并行）
    std::vector<std::vector<bool>> parallel_decode_batch(
        const std::vector<std::vector<bool>>& syndromes,
        bool output_prob = false,
        std::vector<std::unordered_map<std::string, double>>* prob_dists = nullptr,
        int num_threads = 4
    );

    // 一些统计时间获取函数（可选，如果你有这些）
    double total_contraction_time() const { return total_contraction_time_; }
    double total_hypergraph_approximate_time() const { return total_hypergraph_approximate_time_; }
    double total_init_time() const { return total_init_time_; }
    double total_running_time() const { return total_running_time_; }
    double total_order_finder_time() const { return total_order_finder_time_; }

private:
    stim::DetectorErrorModel detector_error_model_;

    std::string approximatestrategy_;
    double approximate_param_;
    int priority_;
    int priority_topk_;
    bool use_heuristic_;
    double alpha_;
    bool openmp_;
    int openmp_num_threads_;

    int num_detectors_;
    int num_observables_;
    int total_length_;

    std::shared_ptr<DetectorErrorModelHypergraph> hypergraph_;
    std::unique_ptr<ApproximateContractionExecutor> contractor_;

    // 计时统计
    double total_contraction_time_;
    double total_hypergraph_approximate_time_;
    double total_init_time_;
    double total_running_time_;
    double total_order_finder_time_;
};
