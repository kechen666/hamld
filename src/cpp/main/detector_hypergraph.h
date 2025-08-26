#pragma once

#include "stim.h"
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <tuple>

class DetectorErrorModelHypergraph {
public:
    stim::DetectorErrorModel detector_error_model;
    std::vector<std::string> nodes;
    std::vector<std::vector<std::string>> hyperedges;
    std::vector<double> weights;
    bool have_logical_observable;
    int detector_number;
    int logical_observable_number;

    // 新增缓存映射：超边（str） -> 权重
    std::vector<std::string> str_hyperedges;
    std::unordered_map<std::string, double> hyperedge_weight_map_;

    

    DetectorErrorModelHypergraph();
    DetectorErrorModelHypergraph(const stim::DetectorErrorModel& dem, bool include_logical = false);

    std::vector<std::string> get_nodes(bool include_logical = true) const;
    std::vector<std::vector<std::string>> get_hyperedges() const;
    std::vector<double> get_weights() const;
    const std::vector<std::string>& get_str_hyperedges() const;
    std::string canonicalize_edge(const std::vector<std::string>& edge) const;

    DetectorErrorModelHypergraph get_new_priority_sub_hypergraph(std::vector<std::string>& selected_nodes, int priority, int topk = -1);
    
    DetectorErrorModelHypergraph get_new_priority_sub_hypergraph_parallel(const std::vector<std::string>& selected_nodes, int priority, int topk = -1, int openmp_num_threads = 1);
    DetectorErrorModelHypergraph remove_nodes(const std::vector<std::string>& remove);

    std::unordered_map<std::string, double> get_hyperedges_weights_dict(const std::vector<std::vector<std::string>>* hyperedges = nullptr) const;

    std::unordered_map<std::string, std::vector<std::string>> get_detector_to_hyperedges_map() const;

    // 打印超图信息的接口
    std::string to_string() const;
    void print() const;
private:
    std::tuple<std::vector<std::string>, std::vector<std::vector<std::string>>, std::vector<double>> detector_error_model_to_hypergraph(const stim::DetectorErrorModel& dem);
    std::vector<std::string> detector_and_logical_observable_number_to_hypernodes(int d, int l);
    std::pair<std::vector<std::vector<std::string>>, std::vector<double>> detector_error_model_to_hyperedge(const stim::DetectorErrorModel& dem);
    std::vector<std::string> error_even_to_hyperedge(const stim::SpanRef<const stim::DemTarget>& targets);
};
