#include "detector_hypergraph.h"
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iomanip>  // 用于设置输出格式
#include <omp.h>

// 默认构造函数
DetectorErrorModelHypergraph::DetectorErrorModelHypergraph()
    : detector_error_model(),  // stim::DetectorErrorModel默认构造
      have_logical_observable(false),
      detector_number(0),
      logical_observable_number(0) {
    nodes.clear();
    hyperedges.clear();
    weights.clear();
    str_hyperedges.clear();
    hyperedge_weight_map_.clear();
}

DetectorErrorModelHypergraph::DetectorErrorModelHypergraph(const stim::DetectorErrorModel& dem, bool include_logical)
    : detector_error_model(dem), have_logical_observable(include_logical) {

    std::tie(nodes, hyperedges, weights) = detector_error_model_to_hypergraph(dem);

    // 添加如下代码以初始化 str_hyperedges 和 hyperedge_weight_map_
    str_hyperedges.clear();
    hyperedge_weight_map_.clear();

    for (size_t i = 0; i < hyperedges.size(); ++i) {
        std::string edge_str = canonicalize_edge(hyperedges[i]);
        str_hyperedges.push_back(edge_str);
        hyperedge_weight_map_[edge_str] = weights[i];
    }
}

std::string DetectorErrorModelHypergraph::canonicalize_edge(const std::vector<std::string>& edge) const {
    std::vector<std::string> sorted_edge = edge;
    std::sort(sorted_edge.begin(), sorted_edge.end());
    std::string result;
    for (size_t i = 0; i < sorted_edge.size(); ++i) {
        if (i > 0) result += ",";
        result += sorted_edge[i];
    }
    return result;
}

std::unordered_map<std::string, std::vector<std::string>> DetectorErrorModelHypergraph::get_detector_to_hyperedges_map() const {
    
    std::unordered_map<std::string, std::vector<std::string>> detector_to_hyperedges_map;

    for (size_t i = 0; i < hyperedges.size(); ++i) {
        const auto& edge = hyperedges[i];
        const auto& edge_str = str_hyperedges[i];

        for (const auto& node : edge) {
            if (!node.empty() && node[0] == 'D') {  // 只统计 D 开头的 detector
                detector_to_hyperedges_map[node].push_back(edge_str);
            }
        }
    }

    return detector_to_hyperedges_map;
}

std::vector<std::string> DetectorErrorModelHypergraph::get_nodes(bool include_logical) const {
    if (include_logical) return nodes;
    std::vector<std::string> filtered;
    for (auto &n : nodes) if (n[0] != 'L') filtered.push_back(n);
    return filtered;
}

std::vector<std::vector<std::string>> DetectorErrorModelHypergraph::get_hyperedges() const {
    return hyperedges;
}

std::vector<double> DetectorErrorModelHypergraph::get_weights() const {
    return weights;
}

const std::vector<std::string>& DetectorErrorModelHypergraph::get_str_hyperedges() const {
    return str_hyperedges;
}


std::unordered_map<std::string, double> DetectorErrorModelHypergraph::get_hyperedges_weights_dict(
    const std::vector<std::vector<std::string>>* hyperedges_input) const {

    std::unordered_map<std::string, double> result;

    if (!hyperedges_input) {
        return hyperedge_weight_map_;
    }

    for (const auto& edge : *hyperedges_input) {
        std::string key = canonicalize_edge(edge);
        auto it = hyperedge_weight_map_.find(key);
        if (it != hyperedge_weight_map_.end()) {
            result[key] = it->second;
        } else {
            throw std::runtime_error("get_hyperedges_weights_dict: edge not found: " + key);
        }
    }

    return result;
}

DetectorErrorModelHypergraph DetectorErrorModelHypergraph::get_new_priority_sub_hypergraph(
    std::vector<std::string>& selected_nodes, int priority, int topk, bool contract_logical_hyperedges = true) {

    // 预先用unordered_set快速判断
    std::unordered_set<std::string> d_nodes;
    for (const auto& n : selected_nodes) {
        if (n[0] == 'D') d_nodes.insert(n);
    }

    std::vector<std::tuple<size_t, double, int>> filtered_edge_indices; // 只存下标，减少复制

    // 可以使用OpenMP进行并行化处理。
    for (size_t i = 0; i < hyperedges.size(); ++i) {
        const auto& edge = hyperedges[i];

        int match = 0, mismatch = 0;
        for (const auto& n : edge) {
            // match and mismatch only conside about D, not about L.
            if (n[0] != 'D') continue;
            if (d_nodes.count(n)) {
                ++match;
            } else {
                ++mismatch;
                if (match == 0 && mismatch > 2) {
                    match = -1;
                    break;
                }
            }
        }
        if (match <= 0) continue;

        // Assign priority to hyperedges based on matching criteria
        // For critical hyperedges (perfect matches or full detector coverage), assign higher priority
        // General case: use difference between matches and mismatches
        if (!contract_logical_hyperedges && match == 0 && mismatch == 0) continue;

        // some very important hyperedges, we give more priority. general is use match - mismatch.
        int edge_priority = ((match != 0 && mismatch == 0) || match == static_cast<int>(d_nodes.size())) ?
                            static_cast<int>(d_nodes.size()) + match :
                            match - mismatch;

        if (edge_priority >= priority) {
            filtered_edge_indices.emplace_back(i, weights[i], edge_priority);
        }
    }

    auto comp = [](const auto& a, const auto& b) {
        if (std::get<2>(a) != std::get<2>(b))
            return std::get<2>(a) > std::get<2>(b);
        return std::get<1>(a) > std::get<1>(b);
    };

    if (topk != -1 && static_cast<int>(filtered_edge_indices.size()) > topk) {
        std::nth_element(filtered_edge_indices.begin(),
                        filtered_edge_indices.begin() + topk,
                        filtered_edge_indices.end(),
                        comp);
        filtered_edge_indices.resize(topk);  // 截断即可，不需要排序
    }

    // 第一步：遍历边，构建新边的同时建立 D 节点的编号映射
    std::unordered_map<std::string, std::string> renaming_map;
    int d_counter = 0;

    std::vector<std::vector<std::string>> new_edges;
    std::vector<std::string> new_str_hyperedges;
    std::vector<double> new_weights;
    std::unordered_set<std::string> final_node_set;

    for (const auto& [idx, w, p] : filtered_edge_indices) {
        const auto& edge = hyperedges[idx];
        std::vector<std::string> new_edge;
        new_edge.reserve(edge.size());

        for (const auto& node : edge) {
            auto it = renaming_map.find(node);
            if (it == renaming_map.end()) {
                std::string new_name;
                if (node[0] == 'D') {
                    new_name = "D" + std::to_string(d_counter++);
                } else {
                    new_name = node;  // L节点不变
                }
                renaming_map[node] = new_name;
                final_node_set.insert(new_name);
                new_edge.push_back(new_name);
            } else {
                new_edge.push_back(it->second);
                final_node_set.insert(it->second);
            }
        }

        new_edges.push_back(std::move(new_edge));
        new_weights.push_back(w);

        std::string edge_str = canonicalize_edge(new_edges.back());
        new_str_hyperedges.push_back(edge_str);
    }

    // 第二步：构造 remapped_selected_nodes
    for (auto& node : selected_nodes) {
        auto it = renaming_map.find(node);
        if (it != renaming_map.end()) {
            node = it->second;
        }
        // 否则保留原始名称（可能是未被选中的节点，也可能是 L 节点）
    }

    // 第三步：构造最终节点列表
    std::vector<std::string> new_nodes(final_node_set.begin(), final_node_set.end());
    std::sort(new_nodes.begin(), new_nodes.end());  // 可选排序

    std::unordered_map<std::string, double> new_weight_map;
    for (size_t i = 0; i < new_str_hyperedges.size(); ++i) {
        new_weight_map[new_str_hyperedges[i]] = new_weights[i];
    }

    // 构建子图对象
    DetectorErrorModelHypergraph sub_hypergraph;
    sub_hypergraph.have_logical_observable = have_logical_observable;
    sub_hypergraph.nodes = std::move(new_nodes);
    sub_hypergraph.hyperedges = std::move(new_edges);
    sub_hypergraph.str_hyperedges = std::move(new_str_hyperedges);
    sub_hypergraph.weights = std::move(new_weights);
    sub_hypergraph.detector_number = d_counter;
    sub_hypergraph.hyperedge_weight_map_ = std::move(new_weight_map);

    // std::cout<< "number of sub Hypergraph nodes:" << sub_hypergraph.nodes.size() << std::endl;
    // std::cout<< "number of sub Hypergraph hyperedges:" << sub_hypergraph.hyperedges.size() << std::endl;
    return sub_hypergraph;
}


DetectorErrorModelHypergraph DetectorErrorModelHypergraph::get_new_priority_sub_hypergraph_parallel(
    const std::vector<std::string>& selected_nodes, int priority, int topk, int openmp_num_threads) {

    std::unordered_set<std::string> d_nodes(selected_nodes.begin(), selected_nodes.end());

    std::vector<std::tuple<size_t, double, int>> filtered_edge_indices;

    #pragma omp parallel num_threads(openmp_num_threads)
    {
        std::vector<std::tuple<size_t, double, int>> local_filtered;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < hyperedges.size(); ++i) {
            const auto& edge = hyperedges[i];

            int match = 0, mismatch = 0;
            bool early_exit = false;

            for (const auto& n : edge) {
                if (n[0] != 'D') continue;
                if (d_nodes.count(n)) ++match;
                else if (++mismatch > 2 && match == 0) {
                    early_exit = true;
                    break;
                }
            }
            if (early_exit) continue;
            if (match == 0 && mismatch > 2) continue;

            int edge_priority = (mismatch == 0 || match == static_cast<int>(d_nodes.size())) ?
                                static_cast<int>(d_nodes.size()) + match :
                                match - mismatch;

            if (edge_priority >= priority) {
                local_filtered.emplace_back(i, weights[i], edge_priority);
            }
        }

        // 合并结果（线程安全）
        #pragma omp critical
        filtered_edge_indices.insert(filtered_edge_indices.end(),
                                    local_filtered.begin(), local_filtered.end());
    }


    // 排序和 topk 筛选
    auto comp = [](const auto& a, const auto& b) {
        if (std::get<2>(a) != std::get<2>(b))
            return std::get<2>(a) > std::get<2>(b);
        return std::get<1>(a) > std::get<1>(b);
    };

    if (topk != -1 && static_cast<int>(filtered_edge_indices.size()) > topk*10) {
        std::partial_sort(filtered_edge_indices.begin(),
                          filtered_edge_indices.begin() + topk,
                          filtered_edge_indices.end(),
                          comp);
        filtered_edge_indices.resize(topk);
    } else {
        std::sort(filtered_edge_indices.begin(), filtered_edge_indices.end(), comp);
    }

    // 构建新的子超图
    std::vector<std::vector<std::string>> new_edges;
    std::vector<double> new_weights;
    std::vector<std::string> new_edge_strs;
    std::unordered_set<std::string> node_set;

    for (const auto& [idx, w, p] : filtered_edge_indices) {
        new_edges.push_back(hyperedges[idx]);
        new_weights.push_back(w);
        new_edge_strs.push_back(str_hyperedges[idx]);
        node_set.insert(hyperedges[idx].begin(), hyperedges[idx].end());
    }

    std::vector<std::string> node_candidates(node_set.begin(), node_set.end());
    std::sort(node_candidates.begin(), node_candidates.end());

    DetectorErrorModelHypergraph sub_hypergraph;
    sub_hypergraph.have_logical_observable = have_logical_observable;
    sub_hypergraph.nodes = std::move(node_candidates);
    sub_hypergraph.hyperedges = std::move(new_edges);
    sub_hypergraph.weights = std::move(new_weights);
    sub_hypergraph.detector_number = static_cast<int>(sub_hypergraph.nodes.size());

    sub_hypergraph.hyperedge_weight_map_.reserve(new_edge_strs.size());
    for (size_t i = 0; i < new_edge_strs.size(); ++i) {
        sub_hypergraph.hyperedge_weight_map_[new_edge_strs[i]] = sub_hypergraph.weights[i];
    }

    sub_hypergraph.str_hyperedges = std::move(new_edge_strs);
    return sub_hypergraph;
}

DetectorErrorModelHypergraph DetectorErrorModelHypergraph::remove_nodes(const std::vector<std::string>& remove) {
    std::unordered_set<std::string> to_remove(remove.begin(), remove.end());
    std::vector<std::vector<std::string>> new_edges;
    std::vector<double> new_weights;
    std::set<std::string> new_nodes;

    for (size_t i = 0; i < hyperedges.size(); ++i) {
        bool has_removed = false;
        for (const auto& n : hyperedges[i]) {
            if (to_remove.count(n)) {
                has_removed = true;
                break;
            }
        }
        if (!has_removed) {
            new_edges.push_back(hyperedges[i]);
            new_weights.push_back(weights[i]);
            new_nodes.insert(hyperedges[i].begin(), hyperedges[i].end());
        }
    }

    DetectorErrorModelHypergraph result(detector_error_model, have_logical_observable);
    result.nodes = {new_nodes.begin(), new_nodes.end()};
    result.hyperedges = new_edges;
    result.weights = new_weights;
    return result;
}

std::tuple<std::vector<std::string>, std::vector<std::vector<std::string>>, std::vector<double>> DetectorErrorModelHypergraph::detector_error_model_to_hypergraph(const stim::DetectorErrorModel& dem) {
    detector_number = dem.count_detectors();
    logical_observable_number = dem.count_observables();
    auto nodes = detector_and_logical_observable_number_to_hypernodes(detector_number, logical_observable_number);
    auto [edges, w] = detector_error_model_to_hyperedge(dem);
    return {nodes, edges, w};
}

std::vector<std::string> DetectorErrorModelHypergraph::detector_and_logical_observable_number_to_hypernodes(int d, int l) {
    std::vector<std::string> n;
    for (int i = 0; i < d; ++i) n.push_back("D" + std::to_string(i));
    if (have_logical_observable) for (int i = 0; i < l; ++i) n.push_back("L" + std::to_string(i));
    return n;
}

std::pair<std::vector<std::vector<std::string>>, std::vector<double>> DetectorErrorModelHypergraph::detector_error_model_to_hyperedge(const stim::DetectorErrorModel& dem) {
    std::vector<std::vector<std::string>> edges;
    std::vector<double> w;

    auto instructions = dem.instructions;
    for (const auto& inst : instructions)  {
        if (inst.type != stim::DemInstructionType::DEM_ERROR) {
            continue;
        }
        auto dem_target = inst.target_data;
        std::vector<std::string> edge = error_even_to_hyperedge(dem_target);
        // size_t start = inst.find("(") + 1;
        // size_t end = inst.find(")");
        // double weight = std::stod(inst.substr(start, end - start));
        double weight = inst.arg_data[0]; 
        edges.push_back(edge);
        w.push_back(weight);
    }
    return {edges, w};
}

std::vector<std::string> DetectorErrorModelHypergraph::error_even_to_hyperedge(const stim::SpanRef<const stim::DemTarget>& targets) {
    std::vector<std::string> edge;
    for (const auto& t : targets) {
        if (t.is_relative_detector_id()) edge.push_back("D" + std::to_string(t.val()));
        else if (have_logical_observable && t.is_observable_id()) edge.push_back("L" + std::to_string(t.val()));
    }
    return edge;
}

std::string DetectorErrorModelHypergraph::to_string() const {
    std::ostringstream oss;
    
    // 打印节点信息
    oss << "Hypergraph Nodes (" << nodes.size() << "):\n";
    for (size_t i = 0; i < nodes.size(); ++i) {
        oss << "  " << i << ": " << nodes[i] << "\n";
    }
    
    // 打印超边信息
    oss << "\nHypergraph Hyperedges (" << hyperedges.size() << "):\n";
    for (size_t i = 0; i < hyperedges.size(); ++i) {
        oss << "  " << i << ": [";
        for (size_t j = 0; j < hyperedges[i].size(); ++j) {
            if (j > 0) oss << ", ";
            oss << hyperedges[i][j];
        }
        oss << "] (weight: " << std::setprecision(6) << weights[i] << ")\n";
    }
    
    // 打印统计信息
    oss << "\nSummary:\n";
    oss << "  Total nodes: " << nodes.size() << "\n";
    oss << "  Total hyperedges: " << hyperedges.size() << "\n";
    oss << "  Detectors: " << detector_number << "\n";
    oss << "  Logical observables: " << logical_observable_number << "\n";
    
    return oss.str();
}

void DetectorErrorModelHypergraph::print() const {
    std::cout << to_string();
}