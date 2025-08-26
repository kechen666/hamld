#include "greedy_mld_order.h"
#include <algorithm>
#include <limits>
#include <cmath>

GreedyMLDOrderFinder::GreedyMLDOrderFinder(ConnectivityGraph& graph)
    : connectivity_graph(graph) {
    all_nodes = graph.nodes;
}

std::vector<std::string> GreedyMLDOrderFinder::find_order() {
    int nodes_number = (int)all_nodes.size();

    for (int step = 0; step < nodes_number; ++step) {
        std::string detector;

        if (step == 0) {
            detector = *std::min_element(
                all_nodes.begin(), all_nodes.end(),
                [&](const std::string& a, const std::string& b) {
                    return connectivity_graph.degree(a) < connectivity_graph.degree(b);
                });
        } else {
            auto prob_dist_changes = compute_contraction_node_prob_dist_dim_change();
            if (prob_dist_changes.empty()) {
                std::unordered_set<std::string> remaining_nodes;
                for (const auto& n : all_nodes) {
                    if (!current_contracted_nodes.count(n)) remaining_nodes.insert(n);
                }
                if (remaining_nodes.empty()) break;
                detector = *std::min_element(
                    remaining_nodes.begin(), remaining_nodes.end(),
                    [&](const std::string& a, const std::string& b) {
                        return connectivity_graph.degree(a) < connectivity_graph.degree(b);
                    });
            } else {
                detector = std::min_element(
                    prob_dist_changes.begin(), prob_dist_changes.end(),
                    [](const auto& a, const auto& b) {
                        return std::tie(a.second, a.first) < std::tie(b.second, b.first);
                    })->first;
            }
        }

        order.push_back(detector);
        contract_node_and_update_candidates(detector);

        current_prob_dist_dimension = (int)(current_prob_dist_related_nodes.size() - current_contracted_nodes.size());

        if (current_prob_dist_dimension > max_prob_dist_dimension) {
            max_prob_dist_dimension = current_prob_dist_dimension;
            max_prob_dist_dimension_count = 1;
            max_contracted_nodes = {detector};
        } else if (current_prob_dist_dimension == max_prob_dist_dimension) {
            ++max_prob_dist_dimension_count;
            max_contracted_nodes.push_back(detector);
        }
    }
    return order;
}

void GreedyMLDOrderFinder::contract_node_and_update_candidates(const std::string& detector) {
    current_contracted_nodes.insert(detector);
    current_prob_dist_related_nodes.insert(detector);

    auto neighboring_nodes = connectivity_graph.neighbors(detector);
    current_prob_dist_related_nodes.insert(neighboring_nodes.begin(), neighboring_nodes.end());

    current_candidate_nodes_for_contraction.clear();
    for (const auto& node : current_prob_dist_related_nodes) {
        if (!current_contracted_nodes.count(node)) {
            current_candidate_nodes_for_contraction.insert(node);
        }
    }
}

std::unordered_map<std::string, int> GreedyMLDOrderFinder::compute_contraction_node_prob_dist_dim_change() {
    std::unordered_map<std::string, int> changes;

    for (const auto& candidate_node : current_candidate_nodes_for_contraction) {
        auto neighboring_nodes = connectivity_graph.neighbors(candidate_node);

        int count = 0;
        for (const auto& node : neighboring_nodes) {
            if (!current_contracted_nodes.count(node) &&
                !current_candidate_nodes_for_contraction.count(node)) {
                ++count;
            }
        }
        changes[candidate_node] = count;
    }

    return changes;
}
