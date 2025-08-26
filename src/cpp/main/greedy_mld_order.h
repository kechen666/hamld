#pragma once

#include "connectivity_graph.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

class GreedyMLDOrderFinder {
public:
    ConnectivityGraph& connectivity_graph;

    int max_prob_dist_dimension = 0;
    int max_prob_dist_dimension_count = 0;
    std::vector<std::string> max_contracted_nodes;
    std::vector<std::string> order;

    std::unordered_set<std::string> all_nodes;
    int current_prob_dist_dimension = 0;
    std::unordered_set<std::string> current_contracted_nodes;
    std::unordered_set<std::string> current_prob_dist_related_nodes;
    std::unordered_set<std::string> current_candidate_nodes_for_contraction;

    GreedyMLDOrderFinder(ConnectivityGraph& graph);

    std::vector<std::string> find_order();
    void contract_node_and_update_candidates(const std::string& detector);
    std::unordered_map<std::string, int> compute_contraction_node_prob_dist_dim_change();
};
