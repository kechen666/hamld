#pragma once

#include "stim.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include "detector_hypergraph.h"
#include <random>
#include <queue>

class ConnectivityGraph {
public:
    stim::DetectorErrorModel detector_error_model;
    DetectorErrorModelHypergraph* hypergraph;

    std::unordered_set<std::string> nodes;
    std::unordered_map<std::string, std::unordered_set<std::string>> adjacency;

    void hypergraph_to_connectivity_graph(DetectorErrorModelHypergraph& hg, bool include_logical = false);
    void dem_to_connectivity_graph(const stim::DetectorErrorModel& dem, bool include_logical = false);
    void add_edges_from_hyperedge(const std::vector<std::string>& hyperedge);

    std::string find_min_degree_node() const;
    std::unordered_set<std::string> neighbors(const std::string& node) const;
    int degree(const std::string& node) const;

    std::vector<std::string> sorted_nodes_by_degree(bool ascending) const;
    
    std::vector<std::string> bfs_random_start() const;
};
