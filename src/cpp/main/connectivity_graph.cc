#include "connectivity_graph.h"
#include <algorithm>
#include <limits>

void ConnectivityGraph::hypergraph_to_connectivity_graph(DetectorErrorModelHypergraph& hg, bool include_logical) {
    hypergraph = &hg;
    nodes.clear();
    adjacency.clear();

    auto node_list = hg.get_nodes(include_logical);
    auto hyperedges = hg.get_hyperedges();

    if (!include_logical && hg.have_logical_observable) {
        for (int i = 0; i < hg.logical_observable_number; ++i) {
            std::string logical_node = "L" + std::to_string(i);
            node_list.erase(std::remove(node_list.begin(), node_list.end(), logical_node), node_list.end());
        }
        for (auto& edge : hyperedges) {
            edge.erase(std::remove_if(edge.begin(), edge.end(), [](const std::string& n) { return n[0] == 'L'; }), edge.end());
        }
    }

    nodes.insert(node_list.begin(), node_list.end());

    for (const auto& edge : hyperedges) {
        add_edges_from_hyperedge(edge);
    }
}

void ConnectivityGraph::dem_to_connectivity_graph(const stim::DetectorErrorModel& dem, bool include_logical) {
    detector_error_model = dem;
    nodes.clear();
    adjacency.clear();

    // 添加检测器节点
    for (size_t i = 0; i < dem.count_detectors(); ++i) {
        nodes.insert("D" + std::to_string(i));
    }

    // 假设DemInstruction列表可以这样访问：
    auto instructions = dem.instructions;  // 这里根据 stim 实际接口替换

    for (const auto& inst : instructions) {
        if (inst.type != stim::DemInstructionType::DEM_ERROR) continue;
        std::vector<std::string> edge;
        for (const auto& t : inst.target_data) {
            if (t.is_relative_detector_id()) {
                edge.push_back("D" + std::to_string(t.val()));
            } else if (include_logical && t.is_observable_id()) {
                edge.push_back("L" + std::to_string(t.val()));
            }
        }
        add_edges_from_hyperedge(edge);
    }
}


void ConnectivityGraph::add_edges_from_hyperedge(const std::vector<std::string>& hyperedge) {
    if (hyperedge.size() <= 1) return;
    for (size_t i = 0; i < hyperedge.size(); ++i) {
        for (size_t j = i + 1; j < hyperedge.size(); ++j) {
            const std::string& a = hyperedge[i];
            const std::string& b = hyperedge[j];
            adjacency[a].insert(b);
            adjacency[b].insert(a);
            nodes.insert(a);
            nodes.insert(b);
        }
    }
}

std::string ConnectivityGraph::find_min_degree_node() const {
    int min_degree = std::numeric_limits<int>::max();
    std::string min_node;
    for (const auto& node : nodes) {
        int degree = adjacency.count(node) ? (int)adjacency.at(node).size() : 0;
        if (degree < min_degree) {
            min_degree = degree;
            min_node = node;
        }
    }
    return min_node;
}

std::unordered_set<std::string> ConnectivityGraph::neighbors(const std::string& node) const {
    if (adjacency.count(node)) return adjacency.at(node);
    return {};
}

int ConnectivityGraph::degree(const std::string& node) const {
    return adjacency.count(node) ? (int)adjacency.at(node).size() : 0;
}

std::vector<std::string> ConnectivityGraph::sorted_nodes_by_degree(bool ascending) const {
    std::vector<std::string> sorted_nodes(nodes.begin(), nodes.end());

    std::sort(sorted_nodes.begin(), sorted_nodes.end(),
        [this, ascending](const std::string& a, const std::string& b) {
            int deg_a = degree(a);
            int deg_b = degree(b);
            return ascending ? (deg_a < deg_b) : (deg_a > deg_b);
        });

    return sorted_nodes;
}

std::vector<std::string> ConnectivityGraph::bfs_random_start() const {
    if (nodes.empty()) return {};

    // 从 nodes 中随机选一个起点
    std::vector<std::string> node_list(nodes.begin(), nodes.end());
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, node_list.size() - 1);
    std::string start_node = node_list[dist(gen)];

    // BFS 遍历
    std::vector<std::string> visit_order;
    std::unordered_set<std::string> visited;
    std::queue<std::string> q;

    visited.insert(start_node);
    q.push(start_node);

    while (!q.empty()) {
        std::string current = q.front();
        q.pop();
        visit_order.push_back(current);

        for (const auto& neighbor : neighbors(current)) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                q.push(neighbor);
            }
        }
    }

    return visit_order;
}