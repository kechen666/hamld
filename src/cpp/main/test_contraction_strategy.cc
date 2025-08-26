#include "detector_hypergraph.h"
#include "connectivity_graph.h"
#include "greedy_mld_order.h"
#include "stim.h"
#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_dem_file>" << std::endl;
        return 1;
    }

    const char* file_path = argv[1];
    FILE* f = fopen(file_path, "rb");
    if (!f) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return 1;
    }

    try {
        stim::DetectorErrorModel dem = stim::DetectorErrorModel::from_file(f);
        fclose(f);

        std::cout << "Loaded DetectorErrorModel from: " << file_path << std::endl;
        std::cout << "Number of detectors: " << dem.count_detectors() << std::endl;
        std::cout << "Number of logical observables: " << dem.count_observables() << std::endl;

        DetectorErrorModelHypergraph hypergraph(dem, /*include_logical=*/true);
        std::cout << "Hypergraph: " << hypergraph.get_nodes().size()
                  << " nodes, " << hypergraph.get_hyperedges().size() << " hyperedges\n";
        
        // 打印超边及其权重
        auto all_nodes = hypergraph.get_nodes();
        const auto& hyperedges = hypergraph.get_hyperedges();
        const auto& weights = hypergraph.get_weights();

        std::cout << "Hyperedges and weights:\n";
        for (size_t i = 0; i < hyperedges.size(); ++i) {
            std::cout << "Hyperedge " << i << " (weight " << weights[i] << "): ";
            for (const auto& node : hyperedges[i]) {
                std::cout << node << " ";
            }
            std::cout << "\n";
        }

        // 测试 get_hyperedges_weights_dict（全部超边）
        std::cout << "\nTesting get_hyperedges_weights_dict with full hyperedges:\n";
        auto hyperedge_weight_map = hypergraph.get_hyperedges_weights_dict();
        for (const auto& [edge, weight] : hyperedge_weight_map) {
            std::cout << "Weight " << weight << " for edge: ";
            // for (const auto& node : edge) {
            //     std::cout << node << " ";
            // }
            std::cout << edge << " ";
            std::cout << "\n";
        }

        // 测试 get_hyperedges_weights_dict（部分超边）
        std::cout << "\nTesting get_hyperedges_weights_dict with partial hyperedges:\n";
        std::vector<std::vector<std::string>> partial_edges;
        if (!hyperedges.empty()) {
            partial_edges.push_back(hyperedges[0]);
            if (hyperedges.size() > 1) partial_edges.push_back(hyperedges[1]);
        }

        try {
            auto partial_map = hypergraph.get_hyperedges_weights_dict(&partial_edges);
            for (const auto& [edge, weight] : partial_map) {
                std::cout << "Weight " << weight << " for partial edge: ";
                // for (const auto& node : edge) {
                //     std::cout << node << " ";
                // }
                std::cout << edge << " ";
                std::cout << "\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in partial get_hyperedges_weights_dict: " << e.what() << "\n";
        }

        // 选择部分节点作为selected_nodes，测试get_new_priority_sub_hypergraph
        // 这里简单选前10个D节点，或者全部节点
        std::vector<std::string> selected_nodes;
        int count = 0;
        for (const auto& n : all_nodes) {
            if (n[0] == 'D') {
                if (++count == 5){
                    selected_nodes.push_back(n);
                    std::cout << "selected node: " << n << "\n";
                    break;
                }
            }
        }
        
        int priority = 0; // 你可以尝试不同的优先级
        int topk = 10;     // 只取前5个优先超边

        DetectorErrorModelHypergraph subgraph = hypergraph.get_new_priority_sub_hypergraph(selected_nodes, priority, topk);

        // 打印子超图的节点、超边和权重
        std::cout << "Subgraph nodes (" << subgraph.get_nodes().size() << "): ";
        for (const auto& n : subgraph.get_nodes()) std::cout << n << " ";
        std::cout << "\n";

        auto sub_edges = subgraph.get_hyperedges();
        auto subweights = subgraph.get_weights();

        std::cout << "Subgraph edges (" << sub_edges.size() << "):\n";
        for (size_t i = 0; i < sub_edges.size(); ++i) {
            std::cout << "  Edge " << i << " (weight=" << subweights[i] << "): ";
            for (const auto& node : sub_edges[i]) std::cout << node << " ";
            std::cout << "\n";
        }

        ConnectivityGraph connectivity;
        connectivity.hypergraph_to_connectivity_graph(hypergraph);
        std::cout << "ConnectivityGraph: " << connectivity.nodes.size()
                  << " nodes, adjacency entries: " << connectivity.adjacency.size() << "\n";

        // 打印邻接表 adjacency
        std::cout << "\nConnectivityGraph adjacency list:\n";
        for (const auto& [node, neighbors] : connectivity.adjacency) {
            std::cout << node << ": ";
            for (const auto& n : neighbors) {
                std::cout << n << " ";
            }
            std::cout << "\n";
        }
        // random order
        auto random_order = connectivity.bfs_random_start();
        std::cout << "\nRandom BFS Order (" << random_order.size() << " nodes):" << std::endl;
        for (size_t i = 0; i < random_order.size(); ++i) {
            std::cout << i + 1 << ": " << random_order[i] << std::endl;
        }

        GreedyMLDOrderFinder order_finder(connectivity);
        auto order = order_finder.find_order();

        std::cout << "\nContraction Order (" << order.size() << " nodes):" << std::endl;
        for (size_t i = 0; i < order.size(); ++i) {
            std::cout << i + 1 << ": " << order[i] << std::endl;
        }

        std::cout << "\nMax Probability Distribution Dimension: " << order_finder.max_prob_dist_dimension << std::endl;
        std::cout << "Count: " << order_finder.max_prob_dist_dimension_count << std::endl;
        std::cout << "Nodes: ";
        for (const auto& node : order_finder.max_contracted_nodes) {
            std::cout << node << " ";
        }
        std::cout << std::endl;

        // ====== 从子图构建新的 DetectorErrorModel 测试 ======
        std::cout << "\n=== Reconstructing stim::DetectorErrorModel from subgraph ===\n";

        // 使用你前面定义的函数导出 stim::DetectorErrorModel
        // stim::DetectorErrorModel new_dem = subgraph.to_detector_error_model();
        stim::DetectorErrorModel new_dem = subgraph.detector_error_model;

        // 将重构出的 DEM 写入文件进行对比检查（可选）
        // std::ofstream out_dem_file("subgraph_reconstructed.dem");
        // if (out_dem_file.is_open()) {
        //     out_dem_file << new_dem;
        //     out_dem_file.close();
        //     std::cout << "Reconstructed subgraph DEM written to subgraph_reconstructed.dem\n";
        // }

        // 打印部分新 DEM 的信息
        std::cout << "New DEM: detectors = " << new_dem.count_detectors()
                << ", observables = " << new_dem.count_observables()
                << ", instructions = " << new_dem.instructions.size() << "\n";

        // 可选：打印 new_dem 的指令（仅前几条）
        size_t max_to_show = 10;
        std::cout << "\nInstructions in new_dem (first " << max_to_show << "):\n";
        for (size_t i = 0; i < std::min(max_to_show, new_dem.instructions.size()); ++i) {
            std::cout << new_dem.instructions[i] << "\n";
        }
    } catch (const std::exception& ex) {
        fclose(f);
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
