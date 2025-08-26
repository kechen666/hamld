#include "approximate_contraction_executor.h"
#include "detector_hypergraph.h"
#include "connectivity_graph.h"
#include "greedy_mld_order.h"
#include "stim.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>

#include <boost/dynamic_bitset.hpp>

std::string bitset_to_string(const boost::dynamic_bitset<> &bs)
{
    std::string s;
    boost::to_string(bs, s); // boost的free function
    return s;
}

void test_approximate_contraction_executor(const std::string &dem_file_path)
{
    try
    {
        // 1. Load the detector error model
        FILE *f = fopen(dem_file_path.c_str(), "rb");
        if (!f)
        {
            throw std::runtime_error("Failed to open file: " + dem_file_path);
        }
        stim::DetectorErrorModel dem = stim::DetectorErrorModel::from_file(f);
        fclose(f);

        std::cout << "\n=== Testing ApproximateContractionExecutor ===\n";
        std::cout << "Loaded DetectorErrorModel from: " << dem_file_path << std::endl;
        std::cout << "Number of detectors: " << dem.count_detectors() << std::endl;
        std::cout << "Number of logical observables: " << dem.count_observables() << std::endl;

        // 2. Create hypergraph
        auto hypergraph = std::make_shared<DetectorErrorModelHypergraph>(dem, true);
        std::cout << "Hypergraph nodes: " << hypergraph->get_nodes().size() << std::endl;
        std::cout << "Hypergraph hyperedges: " << hypergraph->get_hyperedges().size() << std::endl;

        // // 3. Prepare hyperedge weights dictionary
        // std::unordered_map<std::string, double> hyperedge_weights;
        // const auto& hyperedges = hypergraph->get_hyperedges();
        // const auto& weights = hypergraph->get_weights();

        // for (size_t i = 0; i < hyperedges.size(); ++i) {
        //     std::string key;
        //     for (const auto& node : hyperedges[i]) {
        //         if (!key.empty()) key += ",";
        //         key += node;
        //     }
        //     hyperedge_weights[key] = weights[i];
        // }

        // // 4. Create connectivity graph and find order
        // ConnectivityGraph connectivity;
        // connectivity.hypergraph_to_connectivity_graph(*hypergraph, false);
        // GreedyMLDOrderFinder order_finder(connectivity);
        // auto order = order_finder.find_order();

        // std::cout << "\nContraction order (" << order.size() << " nodes):\n";
        // for (size_t i = 0; i < order.size(); ++i) {
        //     std::cout << i + 1 << ": " << order[i] << std::endl;
        // }

        // 5. Create test syndrome (all zeros except first detector)
        std::vector<bool> syndrome(dem.count_detectors(), false);
        // if (!syndrome.empty()) {
        //     syndrome[0] = true; // Flip first detector
        //     syndrome[2] = true;
        // }
        if (syndrome.empty())
        {
            return; // 如果 syndrome 为空，直接返回
        }

        // 需要设置为 true 的索引列表
        // const std::vector<size_t> flip_indices = {
        //     25,  37,  43,  47,  59,  76, 114, 115, 121, 127,
        //     137, 180, 186, 192, 195, 196, 201, 203, 232, 266,
        //     282, 291, 349, 377, 382, 392, 393, 399, 401, 407,
        //     444, 447, 450, 460, 466, 469, 493, 530, 536, 562,
        //     565, 571, 591, 647, 666, 678, 679, 682, 685, 701,
        //     742, 743, 745, 749, 752, 761, 810, 826, 829, 857,
        //     881, 894, 905, 923, 958, 959, 975,1001,1004,1005,
        // 1020,1038,1067,1075,1077,1092,1093,1098,1101,1107,
        // 1113,1156,1162,1206,1212,1221,1233,1236,1257,1280,
        // 1286,1310,1319,1325,1328,1335,1341,1344,1421,1551
        // };

        // 大部分能够解码出来。
        // const std::vector<size_t> flip_indices = {
        //     368, 369, 377, 383, 435, 438, 440, 442, 445, 451,
        //     462, 476, 484, 496, 497, 501, 502, 513, 529, 537,
        //     539, 541, 542, 547, 548, 555, 558, 562, 564, 588,
        //     604, 610, 613, 615, 621, 629, 646, 649, 655, 674,
        //     681, 690, 694, 718, 738, 747, 759, 769, 786, 791,
        //     839, 937, 983, 989, 991, 992, 993, 1076, 1106, 1133,
        //     1135, 1161, 1177, 1220, 1226, 1227, 1243, 1250, 1264, 1269,
        //     1270, 1272, 1278, 1299, 1308, 1312, 1322, 1328, 1343, 1349,
        //     1352, 1399, 1449, 1471, 1473, 1477, 1486, 1536, 1545, 1551,
        //     1568, 1593, 1603, 1609, 1610, 1647
        // };
        
        // 1000 0 1000 能够求解。 -1 -1 的时候似乎又无解了。
        const std::vector<size_t> flip_indices = {
            46, 52, 55, 73, 82, 88, 127, 163, 185, 230,
            239, 245, 262, 265, 303, 330, 339, 359, 432, 448,
            488, 509, 512, 518, 562, 575, 583, 592, 632, 651,
            683, 686, 692, 707, 719, 734, 750, 751, 766, 776,
            796, 804, 809, 814, 837, 840, 846, 853, 861, 862,
            873, 910, 918, 924, 933, 959, 970, 975, 982, 1005,
            1017, 1030, 1046, 1055, 1086, 1087, 1099, 1120, 1123, 1126,
            1129, 1134, 1142, 1149, 1188, 1199, 1201, 1208, 1216, 1231,
            1243, 1253, 1312, 1332, 1349, 1410, 1420, 1423, 1429, 1471,
            1473, 1474, 1488, 1506, 1539, 1554
        };
        
        // HAMLD 1000 0 1000 无解，存在early filp，设置为-1，-1，即获取全部-1的情况，有解，说明该情况是超边考虑太少。
        // const std::vector<size_t> flip_indices ={
        //     109, 110, 126, 168, 174, 182, 183, 233, 291, 297,
        //     300, 322, 323, 326, 343, 347, 349, 353, 358, 372,
        //     373, 377, 395, 403, 432, 434, 441, 445, 450, 451,
        //     498, 533, 536, 542, 547, 595, 610, 650, 651, 659,
        //     667, 696, 724, 745, 781, 787, 798, 799, 803, 808,
        //     814, 853, 854, 872, 873, 912, 918, 927, 928, 985,
        //     1007, 1081, 1122, 1131, 1134, 1137, 1144, 1147, 1195, 1212,
        //     1217, 1229, 1234, 1237, 1243, 1270, 1273, 1310, 1343, 1349,
        //     1352, 1381, 1390, 1396, 1403, 1406, 1412, 1445, 1447, 1454,
        //     1495, 1552, 1554, 1555, 1561, 1591
        // };
        // const std::vector<size_t> flip_indices = {0,2};


        // 遍历所有需要翻转的索引
        for (size_t index : flip_indices)
        {
            if (index < syndrome.size())
            {
                syndrome[index] = true; // 设置为 true
            }
            // 如果索引超出范围，可以选择忽略或抛出异常
            // else {
            //     throw std::out_of_range("Index out of bounds");
            // }
        }

        // 6. Test different approximation strategies
        std::vector<std::pair<std::string, double>> strategies = {
            // {"node_topk", 2},
            {"hyperedge_topk", 1000},
            // {"node_threshold", 1e-6},
            // {"hyperedge_threshold", 1e-6},
            // {"no_no", -1}
        };

        int priority = 0;
        int priority_topk = 1000;
        // int priority = 0;
        // int priority_topk = 1000;
        bool use_heuristric = false;
        // bool use_heuristric = false;
        double alpha = 0.05;

        for (const auto &[strategy, param] : strategies)
        {
            std::cout << "\n=== Testing strategy: " << strategy << " ===" << std::endl;

            // Create executor
            ApproximateContractionExecutor executor(
                dem.count_detectors(),
                dem.count_observables(),
                strategy,
                param,
                hypergraph,
                priority,
                priority_topk,
                use_heuristric = use_heuristric,
                alpha = alpha);

            std::cout << "\n=== Finish init: =======" << std::endl;

            // Perform contraction
            auto start = std::chrono::high_resolution_clock::now();
            auto prob_dist = executor.mld_contraction_no_slicing(syndrome);
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - start);

            std::cout << "\n=== Finish contraction: =======" << std::endl;

            // Validate results
            auto [logical_error, prob_correct] = executor.validate_logical_operator(prob_dist);

            std::cout << "\n=== Finish validate logical operator: =======" << std::endl;

            // Print results
            std::cout << "Contraction time: " << duration.count() << " ms\n";
            std::cout << "Max distribution size: " << executor.get_execution_max_distribution_size() << "\n";
            std::cout << "Probability distribution size: " << prob_dist.size() << "\n";
            std::cout << "Logical error: [";
            for (bool b : logical_error)
                std::cout << (b ? "1" : "0");
            std::cout << "]\n";
            std::cout << "Probability correct: " << prob_correct << "\n";

            // Print top 5 probabilities if distribution is small
            // Print top 5 entries
            std::vector<std::pair<std::string, double>> sorted_probs;
            sorted_probs.reserve(prob_dist.size());
            for (const auto &[key, prob] : prob_dist)
            {
                sorted_probs.emplace_back(bitset_to_string(key), prob); // 👈 显式转换
            }

            // std::sort(sorted_probs.begin(), sorted_probs.end(),
            //           [](const auto &a, const auto &b)
            //           { return a.second > b.second; });
            std::sort(sorted_probs.begin(), sorted_probs.end(),
                      [](const auto &a, const auto &b)
                      { return a.second > b.second; });

            size_t top_k = std::min<size_t>(5, sorted_probs.size());
            std::cout << (prob_dist.size() <= 5 ? "Probability distribution:\n" : "Top 5 probabilities:\n");
            for (size_t i = 0; i < top_k; ++i)
            {
                std::cout << "  " << sorted_probs[i].first << ": " << sorted_probs[i].second << "\n";
            }
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error in test_approximate_contraction_executor: " << ex.what() << std::endl;
        throw;
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <path_to_dem_file>" << std::endl;
        return 1;
    }

    test_approximate_contraction_executor(argv[1]);
    return 0;
}