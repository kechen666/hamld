#include "hamld_cpp.h"
#include "stim.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
// #include "sparse_state.h"  // 需要确保这个头文件被包含


int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_dem_file> [batch_size]" << std::endl;
        return 1;
    }

    const char* file_path = argv[1];

    int batch_size = 10;  // 默认值
    if (argc >= 3) {
        try {
            batch_size = std::stoi(argv[2]);
            if (batch_size <= 0) {
                std::cerr << "Invalid batch_size, must be positive integer." << std::endl;
                return 1;
            }
        } catch (...) {
            std::cerr << "Failed to parse batch_size argument." << std::endl;
            return 1;
        }
    }

    // 以下代码保持不变
    FILE* f = fopen(file_path, "rb");
    if (!f) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return 1;
    }
    stim::DetectorErrorModel dem;

    dem = stim::DetectorErrorModel::from_file(f);
    fclose(f);
    std::cout << "Loaded DetectorErrorModel from: " << file_path << std::endl;
    std::cout << "Number of detectors: " << dem.count_detectors() << std::endl;
    std::cout << "Number of logical observables: " << dem.count_observables() << std::endl;
    auto hypergraph = std::make_shared<DetectorErrorModelHypergraph>(dem, false);


    std::cout << "Loaded DEM with " << dem.count_detectors() << " detectors and "
              << dem.count_observables() << " observables.\n";

    // 2. 初始化HAMLD
    std::string approx_strategy = "hyperedge_topk";
    // double approx_param = 1000;
    double approx_param = 3;
    int priority = 0;
    // int priority_topk = 1000;
    int priority_topk = 3;
    bool use_heuristic = false;
    double alpha = 0.05;

    HAMLDCpp hamld(dem, approx_strategy, approx_param, priority, priority_topk, use_heuristic, alpha);

    // 3. 构造一个测试syndrome（假设至少有3个detector）
    // std::vector<bool> syndrome(dem.count_detectors(), false);
    // if (syndrome.size() > 2) {
    //     syndrome[0] = true;
    //     syndrome[2] = true;
    // }
    std::vector<bool> syndrome(dem.count_detectors(), false);
    for (int i = 0; i < int(dem.count_detectors() / 2); ++i) {
        syndrome[i * 2 - 1] = true;
    }
    // std::vector<bool> syndrome(dem.count_detectors(), false);
    // const std::vector<size_t> flip_indices = {
    //     46, 52, 55, 73, 82, 88, 127, 163, 185, 230,
    //     239, 245, 262, 265, 303, 330, 339, 359, 432, 448,
    //     488, 509, 512, 518, 562, 575, 583, 592, 632, 651,
    //     683, 686, 692, 707, 719, 734, 750, 751, 766, 776,
    //     796, 804, 809, 814, 837, 840, 846, 853, 861, 862,
    //     873, 910, 918, 924, 933, 959, 970, 975, 982, 1005,
    //     1017, 1030, 1046, 1055, 1086, 1087, 1099, 1120, 1123, 1126,
    //     1129, 1134, 1142, 1149, 1188, 1199, 1201, 1208, 1216, 1231,
    //     1243, 1253, 1312, 1332, 1349, 1410, 1420, 1423, 1429, 1471,
    //     1473, 1474, 1488, 1506, 1539, 1554
    // };
    // for (size_t index : flip_indices)
    // {
    //     if (index < syndrome.size())
    //     {
    //         syndrome[index] = true; // 设置为 true
    //     }
    // }

    // 4. 单次解码
    // std::cout << "Single decode ===================="  << std::endl;
    // auto start = std::chrono::steady_clock::now();
    // auto [prediction, prob_dist, prob_correct] = hamld.decode(syndrome);
    // auto end = std::chrono::steady_clock::now();

    // std::cout << "Single decode time: "
    //           << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
    //           << " ms\n";

    // std::cout << "Decoded logical error: ";
    // for (bool b : prediction) std::cout << (b ? "1" : "0");
    // std::cout << "\nProbability correct: " << prob_correct << "\n";

    // // 5. 批量解码测试
    std::cout << "Batch decode ===================="  << std::endl;
    // std::vector<std::vector<bool>> batch_syndromes = {syndrome, syndrome, syndrome};
    std::vector<std::vector<bool>> batch_syndromes;
    // int batch_size = 10;
    // int num_detectors = dem.count_detectors();

    for (int i = 0; i < batch_size; ++i) {
        // std::vector<bool> syn(num_detectors, false);
        // if (num_detectors > 2) {
        //     syn[0] = (i % 2 == 0);     // 偶数 index 设置第0位为1
        //     syn[1] = (i % 3 == 0);     // 每3个设置第1位为1
        //     syn[2] = (i % 5 == 0);     // 每5个设置第2位为1
        // }
        batch_syndromes.push_back(syndrome);
    }

    std::vector<std::unordered_map<std::string, double>> prob_dists;

    auto start = std::chrono::steady_clock::now();
    auto predictions = hamld.decode_batch(batch_syndromes, false, &prob_dists);
    auto end = std::chrono::steady_clock::now();

    std::cout << "Batch decode time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";

    // for (size_t i = 0; i < predictions.size(); ++i) {
    //     std::cout << "Batch " << i << " decoded logical error: ";
    //     for (bool b : predictions[i]) std::cout << (b ? "1" : "0");
    //     std::cout << "\n";
    // }

    // // 6. 并行批量解码测试（4线程）
    // std::cout << "Parallel Batch decode ===================="  << std::endl;
    // int num_threads = 4;
    // prob_dists.clear();

    // start = std::chrono::steady_clock::now();
    // auto par_predictions = hamld.parallel_decode_batch(batch_syndromes, true, &prob_dists, num_threads);
    // end = std::chrono::steady_clock::now();

    // std::cout << "Parallel batch decode time (" << num_threads << " threads): "
    //           << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
    //           << " ms\n";

    // // for (size_t i = 0; i < par_predictions.size(); ++i) {
    // //     std::cout << "Parallel batch " << i << " decoded logical error: ";
    // //     for (bool b : par_predictions[i]) std::cout << (b ? "1" : "0");
    // //     std::cout << "\n";
    // // }

    // std::cout << "Total contraction time (accumulated): " << hamld.total_contraction_time() << " ms\n";

    // if (predictions == par_predictions) {
    //     std::cout << "✅ Sequential and parallel predictions match exactly.\n";
    // } else {
    //     std::cout << "❌ Mismatch between sequential and parallel predictions!\n";
    // }
    
    return 0;
}
