#include "hamld_cpp.h"
#include "stim.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

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
    std::string approx_strategy = "node_topk";
    double approx_param = 2;
    int priority = 0;
    int priority_topk = 10;
    bool use_heuristic = true;
    double alpha = 0.05;

    HAMLDCpp hamld(dem, approx_strategy, approx_param, priority, priority_topk, use_heuristic, alpha);

    // 3. 构造一个测试syndrome（假设至少有3个detector）
    std::vector<bool> syndrome(dem.count_detectors(), false);
    if (syndrome.size() > 2) {
        syndrome[0] = true;
        syndrome[2] = true;
    }

    // 4. 单次解码
    std::cout << "Single decode ===================="  << std::endl;
    auto start = std::chrono::steady_clock::now();
    auto [prediction, prob_dist, prob_correct] = hamld.decode(syndrome);
    auto end = std::chrono::steady_clock::now();

    std::cout << "Single decode time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";

    std::cout << "Decoded logical error: ";
    for (bool b : prediction) std::cout << (b ? "1" : "0");
    std::cout << "\nProbability correct: " << prob_correct << "\n";

    // 5. 批量解码测试
    std::cout << "Batch decode ===================="  << std::endl;
    // std::vector<std::vector<bool>> batch_syndromes = {syndrome, syndrome, syndrome};
    std::vector<std::vector<bool>> batch_syndromes;
    // int batch_size = 1000;
    int num_detectors = dem.count_detectors();

    for (int i = 0; i < batch_size; ++i) {
        std::vector<bool> syn(num_detectors, false);
        if (num_detectors > 2) {
            syn[0] = (i % 2 == 0);     // 偶数 index 设置第0位为1
            syn[1] = (i % 3 == 0);     // 每3个设置第1位为1
            syn[2] = (i % 5 == 0);     // 每5个设置第2位为1
        }
        batch_syndromes.push_back(syn);
    }

    std::vector<std::unordered_map<std::string, double>> prob_dists;

    start = std::chrono::steady_clock::now();
    auto predictions = hamld.decode_batch(batch_syndromes, true, &prob_dists);
    end = std::chrono::steady_clock::now();

    std::cout << "Batch decode time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";

    // for (size_t i = 0; i < predictions.size(); ++i) {
    //     std::cout << "Batch " << i << " decoded logical error: ";
    //     for (bool b : predictions[i]) std::cout << (b ? "1" : "0");
    //     std::cout << "\n";
    // }

    // 6. 并行批量解码测试（4线程）
    std::cout << "Parallel Batch decode ===================="  << std::endl;
    int num_threads = 4;
    prob_dists.clear();

    start = std::chrono::steady_clock::now();
    auto par_predictions = hamld.parallel_decode_batch(batch_syndromes, true, &prob_dists, num_threads);
    end = std::chrono::steady_clock::now();

    std::cout << "Parallel batch decode time (" << num_threads << " threads): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << " ms\n";

    // for (size_t i = 0; i < par_predictions.size(); ++i) {
    //     std::cout << "Parallel batch " << i << " decoded logical error: ";
    //     for (bool b : par_predictions[i]) std::cout << (b ? "1" : "0");
    //     std::cout << "\n";
    // }

    std::cout << "Total contraction time (accumulated): " << hamld.total_contraction_time() << " ms\n";

    if (predictions == par_predictions) {
        std::cout << "✅ Sequential and parallel predictions match exactly.\n";
    } else {
        std::cout << "❌ Mismatch between sequential and parallel predictions!\n";
    }
    
    return 0;
}
