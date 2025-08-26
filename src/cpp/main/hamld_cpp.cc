#include "hamld_cpp.h"
// #include "sparse_state.h"  // 移除此行
#include <thread>
#include <mutex>
#include <algorithm> // std::min
#include <chrono>
#include <boost/dynamic_bitset.hpp>  // 可选，显示引用
#include <string>
// #include <boost/dynamic_bitset/io.hpp> // 包含to_string支持

std::string bitset_to_string(const boost::dynamic_bitset<>& bs) {
    std::string s;
    boost::to_string(bs, s);  // boost的free function
    return s;
}

HAMLDCpp::HAMLDCpp(
    const stim::DetectorErrorModel& detector_error_model,
    const std::string& approximatestrategy,
    double approximate_param,
    int priority,
    int priority_topk,
    bool use_heuristic,
    double alpha,
    bool openmp,
    int openmp_num_threads
) : detector_error_model_(detector_error_model),
    approximatestrategy_(approximatestrategy),
    approximate_param_(approximate_param),
    priority_(priority),
    priority_topk_(priority_topk),
    use_heuristic_(use_heuristic),
    alpha_(alpha),
    openmp_(openmp),
    openmp_num_threads_(openmp_num_threads),
    total_contraction_time_(0.0),
    total_hypergraph_approximate_time_(0.0),
    total_init_time_(0.0),
    total_running_time_(0.0),
    total_order_finder_time_(0.0)
{
    num_detectors_ = detector_error_model_.count_detectors();
    num_observables_ = detector_error_model_.count_observables();
    total_length_ = num_detectors_ + num_observables_;

    hypergraph_ = std::make_shared<DetectorErrorModelHypergraph>(detector_error_model_, true);
    contractor_ = std::make_unique<ApproximateContractionExecutor>(
        num_detectors_,
        num_observables_,
        approximatestrategy_,
        approximate_param_,
        hypergraph_,
        priority_,
        priority_topk_,
        use_heuristic_,
        alpha_,
        openmp_,
        openmp_num_threads_
    );
}

std::tuple<std::vector<bool>, std::unordered_map<std::string, double>, double> HAMLDCpp::decode(const std::vector<bool>& syndrome) {
    auto prob_dist = contractor_->mld_contraction_no_slicing(syndrome); // tsl::robin_map<boost::dynamic_bitset<>, double>
    auto [prediction, prob_correct] = contractor_->validate_logical_operator(prob_dist);

    std::unordered_map<std::string, double> converted_map;
    for (const auto& [state, prob] : prob_dist) {
        converted_map[bitset_to_string(state)] = prob;  // boost::dynamic_bitset has to_string()
    }

    return {prediction, converted_map, prob_correct};
}

std::vector<std::vector<bool>> HAMLDCpp::decode_batch(
    const std::vector<std::vector<bool>>& syndromes,
    bool output_prob,
    std::vector<std::unordered_map<std::string, double>>* prob_dists)
{
    total_running_time_ = 0.0;
    auto thread_start = std::chrono::steady_clock::now();

    size_t n = syndromes.size();
    std::vector<std::vector<bool>> predictions(n);

    if (output_prob && prob_dists) {
        prob_dists->resize(n);
    }

    for (size_t i = 0; i < n; ++i) {
        auto prob_dist = contractor_->mld_contraction_no_slicing(syndromes[i]);
        auto [pred, prob_correct] = contractor_->validate_logical_operator(prob_dist);

        predictions[i] = std::move(pred);

        if (output_prob && prob_dists) {
            std::unordered_map<std::string, double> converted_map;
            for (const auto& [state, prob] : prob_dist) {
                converted_map[bitset_to_string(state)] = prob;
            }
            (*prob_dists)[i] = std::move(converted_map);
        }
    }

    auto thread_end = std::chrono::steady_clock::now();
    total_running_time_ += std::chrono::duration<double>(thread_end - thread_start).count();

    return predictions;
}

std::vector<std::vector<bool>> HAMLDCpp::parallel_decode_batch(
    const std::vector<std::vector<bool>>& syndromes,
    bool output_prob,
    std::vector<std::unordered_map<std::string, double>>* prob_dists,
    int num_threads)
{
    size_t n = syndromes.size();
    std::vector<std::vector<bool>> predictions(n);

    if (output_prob && prob_dists) {
        prob_dists->resize(n);
    }

    total_contraction_time_ = 0.0;
    total_hypergraph_approximate_time_ = 0.0;
    total_init_time_ = 0.0;
    total_running_time_ = 0.0;
    total_order_finder_time_ = 0.0;

    std::mutex result_mutex;

    auto worker = [&](size_t start_idx, size_t end_idx) {

        ApproximateContractionExecutor local_contractor(
            num_detectors_,
            num_observables_,
            approximatestrategy_,
            approximate_param_,
            hypergraph_,
            priority_,
            priority_topk_,
            use_heuristic_,
            alpha_,
            false,  // local contractor disables internal openmp parallelism
            openmp_num_threads_
        );

        double thread_contraction_time = 0.0;
        double thread_init_time = 0.0;
        double thread_hypergraph_approximate_time = 0.0;
        double thread_order_finder_time = 0.0;
        double thread_core_decode_time = 0.0;


        for (size_t i = start_idx; i < end_idx; ++i) {
            auto start_time = std::chrono::high_resolution_clock::now();

            auto prob_dist = local_contractor.mld_contraction_no_slicing(syndromes[i]);

            auto end_time = std::chrono::high_resolution_clock::now();

            auto [prediction, prob_correct] = local_contractor.validate_logical_operator(prob_dist);
            
            double decode_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1e6;

            thread_core_decode_time += decode_time;

            thread_contraction_time += local_contractor.get_execution_contraction_time();
            thread_init_time += local_contractor.get_execution_init_time();
            thread_hypergraph_approximate_time += local_contractor.get_execution_hypergraph_approximate_time();
            thread_order_finder_time += local_contractor.get_execution_order_finder_time();

            {
                std::lock_guard<std::mutex> lock(result_mutex);
                predictions[i] = std::move(prediction);
                if (output_prob && prob_dists) {
                    std::unordered_map<std::string, double> converted_map;
                    for (const auto& [state, prob] : prob_dist) {
                        converted_map[bitset_to_string(state)] = prob;
                    }
                    (*prob_dists)[i] = std::move(converted_map);
                }
            }
        }

        {
            std::lock_guard<std::mutex> lock(result_mutex);
            total_contraction_time_ += thread_contraction_time;
            total_running_time_ += thread_core_decode_time;
            total_hypergraph_approximate_time_ += thread_hypergraph_approximate_time;
            total_init_time_ += thread_init_time;
            total_order_finder_time_ += thread_order_finder_time;
        }
    };

    std::vector<std::thread> threads;
    size_t chunk_size = (n + num_threads - 1) / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        size_t start_idx = t * chunk_size;
        size_t end_idx = std::min(start_idx + chunk_size, n);
        if (start_idx >= end_idx) break;
        threads.emplace_back(worker, start_idx, end_idx);
    }

    for (auto& thread : threads) {
        if (thread.joinable()) thread.join();
    }

    return predictions;
}
