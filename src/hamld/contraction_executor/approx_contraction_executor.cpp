#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <queue>
#include <limits>

namespace py = pybind11;

// 定义四种策略的枚举类型
enum class ApproximateStrategy {
    NODE_TOPK,
    HYPEREDGE_TOPK,
    NODE_THRESHOLD,
    HYPEREDGE_THRESHOLD
};

// 存储每种策略对应的参数
std::unordered_map<ApproximateStrategy, double> strategy_params = {
    {ApproximateStrategy::NODE_TOPK, 1e3},
    {ApproximateStrategy::HYPEREDGE_TOPK, 1e4},
    {ApproximateStrategy::NODE_THRESHOLD, 1e-8},
    {ApproximateStrategy::HYPEREDGE_THRESHOLD, 1e-9}
};

// 将二进制字符串转换为布尔数组
std::vector<bool> str_to_binary_bitwise(const std::string& s) {
    std::vector<bool> result;
    for (char c : s) {
        result.push_back(c == '1');
    }
    return result;
}

// 将布尔数组转换为二进制字符串
std::string binary_to_str(const std::vector<bool>& binary) {
    std::string result;
    for (bool bit : binary) {
        result += bit ? '1' : '0';
    }
    return result;
}

// 检测字符串的指定位是否等于预期值
bool check_str_value(const std::string& str_syndrome, int position, const std::string& expected_value) {
    if (expected_value != "0" && expected_value != "1") {
        throw std::invalid_argument("预期值必须是 0 或 1。");
    }
    return str_syndrome[position] == expected_value[0];
}

class ApproximateContractionExecutorCpp {
public:
    ApproximateContractionExecutorCpp(int detector_number, int logical_number, const std::vector<std::string>& order,
                                        const std::vector<std::string>& sliced_hyperedges,
                                        const std::unordered_map<std::string, double>& contractable_hyperedges_weights_dict,
                                        const std::string& accuracy = "float64", const std::string& approximatestrategy = "node_topk",
                                        double approximate_param = std::numeric_limits<double>::quiet_NaN())
        : detector_number(detector_number), logical_number(logical_number), total_length(detector_number + logical_number),
          order(order), sliced_hyperedges(sliced_hyperedges), accuracy(accuracy),
          contractable_hyperedges_weights_dict(contractable_hyperedges_weights_dict),
          _execution_contraction_time(0), _execution_max_distribution_size(0) {

        // 近似策略以及近似参数
        if (approximatestrategy == "node_topk") {
            this->approximatestrategy = ApproximateStrategy::NODE_TOPK;
        } else if (approximatestrategy == "hyperedge_topk") {
            this->approximatestrategy = ApproximateStrategy::HYPEREDGE_TOPK;
        } else if (approximatestrategy == "node_threshold") {
            this->approximatestrategy = ApproximateStrategy::NODE_THRESHOLD;
        } else if (approximatestrategy == "hyperedge_threshold") {
            this->approximatestrategy = ApproximateStrategy::HYPEREDGE_THRESHOLD;
        } else {
            throw std::invalid_argument("approximatestrategy 必须是以下之一: node_topk, hyperedge_topk, node_threshold, hyperedge_threshold");
        }

        _approximatestrategy_method = (this->approximatestrategy == ApproximateStrategy::NODE_TOPK || this->approximatestrategy == ApproximateStrategy::HYPEREDGE_TOPK) ? "topk" : "threshold";
        _is_topk = _approximatestrategy_method == "topk";
        _is_threshold = _approximatestrategy_method == "threshold";
        _approximate_position = (this->approximatestrategy == ApproximateStrategy::NODE_TOPK || this->approximatestrategy == ApproximateStrategy::NODE_THRESHOLD) ? "node" : "hyperedge";

        // 验证 approximate_param 的有效性
        if (std::isnan(approximate_param)) {
            this->approximate_param = strategy_params[this->approximatestrategy];
        } else {
            if ((this->approximatestrategy == ApproximateStrategy::NODE_TOPK || this->approximatestrategy == ApproximateStrategy::HYPEREDGE_TOPK) && static_cast<int>(approximate_param) != approximate_param) {
                throw std::invalid_argument("当 approximatestrategy 为 topk 时，approximate_param 必须是整数。");
            }
            this->approximate_param = approximate_param;
        }
    }

    // // 近似概率分布
    // std::unordered_map<std::string, double> approximate_distribution(const std::unordered_map<std::string, double>& updated_prob_dist) {
    //     if (_is_topk) {
    //         if (updated_prob_dist.size() <= approximate_param) {
    //             // 日志信息可以根据需要添加
    //             return updated_prob_dist;
    //         }
    //         std::priority_queue<std::pair<double, std::string>, std::vector<std::pair<double, std::string>>, std::greater<std::pair<double, std::string>>> pq;
    //         for (const auto& pair : updated_prob_dist) {
    //             pq.push({pair.second, pair.first});
    //             if (pq.size() > approximate_param) {
    //                 pq.pop();
    //             }
    //         }
    //         std::unordered_map<std::string, double> result;
    //         while (!pq.empty()) {
    //             result[pq.top().second] = pq.top().first;
    //             pq.pop();
    //         }
    //         return result;
    //     } else if (_is_threshold) {
    //         std::unordered_map<std::string, double> result;
    //         for (const auto& pair : updated_prob_dist) {
    //             if (pair.second > approximate_param) {
    //                 result[pair.first] = pair.second;
    //             }
    //         }
    //         return result;
    //     } else {
    //         throw std::invalid_argument("Invalid approximate strategy");
    //     }
    // }
    // 近似概率分布
    std::unordered_map<std::string, double> approximate_distribution(const std::unordered_map<std::string, double>& updated_prob_dist) {
        if (_is_topk) {
            if (updated_prob_dist.size() <= approximate_param) {
                // 日志信息可以根据需要添加
                return updated_prob_dist;
            }
            std::vector<std::pair<std::string, double>> pairs;
            pairs.reserve(updated_prob_dist.size());
            for (const auto& pair : updated_prob_dist) {
                pairs.emplace_back(pair.first, pair.second);
            }
            std::nth_element(pairs.begin(), pairs.begin() + approximate_param, pairs.end(), [](const auto& a, const auto& b) {
                return a.second > b.second;
            });
            std::unordered_map<std::string, double> result;
            for (size_t i = 0; i < approximate_param; ++i) {
                result[pairs[i].first] = pairs[i].second;
            }
            return result;
        } else if (_is_threshold) {
            std::unordered_map<std::string, double> result;
            for (const auto& pair : updated_prob_dist) {
                if (pair.second > approximate_param) {
                    result[pair.first] = pair.second;
                }
            }
            return result;
        } else {
            throw std::invalid_argument("Invalid approximate strategy");
        }
    }

    // 翻转二进制字符串中的指定比特
    std::string flip_bits(const std::string& binary_str, const std::string& hyperedge) {
        std::string result = binary_str;
        std::string bit_label;
        std::stringstream ss(hyperedge);
        while (std::getline(ss, bit_label, ',')) {
            if (bit_label[0] == 'D') {
                int index = std::stoi(bit_label.substr(1));
                result[index] = (result[index] == '0') ? '1' : '0';
            } else if (bit_label[0] == 'L') {
                int index = detector_number + std::stoi(bit_label.substr(1));
                result[index] = (result[index] == '0') ? '1' : '0';
            } else {
                throw std::invalid_argument("Invalid bit label '" + bit_label + "': must start with 'D' or 'L'.");
            }
        }
        return result;
    }

    // 收缩超边并更新概率分布和超边权重字典
    std::pair<std::unordered_map<std::string, double>, std::unordered_map<std::string, double>>
    contract_hyperedge(const std::unordered_map<std::string, double>& prob_dist,
                       std::unordered_map<std::string, double> contractable_hyperedges_weights_dict,
                       const std::string& contracted_hyperedge) {
        double contracted_hyperedge_prob = contractable_hyperedges_weights_dict[contracted_hyperedge];
        contractable_hyperedges_weights_dict.erase(contracted_hyperedge);
        double non_flip_contracted_hyperedge_prob = 1 - contracted_hyperedge_prob;

        std::unordered_map<std::string, double> updated_prob_dist;
        for (const auto& pair : prob_dist) {
            const std::string& binary_str = pair.first;
            double prob = pair.second;
            std::string flipped_str = flip_bits(binary_str, contracted_hyperedge);

            double flipped_prob = prob * contracted_hyperedge_prob;
            double non_flipped_prob = prob * non_flip_contracted_hyperedge_prob;

            updated_prob_dist[flipped_str] += flipped_prob;
            updated_prob_dist[binary_str] += non_flipped_prob;
        }

        if (_approximate_position == "hyperedge") {
            updated_prob_dist = approximate_distribution(updated_prob_dist);
        }

        return {updated_prob_dist, contractable_hyperedges_weights_dict};
    }

    // 获取任务的初始输入
    std::pair<std::unordered_map<std::string, double>, std::unordered_map<std::string, double>>
    get_task_initial_input() {
        std::unordered_map<std::string, double> init_prob_dist;
        std::unordered_map<std::string, double> contractable_hyperedges_weights_dict = this->contractable_hyperedges_weights_dict;

        std::string init_key(total_length, '0');
        init_prob_dist[init_key] = 1;

        return {init_prob_dist, contractable_hyperedges_weights_dict};
    }

    // 获取并行任务的初始输入
    std::pair<std::unordered_map<std::string, double>, std::unordered_map<std::string, double>>
    get_parallel_task_initial_input() {
        auto [parallelizable_init_prob_dist, contractable_hyperedges_weights_dict] = get_task_initial_input();
        for (const std::string& hyperedge : sliced_hyperedges) {
            auto [new_prob_dist, new_weights_dict] = contract_hyperedge(parallelizable_init_prob_dist, contractable_hyperedges_weights_dict, hyperedge);
            parallelizable_init_prob_dist = new_prob_dist;
            contractable_hyperedges_weights_dict = new_weights_dict;
        }
        return {parallelizable_init_prob_dist, contractable_hyperedges_weights_dict};
    }

    // 无切片的 MLD 收缩
    std::unordered_map<std::string, double> mld_contraction_no_slicing(py::array_t<bool> syndrome) {
        auto syndrome_np = syndrome.unchecked<1>();
        std::vector<bool> syndrome_vec(syndrome_np.size());
        for (ssize_t i = 0; i < syndrome_np.size(); ++i) {
            syndrome_vec[i] = syndrome_np(i);
        }

        auto [init_prob_dist, init_contractable_hyperedges_weights_dict] = get_task_initial_input();
        auto start_time = std::chrono::high_resolution_clock::now();
        auto [prob_dist, _] = single_node_online_mld_contraction(syndrome_vec, init_prob_dist, init_contractable_hyperedges_weights_dict);
        auto end_time = std::chrono::high_resolution_clock::now();
        _execution_contraction_time = std::chrono::duration<double>(end_time - start_time).count();
        return prob_dist;
    }

    // 单节点在线 MLD 收缩
    std::pair<std::unordered_map<std::string, double>, std::unordered_map<std::string, double>>
    single_node_online_mld_contraction(const std::vector<bool>& syndrome,
                                       std::unordered_map<std::string, double> prob_dist,
                                       std::unordered_map<std::string, double> contractable_hyperedges_weights) {
        for (int contraction_step = 0; contraction_step < detector_number; ++contraction_step) {
            std::string contract_detector = order[contraction_step];
            int contract_detector_index = std::stoi(contract_detector.substr(1));
            std::string observed_detector_syndrome = std::to_string(syndrome[contract_detector_index] ? 1 : 0);

            std::vector<std::string> relevant_hyperedges;
            for (const auto& pair : contractable_hyperedges_weights) {
                const std::string& hyperedge = pair.first;
                std::string search_str = "," + hyperedge + ",";
                std::string target_str = "," + contract_detector + ",";
                if (search_str.find(target_str) != std::string::npos) {
                    relevant_hyperedges.push_back(hyperedge);
                }
            }

            for (const std::string& hyperedge : relevant_hyperedges) {
                auto [new_prob_dist, new_weights_dict] = contract_hyperedge(prob_dist, contractable_hyperedges_weights, hyperedge);
                prob_dist = new_prob_dist;
                contractable_hyperedges_weights = new_weights_dict;
            }

            std::unordered_map<std::string, double> new_prob_dist;
            for (const auto& pair : prob_dist) {
                const std::string& candidate_syndrome = pair.first;
                double prob = pair.second;
                if (check_str_value(candidate_syndrome, contract_detector_index, observed_detector_syndrome)) {
                    new_prob_dist[candidate_syndrome] = prob;
                }
            }
            prob_dist = new_prob_dist;

            if (_approximate_position == "node") {
                prob_dist = approximate_distribution(prob_dist);
            }
        }

        return {prob_dist, contractable_hyperedges_weights};
    }

    // 验证逻辑运算符
    std::pair<py::array_t<bool>, double> validate_logical_operator(const std::unordered_map<std::string, double>& prob_dist) {
        if (prob_dist.size() == 1) {
            bool logical_error_detected = prob_dist.begin()->first.back() == '1';
            py::array_t<bool> result = py::array_t<bool>(1);
            result.mutable_at(0) = logical_error_detected;
            return {result, 1.0};
        } else if (prob_dist.size() == 0) {
            py::array_t<bool> result = py::array_t<bool>(1);
            result.mutable_at(0) = false;
            return {result, 0.5};
        }

        double p_1 = 0, p_2 = 0;
        for (const auto& pair : prob_dist) {
            const std::string& key = pair.first;
            double prob = pair.second;
            if (key.back() == '0') {
                p_1 = prob;
            } else if (key.back() == '1') {
                p_2 = prob;
            }
        }

        if (p_1 == 0 || p_2 == 0) {
            throw std::invalid_argument("prob_dist must contain keys with the last element being True and False.");
        }

        bool logical_error_detected = p_2 > p_1;
        double prob_correct_correction = std::max(p_1, p_2) / (p_1 + p_2);

        py::array_t<bool> result = py::array_t<bool>(1);
        result.mutable_at(0) = logical_error_detected;

        return {result, prob_correct_correction};
    }

    double get_execution_contraction_time() const {
        return _execution_contraction_time;
    }

private:
    int detector_number;
    int logical_number;
    int total_length;
    std::vector<std::string> order;
    std::vector<std::string> sliced_hyperedges;
    std::string accuracy;
    std::unordered_map<std::string, double> contractable_hyperedges_weights_dict;
    double _execution_contraction_time;
    int _execution_max_distribution_size;
    ApproximateStrategy approximatestrategy;
    std::string _approximatestrategy_method;
    bool _is_topk;
    bool _is_threshold;
    std::string _approximate_position;
    double approximate_param;
};

PYBIND11_MODULE(approx_contraction_executor_cpp, m) {
    py::class_<ApproximateContractionExecutorCpp>(m, "ApproximateContractionExecutorCpp")
        .def(py::init<int, int, const std::vector<std::string>&, const std::vector<std::string>&, const std::unordered_map<std::string, double>&, const std::string&, const std::string&, double>())
        .def("mld_contraction_no_slicing", &ApproximateContractionExecutorCpp::mld_contraction_no_slicing)
        .def("validate_logical_operator", &ApproximateContractionExecutorCpp::validate_logical_operator)
        .def("get_execution_contraction_time", &ApproximateContractionExecutorCpp::get_execution_contraction_time);
}    