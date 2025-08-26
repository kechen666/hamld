#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // 添加这一行
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <future>

namespace py = pybind11;

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

class ContractionExecutorCpp {
public:
    ContractionExecutorCpp(int detector_number, int logical_number, const std::vector<std::string>& order,
                           const std::vector<std::string>& sliced_hyperedges,
                           const std::unordered_map<std::string, double>& contractable_hyperedges_weights_dict,
                           const std::string& accuracy = "float64")
        : detector_number(detector_number), logical_number(logical_number), total_length(detector_number + logical_number),
          order(order), sliced_hyperedges(sliced_hyperedges), accuracy(accuracy),
          contractable_hyperedges_weights_dict(contractable_hyperedges_weights_dict),
          _execution_contraction_time(0), _execution_max_distribution_size(0) {}

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

    // 串行的 MLD 收缩
    std::unordered_map<std::string, double> mld_contraction_serial(py::array_t<bool> syndrome) {
        auto syndrome_np = syndrome.unchecked<1>();
        std::vector<bool> syndrome_vec(syndrome_np.size());
        for (ssize_t i = 0; i < syndrome_np.size(); ++i) {
            syndrome_vec[i] = syndrome_np(i);
        }

        auto [parallelizable_init_prob_dist, init_contractable_hyperedges_weights_dict] = get_parallel_task_initial_input();
        auto start_time = std::chrono::high_resolution_clock::now();
        auto [prob_dist, _] = single_node_online_mld_contraction(syndrome_vec, parallelizable_init_prob_dist, init_contractable_hyperedges_weights_dict);
        auto end_time = std::chrono::high_resolution_clock::now();
        _execution_contraction_time = std::chrono::duration<double>(end_time - start_time).count();
        return prob_dist;
    }

    // 并行的 MLD 收缩
    std::unordered_map<std::string, double> mld_contraction_parallel_concurrent(py::array_t<bool> syndrome, int max_thread = 4) {
        auto syndrome_np = syndrome.unchecked<1>();
        std::vector<bool> syndrome_vec(syndrome_np.size());
        for (ssize_t i = 0; i < syndrome_np.size(); ++i) {
            syndrome_vec[i] = syndrome_np(i);
        }

        auto [parallelizable_init_prob_dist, init_contractable_hyperedges_weights_dict] = get_parallel_task_initial_input();

        std::vector<std::pair<std::string, double>> tasks;
        for (const auto& pair : parallelizable_init_prob_dist) {
            tasks.emplace_back(pair);
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<std::future<std::pair<std::unordered_map<std::string, double>, std::unordered_map<std::string, double>>>> futures;
        for (const auto& task : tasks) {
            futures.emplace_back(std::async(std::launch::async, [this, syndrome_vec, task, init_contractable_hyperedges_weights_dict]() {
                return single_node_online_mld_contraction(syndrome_vec, {{task.first, task.second}}, init_contractable_hyperedges_weights_dict);
            }));
        }

        std::unordered_map<std::string, double> merged_prob_dist;
        for (auto& future : futures) {
            auto [result_prob_dist, _] = future.get();
            for (const auto& pair : result_prob_dist) {
                merged_prob_dist[pair.first] += pair.second;
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        _execution_contraction_time = std::chrono::duration<double>(end_time - start_time).count();

        return merged_prob_dist;
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
        }

        return {prob_dist, contractable_hyperedges_weights};
    }

    // 验证逻辑运算符
    std::pair<py::array_t<bool>, double> validate_logical_operator(const std::unordered_map<std::string, double>& prob_dist) {
        if (prob_dist.size() != 2) {
            throw std::invalid_argument("prob_dist must contain exactly two keys, with the last element being True and False.");
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
};

PYBIND11_MODULE(contraction_executor_cpp, m) {
    py::class_<ContractionExecutorCpp>(m, "ContractionExecutorCpp")
        .def(py::init<int, int, const std::vector<std::string>&, const std::vector<std::string>&, const std::unordered_map<std::string, double>&, const std::string&>())
        .def("mld_contraction_no_slicing", &ContractionExecutorCpp::mld_contraction_no_slicing)
        .def("mld_contraction_serial", &ContractionExecutorCpp::mld_contraction_serial)
        .def("mld_contraction_parallel_concurrent", &ContractionExecutorCpp::mld_contraction_parallel_concurrent)
        .def("validate_logical_operator", &ContractionExecutorCpp::validate_logical_operator)
        .def("get_execution_contraction_time", &ContractionExecutorCpp::get_execution_contraction_time);
}
    