#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fstream>
#include "stim.h"
#include "hamld_cpp.h"

namespace py = pybind11;

PYBIND11_MODULE(hamld_pybind11, m) {
    py::class_<HAMLDCpp>(m, "HAMLDCpp")
        .def("decode", [](HAMLDCpp& self, const std::vector<bool>& syndrome) {
            try {
                return self.decode(syndrome);
            } catch (const std::exception& e) {
                throw py::value_error(std::string("decode() failed: ") + e.what());
            }
        }, py::arg("syndrome"))

        .def("decode_batch", [](HAMLDCpp& self,
                                const std::vector<std::vector<bool>>& syndromes,
                                bool output_prob) -> py::object {
            try {
                std::vector<std::unordered_map<std::string, double>> prob_dists;
                if (output_prob) {
                    auto results = self.decode_batch(syndromes, true, &prob_dists);
                    return py::make_tuple(results, prob_dists);
                } else {
                    auto results = self.decode_batch(syndromes, false, &prob_dists);
                    return py::cast(results);
                }
            } catch (const std::exception& e) {
                throw py::value_error(std::string("decode_batch() failed: ") + e.what());
            }
        }, py::arg("syndromes"), py::arg("output_prob") = false)

        .def("parallel_decode_batch", [](HAMLDCpp& self,
                                         const std::vector<std::vector<bool>>& syndromes,
                                         bool output_prob,
                                         int num_threads) -> py::object {
            try {
                std::vector<std::unordered_map<std::string, double>> prob_dists;
                if (output_prob) {
                    auto results = self.parallel_decode_batch(syndromes, true, &prob_dists, num_threads);
                    return py::make_tuple(results, prob_dists);
                } else {
                    auto results = self.parallel_decode_batch(syndromes, false, &prob_dists, num_threads);
                    return py::cast(results);
                }
            } catch (const std::exception& e) {
                throw py::value_error(std::string("parallel_decode_batch() failed: ") + e.what());
            }
        }, py::arg("syndromes"), py::arg("output_prob") = false, py::arg("num_threads") = 4)

        .def("total_contraction_time", &HAMLDCpp::total_contraction_time)
        .def("total_running_time", &HAMLDCpp::total_running_time)
        .def("total_hypergraph_approximate_time", &HAMLDCpp::total_hypergraph_approximate_time)
        .def("total_init_time", &HAMLDCpp::total_init_time)
        .def("total_order_finder_time", &HAMLDCpp::total_order_finder_time);

    m.def("HAMLDCpp_from_file", [](const std::string& dem_path,
                                   const std::string& approx_strategy,
                                   double approx_param,
                                   int priority,
                                   int priority_topk,
                                   bool use_heuristic,
                                   double alpha,
                                   bool openmp,
                                   int openmp_num_threads) {
        FILE* f = fopen(dem_path.c_str(), "rb");
        if (!f) {
            throw std::runtime_error("Failed to open DEM file: " + dem_path);
        }
        stim::DetectorErrorModel dem = stim::DetectorErrorModel::from_file(f);
        fclose(f);
        return std::make_unique<HAMLDCpp>(
            dem, approx_strategy, approx_param, priority, priority_topk,
            use_heuristic, alpha, openmp, openmp_num_threads);
    }, py::arg("dem_path"),
       py::arg("approx_strategy") = "hyperedge_topk",
       py::arg("approx_param") = -1.0,
       py::arg("priority") = 1,
       py::arg("priority_topk") = 10,
       py::arg("use_heuristic") = true,
       py::arg("alpha") = 0.05,
       py::arg("openmp") = false,
       py::arg("openmp_num_threads") = 1);
}
