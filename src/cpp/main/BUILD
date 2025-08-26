load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")
load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "detector_hypergraph",
    srcs = ["detector_hypergraph.cc"],
    hdrs = ["detector_hypergraph.h"],
    deps = ["@stim//:stim_lib"],
    copts = ["-fopenmp"],
    linkopts = ["-fopenmp"],
)

cc_library(
    name = "connectivity_graph",
    srcs = ["connectivity_graph.cc"],
    hdrs = ["connectivity_graph.h"],
    deps = [
        ":detector_hypergraph",  # 添加这个依赖
        "@stim//:stim_lib",
    ],
)

cc_library(
    name = "greedy_mld_order",
    srcs = ["greedy_mld_order.cc"],
    hdrs = ["greedy_mld_order.h"],
    deps = [
        ":connectivity_graph",  # 这里添加依赖！
        "@stim//:stim_lib",
    ],
)

cc_binary(
    name = "test_contraction_strategy",
    srcs = ["test_contraction_strategy.cc"],
    deps = [
        ":detector_hypergraph",
        ":connectivity_graph",
        ":greedy_mld_order",
        "@stim//:stim_lib",
    ],
)

cc_library(
    name = "sparse_state",
    srcs = ["sparse_state.cc"],
    hdrs = ["sparse_state.h"],
    deps = [],
)


cc_library(
    name = "approximate_contraction_executor",
    srcs = ["approximate_contraction_executor.cc"],
    hdrs = ["approximate_contraction_executor.h"],
    deps = [
        ":detector_hypergraph",
        ":connectivity_graph",
        ":greedy_mld_order",
        "//:tsl_robin_map",
        "@stim//:stim_lib",
    ],
    # 如果使用OpenMP可以打开以下两行
    # copts = ["-fopenmp"],
    # linkopts = ["-fopenmp"],
)

cc_binary(
    name = "test_approximate_contraction_executor",
    srcs = ["test_approximate_contraction_executor.cc"],
    deps = [
        ":detector_hypergraph",
        ":connectivity_graph",
        ":greedy_mld_order",
        "approximate_contraction_executor",
        "@stim//:stim_lib",
        "//:tsl_robin_map",
    ],
)

cc_library(
    name = "hamld_cpp",
    srcs = ["hamld_cpp.cc"],
    hdrs = ["hamld_cpp.h"],
    deps = [
        ":detector_hypergraph",
        ":approximate_contraction_executor",
        "@stim//:stim_lib",
    ],
)

cc_binary(
    name = "test_hamld_cpp",
    srcs = ["test_hamld_cpp.cc"],
    deps = [
        ":hamld_cpp",
        "@stim//:stim_lib",
    ],
)

cc_binary(
    name = "test_hamld_cpp_perf",
    srcs = ["test_hamld_cpp_perf.cc"],
    deps = [
        ":hamld_cpp",
        "@stim//:stim_lib",
    ],
)

pybind_extension(
    name = "hamld_pybind11",
    srcs = ["hamld_pybind11.cc"],
    deps = [
        ":hamld_cpp",  # 你已有的 C++ 库
        "@stim//:stim_lib",
    ],
)