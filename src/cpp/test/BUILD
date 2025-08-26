cc_binary(
    name = "map_benchmark",
    srcs = ["map_benchmark.cpp"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "//:tsl_robin_map",
        "//:robin_hood",
    ],
    copts = [
        "-std=c++17",
        "-O2",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "fast_benchmark",
    srcs = ["fast_benchmark.cpp"],
    deps = [
        "//:tsl_robin_map",
    ],
    copts = [
        "-std=c++17",
        "-O2",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "test_find_fast_benchmark",
    srcs = ["test_find_fast_benchmark.cpp"],
    deps = [
        "//:tsl_robin_map",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "test_approximate",
    srcs = ["test_approximate.cpp"],
    deps = [
        "//:tsl_robin_map",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "test_approximate_performance",
    srcs = ["test_approximate_performance.cc"],
    deps = [
        "//:tsl_robin_map",
    ],
)

cc_binary(
    name = "test_approximate_performance_h",
    srcs = ["test_approximate_performance_h.cc"],
    deps = [
        "//:tsl_robin_map",
    ],
)