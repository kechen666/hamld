import stim
import time
import sys
sys.path.insert(0, "/home/normaluser/ck/epmld/src/cpp/bazel-bin/main")

import eamld_pybind11  # 模块名必须和 C++ 中 PYBIND11_MODULE 定义一致

if __name__ == "__main__":
    dem_file = "/home/normaluser/ck/epmld/data/external/epmld_experiment_data/experiment_1_surface_code_fault_tolerance_threshold/X/d3_r1/detector_error_model_sd_p10.dem"
    dem = stim.DetectorErrorModel.from_file(dem_file)
    decoder = eamld_pybind11.EAMLDCpp_from_file(dem_path=dem_file, approx_strategy="node_topk", approx_param=10)

    print("Number of detectors:", dem.num_detectors)

    # 单次解码
    syndrome = [False] * dem.num_detectors
    syndrome[0] = True
    syndrome[2] = True
    start = time.perf_counter()
    pred, prob_dist, prob_correct = decoder.decode(syndrome)
    end = time.perf_counter()
    print(f"Single decode time: {(end - start)*1000:.3f} ms")
    print("Single decode prediction:", pred)
    print("Single decode prob_correct:", prob_correct)

    # 批量解码
    syndromes = [syndrome]*1000
    prob_dists = []
    start = time.perf_counter()
    predictions = decoder.decode_batch(syndromes, False, prob_dists)
    end = time.perf_counter()
    print(f"Batch decode time: {(end - start)*1000:.3f} ms")
    # print("Batch predictions:", predictions)

    # 并行解码
    prob_dists_parallel = []
    start = time.perf_counter()
    predictions_parallel = decoder.parallel_decode_batch(syndromes, False, prob_dists_parallel, 12)
    end = time.perf_counter()
    print(f"Parallel batch decode time: {(end - start)*1000:.3f} ms")
    print(f"Parallel total_contraction_time_ms: {decoder.total_contraction_time_ms()*1000} ms")
