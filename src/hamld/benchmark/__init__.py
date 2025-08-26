# src/hamld/benchmark/__init__.py

"""
Benchmarking module for HAMLD.

This module provides benchmarking tools for contraction strategies and other processes.
"""

from .contraction_strategy_benchmarking import contraction_strategy_benchmarking
from .logical_error_rate_benchmarking import LogicalErrorRateBenchmark
from .decoder_speed_benchmarking import DecoderSpeedBenchmark
from .logical_error_expectation_benchmarking import LogicalErrorExpectationBenchmark
from .utility import generate_syndrome_and_observables, generate_qldpc_syndrome_and_observables, generate_detector_error_model, generate_qldpc_detector_error_model, b8_to_array, parse_b8, generate_all_possible_syndromes, generate_syndromes_generator, generate_qldpc_detector_error_model_path, generate_detector_error_model_path

__all__ = [
    'contraction_strategy_benchmarking',
    'LogicalErrorRateBenchmark',
    'DecoderSpeedBenchmark',
    'LogicalErrorExpectationBenchmark',
    'generate_syndrome_and_observables',
    'generate_qldpc_syndrome_and_observables',
    'generate_detector_error_model',
    'generate_qldpc_detector_error_model',
    'b8_to_array',
    'parse_b8',
    'generate_all_possible_syndromes',
    'generate_syndromes_generator',
    'generate_qldpc_detector_error_model_path',
    'generate_detector_error_model_path'
]
