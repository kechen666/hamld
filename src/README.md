# HAMLD - Hypergraph-based Approximate Maximum Likelihood Decoder

## Installation

To install the required dependencies, run:

```bash
python -m pip install jupyter stim -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install networkx matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Project Structure

The HAMLD source code consists of three core modules, supporting components and c++ version:

### Core Modules
- `benchmark/`: Contains performance evaluation scripts for assessing HAMLD algorithm performance
- `contraction_strategies/`: Implements tensor contraction strategies and optimization methods
- `contraction_executor/`: Provides the execution engine for tensor contraction operations
- `cpp/`: Contains C++ source code for performance-critical components

### Supporting Components
- `hamld.py`: Core implementation of the HAMLD algorithm
- `utils.py`: General utility functions
- `hamld_utility.py`: HAMLD-specific helper functions
- `logging_config.py`: Default logging configuration
- `circuit_generator.py`: Quantum circuit generators (currently supports Surface code)
- `sample_decoder.py`: Quantum bit sampling decoders

### Deprecated
- `config.py`: Legacy configuration parameters (currently unused)

## About C++ Files

Before using the C++ files in this project, you need to install Bazel. Please note the following:

This project uses Bazel version 8.2.1. Other versions may have potential compatibility issues.

The bazel_downloader.cfg file specifies mirror sources because accessing Bazel's official repository can be slow in China. If you encounter download speed issues, you may consider modifying or removing the mirror addresses in this configuration file.

To use the C++ code, you first need to compile `hamd_pybind11`, then place the resulting files in the `src/hamld` directory and package them via `__init__.py`.

The compilation command is:

```bash

bazel build //main:hamld_pybind11
```

## Key Features
- Modular architecture for quantum error correction analysis
- Optimized hypergraph contraction strategies
- Performance benchmarking framework
- Support for surface code circuit generation