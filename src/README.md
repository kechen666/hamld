# EAMLD - Efficient Approximate Maximum Likelihood Decoder

## Installation

To install the required dependencies, run:

```bash
python -m pip install jupyter stim -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install networkx matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Project Structure

The EAMLD source code consists of three core modules and supporting components:

### Core Modules
- `benchmark/`: Contains performance evaluation scripts for assessing EAMLD algorithm performance
- `contraction_strategies/`: Implements tensor contraction strategies and optimization methods
- `contraction_executor/`: Provides the execution engine for tensor contraction operations

### Supporting Components
- `eamld.py`: Core implementation of the EAMLD algorithm
- `utils.py`: General utility functions
- `eamld_utility.py`: EAMLD-specific helper functions
- `logging_config.py`: Default logging configuration
- `circuit_generator.py`: Quantum circuit generators (currently supports Surface code)
- `sample_decoder.py`: Quantum bit sampling decoders

### Deprecated
- `config.py`: Legacy configuration parameters (currently unused)

## Key Features
- Modular architecture for quantum error correction analysis
- Optimized hypergraph contraction strategies
- Performance benchmarking framework
- Support for surface code circuit generation
