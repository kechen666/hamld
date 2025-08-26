# HAMLD (hypergraph-based approximate maximum likelihood decoding)

* [What is HAMLD?](#1)
* [How do I use HAMLD?](#2)
* [How does HAMLD work?](#3)
* [How do I cite HAMLD?](#4)

<a id="1"></a>

## What is HAMLD?

HAMLD (Efficient Approximate Maximum Likelihood Decoding) is an innovative decoder designed for quantum error-correcting codes (QEC), specifically optimized for high-accuracy decoding under circuit-level noise models. It achieves efficient decoding through the following core innovations:

### Core Innovations:

* **Support for arbitrary QEC codes and noise models**
  Supports arbitrary quantum error-correcting codes and noise models, including all noise circuits constructible via Stim, as well as special noise types not yet implemented in Stim (e.g., biased measurement noise).

* **Approximation-based acceleration**
  Introduces hypergraph and contraction approximations to eliminate computational overhead unrelated to decoding. This significantly boosts speed while maintaining accuracy, achieving higher efficiency compared with commonly used TN-MLD methods.

### Performance Advantages:

* **Lower complexity**
  Reduces the exponential complexity of traditional MLD decoders to a polynomial scale.

* **Higher decoding accuracy**
  In scenarios using only ancilla qubit measurements (non-destructive error correction), HAMLD significantly outperforms traditional decoders such as MWPM and BP+OSD in terms of logical error rate.

### Current Limitations:

* For **surface codes with data qubit measurement syndromes**, accuracy improvement over MWPM is limited, though still favorable.
* For **QLDPC codes with data qubit syndrome**, HAMLD may not outperform BP+OSD in accuracy, but provides lower decoding complexity.
* **Real-time decoding** requirements for superconducting quantum processors are not yet fully met.

<a id="2"></a>

## How do I use HAMLD?

### Installation

Create a Python virtual environment (Anaconda recommended):

```bash
conda create -n hamld python=3.10
conda activate hamld
```

We use a modern `pyproject.toml` build system powered by **hatch**. Install it with:

```bash
pip install hatch -i https://pypi.tuna.tsinghua.edu.cn/simple
```

First, build the package:

```bash
hatch build  # Builds the package into the dist/ folder
```

Then, choose one of the installation methods:

```bash
pip install -e .  # Development mode
# OR
pip install ./dist/hamld-0.0.0-py3-none-any.whl  # Install from build
```

High-performance decoding is recommended for Linux systems with sufficient CPU and memory.

Navigate to the C++ source folder:

```bash
cd hamld/src/cpp
```

Compile and test (bazel version is 8.2.1):

```bash
bazel build //main:test_contraction_strategy

./bazel-bin/main/test_contraction_strategy <PROJECT_ROOT>/hamld/data/external/epmld_experiment_data/epmld_paper_experiment/overall_performance/surface_code/X/d3_r1/detector_error_model_si1000_p10_no_stabilizer.dem
```

> **Note:** Replace `<PROJECT_ROOT>` with your project root path. Using an SSD and multi-core CPUs will significantly improve compilation and runtime performance.

Install Bit Set dependencies:

```bash
cd hamld/src/cpp
mkdir -p bit_set && cd bit_set

# Manually install dependencies (automation via http_archive possible in future)
git clone https://github.com/abseil/abseil-cpp.git
git clone https://github.com/Tessil/robin-map.git
git clone https://github.com/martinus/robin-hood-hashing.git
```

Compile `hamld_pybind11` (Linux example):

```bash
bazel build //main:hamld_pybind11

ln -sf <PROJECT_ROOT>/hamld/src/cpp/bazel-bin/main/hamld_pybind11.so <PROJECT_ROOT>/hamld/src/hamld/hamld_pybind11.so
```

> Replace `<PROJECT_ROOT>` with your project root. High-performance CPUs and enough RAM help for faster pybind11 compilation.

Enable Python API by editing `hamld/src/hamld/__init__.py` and uncommenting:

```python
from .hamld_pybind11 import HAMLDCpp_from_file
```

Follow the instructions in `HAMLD_Tutorial.ipynb` for example usage and decoder API calls.

### Verify Installation

Check that the package is properly installed:

```python
import hamld  # Should import without error
```

### Example Code

Here’s a basic usage example:

```python
import numpy as np
import stim

# Generate noisy surface code circuit
circuit = stim.Circuit.generated("surface_code:rotated_memory_x", 
                                 distance=3, 
                                 rounds=1, 
                                 after_clifford_depolarization=0.05)
num_shots = 1000
model = circuit.detector_error_model(decompose_errors=False, flatten_loops=True)
sampler = circuit.compile_detector_sampler()
syndrome, actual_observables = sampler.sample(shots=num_shots, separate_observables=True)

# Decode using HAMLD
import hamld
mld_decoder = hamld.HAMLD(detector_error_model=model, order_method='mld', slice_method='no_slice')
predicted_observables = mld_decoder.decode_batch(syndrome)
num_mistakes = np.sum(np.any(predicted_observables != actual_observables, axis=1))

print(f"{num_mistakes}/{num_shots}")
```

### Quick Start

We provide detailed tutorials to help you get started:

* Run the `HAMLD_Tutorial.ipynb` notebook
* Explore and test the API via example code
* Refer to the `hamld` package’s `README.md` for full documentation

<a id="3"></a>

## How does HAMLD work?

Based on the underlying research, HAMLD offers several key advantages over traditional MLD decoders and other decoding approaches:

### 1. Scalable Approximation Strategies

HAMLD uses innovative approximations to achieve efficient decoding, with a time complexity of
**O(rd² + C)**, where:

* `r` is the number of rounds
* `d` is the code distance
* `C` is a constant determined by approximation and noise structure

This enables scalable performance improvements through parameter tuning.

### 2. Near-MLD Accuracy

In ancilla-only syndrome decoding (non-destructive error correction):

* HAMLD achieves **higher accuracy** than MWPM and BP+OSD
* For surface codes, HAMLD reaches **near-optimal decoding performance**, close to full MLD
* (Note: For QLDPC codes, full MLD baselines are not well-defined, so no direct comparison is made.)

### 3. Adaptability to Biased Measurement Noise

HAMLD features a dedicated mechanism for handling **measurement bias**, i.e., asymmetric 0/1 measurement error rates—a feature lacking in many existing decoders and even in `stim`.

### Additional Tools

* **Hypergraph-based DEM analysis**
  HAMLD includes a novel visual and structural analysis tool using hypergraph and connectivity graph representations of detector error models.

* **Noisy quantum circuit generator**
  Currently supports surface code noise circuits via `stim`. Future support for other QEC codes is planned.

<a id="4"></a>

## TODO
* Implement the core code in C++. (To be completed)

* Matrixize the calculation process and add parallel and vectorized calculations to achieve acceleration. (To be completed)

## How do I cite HAMLD?

If you use HAMLD in your research, please cite:

```bibtex
xxxxx
```

## Other Good Open-Source for QEC decoding

* [Stim](https://github.com/quantumlib/Stim.git)
* [PyMatching](https://github.com/oscarhiggott/PyMatching/tree/master)
* [LDPC](https://github.com/quantumgizmos/ldpc)
* [stimbposd](https://github.com/oscarhiggott/stimbposd)
* [tesseract-decoder](https://github.com/quantumlib/tesseract-decoder)
* [qtcodes](https://github.com/yaleqc/qtcodes)
