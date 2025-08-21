# HAMLD (hypergraph-based approximate maximum likelihood decoding)

* [What is EAMLD?](#1)
* [How do I use EAMLD?](#2)
* [How does EAMLD work?](#3)
* [How do I cite EAMLD?](#4)

<a id="1"></a>

## What is EAMLD?

EAMLD (Efficient Approximate Maximum Likelihood Decoding) is an innovative decoder designed for quantum error-correcting codes (QEC), specifically optimized for high-accuracy decoding under circuit-level noise models. It achieves efficient decoding through the following core innovations:

### Core Innovations:

* **Support for arbitrary QEC codes and noise models**
  EAMLD supports any QEC code expressible via `stim` noisy circuits, including advanced noise types like biased measurement noise not yet supported by `stim`.

* **Approximation-based acceleration**
  EAMLD applies hypergraph and contraction approximations to remove decoding-irrelevant computational overhead, significantly speeding up decoding while maintaining accuracy.

* **Optimized contraction order**
  Advanced contraction order algorithms reduce decoding complexity without compromising accuracy.

### Performance Advantages:

* **Lower complexity**
  Reduces the exponential complexity of traditional MLD decoders to a polynomial scale.

* **Higher decoding accuracy**
  In scenarios using only ancilla qubit measurements (non-destructive error correction), EAMLD significantly outperforms traditional decoders such as MWPM and BP+OSD in terms of logical error rate.

### Current Limitations:

* For **surface codes with data qubit measurement syndromes**, accuracy improvement over MWPM is limited, though still favorable.
* For **QLDPC codes with data qubit syndrome**, EAMLD may not outperform BP+OSD in accuracy, but provides lower decoding complexity.
* **Real-time decoding** requirements for superconducting quantum processors are not yet fully met.

<a id="2"></a>

## How do I use EAMLD?

### Installation

Currently, EAMLD supports **local installation only**. Follow these steps:

1. **Create a Python virtual environment** (Anaconda recommended):

```bash
conda create -n eamld python=3.10
conda activate eamld
```

2. **Install build tools**
   We use a modern `pyproject.toml` build system powered by `hatch`:

```bash
pip install hatch -i https://pypi.tuna.tsinghua.edu.cn/simple
```

3. **Build and install the project**

```bash
hatch build  # Builds the package into the dist/ folder
```

Choose one of the installation methods:

```bash
pip install -e .  # Development mode
# OR
pip install ./dist/eamld-0.0.1-py3-none-any.whl  # Install from build
```

### Verify Installation

Check that the package is properly installed:

```python
import eamld  # Should import without error
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

# Decode using EAMLD
import eamld
mld_decoder = eamld.EAMLD(detector_error_model=model, order_method='mld', slice_method='no_slice')
predicted_observables = mld_decoder.decode_batch(syndrome)
num_mistakes = np.sum(np.any(predicted_observables != actual_observables, axis=1))

print(f"{num_mistakes}/{num_shots}")
```

### Quick Start

We provide detailed tutorials to help you get started:

* Run the `EAMLD_Tutorial.ipynb` notebook
* Explore and test the API via example code
* Refer to the `eamld` package’s `README.md` for full documentation

<a id="3"></a>

## How does EAMLD work?

Based on the underlying research, EAMLD offers several key advantages over traditional MLD decoders and other decoding approaches:

### 1. Scalable Approximation Strategies

EAMLD uses innovative approximations to achieve efficient decoding, with a time complexity of
**O(rd² + C)**, where:

* `r` is the number of rounds
* `d` is the code distance
* `C` is a constant determined by approximation and noise structure

This enables scalable performance improvements through parameter tuning.

### 2. Near-MLD Accuracy

In ancilla-only syndrome decoding (non-destructive error correction):

* EAMLD achieves **higher accuracy** than MWPM and BP+OSD
* For surface codes, EAMLD reaches **near-optimal decoding performance**, close to full MLD
* (Note: For QLDPC codes, full MLD baselines are not well-defined, so no direct comparison is made.)

### 3. Adaptability to Biased Measurement Noise

EAMLD features a dedicated mechanism for handling **measurement bias**, i.e., asymmetric 0/1 measurement error rates—a feature lacking in many existing decoders and even in `stim`.

### Additional Tools

* **Hypergraph-based DEM analysis**
  EAMLD includes a novel visual and structural analysis tool using hypergraph and connectivity graph representations of detector error models.

* **Noisy quantum circuit generator**
  Currently supports surface code noise circuits via `stim`. Future support for other QEC codes is planned.

<a id="4"></a>

## TODO
* Implement the core code in C++. (To be completed)

* Matrixize the calculation process and add parallel and vectorized calculations to achieve acceleration. (To be completed)

## How do I cite EAMLD?

If you use EAMLD in your research, please cite:

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
