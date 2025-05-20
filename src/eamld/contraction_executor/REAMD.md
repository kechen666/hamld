# Contraction Engine Design

## Exact Contraction Implementations (use_approx=False)

We have developed and optimized several implementations of the exact contraction algorithm from both data structure and programming language perspectives:

1. **Basic Python Implementation** (`contraction_executor.py`):
   - Uses `Tuple[Bool]` for syndrome representation
   - Higher memory overhead but straightforward implementation

2. **Optimized Python Implementations**:
   - `contraction_executor_cpp_py.py` (EMLD):
     - Uses `str` type for syndrome representation
     - Easier conversion to C++ code
   - `contraction_executor_int.py` (INT):
     - Uses `int` type for syndrome representation
     - More memory efficient for certain cases

3. **C++ Implementation** (`contraction_executor.cpp`):
   - Requires compilation with:
     ```bash
     g++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) contraction_executor.cpp -o contraction_executor_cpp.so
     ```
   - Note: Currently disabled by default in favor of Python implementations

## Approximate Contraction Strategies (use_approx=True)

We evaluated multiple approximation strategies:

### QLDPC-Optimized Approaches
1. **EAMLD Method** (`new_priority_approx_contraction_executor_qldpc.py`):
   - Implements the priority-based contraction described in our paper
   - Our primary recommended approach

2. **Hierarchical Methods** (`hierarchical_approx_contraction_executor_qldpc.py`):
   - **First-layer contraction**: Only contracts hyperedges connected to flipped nodes (first layer)
   - **Second-layer contraction**: Focuses on secondary connections (second layer)

3. **Alternative Strategies**:
   - **Priority-based** (`priority_approx_contraction_executor_qldpc.py`):
     - Prioritizes by number of connected flipped nodes
     - Underperforms compared to EAMLD for QLDPC codes
   - **Logarithmic Scaling** (`log_approx_contraction_executor_qldpc.py`):
     - Applies log function to probability distributions
     - Shows limited practical improvement

### General Approximate Implementations
1. **Python Variants**:
   - `approx_contraction_executor.py`: Derived from the basic Python implementation
   - `approx_contraction_executor_cpp_py.py`: Python version with C++ compatibility

2. **C++ Implementation** (`approx_contraction_executor.cpp`):
   - Requires compilation with:
     ```bash
     g++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) approx_contraction_executor.cpp -o approx_contraction_executor_cpp.so
     ```
   - Offers potential performance benefits for large-scale computations

Note: The EAMLD method (priority-based approach) generally provides the best balance of accuracy and performance for most use cases.

<!-- # 收缩器 -->

<!-- 对于收缩器的设计，我们进行了许多的思考和改进（use_approx = False）：

在非近似的代码中，我们尝试了一些从数据结构以及代码语言角度进行优化收缩的性能。

- contraction_executor.py（normal）：使用Tuple[Bool]类型来表示syndrome，效率和内存开销更多。
- contraction_executor_cpp_py.py（emld）或者contraction_executor_int.py（int）均可以实现EAMLD的效果，只不过emld中使用str类型表示syndrome，而int使用int类型表示syndrome。相较而言，str类型更易转换为C++代码。
- contraction_executor.cpp为contraction_executor_cpp_py.py对应的C++代码，其中我们需要依赖下述代码在自己的操作平台上进行编译，进行才可使用。`g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) contraction_executor.cpp -o contraction_executor_cpp.so`。目前代码中，默认注释掉C++代码的使用，使用python代码。


在近似中，我们在最初也测试了多种的近似策略（use_approx = True）：

- new_priority_approx_contraction_executor_qldpc.py（eamld）：实现了论文描述的优先级，也是论文中描述的EAMLD方法。
- hierarchical_approx_contraction_executor_qldpc.py（qldpc-hierarchical-first）：分层收缩，将超边分为两层，连接翻转节点的作为first层，我们仅仅针对first的超边进行收缩。
- hierarchical_approx_contraction_executor_qldpc.py（qldpc-hierarchical-second）：分层收缩，将超边分为两层，连接翻转节点的作为second层，我们仅仅针对second的超边进行收缩。
- priority_approx_contraction_executor_qldpc.py（qldpc-priority）：优先级定义为连接的翻转节点的数量，其他类似eamld，在qldpc code中测试不如eamld中的优先级。
- log_approx_contraction_executor_qldpc.py（qldpc-log）：使用log函数对概率分布进行处理，期待使得值的区分度更好，有效效果不明显。

- approx_contraction_executor.py（normal）：由contraction_executor.py添加收缩近似得到。
- approx_contraction_executor_cpp_py.py(cpp-py)：由contraction_executor_cpp_py.py添加收缩近似得到。
- approx_contraction_executor.cpp(cpp)：由approx_contraction_executor_cpp_py修改为C++代码得到。 -->
