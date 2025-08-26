# Contraction Executor

Several test versions and corresponding C++ versions are provided internally, allowing users to choose according to their needs.

Note that the C++ versions of the contraction executors need to be compiled by the user:

```bash
g++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) contraction_executor.cpp -o contraction_executor_cpp.so

g++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) approx_contraction_executor.cpp -o approx_contraction_executor_cpp.so
```

Additionally, to use the C++ versions, please **uncomment the relevant lines in the `__init__.py` file**.