"""
numpy_array_comparison.py

Compares the performance of NumPy and native Python arrays using a list of 100000 elements.

Usage:
    pytest tests/performance/ -v --benchmark-autosave
"""

import pytest
import numpy as np

ARRAY_SIZE = 100_000


@pytest.fixture
def python_list():
    return list(range(ARRAY_SIZE))


@pytest.fixture
def numpy_array():
    return np.arange(ARRAY_SIZE)


def test_list_elementwise_math(benchmark, python_list):
    """
    Benchmarks a simple addition operation across 100k elements
    using a standard Python list comprehension.
    """

    def list_ops():
        return [x + 1 for x in python_list]

    benchmark(list_ops)


def test_numpy_elementwise_math(benchmark, numpy_array):
    """
    Benchmarks the same addition operation using NumPy's
    vectorized operations (optimized C-code).
    """

    def numpy_ops():
        return numpy_array + 1

    benchmark(numpy_ops)
