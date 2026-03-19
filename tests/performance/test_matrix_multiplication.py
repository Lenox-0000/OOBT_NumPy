"""
test_matrix_multiplication.py

Benchmarks NumPy matrix multiplication performance using pytest-benchmark.

Usage:
    pytest tests/performance/ -v --benchmark-autosave
"""

import pytest
import numpy as np

MATRIX_SIZES = [100, 500, 1000, 2000]


@pytest.fixture(params=MATRIX_SIZES, ids=[f"{s}x{s}" for s in MATRIX_SIZES])
def matrix_pair(request):
    """
    Fixture that generates a pair of random square float64 matrices
    for each size defined in MATRIX_SIZES.
    """
    size = request.param
    A = np.random.rand(size, size).astype(np.float64)
    B = np.random.rand(size, size).astype(np.float64)
    return A, B


def test_matrix_multiplication(benchmark, matrix_pair):
    """
    Benchmarks np.dot() matrix multiplication across multiple matrix sizes.
    pytest-benchmark handles warm-up, multiple rounds, and result export.
    """
    A, B = matrix_pair

    result = benchmark(np.dot, A, B)

    # Sanity check — result shape should match input dimensions
    assert result.shape == A.shape
