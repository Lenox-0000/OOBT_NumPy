"""
test_matrix_multiplication.py
Benchmarks NumPy matrix multiplication performance using pytest-benchmark.
Usage:
    pytest tests/performance/ -v --benchmark-autosave
"""
import pytest
import numpy as np

MATRIX_SIZES = [
    (100, 100),
    (500, 500),
    (1000, 1000),
    (2000, 2000),
    (100, 1000),  
    (1000, 100),   
]

@pytest.fixture(params=MATRIX_SIZES, ids=[f"{r}x{c}" for r, c in MATRIX_SIZES])
def matrix_pair(request):
    """
    Fixture that generates a pair of random float64 matrices for each size.
    Supports both square and rectangular shapes to test non-square dot products.
    Seed is fixed for reproducibility across runs.
    """
    rows, cols = request.param
    rng = np.random.default_rng(seed=42)
    A = rng.random((rows, cols)).astype(np.float64)
    B = rng.random((cols, rows)).astype(np.float64)  
    return A, B

def test_matrix_multiplication(benchmark, matrix_pair):
    """
    Benchmarks np.dot() matrix multiplication across multiple matrix sizes and shapes.
    pytest-benchmark handles warm-up, multiple rounds, and result export.
    """
    A, B = matrix_pair
    result = benchmark(np.dot, A, B)
    assert result.shape == (A.shape[0], A.shape[0])