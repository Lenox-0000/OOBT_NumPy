"""
test_matrix_multiplication.py

Benchmarks NumPy matrix multiplication performance using pytest-benchmark.
Covers performance test scenario 2.1 (Matrix Multiplication Benchmark).

Usage:
    pytest tests/performance/ -v --benchmark-autosave
"""

import pytest
import numpy as np

# The primary acceptance scenario (2.1) targets 1000x1000.
# Additional sizes are included to track regression across the full range.
MATRIX_SIZES = [
    (100, 100),
    (500, 500),
    (1000, 1000),   # Primary scenario-2.1 size; regression baseline is tracked here
    (2000, 2000),
    (100, 1000),    # Rectangular: wide
    (1000, 100),    # Rectangular: tall
]


@pytest.fixture(params=MATRIX_SIZES, ids=[f"{r}x{c}" for r, c in MATRIX_SIZES])
def matrix_pair(request):
    """
    Generates a pair of random float64 matrices for each parametrised size.

    B is shaped so that A @ B is always valid and square (cols_A == rows_B,
    cols_B == rows_A), matching the shape assertion in the test body.
    Seed is fixed for reproducibility across runs and baseline comparisons.
    """
    rows, cols = request.param
    rng = np.random.default_rng(seed=42)
    A = rng.random((rows, cols)).astype(np.float64)
    B = rng.random((cols, rows)).astype(np.float64)
    return A, B


def test_matrix_multiplication(benchmark, matrix_pair):
    """
    Scenario 2.1 – NumPy matmul benchmark.

    Benchmarks np.dot() across multiple matrix sizes and shapes.
    pytest-benchmark handles warm-up, multiple rounds, and JSON export
    to benchmark_results.json (pipeline artifact).

    Acceptance threshold: no unhandled exception; results saved and visible
    in the pipeline log (mean time reported per size by the summary step).
    """
    A, B = matrix_pair

    result = benchmark(np.dot, A, B)

    # Shape correctness: result must be (rows_A, rows_A) because B is (cols_A, rows_A)
    assert result.shape == (A.shape[0], A.shape[0]), (
        f"Expected output shape {(A.shape[0], A.shape[0])}, got {result.shape}"
    )
    # Dtype must be preserved as float64 (no silent downcast)
    assert result.dtype == np.float64, (
        f"Expected float64 output dtype, got {result.dtype}"
    )
    # Sanity-check: result must contain only finite values
    assert np.all(np.isfinite(result)), "Result contains non-finite values (NaN or Inf)"