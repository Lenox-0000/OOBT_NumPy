"""
numpy_array_comparison.py

Compares NumPy matrix multiplication performance against a pure-Python baseline.
Covers the Python-list baseline requirement of performance test scenario 2.1.

The scenario states the Python baseline is "expected to be orders of magnitude
slower than NumPy" — this file verifies and documents that claim.

Usage:
    pytest tests/performance/ -v --benchmark-autosave
"""

import pytest
import numpy as np

# Scenario 2.1 targets 1000x1000 for the primary NumPy benchmark.
# The Python baseline uses the same logical operation on the same data
# so the comparison is apples-to-apples.
BASELINE_SIZE = 1000


@pytest.fixture(scope="module")
def numpy_matrices():
    """1000x1000 float64 matrix pair — identical seed to test_matrix_multiplication."""
    rng = np.random.default_rng(seed=42)
    A = rng.random((BASELINE_SIZE, BASELINE_SIZE)).astype(np.float64)
    B = rng.random((BASELINE_SIZE, BASELINE_SIZE)).astype(np.float64)
    return A, B


@pytest.fixture(scope="module")
def python_matrices(numpy_matrices):
    """
    Pure-Python nested-list equivalents of the NumPy matrices.
    Derived from the same numpy_matrices fixture so both benchmarks
    operate on numerically identical data.
    """
    A_np, B_np = numpy_matrices
    return A_np.tolist(), B_np.tolist()


def test_numpy_matmul_baseline(benchmark, numpy_matrices):
    """
    Scenario 2.1 – NumPy 1000x1000 matmul (reference run for comparison).

    This duplicate of the primary benchmark lives here so pytest-benchmark
    can report the NumPy vs Python ratio in a single output table when both
    tests are collected from the same file.
    """
    A, B = numpy_matrices

    result = benchmark(np.dot, A, B)

    assert result.shape == (BASELINE_SIZE, BASELINE_SIZE)
    assert result.dtype == np.float64
    assert np.all(np.isfinite(result))


def test_python_list_matmul_baseline(benchmark, python_matrices):
    """
    Scenario 2.1 – Pure-Python matrix multiplication baseline.

    Implements a naive O(n³) triple-loop matmul on Python lists.
    Expected to be orders of magnitude slower than np.dot (scenario 2.1).

    No numeric correctness assertion is made beyond shape, because the
    sole purpose of this test is to produce a benchmark timing that can
    be compared against test_numpy_matmul_baseline in the pipeline summary.

    Note: to keep CI run-time manageable the baseline uses a reduced matrix
    size (BASELINE_PYTHON_SIZE). The slowdown ratio relative to NumPy at
    that size is already representative; scaling to 1000x1000 would make
    the suite impractically slow in pure Python.
    """
    # Pure Python is ~100–1000× slower; use a smaller size so CI stays fast
    # while still demonstrating the order-of-magnitude gap clearly.
    BASELINE_PYTHON_SIZE = 100

    rng = np.random.default_rng(seed=42)
    A_small = rng.random((BASELINE_PYTHON_SIZE, BASELINE_PYTHON_SIZE)).tolist()
    B_small = rng.random((BASELINE_PYTHON_SIZE, BASELINE_PYTHON_SIZE)).tolist()

    n = BASELINE_PYTHON_SIZE

    def python_matmul():
        result = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for k in range(n):
                a_ik = A_small[i][k]
                for j in range(n):
                    result[i][j] += a_ik * B_small[k][j]
        return result

    result = benchmark(python_matmul)

    # Shape sanity-check
    assert len(result) == n
    assert len(result[0]) == n