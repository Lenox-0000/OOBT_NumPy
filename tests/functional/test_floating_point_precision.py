"""
Module: test_floating_point_precision.py
Purpose: Validate NumPy's handling of floating-point arithmetic, precision limits,
         and accumulation errors across different data types (float32, float64).
Scope: Covers machine epsilon limits, catastrophic cancellation, accumulation
       stability in multidimensional arrays, and memory-layout preservation during maths.

Usage:
    pytest test_floating_point_precision.py -v
"""

import pytest
import numpy as np


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def f_contiguous_3d_float64():
    """
    Provides a 3D Fortran-contiguous array of float64 to ensure complex
    memory layouts are tested for mathematical and memory stability.
    """
    # Create a 2x3x4 array filled with small fractional values
    arr = np.linspace(0.1, 1.0, 24, dtype=np.float64).reshape((2, 3, 4), order='F')
    return arr


@pytest.fixture
def scalar_like_float32():
    """Provides a zero-dimensional (scalar-like) float32 array."""
    return np.array(0.123456789, dtype=np.float32)


@pytest.fixture
def catastrophic_cancellation_data():
    """Provides two very close float64 arrays to test precision loss on subtraction."""
    a = np.array([1.000000000000001, 1000000.000000001], dtype=np.float64)
    b = np.array([1.000000000000000, 1000000.000000000], dtype=np.float64)
    return a, b


# -----------------------------------------------------------------------------
# Test Precision Basics
# -----------------------------------------------------------------------------

class TestPrecisionBasics:
    """Groups test cases related to fundamental floating-point type limits."""

    # Machine Epsilon and Resolution ---

    def test_float32_truncation(self, scalar_like_float32):
        """
        Validates that float32 downcasting intrinsically truncates precision
        beyond the 7th decimal place compared to theoretical true values.
        """
        # Arrange
        # The float32 machine epsilon is ~1.192e-07. Digits beyond this limit are lost.
        true_value = 0.123456789

        # Act
        actual_val = scalar_like_float32.item()

        # Assert
        # Check that the difference exceeds the float64 epsilon but is within float32 limits
        assert abs(true_value - actual_val) > np.finfo(np.float64).eps
        np.testing.assert_allclose(actual_val, true_value, rtol=1e-6)

        # Memory-level check for edge case: ensure it is strictly 0-dimensional
        assert scalar_like_float32.ndim == 0

    # Subtraction and Tolerance ---

    def test_catastrophic_cancellation(self, catastrophic_cancellation_data):
        """
        Ensures that operations known to cause catastrophic cancellation
        retain accuracy within expected float64 absolute tolerances.
        """
        # Arrange
        arr_a, arr_b = catastrophic_cancellation_data
        # We expect the difference to be roughly 1e-15, but the second element
        # suffers from scaling magnitude precision loss (expected ~1e-9 difference).
        expected = np.array([1e-15, 1e-9])

        # Act
        result = arr_a - arr_b

        # Assert
        # Because we subtract nearly equal numbers, relative error blows up.
        # Therefore, rtol is forced to 0.0, and we rely purely on an absolute tolerance.
        np.testing.assert_allclose(result, expected, rtol=0.0, atol=1e-10)


# -----------------------------------------------------------------------------
# Accumulation and Memory Layout
# -----------------------------------------------------------------------------

class TestAccumulationAndMemory:
    """Groups test cases evaluating precision over aggregations and memory layouts."""

    # Multidimensional Accumulation

    def test_fortran_layout_summation_stability(self, f_contiguous_3d_float64):
        """
        Contracts that summing a 3D F-contiguous array along its first axis yields
        the correct precision and preserves the F-contiguous memory layout for
        the remaining dimensions.
        """
        # Arrange
        arr = f_contiguous_3d_float64
        # Magic number derivation: the analytical sum of linspace(0.1, 1.0, 24) is exactly 13.2
        # We test summation over axis 0, which has dimension 2.

        # Act
        result = np.sum(arr, axis=0)

        # Assert
        # 1. Exactness and Tolerances
        assert result.shape == (3, 4)
        np.testing.assert_allclose(result.sum(), 13.2, rtol=1e-14)

        # 2. Memory-level validation
        assert arr.flags.f_contiguous is True

        # Aggregation over axis 0 of an F-contiguous array preserves the
        # relative stride ordering of the remaining axes (16, 48),
        # resulting in an F-contiguous 2D array.
        assert result.flags.f_contiguous is True
        assert result.flags.c_contiguous is False
        assert not np.shares_memory(arr, result)

    def test_view_precision_sharing(self, f_contiguous_3d_float64):
        """
        Validates that taking a flattened view of an array preserves exact
        floating-point representation and successfully shares underlying memory.
        """
        # Arrange
        original = f_contiguous_3d_float64

        # Act
        # ravel('K') reads elements in the order they occur in memory,
        # ensuring no copy is made even for Fortran arrays.
        view = original.ravel(order='K')

        # Assert
        # 1. Exact precision check using array_equal
        np.testing.assert_array_equal(view.reshape((2, 3, 4), order='F'), original)

        # 2. Memory-level validation
        assert np.shares_memory(original, view) is True

    # Edge Cases and Exceptions

    def test_invalid_floating_operations_raise(self):
        """
        Verifies that strict floating-point error modes trigger a FloatingPointError
        when mathematically undefined operations (like 0.0 / 0.0) are attempted.
        """
        # Arrange
        zeros = np.zeros(3, dtype=np.float64)

        # Act & Assert
        # Temporarily strictly enforce floating point error raising
        with np.errstate(invalid='raise'):
            with pytest.raises(FloatingPointError, match="invalid value encountered"):
                # Attempting 0.0 / 0.0 generates a NaN and triggers the exception
                _ = zeros / zeros
