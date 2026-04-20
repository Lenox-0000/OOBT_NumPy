"""
test_broadcasting_functional.py
Functional tests for NumPy broadcasting rules.
Verifies the ability of NumPy to perform arithmetic operations on 
arrays of different but compatible shapes, including scalars, 
1D-to-2D, and multi-dimensional broadcasting.

Usage:
    pytest tests/functional/test_broadcasting_functional.py -v
"""
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scalar():
    """Simple scalar value."""
    return 5.0


@pytest.fixture
def array_1d():
    """1-D array of shape (3,)."""
    return np.array([1.0, 2.0, 3.0], dtype=np.float64)


@pytest.fixture
def matrix_2d():
    """2-D array of shape (2, 3)."""
    return np.array([[10, 20, 30],
                     [40, 50, 60]], dtype=np.float64)


@pytest.fixture
def col_vector():
    """2-D column vector of shape (2, 1)."""
    return np.array([[1],
                     [2]], dtype=np.float64)


@pytest.fixture
def tensor_3d():
    """3-D array of shape (2, 2, 3)."""
    return np.arange(12, dtype=np.float64).reshape(2, 2, 3)


# ---------------------------------------------------------------------------
# Broadcasting Tests
# ---------------------------------------------------------------------------

class TestBasicBroadcasting:
    """Verifies fundamental broadcasting: scalars and 1D-to-2D."""

    def test_scalar_broadcast_to_2d(self, matrix_2d, scalar):
        """A scalar should be added to every element of a 2D matrix."""
        # Act
        result = matrix_2d + scalar

        # Assert
        expected = np.array([[15.0, 25.0, 35.0],
                             [45.0, 55.0, 65.0]])
        np.testing.assert_allclose(result, expected)

    def test_1d_to_2d_row_broadcast(self, matrix_2d, array_1d):
        """A (3,) array should broadcast across a (2, 3) matrix (row-wise)."""
        # Act
        result = matrix_2d + array_1d

        # Assert
        expected = np.array([[11.0, 22.0, 33.0],
                             [41.0, 52.0, 63.0]])
        np.testing.assert_allclose(result, expected)

    def test_column_vector_to_2d_broadcast(self, matrix_2d, col_vector):
        """A (2, 1) array should broadcast across a (2, 3) matrix (column-wise)."""
        # Act
        result = matrix_2d * col_vector

        # Assert
        # Row 0 multiplied by 1, Row 1 multiplied by 2
        expected = np.array([[10.0, 20.0, 30.0],
                             [80.0, 100.0, 120.0]])
        np.testing.assert_allclose(result, expected)


class TestMultiDimensionalBroadcasting:
    """Verifies broadcasting in higher dimensions and complex shape stretching."""

    def test_2d_to_3d_broadcasting(self, tensor_3d, matrix_2d):
        """Broadcasting (2, 3) into (2, 2, 3) tensor."""
        # Act
        result = tensor_3d + matrix_2d

        # Assert
        # tensor_3d[0] is (2,3), tensor_3d[1] is (2,3)
        # matrix_2d is (2,3)
        assert result.shape == (2, 2, 3)
        # Check a specific slice to ensure logic holds
        np.testing.assert_allclose(result[0, 0, :], tensor_3d[0, 0, :] + matrix_2d[0, :])

    def test_trailing_dimensions_rule(self):
        """Broadcasting (4, 1) and (3,) to produce (4, 3)."""
        # Arrange
        a = np.arange(4).reshape(4, 1)
        b = np.arange(3)

        # Act
        result = a + b

        # Assert
        assert result.shape == (4, 3)
        for i in range(4):
            for j in range(3):
                assert result[i, j] == a[i, 0] + b[j]


class TestBroadcastingConstraints:
    """Verifies that NumPy correctly rejects incompatible shapes."""

    def test_incompatible_shapes_raises_value_error(self, matrix_2d):
        """Attempting to broadcast (2, 3) and (2, 2) must raise ValueError."""
        # Arrange
        bad_shape_matrix = np.ones((2, 2))

        # Act / Assert
        with pytest.raises(ValueError, match="operands could not be broadcast together"):
            _ = matrix_2d + bad_shape_matrix

    def test_incompatible_1d_broadcast(self, array_1d):
        """Attempting to broadcast (3,) with (4,) must fail."""
        # Arrange
        other_1d = np.array([1, 2, 3, 4])

        # Act / Assert
        with pytest.raises(ValueError):
            _ = array_1d + other_1d


class TestInPlaceBroadcasting:
    """Verifies that in-place operations also follow broadcasting rules."""

    def test_inplace_addition_broadcast(self, matrix_2d, array_1d):
        """Using += with a broadcastable array."""
        # Act
        matrix_2d += array_1d

        # Assert
        expected = np.array([[11.0, 22.0, 33.0],
                             [41.0, 52.0, 63.0]])
        np.testing.assert_allclose(matrix_2d, expected)

    def test_inplace_invalid_broadcast_raises(self, matrix_2d):
        """In-place operations cannot change the shape of the original array."""
        # Arrange
        # While (3,1) can broadcast with (2,3) to make a (2,3), 
        # (3,1) cannot be broadcast INTO a (2,3) via += because the result 
        # must fit in the original (2,3) shape, but the axes must match exactly 
        # or be 1.
        col_vector_bad = np.array([[1], [2], [3]])  # (3, 1)

        # Act / Assert
        with pytest.raises(ValueError):
            matrix_2d += col_vector_bad
