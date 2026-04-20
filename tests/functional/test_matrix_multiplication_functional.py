"""
test_matrix_multiplication.py
Functional tests for NumPy matrix multiplication operations.
Verifies correctness of np.dot, np.matmul / @ operator, and np.linalg.multi_dot
across various scenarios including different dtypes, shapes, edge cases,
broadcasting behaviour, and numerical accuracy.

Usage:
    pytest tests/functional/ -v
"""
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def square_int():
    """Simple 2x2 integer matrix with known multiplication result."""
    return np.array([[1, 2],
                     [3, 4]], dtype=np.int32)


@pytest.fixture
def square_float():
    """Simple 2x2 float64 matrix with known multiplication result."""
    return np.array([[1.0, 2.0],
                     [3.0, 4.0]], dtype=np.float64)


@pytest.fixture
def identity_3x3():
    """3x3 identity matrix — multiplication by it must be a no-op."""
    return np.eye(3, dtype=np.float64)


@pytest.fixture
def rect_A():
    """Rectangular matrix A shaped (3, 4) for non-square multiplication."""
    return np.arange(12, dtype=np.float64).reshape(3, 4)


@pytest.fixture
def rect_B():
    """Rectangular matrix B shaped (4, 2) compatible with rect_A."""
    return np.arange(8, dtype=np.float64).reshape(4, 2)


@pytest.fixture
def vector_1d():
    """1-D float64 vector of length 3."""
    return np.array([1.0, 2.0, 3.0], dtype=np.float64)


@pytest.fixture
def matrix_3x3():
    """3x3 float64 matrix for vector–matrix tests."""
    return np.arange(1, 10, dtype=np.float64).reshape(3, 3)


@pytest.fixture
def batch_matrices():
    """Stack of four 3x3 identity matrices shaped (4, 3, 3) for batched matmul."""
    return np.stack([np.eye(3, dtype=np.float64)] * 4)  # shape (4, 3, 3)


@pytest.fixture
def large_square():
    """Reproducible 100x100 random matrix for numerical accuracy checks."""
    rng = np.random.default_rng(seed=0)
    return rng.standard_normal((100, 100)).astype(np.float64)


@pytest.fixture
def complex_matrix():
    """2x2 complex128 matrix to verify dtype support."""
    return np.array([[1 + 2j, 3 + 4j],
                     [5 + 6j, 7 + 8j]], dtype=np.complex128)


# ---------------------------------------------------------------------------
# np.dot tests
# ---------------------------------------------------------------------------

class TestDot:
    # --- Basic correctness ---------------------------------------------------

    def test_dot_square_float_known_result(self, square_float):
        """np.dot of [[1,2],[3,4]] with itself must equal [[7,10],[15,22]]."""
        # Arrange
        A = square_float
        expected = np.array([[7.0, 10.0],
                             [15.0, 22.0]], dtype=np.float64)

        # Act
        result = np.dot(A, A)

        # Assert
        np.testing.assert_allclose(result, expected)

    def test_dot_rectangular_shape(self, rect_A, rect_B):
        """np.dot of (3,4) @ (4,2) must produce a (3,2) result."""
        # Arrange / Act
        result = np.dot(rect_A, rect_B)

        # Assert
        assert result.shape == (3, 2)

    def test_dot_rectangular_known_result(self):
        """Verify element values for a small rectangular multiplication."""
        # Arrange
        A = np.array([[1, 0], [0, 1], [2, 3]], dtype=np.float64)
        B = np.array([[4, 5, 6], [7, 8, 9]], dtype=np.float64)
        expected = np.array([[4, 5, 6],
                             [7, 8, 9],
                             [29, 34, 39]], dtype=np.float64)

        # Act
        result = np.dot(A, B)

        # Assert
        np.testing.assert_allclose(result, expected)

    def test_dot_identity_is_noop(self, matrix_3x3, identity_3x3):
        """Multiplying any matrix by the identity must return that matrix."""
        # Arrange
        A = matrix_3x3

        # Act
        result = np.dot(A, identity_3x3)

        # Assert
        np.testing.assert_allclose(result, A)

    def test_dot_by_zero_matrix_gives_zeros(self, square_float):
        """Multiplying by an all-zero matrix must yield an all-zero result."""
        # Arrange
        zeros = np.zeros_like(square_float)

        # Act
        result = np.dot(square_float, zeros)

        # Assert
        np.testing.assert_array_equal(result, zeros)

    def test_dot_1d_vectors_gives_scalar(self, vector_1d):
        """np.dot on two 1-D vectors must return the scalar inner product."""
        # Arrange / Act
        result = np.dot(vector_1d, vector_1d)

        # Assert — [1,2,3]·[1,2,3] = 1+4+9 = 14
        assert isinstance(result, (float, np.floating))
        assert result == pytest.approx(14.0)

    def test_dot_matrix_vector_shape(self, matrix_3x3, vector_1d):
        """np.dot of a (3,3) matrix with a (3,) vector must return a (3,) vector."""
        # Arrange / Act
        result = np.dot(matrix_3x3, vector_1d)

        # Assert
        assert result.shape == (3,)

    def test_dot_integer_dtype_preserved(self, square_int):
        """np.dot on integer arrays must keep an integer dtype."""
        # Arrange / Act
        result = np.dot(square_int, square_int)

        # Assert
        assert np.issubdtype(result.dtype, np.integer)

    def test_dot_float_result_dtype(self, square_float):
        """np.dot on float64 arrays must return a float64 result."""
        # Arrange / Act
        result = np.dot(square_float, square_float)

        # Assert
        assert result.dtype == np.float64

    def test_dot_complex_matrix(self, complex_matrix):
        """np.dot must handle complex128 inputs without raising."""
        # Arrange
        A = complex_matrix

        # Act
        result = np.dot(A, A)

        # Assert — result must have same complex dtype
        assert result.dtype == np.complex128
        assert result.shape == (2, 2)

    def test_dot_is_not_commutative(self, square_float):
        """Matrix multiplication is generally not commutative: A@B ≠ B@A."""
        # Arrange
        A = square_float
        B = np.array([[2.0, 0.0],
                      [1.0, 3.0]], dtype=np.float64)

        # Act
        AB = np.dot(A, B)
        BA = np.dot(B, A)

        # Assert — results have the same shape but different values
        assert AB.shape == BA.shape
        assert not np.array_equal(AB, BA)

    def test_dot_incompatible_shapes_raises(self):
        """np.dot on shape-incompatible matrices must raise ValueError."""
        # Arrange
        A = np.ones((3, 4), dtype=np.float64)
        B = np.ones((5, 2), dtype=np.float64)  # inner dims 4 ≠ 5

        # Act / Assert
        with pytest.raises(ValueError):
            np.dot(A, B)

    def test_dot_associativity(self, matrix_3x3):
        """Matrix multiplication must be associative: (A@B)@C == A@(B@C)."""
        # Arrange
        A = B = C = matrix_3x3

        # Act
        left = np.dot(np.dot(A, B), C)
        right = np.dot(A, np.dot(B, C))

        # Assert
        np.testing.assert_allclose(left, right, rtol=1e-10)

    def test_dot_large_matrix_numerical_accuracy(self, large_square):
        """A @ inv(A) must be close to the identity for a well-conditioned matrix."""
        # Arrange
        A = large_square
        A_inv = np.linalg.inv(A)

        # Act
        result = np.dot(A, A_inv)

        # Assert — off-diagonal elements must be very close to 0
        np.testing.assert_allclose(result, np.eye(100), atol=1e-8)


# ---------------------------------------------------------------------------
# np.matmul / @ operator tests
# ---------------------------------------------------------------------------

class TestMatmul:
    # --- Equivalence with @ operator -----------------------------------------

    def test_matmul_equals_at_operator(self, square_float):
        """np.matmul(A, B) must produce the same result as A @ B."""
        # Arrange
        A = square_float

        # Act
        via_function = np.matmul(A, A)
        via_operator = A @ A

        # Assert
        np.testing.assert_array_equal(via_function, via_operator)

    def test_matmul_equals_dot_for_2d(self, rect_A, rect_B):
        """For 2-D inputs, np.matmul must agree with np.dot."""
        # Arrange / Act
        result_matmul = np.matmul(rect_A, rect_B)
        result_dot = np.dot(rect_A, rect_B)

        # Assert
        np.testing.assert_array_equal(result_matmul, result_dot)

    # --- Basic correctness ---------------------------------------------------

    def test_matmul_square_known_result(self, square_float):
        """np.matmul of [[1,2],[3,4]] with itself must equal [[7,10],[15,22]]."""
        # Arrange
        A = square_float
        expected = np.array([[7.0, 10.0],
                             [15.0, 22.0]], dtype=np.float64)

        # Act
        result = np.matmul(A, A)

        # Assert
        np.testing.assert_allclose(result, expected)

    def test_matmul_identity_is_noop(self, matrix_3x3, identity_3x3):
        """Multiplying any matrix by the identity via @ must return that matrix."""
        # Arrange
        A = matrix_3x3

        # Act
        result = A @ identity_3x3

        # Assert
        np.testing.assert_allclose(result, A)

    def test_matmul_rectangular_output_shape(self, rect_A, rect_B):
        """np.matmul of (3,4) @ (4,2) must produce shape (3,2)."""
        # Arrange / Act
        result = np.matmul(rect_A, rect_B)

        # Assert
        assert result.shape == (3, 2)

    def test_matmul_matrix_vector(self, matrix_3x3, vector_1d):
        """np.matmul of (3,3) @ (3,) must return a (3,) vector."""
        # Arrange / Act
        result = np.matmul(matrix_3x3, vector_1d)

        # Assert
        assert result.shape == (3,)

    def test_matmul_known_matrix_vector_values(self):
        """Verify element values for a matrix–vector product."""
        # Arrange
        A = np.array([[2, 0], [1, 3]], dtype=np.float64)
        v = np.array([4.0, 5.0])
        expected = np.array([8.0, 19.0])

        # Act
        result = A @ v

        # Assert
        np.testing.assert_allclose(result, expected)

    # --- Batched (3-D) matmul ------------------------------------------------

    def test_matmul_batched_shape(self, batch_matrices, square_float):
        """np.matmul of (4,3,3) @ (3,3) must broadcast to (4,3,3)."""
        # Arrange
        batch = batch_matrices  # (4, 3, 3)
        M = np.eye(3, dtype=np.float64)

        # Act
        result = np.matmul(batch, M)

        # Assert
        assert result.shape == (4, 3, 3)

    def test_matmul_batched_identity_is_noop(self, batch_matrices):
        """Batched matmul of identity stacks with themselves must stay identity."""
        # Arrange
        batch = batch_matrices  # (4, 3, 3) — each slice is an identity

        # Act
        result = np.matmul(batch, batch)

        # Assert — result must still be all identity matrices
        for i in range(4):
            np.testing.assert_allclose(result[i], np.eye(3))

    def test_matmul_batch_known_result(self):
        """Verify element values for a small batched multiplication."""
        # Arrange
        A = np.array([[[1, 2], [3, 4]],
                      [[5, 6], [7, 8]]], dtype=np.float64)  # (2,2,2)
        B = np.eye(2, dtype=np.float64)
        expected = A  # multiplying by identity returns self

        # Act
        result = np.matmul(A, B)

        # Assert
        np.testing.assert_allclose(result, expected)

    # --- Error cases ---------------------------------------------------------

    def test_matmul_scalar_raises(self):
        """np.matmul must raise ValueError when given scalar (0-D) inputs."""
        # Arrange
        scalar = np.float64(3.0)

        # Act / Assert
        with pytest.raises(ValueError):
            np.matmul(scalar, scalar)

    def test_matmul_incompatible_shapes_raises(self):
        """np.matmul on shape-incompatible matrices must raise ValueError."""
        # Arrange
        A = np.ones((3, 4), dtype=np.float64)
        B = np.ones((5, 2), dtype=np.float64)  # inner dims 4 ≠ 5

        # Act / Assert
        with pytest.raises(ValueError):
            np.matmul(A, B)

    # --- Dtype handling ------------------------------------------------------

    def test_matmul_integer_dtype(self, square_int):
        """np.matmul on integer arrays must preserve an integer dtype."""
        # Arrange / Act
        result = np.matmul(square_int, square_int)

        # Assert
        assert np.issubdtype(result.dtype, np.integer)

    def test_matmul_complex_dtype(self, complex_matrix):
        """np.matmul on complex128 inputs must return a complex128 result."""
        # Arrange / Act
        result = np.matmul(complex_matrix, complex_matrix)

        # Assert
        assert result.dtype == np.complex128

    # --- Numerical accuracy --------------------------------------------------

    def test_matmul_large_matrix_accuracy(self, large_square):
        """A @ inv(A) must be close to the identity for a well-conditioned matrix."""
        # Arrange
        A = large_square
        A_inv = np.linalg.inv(A)

        # Act
        result = A @ A_inv

        # Assert
        np.testing.assert_allclose(result, np.eye(100), atol=1e-8)

    def test_matmul_transpose_identity(self, large_square):
        """For any matrix A, (A @ A.T) must be symmetric."""
        # Arrange
        A = large_square

        # Act
        result = A @ A.T

        # Assert — symmetric: result == result.T
        np.testing.assert_allclose(result, result.T, atol=1e-10)


# ---------------------------------------------------------------------------
# np.linalg.multi_dot tests
# ---------------------------------------------------------------------------

class TestMultiDot:
    def test_multi_dot_two_matrices_matches_dot(self, rect_A, rect_B):
        """multi_dot of two matrices must match np.dot."""
        # Arrange / Act
        result_multi = np.linalg.multi_dot([rect_A, rect_B])
        result_dot = np.dot(rect_A, rect_B)

        # Assert
        np.testing.assert_allclose(result_multi, result_dot)

    def test_multi_dot_three_matrices_shape(self):
        """multi_dot of (2,3) @ (3,4) @ (4,2) must produce shape (2,2)."""
        # Arrange
        A = np.ones((2, 3), dtype=np.float64)
        B = np.ones((3, 4), dtype=np.float64)
        C = np.ones((4, 2), dtype=np.float64)

        # Act
        result = np.linalg.multi_dot([A, B, C])

        # Assert
        assert result.shape == (2, 2)

    def test_multi_dot_three_matrices_known_result(self):
        """Verify element values for a chained 3-matrix product."""
        # Arrange
        A = np.array([[1, 2], [3, 4]], dtype=np.float64)
        B = np.eye(2, dtype=np.float64)
        C = np.array([[2, 0], [0, 2]], dtype=np.float64)
        expected = np.dot(np.dot(A, B), C)  # reference via ordinary dot

        # Act
        result = np.linalg.multi_dot([A, B, C])

        # Assert
        np.testing.assert_allclose(result, expected)

    def test_multi_dot_matches_sequential_dot(self, matrix_3x3):
        """multi_dot chain must equal left-to-right sequential np.dot calls."""
        # Arrange
        A = B = C = matrix_3x3
        sequential = np.dot(np.dot(A, B), C)

        # Act
        result = np.linalg.multi_dot([A, B, C])

        # Assert
        np.testing.assert_allclose(result, sequential, rtol=1e-10)

    def test_multi_dot_with_vectors(self, vector_1d, matrix_3x3):
        """multi_dot must support leading/trailing 1-D vectors."""
        # Arrange
        v = vector_1d  # (3,)
        M = matrix_3x3  # (3,3)

        # Act — v @ M @ v should yield a scalar
        result = np.linalg.multi_dot([v, M, v])

        # Assert
        assert result.ndim == 0  # scalar output
