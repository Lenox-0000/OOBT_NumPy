"""
test_3d_dimension_manipulation.py
Functional tests for NumPy 3D dimension manipulation operations.
Verifies correctness of np.reshape, np.transpose, and memory layout
behaviour (C-order vs Fortran-order, contiguity, views vs copies)
across various shapes, dtypes, and edge cases.

Usage:
    pytest tests/functional/ -v
"""
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_24():
    """1-D array of 24 sequential integers — reshapable into many 3-D forms."""
    return np.arange(24, dtype=np.float64)


@pytest.fixture
def cube_3d():
    """3-D array shaped (2, 3, 4) with sequential float values."""
    return np.arange(24, dtype=np.float64).reshape(2, 3, 4)


@pytest.fixture
def square_3d():
    """Cubic 3-D array shaped (3, 3, 3) with sequential float values."""
    return np.arange(27, dtype=np.float64).reshape(3, 3, 3)


@pytest.fixture
def integer_3d():
    """3-D int32 array shaped (2, 3, 4) to verify dtype preservation."""
    return np.arange(24, dtype=np.int32).reshape(2, 3, 4)


@pytest.fixture
def fortran_3d():
    """3-D array allocated in Fortran (column-major) memory order."""
    return np.asfortranarray(np.arange(24, dtype=np.float64).reshape(2, 3, 4))


@pytest.fixture
def ones_3d():
    """3-D array of ones shaped (4, 5, 6) — used for shape/size checks."""
    return np.ones((4, 5, 6), dtype=np.float64)


# ---------------------------------------------------------------------------
# np.reshape tests
# ---------------------------------------------------------------------------

class TestReshape:
    # --- Shape correctness ---------------------------------------------------

    def test_reshape_1d_to_3d(self, flat_24):
        """Reshaping a flat 24-element array into (2, 3, 4) must succeed."""
        # Arrange
        arr = flat_24

        # Act
        result = arr.reshape(2, 3, 4)

        # Assert
        assert result.shape == (2, 3, 4)

    def test_reshape_3d_to_1d(self, cube_3d):
        """Flattening a (2, 3, 4) array back to 1-D must restore all 24 elements."""
        # Arrange
        arr = cube_3d

        # Act
        result = arr.reshape(-1)

        # Assert
        assert result.shape == (24,)
        np.testing.assert_array_equal(result, np.arange(24, dtype=np.float64))

    def test_reshape_3d_to_2d(self, cube_3d):
        """Reshaping (2, 3, 4) to (6, 4) must halve the leading dimensions."""
        # Arrange
        arr = cube_3d

        # Act
        result = arr.reshape(6, 4)

        # Assert
        assert result.shape == (6, 4)

    def test_reshape_with_minus_one_inference(self, cube_3d):
        """Using -1 in reshape must let NumPy infer the missing dimension."""
        # Arrange
        arr = cube_3d

        # Act
        result = arr.reshape(4, -1, 2)

        # Assert — 24 elements / (4 * 2) = 3
        assert result.shape == (4, 3, 2)

    def test_reshape_preserves_total_elements(self, cube_3d):
        """Element count must be identical before and after reshape."""
        # Arrange
        arr = cube_3d

        # Act
        result = arr.reshape(4, 2, 3)

        # Assert
        assert result.size == arr.size

    def test_reshape_preserves_dtype(self, integer_3d):
        """Reshape must not silently cast the array's dtype."""
        # Arrange
        arr = integer_3d

        # Act
        result = arr.reshape(4, 6)

        # Assert
        assert result.dtype == np.int32

    def test_reshape_preserves_element_order(self, flat_24):
        """Element values must follow C (row-major) order after reshape."""
        # Arrange
        arr = flat_24

        # Act
        result = arr.reshape(2, 3, 4)

        # Assert — first row of first slice should be [0, 1, 2, 3]
        np.testing.assert_array_equal(result[0, 0, :], [0.0, 1.0, 2.0, 3.0])

    def test_reshape_invalid_size_raises(self, cube_3d):
        """Reshape to an incompatible size must raise ValueError."""
        # Arrange
        arr = cube_3d

        # Act / Assert
        with pytest.raises(ValueError):
            arr.reshape(5, 5, 5)   # 125 ≠ 24

    def test_reshape_single_element(self):
        """A scalar-like array must reshape to (1, 1, 1) without error."""
        # Arrange
        arr = np.array([42.0])

        # Act
        result = arr.reshape(1, 1, 1)

        # Assert
        assert result.shape == (1, 1, 1)
        assert result[0, 0, 0] == pytest.approx(42.0)

    # --- Memory layout -------------------------------------------------------

    def test_reshape_returns_view_when_possible(self, flat_24):
        """Reshape of a contiguous C-order array should return a view, not a copy."""
        # Arrange
        arr = flat_24

        # Act
        result = arr.reshape(2, 3, 4)

        # Assert — modifying result must affect arr (shared memory)
        assert np.shares_memory(arr, result)

    def test_reshape_c_order(self, flat_24):
        """Explicit order='C' must iterate last axis fastest."""
        # Arrange
        arr = flat_24

        # Act
        result = arr.reshape(2, 3, 4, order="C")

        # Assert — elements 0-3 lie along the innermost axis
        np.testing.assert_array_equal(result[0, 0, :], [0.0, 1.0, 2.0, 3.0])

    def test_reshape_f_order(self, flat_24):
        """Explicit order='F' must iterate first axis fastest."""
        # Arrange
        arr = flat_24

        # Act
        result = arr.reshape(2, 3, 4, order="F")

        # Assert — elements 0 and 1 lie along the first axis
        assert result[0, 0, 0] == pytest.approx(0.0)
        assert result[1, 0, 0] == pytest.approx(1.0)

    def test_reshape_result_is_c_contiguous_by_default(self, flat_24):
        """A default (C-order) reshape must produce a C-contiguous array."""
        # Arrange / Act
        result = flat_24.reshape(2, 3, 4)

        # Assert
        assert result.flags["C_CONTIGUOUS"]

    def test_reshape_copy_on_non_contiguous_input(self, cube_3d):
        """Reshape of a non-contiguous slice must produce a copy, not a view."""
        # Arrange — slicing with step breaks contiguity
        arr = cube_3d[:, ::2, :]  # shape (2, 2, 4), non-contiguous

        # Act
        result = arr.reshape(4, 4)

        # Assert — must not share memory with the original cube
        assert not np.shares_memory(cube_3d, result)


# ---------------------------------------------------------------------------
# np.transpose tests
# ---------------------------------------------------------------------------

class TestTranspose:
    # --- Shape correctness ---------------------------------------------------

    def test_transpose_default_reverses_axes(self, cube_3d):
        """Default transpose of (2, 3, 4) must produce (4, 3, 2)."""
        # Arrange
        arr = cube_3d

        # Act
        result = arr.transpose()

        # Assert
        assert result.shape == (4, 3, 2)

    def test_transpose_custom_axes(self, cube_3d):
        """Transposing with axes=(1, 2, 0) must permute shape accordingly."""
        # Arrange
        arr = cube_3d  # (2, 3, 4)

        # Act
        result = arr.transpose(1, 2, 0)

        # Assert — (3, 4, 2)
        assert result.shape == (3, 4, 2)

    def test_transpose_identity_axes(self, cube_3d):
        """Transposing with axes=(0, 1, 2) must return an identical-shape array."""
        # Arrange
        arr = cube_3d

        # Act
        result = arr.transpose(0, 1, 2)

        # Assert
        assert result.shape == arr.shape
        np.testing.assert_array_equal(result, arr)

    def test_transpose_twice_is_identity(self, cube_3d):
        """Applying default transpose twice must recover the original array."""
        # Arrange
        arr = cube_3d

        # Act
        result = arr.transpose().transpose()

        # Assert
        assert result.shape == arr.shape
        np.testing.assert_array_equal(result, arr)

    def test_transpose_preserves_values(self, cube_3d):
        """Every element must be reachable under permuted indices after transpose."""
        # Arrange
        arr = cube_3d  # (2, 3, 4)

        # Act
        result = arr.transpose(1, 2, 0)  # (3, 4, 2)

        # Assert — arr[i, j, k] == result[j, k, i]
        for i in range(2):
            for j in range(3):
                for k in range(4):
                    assert arr[i, j, k] == result[j, k, i]

    def test_transpose_preserves_dtype(self, integer_3d):
        """Transpose must not alter the array's dtype."""
        # Arrange
        arr = integer_3d

        # Act
        result = arr.transpose()

        # Assert
        assert result.dtype == np.int32

    def test_transpose_preserves_element_count(self, cube_3d):
        """Total number of elements must be unchanged after transpose."""
        # Arrange
        arr = cube_3d

        # Act
        result = arr.transpose()

        # Assert
        assert result.size == arr.size

    def test_transpose_square_cube_diagonal_preserved(self, square_3d):
        """Space diagonal of a cubic array must be unchanged under default transpose."""
        # Arrange
        arr = square_3d  # (3, 3, 3)

        # Act
        result = arr.transpose()

        # Assert — diagonal elements satisfy arr[i,i,i] == result[i,i,i]
        for i in range(3):
            assert arr[i, i, i] == result[i, i, i]

    def test_np_transpose_function_equals_method(self, cube_3d):
        """np.transpose(arr) and arr.transpose() must produce identical results."""
        # Arrange
        arr = cube_3d

        # Act
        via_function = np.transpose(arr)
        via_method = arr.transpose()

        # Assert
        np.testing.assert_array_equal(via_function, via_method)

    def test_transpose_invalid_axes_raises(self, cube_3d):
        """Passing an out-of-range axis index must raise ValueError or AxisError."""
        # Arrange
        arr = cube_3d

        # Act / Assert
        with pytest.raises((ValueError, np.exceptions.AxisError)):
            arr.transpose(0, 1, 5)  # axis 5 does not exist for ndim=3

    # --- Memory layout -------------------------------------------------------

    def test_transpose_returns_view(self, cube_3d):
        """Transpose must return a view sharing memory with the original."""
        # Arrange
        arr = cube_3d

        # Act
        result = arr.transpose()

        # Assert
        assert np.shares_memory(arr, result)

    def test_transpose_is_not_c_contiguous(self, cube_3d):
        """A transposed 3-D array is generally not C-contiguous."""
        # Arrange
        arr = cube_3d

        # Act
        result = arr.transpose()

        # Assert
        assert not result.flags["C_CONTIGUOUS"]

    def test_transpose_is_f_contiguous(self, cube_3d):
        """Default transpose of a C-contiguous array should be F-contiguous."""
        # Arrange
        arr = cube_3d

        # Act
        result = arr.transpose()

        # Assert
        assert result.flags["F_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# Memory layout tests (C-order vs Fortran-order, strides, contiguity)
# ---------------------------------------------------------------------------

class TestMemoryLayout:
    # --- Contiguity flags after construction ---------------------------------

    def test_c_order_array_is_c_contiguous(self, cube_3d):
        """An array created with default (C) order must be C-contiguous."""
        assert cube_3d.flags["C_CONTIGUOUS"]
        assert not cube_3d.flags["F_CONTIGUOUS"]

    def test_f_order_array_is_f_contiguous(self, fortran_3d):
        """An array created with Fortran order must be F-contiguous."""
        assert fortran_3d.flags["F_CONTIGUOUS"]
        assert not fortran_3d.flags["C_CONTIGUOUS"]

    # --- Stride relationships ------------------------------------------------

    def test_c_order_strides_decrease(self, cube_3d):
        """C-order strides must decrease from axis 0 to the last axis."""
        strides = cube_3d.strides
        assert strides[0] > strides[1] > strides[2]

    def test_f_order_strides_increase(self, fortran_3d):
        """Fortran-order strides must increase from axis 0 to the last axis."""
        strides = fortran_3d.strides
        assert strides[0] < strides[1] < strides[2]

    def test_strides_consistent_with_dtype_itemsize(self, cube_3d):
        """Innermost C-order stride must equal the dtype itemsize."""
        assert cube_3d.strides[-1] == cube_3d.itemsize

    # --- np.ascontiguousarray / np.asfortranarray ----------------------------

    def test_ascontiguousarray_restores_c_contiguity(self, fortran_3d):
        """np.ascontiguousarray on an F-order array must return a C-contiguous copy."""
        # Arrange
        arr = fortran_3d

        # Act
        result = np.ascontiguousarray(arr)

        # Assert
        assert result.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(result, arr)

    def test_asfortranarray_restores_f_contiguity(self, cube_3d):
        """np.asfortranarray on a C-order array must return an F-contiguous copy."""
        # Arrange
        arr = cube_3d

        # Act
        result = np.asfortranarray(arr)

        # Assert
        assert result.flags["F_CONTIGUOUS"]
        np.testing.assert_array_equal(result, arr)

    def test_contiguous_copy_does_not_share_memory(self, fortran_3d):
        """Converting memory order must produce a copy, not a view."""
        # Arrange
        arr = fortran_3d

        # Act
        result = np.ascontiguousarray(arr)

        # Assert
        assert not np.shares_memory(arr, result)

    # --- Slicing and contiguity ----------------------------------------------

    def test_slice_along_first_axis_is_c_contiguous(self, cube_3d):
        """A slice selecting a single index along axis 0 must stay C-contiguous."""
        # Arrange / Act
        sliced = cube_3d[0, :, :]  # shape (3, 4)

        # Assert
        assert sliced.flags["C_CONTIGUOUS"]

    def test_slice_with_step_breaks_contiguity(self, cube_3d):
        """Slicing with a step > 1 must break both C and F contiguity."""
        # Arrange / Act
        sliced = cube_3d[:, ::2, :]  # every other row — non-contiguous

        # Assert
        assert not sliced.flags["C_CONTIGUOUS"]
        assert not sliced.flags["F_CONTIGUOUS"]

    def test_reshape_after_transpose_on_copy(self, cube_3d):
        """Reshape after transpose requires a contiguous copy; values must survive."""
        # Arrange
        arr = cube_3d.transpose()          # non C-contiguous view

        # Act — np.reshape on a non-contiguous array returns a copy
        result = np.reshape(arr, (-1,))    # flatten

        # Assert
        assert result.size == arr.size
        np.testing.assert_array_equal(np.sort(result), np.arange(24.0))

    # --- np.ndarray.copy order argument --------------------------------------

    def test_copy_c_order_is_c_contiguous(self, fortran_3d):
        """arr.copy(order='C') must produce a C-contiguous array."""
        result = fortran_3d.copy(order="C")
        assert result.flags["C_CONTIGUOUS"]

    def test_copy_f_order_is_f_contiguous(self, cube_3d):
        """arr.copy(order='F') must produce an F-contiguous array."""
        result = cube_3d.copy(order="F")
        assert result.flags["F_CONTIGUOUS"]

    def test_copy_preserves_values_regardless_of_order(self, cube_3d):
        """Changing memory order via copy must not alter element values."""
        result = cube_3d.copy(order="F")
        np.testing.assert_array_equal(result, cube_3d)
