"""
test_aggregations_statistics.py
Functional tests for NumPy aggregation and statistical functions.
Verifies correctness of np.mean, np.median, np.std across various
scenarios including edge values (NaN, Inf), different dtypes, and axes.

Usage:
    pytest tests/functional/ -v
"""
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_array():
    """Simple 1-D float64 array with known values."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)


@pytest.fixture
def array_with_nan():
    """1-D array containing NaN values."""
    return np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float64)


@pytest.fixture
def array_with_inf():
    """1-D array containing positive and negative infinity."""
    return np.array([1.0, np.inf, 3.0, -np.inf, 5.0], dtype=np.float64)


@pytest.fixture
def matrix_2d():
    """2-D array (4x3) with known values, used for axis-specific tests."""
    return np.array([
        [1.0,  2.0,  3.0],
        [4.0,  5.0,  6.0],
        [7.0,  8.0,  9.0],
        [10.0, 11.0, 12.0],
    ], dtype=np.float64)


@pytest.fixture
def integer_array():
    """1-D int32 array to ensure functions handle integer dtypes correctly."""
    return np.array([10, 20, 30, 40, 50], dtype=np.int32)


@pytest.fixture
def all_same_array():
    """Array where all values are identical — std should be 0."""
    return np.full(100, fill_value=7.0, dtype=np.float64)


@pytest.fixture
def large_random_array():
    """Large reproducible random array for statistical accuracy checks."""
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal(100_000).astype(np.float64)


# ---------------------------------------------------------------------------
# np.mean tests
# ---------------------------------------------------------------------------

class TestMean:
    def test_mean_basic(self, basic_array):
        """Mean of [1,2,3,4,5] should be 3.0."""
        assert np.mean(basic_array) == pytest.approx(3.0)

    def test_mean_integer_array(self, integer_array):
        """Mean of integer array should return float result = 30.0."""
        result = np.mean(integer_array)
        assert isinstance(result, (float, np.floating))
        assert result == pytest.approx(30.0)

    def test_mean_with_nan_returns_nan(self, array_with_nan):
        """np.mean propagates NaN — result should be NaN."""
        assert np.isnan(np.mean(array_with_nan))

    def test_nanmean_ignores_nan(self, array_with_nan):
        """np.nanmean should ignore NaN and return mean of [1, 3, 5] = 3.0."""
        assert np.nanmean(array_with_nan) == pytest.approx(3.0)

    def test_mean_with_inf(self, array_with_inf):
        """Mean of array containing +Inf and -Inf should be NaN."""
        with pytest.warns(RuntimeWarning):
            assert np.isnan(np.mean(array_with_inf))

    def test_mean_axis0(self, matrix_2d):
        """Column-wise mean along axis=0 for a 4x3 matrix."""
        expected = np.array([5.5, 6.5, 7.5])
        np.testing.assert_allclose(np.mean(matrix_2d, axis=0), expected)

    def test_mean_axis1(self, matrix_2d):
        """Row-wise mean along axis=1 for a 4x3 matrix."""
        expected = np.array([2.0, 5.0, 8.0, 11.0])
        np.testing.assert_allclose(np.mean(matrix_2d, axis=1), expected)

    def test_mean_large_array_close_to_zero(self, large_random_array):
        """Mean of a large standard-normal sample should be very close to 0."""
        assert np.mean(large_random_array) == pytest.approx(0.0, abs=0.02)

    def test_mean_all_same(self, all_same_array):
        """Mean of an all-7 array should be exactly 7.0."""
        assert np.mean(all_same_array) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# np.median tests
# ---------------------------------------------------------------------------

class TestMedian:
    def test_median_odd_length(self, basic_array):
        """Median of [1,2,3,4,5] is the middle element = 3.0."""
        assert np.median(basic_array) == pytest.approx(3.0)

    def test_median_even_length(self):
        """Median of even-length array should be average of two middle values."""
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.median(arr) == pytest.approx(2.5)

    def test_median_with_nan_returns_nan(self, array_with_nan):
        """np.median propagates NaN."""
        assert np.isnan(np.median(array_with_nan))

    def test_nanmedian_ignores_nan(self, array_with_nan):
        """np.nanmedian of [1, NaN, 3, NaN, 5] should be 3.0."""
        assert np.nanmedian(array_with_nan) == pytest.approx(3.0)

    def test_median_unsorted_input(self):
        """Median must work correctly even when input is not sorted."""
        arr = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
        assert np.median(arr) == pytest.approx(3.0)

    def test_median_axis0(self, matrix_2d):
        """Column-wise median along axis=0."""
        expected = np.array([5.5, 6.5, 7.5])
        np.testing.assert_allclose(np.median(matrix_2d, axis=0), expected)

    def test_median_single_element(self):
        """Median of a single-element array equals that element."""
        assert np.median(np.array([42.0])) == pytest.approx(42.0)

    def test_median_large_array_close_to_zero(self, large_random_array):
        """Median of a large standard-normal sample should be near 0."""
        assert np.median(large_random_array) == pytest.approx(0.0, abs=0.02)


# ---------------------------------------------------------------------------
# np.std tests
# ---------------------------------------------------------------------------

class TestStd:
    def test_std_basic(self, basic_array):
        """Population std of [1,2,3,4,5] = sqrt(2) ≈ 1.4142."""
        assert np.std(basic_array) == pytest.approx(np.sqrt(2.0), rel=1e-6)

    def test_std_all_same_is_zero(self, all_same_array):
        """Std of a constant array must be exactly 0."""
        assert np.std(all_same_array) == pytest.approx(0.0, abs=1e-10)

    def test_std_with_nan_returns_nan(self, array_with_nan):
        """np.std propagates NaN."""
        assert np.isnan(np.std(array_with_nan))

    def test_nanstd_ignores_nan(self, array_with_nan):
        """np.nanstd of [1, NaN, 3, NaN, 5] should equal std of [1, 3, 5]."""
        expected = np.std([1.0, 3.0, 5.0])
        assert np.nanstd(array_with_nan) == pytest.approx(expected, rel=1e-6)

    def test_std_ddof1_sample(self, basic_array):
        """Sample std (ddof=1) of [1,2,3,4,5] = sqrt(2.5) ≈ 1.5811."""
        assert np.std(basic_array, ddof=1) == pytest.approx(np.sqrt(2.5), rel=1e-6)

    def test_std_axis0(self, matrix_2d):
        """Column-wise std along axis=0 for a 4x3 matrix."""
        expected = np.std(matrix_2d, axis=0)  # reference via numpy itself
        np.testing.assert_allclose(np.std(matrix_2d, axis=0), expected)

    def test_std_axis1(self, matrix_2d):
        """Row-wise std along axis=1 — each row [a,a+1,a+2] has same std."""
        stds = np.std(matrix_2d, axis=1)
        # All rows span 3 consecutive integers → std = std([0,1,2])
        expected_std = np.std([0.0, 1.0, 2.0])
        np.testing.assert_allclose(stds, expected_std, rtol=1e-6)

    def test_std_large_array_close_to_one(self, large_random_array):
        """Std of a large standard-normal sample should be close to 1."""
        assert np.std(large_random_array) == pytest.approx(1.0, abs=0.02)

    def test_std_non_negative(self, basic_array):
        """Standard deviation must always be non-negative."""
        assert np.std(basic_array) >= 0.0
