# Test Scenarios – OOB Testing System for NumPy

This document describes the acceptance test scenarios for the OOB (Out Of The Box) testing
system for the NumPy library. It also provides an overview of the functional and performance
test scenarios implemented in the `tests/` directory.

---

## 1. Functional Test Scenarios

Functional tests are located in `tests/functional/` and are run automatically by the CI
pipeline (`pytest tests/functional/ -v`). Each file targets one specific area of NumPy
functionality.

---

### 1.1 Matrix Multiplication

**File:** `tests/functional/test_matrix_multiplication_functional.py`

**Goal:** Verify the correctness of NumPy matrix multiplication operations across a wide
range of inputs, dtypes, and edge cases.

**Functions under test:** `np.dot`, `np.matmul`, `@` operator, `np.linalg.multi_dot`

**Test classes and selected scenarios:**

| Class | Scenario | Expected result |
|---|---|---|
| `TestDot` | `[[1,2],[3,4]] @ [[1,2],[3,4]]` | `[[7,10],[15,22]]` |
| `TestDot` | Dot product of two 1-D vectors `[1,2,3]` | Scalar `14.0` |
| `TestDot` | Multiply any matrix by the identity matrix | Original matrix unchanged |
| `TestDot` | Multiply any matrix by the zero matrix | All-zero result |
| `TestDot` | Incompatible shapes `(3,4) @ (5,2)` | `ValueError` raised |
| `TestDot` | `A @ inv(A)` for a 100x100 matrix | Close to identity (`atol=1e-8`) |
| `TestDot` | Integer dtype input | Integer dtype preserved in output |
| `TestMatmul` | `np.matmul(A, A)` vs `A @ A` | Results are identical |
| `TestMatmul` | 2-D inputs: `np.matmul` vs `np.dot` | Results are identical |
| `TestMatmul` | Batched `(4,3,3) @ (3,3)` | Output shape `(4,3,3)` |
| `TestMatmul` | `np.matmul` on scalar input | `ValueError` raised |
| `TestMatmul` | `A @ A.T` for any matrix A | Result is symmetric |
| `TestMultiDot` | Two-matrix chain vs `np.dot` | Results are identical |
| `TestMultiDot` | `v @ M @ v` where v is 1-D | Scalar output (0-D array) |

**Pass criterion:** All assertions pass with `np.testing.assert_allclose` tolerances
(`rtol=1e-10`, `atol=1e-8` for large-matrix tests).

---

### 1.2 Aggregations and Statistics

**File:** `tests/functional/test_aggregations_statistics.py`

**Goal:** Verify the correctness of NumPy statistical aggregation functions, including
correct behaviour on NaN and Inf values.

**Functions under test:** `np.mean`, `np.nanmean`, `np.median`, `np.nanmedian`,
`np.std`, `np.nanstd`

**Test classes and selected scenarios:**

| Class | Scenario | Expected result |
|---|---|---|
| `TestMean` | `mean([1,2,3,4,5])` | `3.0` |
| `TestMean` | `mean` of array with NaN | NaN propagated |
| `TestMean` | `nanmean([1, NaN, 3, NaN, 5])` | `3.0` (NaN ignored) |
| `TestMean` | `mean` of array with `+Inf` and `-Inf` | NaN (with `RuntimeWarning`) |
| `TestMean` | Column-wise mean of 4x3 matrix (axis=0) | `[5.5, 6.5, 7.5]` |
| `TestMean` | Large standard-normal sample (100 000 elements) | Approximately `0.0` (`abs=0.02`) |
| `TestMedian` | `median([1,2,3,4,5])` | `3.0` |
| `TestMedian` | `median([1,2,3,4])` (even length) | `2.5` |
| `TestMedian` | `median` on unsorted input | Correct value regardless of order |
| `TestMedian` | `nanmedian([1, NaN, 3, NaN, 5])` | `3.0` |
| `TestStd` | `std([1,2,3,4,5])` (population) | `sqrt(2) ~= 1.4142` |
| `TestStd` | `std` of all-same-value array | `0.0` |
| `TestStd` | `std` with `ddof=1` (sample std) | `sqrt(2.5) ~= 1.5811` |
| `TestStd` | `nanstd([1, NaN, 3, NaN, 5])` | Equal to `std([1, 3, 5])` |
| `TestStd` | Large standard-normal sample | Approximately `1.0` (`abs=0.02`) |

**Pass criterion:** All assertions pass with `pytest.approx` or `np.testing.assert_allclose`.

---

### 1.3 3D Dimension Manipulation

**File:** `tests/functional/test_3d_dimension_manipulation.py`

**Goal:** Verify reshape, transpose, and memory layout behaviour for 3-D arrays,
including C-order vs Fortran-order, views vs copies, and stride correctness.

**Functions under test:** `np.reshape`, `np.transpose`, `np.ascontiguousarray`,
`np.asfortranarray`

**Test classes and selected scenarios:**

| Class | Scenario | Expected result |
|---|---|---|
| `TestReshape` | Flat 24-element array to `(2,3,4)` | Shape `(2,3,4)`, 24 elements |
| `TestReshape` | `(2,3,4)` flattened to `(-1,)` | Shape `(24,)`, values `0..23` |
| `TestReshape` | `-1` dimension inferred as `3` in `(4,-1,2)` | Shape `(4,3,2)` |
| `TestReshape` | Incompatible size `(5,5,5)` on 24-element array | `ValueError` raised |
| `TestReshape` | Reshape of contiguous C-order array | Returns a view (shared memory) |
| `TestReshape` | Reshape with `order='F'` | First axis iterates fastest |
| `TestReshape` | Integer dtype array | dtype `int32` preserved |
| `TestTranspose` | Default transpose of `(2,3,4)` | Shape `(4,3,2)` |
| `TestTranspose` | Custom axes `(1,2,0)` on `(2,3,4)` | Shape `(3,4,2)` |
| `TestTranspose` | Double transpose | Original shape and values recovered |
| `TestTranspose` | Element positions after `(1,2,0)` permutation | `arr[i,j,k] == result[j,k,i]` |
| `TestTranspose` | Transpose returns a view | Shared memory with original |
| `TestTranspose` | Transposed array contiguity | Not C-contiguous; is F-contiguous |
| `TestMemoryLayout` | C-order array flags | `C_CONTIGUOUS=True`, `F_CONTIGUOUS=False` |
| `TestMemoryLayout` | F-order array flags | `F_CONTIGUOUS=True`, `C_CONTIGUOUS=False` |
| `TestMemoryLayout` | C-order strides | Decrease from axis 0 to last axis |
| `TestMemoryLayout` | Slice with step > 1 | Breaks both C and F contiguity |

**Pass criterion:** All shape, value, memory-sharing, and flag assertions pass.

---

### 1.4 Broadcasting

**File:** `tests/functional/test_broadcasting_functional.py`

**Goal:** Verify that NumPy applies broadcasting rules correctly for arithmetic operations
on arrays of different but compatible shapes, and raises errors for incompatible shapes.

**Test classes and selected scenarios:**

| Class | Scenario | Expected result |
|---|---|---|
| `TestBasicBroadcasting` | Scalar `5.0` added to `(2,3)` matrix | Every element increased by 5 |
| `TestBasicBroadcasting` | `(3,)` array added to `(2,3)` matrix | Row-wise addition |
| `TestBasicBroadcasting` | `(2,1)` column vector x `(2,3)` matrix | Column-wise multiplication |
| `TestMultiDimensionalBroadcasting` | `(2,3)` added to `(2,2,3)` tensor | Output shape `(2,2,3)` |
| `TestMultiDimensionalBroadcasting` | `(4,1)` + `(3,)` | Output shape `(4,3)`, values verified element-by-element |
| `TestBroadcastingConstraints` | `(2,3)` + `(2,2)` | `ValueError` raised |
| `TestBroadcastingConstraints` | `(3,)` + `(4,)` | `ValueError` raised |
| `TestInPlaceBroadcasting` | `matrix_2d += array_1d` (compatible shapes) | In-place result correct |
| `TestInPlaceBroadcasting` | `matrix_2d += col_vector_bad` (incompatible) | `ValueError` raised |

**Pass criterion:** All expected values match and all expected exceptions are raised.

---

## 2. Performance Test Scenarios

Performance tests are located in `tests/performance/` and are run with `pytest-benchmark`.
Results are saved to `benchmark_results.json` and uploaded as a pipeline artifact.

---

### 2.1 Matrix Multiplication Benchmark

**File:** `tests/performance/test_matrix_multiplication_perf.py`

**Goal:** Measure the execution time of NumPy matrix multiplication on large matrices and
compare against a Python-native baseline to demonstrate NumPy's performance advantage.

**Scenarios:**

| Scenario | Description | Acceptance threshold |
|---|---|---|
| NumPy 1000x1000 matmul | `np.dot` on two 1000x1000 float64 matrices | Result logged; regression tracked via baseline |
| Python list baseline | Element-wise multiplication on equivalent Python lists | Expected to be orders of magnitude slower than NumPy |

**Reporting:** Mean time (ms) for each benchmark is printed in the pipeline summary step.
If `baseline_benchmark.json` exists, a regression comparison is performed automatically.

**Pass criterion:** No unhandled exception during execution; results saved to
`benchmark_results.json` and visible in the pipeline log.

---

### 2.2 NumPy vs Pure-Python Array Comparison Benchmark

**File:** `tests/performance/test_array_comparison.py`

**Goal:** Quantify the performance gap between NumPy's `np.dot` and a naive pure-Python
triple-loop matrix multiplication, validating the claim from scenario 2.1 that NumPy is
"orders of magnitude faster" than a Python-native baseline.

**Functions under test:** `np.dot`, custom pure-Python `python_matmul` (nested-list triple-loop)

**Scenarios:**

| Scenario | Description | Acceptance threshold |
|---|---|---|
| NumPy 1000×1000 matmul (reference) | `np.dot` on two 1000×1000 float64 matrices seeded with `rng=42` | Result shape `(1000,1000)`, dtype `float64`, all values finite |
| Pure-Python 100×100 matmul (baseline) | Naive O(n³) triple-loop on equivalent Python lists; reduced size to keep CI runtime manageable | Result shape `(100,100)`; timing must demonstrate an order-of-magnitude slowdown vs NumPy |

> **Note on matrix sizes:** The Python baseline intentionally uses 100×100 matrices instead
> of 1000×1000. Scaling to full size would make the CI run impractically slow; the 100×100
> result is already representative of the slowdown ratio.

**Reporting:** Both benchmarks are collected in the same file so pytest-benchmark can print
a unified comparison table. Mean timings (ms) for each scenario appear in the pipeline
summary. Results are saved to `benchmark_results.json`.

**Pass criterion:** No unhandled exception during execution; both benchmarks complete and
their results appear in `benchmark_results.json`; the Python baseline mean time is visibly
(at least one order of magnitude) larger than the NumPy reference time in the pipeline log.

---

## 3. Acceptance Test Scenarios

The following three scenarios are high-level end-to-end checks that validate the OOB
testing system as a whole. They can be verified by triggering the pipeline manually.

---

### AT-01 – Successful installation and import of NumPy

**Goal:** Confirm that the pipeline can install (or compile) NumPy from source and import
it successfully inside a fresh runner environment.

**Preconditions:**
- `requirements.txt` lists all necessary dependencies.
- `scripts/build_numpy.py` is present and correct.

**Steps:**
1. Trigger the pipeline manually via the GitHub Actions UI (`workflow_dispatch`).
2. Observe the **Install pipeline dependencies** step.
3. Observe the **Build NumPy from source** step.

**Expected result:** Both steps complete with exit code 0. No import error appears in
subsequent test steps.

**Pass criterion:** All steps following the build step are able to `import numpy` without
raising `ModuleNotFoundError` or `ImportError`.

---

### AT-02 – Full test suite passes and results are reported

**Goal:** Confirm that all functional and performance tests run to completion and that their
results are visible in the pipeline output in a readable format.

**Preconditions:**
- AT-01 has passed (NumPy is installed).
- All files under `tests/functional/` and `tests/performance/` are present on `main`.

**Steps:**
1. Trigger the pipeline manually.
2. Observe the **Run functional tests** step.
3. Observe the **Run performance tests** step.
4. Observe the **Display test summary** step.

**Expected result:**
- Functional tests: all collected test functions pass (0 failures reported by pytest).
- Performance tests: benchmarks execute and `benchmark_results.json` is produced.
- Test summary: the pipeline prints the OOB NumPy Test Summary block and the benchmark
  mean timings per test.

**Pass criterion:** Exit code 0 for both test steps; `benchmark_results.json` is uploaded
as a pipeline artifact; timing values appear in the summary log.

---

### AT-03 – Pipeline output is readable after manual trigger

**Goal:** Confirm that a person with no prior knowledge of the project can read the pipeline
output and understand which tests passed, which (if any) failed, and what performance was
measured.

**Preconditions:**
- AT-01 and AT-02 have passed.

**Steps:**
1. Open the completed pipeline run in the GitHub Actions UI.
2. Read the **Run functional tests** log — test names and pass/fail status must be visible.
3. Read the **Display test summary** log — the summary block must list per-benchmark
   mean times.
4. Download the `benchmark-results-*` artifact and open `benchmark_results.json`.

**Expected result:**
- The functional test log shows individual test names and their pass/fail markers.
- The summary block lists at least one benchmark with a numeric mean time value.
- `benchmark_results.json` is a valid JSON file containing a `benchmarks` array.

**Pass criterion:** A reviewer unfamiliar with the codebase can determine the overall health
of the tested NumPy build from the pipeline output alone, without needing to run the code
locally.
