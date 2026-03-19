import numpy as np
import time
import json

def measure_matrix_multiplication(size=2000):
    """
    Measures the time taken to multiply two square matrices of given size.
    """
    print(f"--- Starting Matrix Multiplication Performance Test (Size: {size}x{size}) ---")
    
    # Initialize matrices with random floats
    A = np.random.rand(size, size).astype(np.float64)
    B = np.random.rand(size, size).astype(np.float64)

    # Warm-up (to ensure any JIT or library caching is active)
    _ = np.dot(A[:100, :100], B[:100, :100])

    start_time = time.perf_counter()
    result = np.dot(A, B)
    end_time = time.perf_counter()

    duration = end_time - start_time
    print(f"Completed in: {duration:.4f} seconds")
    
    return {
        "operation": "matrix_multiplication",
        "matrix_size": size,
        "duration_seconds": duration,
        "numpy_version": np.__version__
    }

if __name__ == "__main__":
    stats = measure_matrix_multiplication()
    # Export results to a JSON for the CI pipeline to read/log
    with open("perf_results.json", "w") as f:
        json.dump(stats, f, indent=4)