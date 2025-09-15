#!/usr/bin/env python3
"""
Quick performance test for numba JIT compilation
"""
import time
import numpy as np

# Test the JIT functions we added
from numba import jit

@jit(nopython=True)
def _sample_array_jit(arr, ux, uy, uz, N):
    """JIT-compiled trilinear sampling core."""
    i0 = int(np.floor(ux)) % N
    j0 = int(np.floor(uy)) % N
    k0 = int(np.floor(uz)) % N
    i1 = (i0 + 1) % N
    j1 = (j0 + 1) % N
    k1 = (k0 + 1) % N
    fx = ux - np.floor(ux)
    fy = uy - np.floor(uy)
    fz = uz - np.floor(uz)

    c000 = arr[i0, j0, k0]; c100 = arr[i1, j0, k0]
    c010 = arr[i0, j1, k0]; c110 = arr[i1, j1, k0]
    c001 = arr[i0, j0, k1]; c101 = arr[i1, j0, k1]
    c011 = arr[i0, j1, k1]; c111 = arr[i1, j1, k1]

    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx

    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy

    return c0 * (1.0 - fz) + c1 * fz

def sample_array_python(arr, ux, uy, uz, N):
    """Pure Python version for comparison."""
    i0 = int(np.floor(ux)) % N
    j0 = int(np.floor(uy)) % N
    k0 = int(np.floor(uz)) % N
    i1 = (i0 + 1) % N
    j1 = (j0 + 1) % N
    k1 = (k0 + 1) % N
    fx = ux - np.floor(ux)
    fy = uy - np.floor(uy)
    fz = uz - np.floor(uz)

    c000 = arr[i0, j0, k0]; c100 = arr[i1, j0, k0]
    c010 = arr[i0, j1, k0]; c110 = arr[i1, j1, k0]
    c001 = arr[i0, j0, k1]; c101 = arr[i1, j0, k1]
    c011 = arr[i0, j1, k1]; c111 = arr[i1, j1, k1]

    c00 = c000 * (1.0 - fx) + c100 * fx
    c10 = c010 * (1.0 - fx) + c110 * fx
    c01 = c001 * (1.0 - fx) + c101 * fx
    c11 = c011 * (1.0 - fx) + c111 * fx

    c0 = c00 * (1.0 - fy) + c10 * fy
    c1 = c01 * (1.0 - fy) + c11 * fy

    return c0 * (1.0 - fz) + c1 * fz

def benchmark():
    N = 64
    arr = np.random.random((N, N, N)).astype(np.float64)
    n_tests = 10000

    # Random sampling points
    ux = np.random.random(n_tests) * N
    uy = np.random.random(n_tests) * N
    uz = np.random.random(n_tests) * N

    print(f"Benchmarking trilinear sampling on {N}^3 grid with {n_tests} samples...")

    # Warm up JIT
    _sample_array_jit(arr, ux[0], uy[0], uz[0], N)

    # Test Python version
    t0 = time.time()
    for i in range(n_tests):
        result_py = sample_array_python(arr, ux[i], uy[i], uz[i], N)
    t_python = time.time() - t0

    # Test JIT version
    t0 = time.time()
    for i in range(n_tests):
        result_jit = _sample_array_jit(arr, ux[i], uy[i], uz[i], N)
    t_jit = time.time() - t0

    # Verify results are identical
    test_result_py = sample_array_python(arr, ux[0], uy[0], uz[0], N)
    test_result_jit = _sample_array_jit(arr, ux[0], uy[0], uz[0], N)

    print(f"Python time: {t_python:.4f}s")
    print(f"Numba JIT time: {t_jit:.4f}s")
    print(f"Speedup: {t_python/t_jit:.1f}x")
    print(f"Results match: {np.allclose(test_result_py, test_result_jit)}")

if __name__ == "__main__":
    benchmark()