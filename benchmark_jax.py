#!/usr/bin/env python3
"""Benchmark JAX vs NumPy performance for trilinear interpolation."""

import sys
import time
import numpy as np
sys.path.insert(0, 'src')
from don_emergent_collapse_3d import CollapseField3D

def benchmark_interpolation(use_jax, N=64, num_samples=10000):
    """Benchmark interpolation performance."""

    # Create field
    field = CollapseField3D(N=N, L=32.0)
    field.use_jax = use_jax

    # Create test array
    test_arr = np.random.randn(N, N, N)

    # Generate random sample points
    np.random.seed(42)
    points = np.random.uniform(0, 32.0, (num_samples, 3))

    # Warm-up (especially important for JAX JIT compilation)
    for i in range(10):
        field.sample_array(test_arr, points[i,0], points[i,1], points[i,2])
        field.sample_grad_trilinear(test_arr, points[i,0], points[i,1], points[i,2])

    # Benchmark sample_array
    start = time.perf_counter()
    for i in range(num_samples):
        field.sample_array(test_arr, points[i,0], points[i,1], points[i,2])
    sample_time = time.perf_counter() - start

    # Benchmark grad_array
    start = time.perf_counter()
    for i in range(num_samples):
        field.sample_grad_trilinear(test_arr, points[i,0], points[i,1], points[i,2])
    grad_time = time.perf_counter() - start

    return sample_time, grad_time

def main():
    print("Benchmarking trilinear interpolation performance...")
    print("=" * 60)

    # Test different grid sizes
    for N in [32, 64, 128]:
        print(f"\nGrid size: {N}×{N}×{N}")
        print("-" * 40)

        # NumPy benchmark
        sample_np, grad_np = benchmark_interpolation(use_jax=False, N=N)
        print(f"NumPy:")
        print(f"  sample_array: {sample_np:.3f}s ({1000*sample_np/10000:.3f}ms per call)")
        print(f"  grad_array:   {grad_np:.3f}s ({1000*grad_np/10000:.3f}ms per call)")

        # JAX benchmark
        sample_jax, grad_jax = benchmark_interpolation(use_jax=True, N=N)
        print(f"JAX:")
        print(f"  sample_array: {sample_jax:.3f}s ({1000*sample_jax/10000:.3f}ms per call)")
        print(f"  grad_array:   {grad_jax:.3f}s ({1000*grad_jax/10000:.3f}ms per call)")

        # Speedup
        print(f"Speedup:")
        print(f"  sample_array: {sample_np/sample_jax:.2f}x")
        print(f"  grad_array:   {grad_np/grad_jax:.2f}x")

if __name__ == "__main__":
    main()