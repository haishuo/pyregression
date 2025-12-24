"""
Test GPU backend implementations.

Validates that GPU backends produce statistically equivalent results to CPU.
"""

import pytest
import numpy as np

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available() or (
        hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    )
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch GPU not available")
class TestGPUBackends:
    """Test GPU backend QR decomposition."""
    
    def test_backend_creation(self):
        """Test that GPU backend can be created."""
        from pyregression._backends.gpu_fp32_backend import PyTorchBackendFP32
        
        backend = PyTorchBackendFP32()
        assert backend.name == "pytorch_fp32"
        assert backend.precision == "fp32"
        
        info = backend.get_device_info()
        assert info['backend'] == 'gpu'
        assert info['precision'] == 'fp32'
        print(f"\n✓ GPU Backend: {info['device']}")
    
    def test_simple_qr(self):
        """Test QR decomposition on simple matrix."""
        from pyregression._backends import get_backend
        
        # Simple 5x3 matrix
        np.random.seed(42)
        X = np.random.randn(5, 3)
        
        # CPU reference
        cpu_backend = get_backend('cpu')
        cpu_result = cpu_backend.qr_with_pivoting(X)
        
        # GPU result
        gpu_backend = get_backend('gpu', use_fp64=False)
        gpu_result = gpu_backend.qr_with_pivoting(X)
        
        print(f"\nCPU rank: {cpu_result.rank}")
        print(f"GPU rank: {gpu_result.rank}")
        
        # Rank must match exactly
        assert cpu_result.rank == gpu_result.rank
        
        # Pivots must match exactly
        np.testing.assert_array_equal(cpu_result.pivot, gpu_result.pivot)
        
        # R matrices should be close (FP32 tolerance)
        np.testing.assert_allclose(
            cpu_result.R, gpu_result.R, 
            rtol=1e-5, atol=1e-6,
            err_msg="R matrices differ too much"
        )
        
        print("✓ Simple QR test passed")
    
    def test_full_regression(self):
        """Test full regression workflow CPU vs GPU."""
        from pyregression import LinearModel
        
        # Generate test data
        np.random.seed(42)
        n, p = 100, 5
        X = np.random.randn(n, p)
        true_coef = np.random.randn(p)
        y = X @ true_coef + np.random.randn(n) * 0.1
        
        # Fit with CPU
        cpu_model = LinearModel(backend='cpu')
        cpu_result = cpu_model.fit(X, y)
        
        # Fit with GPU
        gpu_model = LinearModel(backend='gpu', use_fp64=False)
        gpu_result = gpu_model.fit(X, y)
        
        print(f"\nCPU R²: {cpu_result.r_squared:.6f}")
        print(f"GPU R²: {gpu_result.r_squared:.6f}")
        
        # Rank must match
        assert cpu_result.rank == gpu_result.rank
        
        # Coefficients should be statistically equivalent
        # (within a few standard errors)
        coef_diff = np.abs(cpu_result.coef - gpu_result.coef)
        max_diff_in_ses = np.max(coef_diff / cpu_result.se)
        
        print(f"Max coef difference: {max_diff_in_ses:.2f} SEs")
        assert max_diff_in_ses < 5.0, "Coefficients differ by too many SEs"
        
        # R² should be very close
        r2_diff = abs(cpu_result.r_squared - gpu_result.r_squared)
        assert r2_diff < 0.001, f"R² differs by {r2_diff:.6f}"
        
        print("✓ Full regression test passed")
    
    def test_rank_deficient(self):
        """Test rank-deficient design matrix."""
        from pyregression import LinearModel
        
        np.random.seed(42)
        n, p = 50, 5
        X = np.random.randn(n, p)
        
        # Make column 3 = 2 * column 1 (perfect collinearity)
        X[:, 2] = 2.0 * X[:, 0]
        
        y = np.random.randn(n)
        
        # CPU
        cpu_model = LinearModel(backend='cpu')
        cpu_result = cpu_model.fit(X, y)
        
        # GPU
        gpu_model = LinearModel(backend='gpu', use_fp64=False)
        gpu_result = gpu_model.fit(X, y)
        
        print(f"\nCPU rank: {cpu_result.rank} (expected < {p+1})")
        print(f"GPU rank: {gpu_result.rank} (expected < {p+1})")
        
        # Both should detect rank deficiency
        assert cpu_result.rank < p + 1  # +1 for intercept
        assert cpu_result.rank == gpu_result.rank
        
        print("✓ Rank deficient test passed")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch GPU not available")
class TestGPUPerformance:
    """Test GPU performance on various problem sizes."""
    
    def test_medium_problem(self):
        """Test medium-sized problem."""
        import time
        from pyregression import LinearModel
        
        np.random.seed(42)
        n, p = 5000, 50
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        print(f"\nMedium problem: n={n}, p={p}")
        
        # CPU
        start = time.time()
        cpu_result = LinearModel(backend='cpu').fit(X, y)
        cpu_time = time.time() - start
        
        # GPU
        start = time.time()
        gpu_result = LinearModel(backend='gpu', use_fp64=False).fit(X, y)
        gpu_time = time.time() - start
        
        print(f"CPU: {cpu_time:.3f}s")
        print(f"GPU: {gpu_time:.3f}s")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        
        # Results should be equivalent
        assert cpu_result.rank == gpu_result.rank
        
        coef_diff = np.abs(cpu_result.coef - gpu_result.coef)
        max_diff = np.max(coef_diff / cpu_result.se)
        assert max_diff < 5.0
        
        print(f"✓ Results statistically equivalent (max diff: {max_diff:.2f} SEs)")
    
    @pytest.mark.slow
    def test_large_problem(self):
        """Test large problem where GPU should dominate."""
        import time
        from pyregression import LinearModel
        
        np.random.seed(42)
        n, p = 50000, 100
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        print(f"\nLarge problem: n={n}, p={p}")
        
        # CPU
        start = time.time()
        cpu_result = LinearModel(backend='cpu').fit(X, y)
        cpu_time = time.time() - start
        
        # GPU
        start = time.time()
        gpu_result = LinearModel(backend='gpu', use_fp64=False).fit(X, y)
        gpu_time = time.time() - start
        
        print(f"CPU: {cpu_time:.3f}s")
        print(f"GPU: {gpu_time:.3f}s")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        
        # GPU should be faster on large problems
        assert gpu_time < cpu_time, "GPU should be faster on large problems!"
        
        # Results should be equivalent
        assert cpu_result.rank == gpu_result.rank
        
        print("✓ GPU faster on large problem")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])