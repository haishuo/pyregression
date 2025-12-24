"""
Test backend implementations with auto-detection.

Tests appropriate backends based on available hardware:
- CPU: Always tested
- PyTorch CUDA: Tested if NVIDIA GPU available
- MLX Metal: Tested if Apple Silicon available
"""

import pytest
import numpy as np
from pyregression._backends import (
    get_backend,
    list_available_backends,
    print_backend_info
)
from pyregression._backends.precision_detector import detect_gpu_capabilities


# Detect hardware once at module level
GPU_CAPS = detect_gpu_capabilities()
HAS_NVIDIA = GPU_CAPS.gpu_type == 'nvidia'
HAS_APPLE = GPU_CAPS.gpu_type == 'mlx'  # Detection returns 'mlx' for Apple Silicon
HAS_ANY_GPU = GPU_CAPS.has_gpu


class TestBackendDetection:
    """Test hardware detection and backend availability."""
    
    def test_detect_gpu_capabilities(self):
        """Test GPU detection returns valid capabilities."""
        caps = detect_gpu_capabilities()
        assert caps.gpu_name is not None
        assert caps.gpu_type in ['nvidia', 'mlx', 'none']
        assert caps.has_gpu == (caps.gpu_type != 'none')
    
    def test_list_backends(self):
        """Test backend listing."""
        backends = list_available_backends()
        assert isinstance(backends, list)
        assert 'cpu' in backends  # CPU always available
        
        # Check GPU backends match detection
        if HAS_NVIDIA:
            assert 'pytorch' in backends
        if HAS_APPLE:
            assert 'mlx' in backends
    
    def test_print_backend_info(self, capsys):
        """Test diagnostic printing."""
        print_backend_info()
        captured = capsys.readouterr()
        assert 'Backend Status' in captured.out
        assert 'CPU' in captured.out


class TestCPUBackend:
    """Test CPU backend (always available)."""
    
    def test_cpu_backend_creation(self):
        """Test CPU backend initializes correctly."""
        backend = get_backend('cpu')
        assert backend is not None
        assert backend.name == 'cpu_fp64'
        assert backend.precision == 'fp64'
    
    def test_cpu_device_info(self):
        """Test CPU backend device info."""
        backend = get_backend('cpu')
        info = backend.get_device_info()
        assert info['backend'] == 'cpu'
        assert info['precision'] == 'fp64'
    
    def test_cpu_simple_regression(self):
        """Test simple regression on CPU."""
        backend = get_backend('cpu')
        
        # Simple test data
        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)
        beta_true = np.array([1.0, 2.0, -1.5])
        y = X @ beta_true + 0.1 * np.random.randn(n)
        
        # Fit
        result = backend.fit_linear_model(X, y)
        
        # Check result structure
        assert result.coef.shape == (p + 1,)  # +1 for intercept
        assert result.residuals.shape == (n,)
        assert result.fitted_values.shape == (n,)
        assert result.rank > 0
        assert result.rank <= p + 1
        
        # Check numerical sanity
        assert np.allclose(result.coef[1:], beta_true, atol=0.5)
        assert np.mean(result.residuals**2) < 1.0  # Low MSE
    
    def test_cpu_weighted_regression(self):
        """Test weighted regression on CPU."""
        backend = get_backend('cpu')
        
        np.random.seed(42)
        n, p = 50, 2
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        weights = np.random.uniform(0.5, 1.5, n)
        
        result = backend.fit_linear_model(X, y, weights=weights)
        
        assert result.coef.shape == (p + 1,)
        assert result.rank > 0
    
    def test_cpu_with_offset(self):
        """Test regression with offset on CPU."""
        backend = get_backend('cpu')
        
        np.random.seed(42)
        n, p = 50, 2
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        offset = np.random.randn(n)
        
        result = backend.fit_linear_model(X, y, offset=offset)
        
        assert result.coef.shape == (p + 1,)
        assert result.fitted_values.shape == (n,)
        
        # Fitted values should include offset
        assert not np.allclose(result.fitted_values, y)


@pytest.mark.skipif(not HAS_NVIDIA, reason="NVIDIA GPU not available")
class TestPyTorchBackend:
    """Test PyTorch CUDA backend (NVIDIA GPUs only)."""
    
    def test_pytorch_backend_creation(self):
        """Test PyTorch backend initializes on CUDA."""
        backend = get_backend('pytorch')
        assert backend is not None
        assert 'pytorch' in backend.name
        assert backend.precision in ['fp32', 'fp64']
    
    def test_pytorch_device_info(self):
        """Test PyTorch backend device info."""
        backend = get_backend('pytorch')
        info = backend.get_device_info()
        assert info['backend'] == 'gpu'
        assert 'cuda' in str(info['device']).lower()
    
    def test_pytorch_simple_regression(self):
        """Test simple regression on PyTorch CUDA."""
        backend = get_backend('pytorch')
        
        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)
        beta_true = np.array([1.0, 2.0, -1.5])
        y = X @ beta_true + 0.1 * np.random.randn(n)
        
        result = backend.fit_linear_model(X, y)
        
        # Check structure
        assert result.coef.shape == (p + 1,)
        assert result.residuals.shape == (n,)
        assert result.fitted_values.shape == (n,)
        
        # Check numerical accuracy (FP32 or FP64)
        assert np.allclose(result.coef[1:], beta_true, atol=0.5)
    
    def test_pytorch_vs_cpu_consistency(self):
        """Test PyTorch gives similar results to CPU."""
        cpu_backend = get_backend('cpu')
        gpu_backend = get_backend('pytorch')
        
        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        cpu_result = cpu_backend.fit_linear_model(X, y)
        gpu_result = gpu_backend.fit_linear_model(X, y)
        
        # Coefficients should be close (accounting for FP32 vs FP64)
        assert np.allclose(cpu_result.coef, gpu_result.coef, rtol=1e-4, atol=1e-4)
        assert np.allclose(cpu_result.residuals, gpu_result.residuals, rtol=1e-4, atol=1e-4)
        assert cpu_result.rank == gpu_result.rank
    
    def test_pytorch_rejects_mps(self):
        """Test PyTorch backend rejects MPS device."""
        with pytest.raises(ValueError, match="does not support.*MPS"):
            get_backend('pytorch')
            # Try to create with MPS device
            from pyregression._backends.gpu_fp32_backend import PyTorchBackendFP32
            PyTorchBackendFP32(device='mps')


@pytest.mark.skipif(not HAS_APPLE, reason="Apple Silicon not available")
class TestMPSBackend:
    """Test MPS ridge backend (Apple Silicon only).
    
    Note: MPS backend uses ridge regression instead of QR decomposition.
    """
    
    def test_mps_backend_creation(self):
        """Test MPS backend initializes."""
        backend = get_backend('mps')
        assert backend is not None
        assert 'mps' in backend.name
        assert 'ridge' in backend.name
        assert backend.precision == 'fp32'
    
    def test_mps_device_info(self):
        """Test MPS backend device info."""
        backend = get_backend('mps')
        info = backend.get_device_info()
        assert info['backend'] == 'gpu'
        assert 'MPS' in info['device'] or 'Metal' in info['device']
        assert info['algorithm'] == 'Ridge regression via Cholesky'
    
    def test_mps_simple_regression(self):
        """Test simple regression on MPS."""
        backend = get_backend('mps')
        
        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)
        beta_true = np.array([1.0, 2.0, -1.5])
        y = X @ beta_true + 0.1 * np.random.randn(n)
        
        # Expect warning about ridge usage
        with pytest.warns(UserWarning, match="ridge regression"):
            result = backend.fit_linear_model(X, y)
        
        # Check structure
        assert result.coef.shape == (p + 1,)
        assert result.residuals.shape == (n,)
        assert result.fitted_values.shape == (n,)
        assert result.rank > 0
        
        # Check numerical accuracy (FP32 + ridge)
        # More lenient tolerance due to ridge penalty
        assert np.allclose(result.coef[1:], beta_true, atol=0.6)
        assert np.mean(result.residuals**2) < 1.0
    
    def test_mps_vs_cpu_consistency(self):
        """Test MPS gives similar results to CPU.
        
        Note: Results won't be identical due to ridge penalty,
        but should be very close for well-conditioned problems.
        """
        cpu_backend = get_backend('cpu')
        mps_backend = get_backend('mps')
        
        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        cpu_result = cpu_backend.fit_linear_model(X, y)
        
        with pytest.warns(UserWarning, match="ridge regression"):
            mps_result = mps_backend.fit_linear_model(X, y)
        
        # Coefficients should be close (accounting for small ridge penalty)
        # More lenient than PyTorch due to ridge
        assert np.allclose(cpu_result.coef, mps_result.coef, rtol=1e-3, atol=1e-3)
        assert np.allclose(cpu_result.residuals, mps_result.residuals, rtol=1e-3, atol=1e-3)
    
    def test_mps_weighted_regression(self):
        """Test weighted regression on MPS."""
        backend = get_backend('mps')
        
        np.random.seed(42)
        n, p = 50, 2
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        weights = np.random.uniform(0.5, 1.5, n)
        
        with pytest.warns(UserWarning, match="ridge regression"):
            result = backend.fit_linear_model(X, y, weights=weights)
        
        assert result.coef.shape == (p + 1,)
        assert result.rank > 0
    
    def test_mps_with_offset(self):
        """Test regression with offset on MPS."""
        backend = get_backend('mps')
        
        np.random.seed(42)
        n, p = 50, 2
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        offset = np.random.randn(n)
        
        with pytest.warns(UserWarning, match="ridge regression"):
            result = backend.fit_linear_model(X, y, offset=offset)
        
        assert result.coef.shape == (p + 1,)
        assert result.fitted_values.shape == (n,)
    
    def test_mps_rejects_ill_conditioned(self):
        """Test MPS rejects severely ill-conditioned problems."""
        backend = get_backend('mps')
        
        # Create severely collinear design
        n, p = 100, 3
        X = np.random.randn(n, 2)
        # Third column is linear combination + tiny noise
        X = np.column_stack([X, X[:, 0] + X[:, 1] + 1e-15 * np.random.randn(n)])
        y = np.random.randn(n)
        
        # Should raise ValueError for severe multicollinearity
        with pytest.raises(ValueError, match="multicollinearity"):
            backend.fit_linear_model(X, y)


class TestAutoBackend:
    """Test automatic backend selection."""
    
    def test_auto_backend_selects_something(self):
        """Test auto backend returns valid backend."""
        backend = get_backend('auto')
        assert backend is not None
        assert hasattr(backend, 'fit_linear_model')
    
    def test_auto_backend_consistency(self):
        """Test auto backend gives consistent results."""
        backend = get_backend('auto')
        
        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        # Run twice - should be deterministic
        result1 = backend.fit_linear_model(X, y)
        result2 = backend.fit_linear_model(X, y)
        
        assert np.allclose(result1.coef, result2.coef)
        assert np.allclose(result1.residuals, result2.residuals)
    
    @pytest.mark.skipif(not HAS_ANY_GPU, reason="No GPU available")
    def test_gpu_backend_selection(self):
        """Test 'gpu' backend routes to correct GPU type."""
        backend = get_backend('gpu')
        
        if HAS_NVIDIA:
            assert 'pytorch' in backend.name and 'cuda' in backend.name
        elif HAS_APPLE:
            assert 'mps' in backend.name and 'ridge' in backend.name


class TestBackendErrors:
    """Test error handling in backend selection."""
    
    def test_invalid_backend_name(self):
        """Test error on invalid backend name."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend('invalid_backend')
    
    def test_gpu_backend_without_gpu(self):
        """Test error when requesting GPU without GPU."""
        if not HAS_ANY_GPU:
            with pytest.raises(ValueError, match="No GPU detected"):
                get_backend('gpu')
    
    @pytest.mark.skipif(HAS_NVIDIA, reason="Test requires no NVIDIA GPU")
    def test_pytorch_without_nvidia(self):
        """Test error when requesting PyTorch without NVIDIA GPU."""
        with pytest.raises(RuntimeError):
            get_backend('pytorch')
    
    @pytest.mark.skipif(HAS_APPLE, reason="Test requires no Apple Silicon")
    def test_mlx_without_apple(self):
        """Test error when requesting MLX without Apple Silicon."""
        with pytest.raises(RuntimeError):
            get_backend('mlx')


class TestNumericalCorrectness:
    """Test numerical correctness across backends."""
    
    def test_perfect_fit(self):
        """Test all backends handle perfect fit correctly."""
        backends_to_test = ['cpu']
        if HAS_NVIDIA:
            backends_to_test.append('pytorch')
        if HAS_APPLE:
            backends_to_test.append('mlx')
        
        np.random.seed(42)
        n, p = 50, 3
        X = np.random.randn(n, p)
        beta_true = np.array([2.0, -1.0, 0.5])
        y = X @ beta_true  # Perfect fit, no noise
        
        for backend_name in backends_to_test:
            backend = get_backend(backend_name)
            
            # MLX will warn about ridge usage
            if backend_name == 'mlx':
                with pytest.warns(UserWarning, match="ridge regression"):
                    result = backend.fit_linear_model(X, y)
            else:
                result = backend.fit_linear_model(X, y)
            
            # Should recover true coefficients
            # MLX uses ridge, so slightly less accurate
            if backend_name == 'mlx':
                atol = 1e-2  # Ridge penalty causes small deviation
            else:
                atol = 1e-3
            
            assert np.allclose(result.coef[1:], beta_true, atol=atol)
            # Residuals should be near zero
            assert np.allclose(result.residuals, 0, atol=1e-3)
    
    def test_rank_deficient(self):
        """Test all backends handle rank deficiency.
        
        Note: MLX backend will reject severely rank-deficient problems.
        """
        backends_to_test = ['cpu']
        if HAS_NVIDIA:
            backends_to_test.append('pytorch')
        # MLX is NOT tested here - it rejects rank-deficient problems
        
        # Create rank-deficient matrix
        n, p = 50, 3
        X = np.random.randn(n, 2)
        X = np.column_stack([X, X[:, 0] + X[:, 1]])  # Third col is dependent
        y = np.random.randn(n)
        
        for backend_name in backends_to_test:
            backend = get_backend(backend_name)
            result = backend.fit_linear_model(X, y, singular_ok=True)
            
            # Rank should be less than p+1 (accounting for intercept)
            assert result.rank < p + 1
            # Should have NaN for aliased coefficient
            assert np.any(np.isnan(result.coef))


if __name__ == '__main__':
    # Print hardware detection
    print("\n" + "="*60)
    print("HARDWARE DETECTION")
    print("="*60)
    print_backend_info()
    print("\n" + "="*60)
    print("RUNNING TESTS")
    print("="*60 + "\n")
    
    # Run pytest
    pytest.main([__file__, '-v', '-s'])