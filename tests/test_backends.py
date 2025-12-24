"""
Test backend implementations with auto-detection.

Tests appropriate backends based on available hardware:
- CPU: Always tested
- PyTorch CUDA: Tested if NVIDIA GPU available
- MPS: Tested if Apple Silicon available
"""

import pytest
import numpy as np
from pyregression._backends import (
    get_backend,
    list_available_backends,
    print_backend_info,
    CPU_AVAILABLE,
    PYTORCH_FP32_AVAILABLE,
    PYTORCH_FP64_AVAILABLE,
    MPS_AVAILABLE,
)
from pyregression._backends.precision_detector import detect_gpu_capabilities


# Detect hardware once at module level
GPU_CAPS = detect_gpu_capabilities()
HAS_NVIDIA = GPU_CAPS.gpu_type == 'nvidia'
HAS_MPS = GPU_CAPS.gpu_type == 'mps'
HAS_ANY_GPU = GPU_CAPS.has_gpu


class TestBackendDetection:
    """Test hardware detection and backend availability."""
    
    def test_detect_gpu_capabilities(self):
        """Test GPU detection returns valid capabilities."""
        caps = detect_gpu_capabilities()
        assert caps.gpu_name is not None
        assert caps.gpu_type in ['nvidia', 'mps', 'none']
        assert caps.has_gpu == (caps.gpu_type != 'none')
    
    def test_list_backends(self):
        """Test backend listing."""
        backends = list_available_backends()
        assert isinstance(backends, list)
        assert 'cpu' in backends  # CPU always available
        
        # Check GPU backends match detection
        if HAS_NVIDIA:
            assert 'pytorch' in backends
        if HAS_MPS:
            assert 'mps' in backends
    
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
        
        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)
        beta_true = np.array([1.5, -2.0, 0.75])
        y = X @ beta_true + 0.1 * np.random.randn(n)
        
        result = backend.fit_linear_model(X, y)
        
        # Should recover coefficients accurately
        assert np.allclose(result.coef[1:], beta_true, atol=0.2)
        assert result.rank == p + 1  # Full rank with intercept
    
    def test_cpu_weighted_regression(self):
        """Test weighted regression on CPU."""
        backend = get_backend('cpu')
        
        np.random.seed(42)
        n, p = 100, 2
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        weights = np.random.uniform(0.5, 2.0, n)
        
        result = backend.fit_linear_model(X, y, weights=weights)
        assert result.coef.shape == (p + 1,)
    
    def test_cpu_with_offset(self):
        """Test regression with offset on CPU."""
        backend = get_backend('cpu')
        
        np.random.seed(42)
        n, p = 100, 2
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        offset = np.random.randn(n) * 0.5
        
        result = backend.fit_linear_model(X, y, offset=offset)
        assert result.coef.shape == (p + 1,)
    
    def test_cpu_rejects_ill_conditioned(self):
        """Test CPU backend rejects severely ill-conditioned matrices."""
        backend = get_backend('cpu')
        
        # Create nearly singular matrix
        X = np.random.randn(50, 3)
        X[:, 2] = X[:, 0] + 1e-15 * np.random.randn(50)
        y = np.random.randn(50)
        
        with pytest.raises(ValueError, match="[Ii]ll.conditioned|[Ss]ingular"):
            backend.fit_linear_model(X, y, singular_ok=False)


@pytest.mark.skipif(not HAS_NVIDIA, reason="Requires NVIDIA GPU")
class TestPyTorchBackend:
    """Test PyTorch CUDA backend (NVIDIA only)."""
    
    def test_pytorch_backend_creation(self):
        """Test PyTorch backend initializes correctly."""
        backend = get_backend('pytorch')
        assert backend is not None
        assert 'pytorch' in backend.name
        assert 'cuda' in backend.name
    
    def test_pytorch_device_info(self):
        """Test PyTorch backend device info."""
        backend = get_backend('pytorch')
        info = backend.get_device_info()
        assert 'pytorch' in info['backend']
        assert 'cuda' in info['device']
    
    def test_pytorch_simple_regression(self):
        """Test simple regression on PyTorch."""
        backend = get_backend('pytorch')
        
        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)
        beta_true = np.array([1.5, -2.0, 0.75])
        y = X @ beta_true + 0.1 * np.random.randn(n)
        
        result = backend.fit_linear_model(X, y)
        
        # Should recover coefficients accurately
        assert np.allclose(result.coef[1:], beta_true, atol=0.2)
    
    def test_pytorch_vs_cpu_consistency(self):
        """Test PyTorch matches CPU results."""
        cpu_backend = get_backend('cpu')
        gpu_backend = get_backend('pytorch')
        
        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        cpu_result = cpu_backend.fit_linear_model(X, y)
        gpu_result = gpu_backend.fit_linear_model(X, y)
        
        # Should match to high precision
        assert np.allclose(cpu_result.coef, gpu_result.coef, rtol=1e-5, atol=1e-6)
        assert np.allclose(cpu_result.residuals, gpu_result.residuals, rtol=1e-5, atol=1e-5)
    
    def test_pytorch_weighted_regression(self):
        """Test weighted regression on PyTorch."""
        backend = get_backend('pytorch')
        
        np.random.seed(42)
        n, p = 100, 2
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        weights = np.random.uniform(0.5, 2.0, n)
        
        result = backend.fit_linear_model(X, y, weights=weights)
        assert result.coef.shape == (p + 1,)
    
    def test_pytorch_with_offset(self):
        """Test regression with offset on PyTorch."""
        backend = get_backend('pytorch')
        
        np.random.seed(42)
        n, p = 100, 2
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        offset = np.random.randn(n) * 0.5
        
        result = backend.fit_linear_model(X, y, offset=offset)
        assert result.coef.shape == (p + 1,)
    
    def test_pytorch_rejects_ill_conditioned(self):
        """Test PyTorch backend rejects severely ill-conditioned matrices."""
        backend = get_backend('pytorch')
        
        # Create nearly singular matrix
        X = np.random.randn(50, 3)
        X[:, 2] = X[:, 0] + 1e-15 * np.random.randn(50)
        y = np.random.randn(50)
        
        with pytest.raises(ValueError, match="[Ii]ll.conditioned|[Ss]ingular"):
            backend.fit_linear_model(X, y, singular_ok=False)


@pytest.mark.skipif(not HAS_MPS, reason="Requires Apple Silicon with MPS")
class TestMPSBackend:
    """Test MPS backend (Apple Silicon only)."""
    
    def test_mps_backend_creation(self):
        """Test MPS backend initializes correctly."""
        backend = get_backend('mps')
        assert backend is not None
        assert 'mps' in backend.name
        assert 'ridge' in backend.name
    
    def test_mps_device_info(self):
        """Test MPS backend device info."""
        backend = get_backend('mps')
        info = backend.get_device_info()
        # MPS backend returns 'gpu' as backend type
        assert info['backend'] == 'gpu'
        # Check for MPS/Metal in device string
        assert 'MPS' in info['device'] or 'Metal' in info['device']
    
    def test_mps_simple_regression(self):
        """Test simple regression on MPS."""
        backend = get_backend('mps')
        
        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)
        beta_true = np.array([1.5, -2.0, 0.75])
        y = X @ beta_true + 0.1 * np.random.randn(n)
        
        # MPS uses ridge, so expect warning
        with pytest.warns(UserWarning, match="ridge regression"):
            result = backend.fit_linear_model(X, y)
        
        # Ridge has slight bias, so looser tolerance
        assert np.allclose(result.coef[1:], beta_true, atol=0.3)
    
    def test_mps_vs_cpu_consistency(self):
        """Test MPS is close to CPU results (with ridge tolerance)."""
        cpu_backend = get_backend('cpu')
        mps_backend = get_backend('mps')
        
        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        cpu_result = cpu_backend.fit_linear_model(X, y)
        
        with pytest.warns(UserWarning, match="ridge regression"):
            mps_result = mps_backend.fit_linear_model(X, y)
        
        # Ridge introduces small differences
        assert np.allclose(cpu_result.coef, mps_result.coef, rtol=1e-2, atol=1e-2)
    
    def test_mps_weighted_regression(self):
        """Test weighted regression on MPS."""
        backend = get_backend('mps')
        
        np.random.seed(42)
        n, p = 100, 2
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        weights = np.random.uniform(0.5, 2.0, n)
        
        with pytest.warns(UserWarning, match="ridge regression"):
            result = backend.fit_linear_model(X, y, weights=weights)
        assert result.coef.shape == (p + 1,)
    
    def test_mps_with_offset(self):
        """Test regression with offset on MPS."""
        backend = get_backend('mps')
        
        np.random.seed(42)
        n, p = 100, 2
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        offset = np.random.randn(n) * 0.5
        
        with pytest.warns(UserWarning, match="ridge regression"):
            result = backend.fit_linear_model(X, y, offset=offset)
        assert result.coef.shape == (p + 1,)
    
    def test_mps_rejects_ill_conditioned(self):
        """Test MPS backend rejects severely ill-conditioned matrices."""
        backend = get_backend('mps')
        
        # Create nearly singular matrix
        X = np.random.randn(50, 3)
        X[:, 2] = X[:, 0] + 1e-15 * np.random.randn(50)
        y = np.random.randn(50)
        
        # MPS backend raises ValueError with "multicollinearity" message
        with pytest.raises(ValueError, match="multicollinearity"):
            backend.fit_linear_model(X, y, singular_ok=False)


class TestAutoBackend:
    """Test auto backend selection."""
    
    def test_auto_backend_selects_something(self):
        """Test auto backend returns valid backend."""
        backend = get_backend('auto')
        assert backend is not None
        
        if HAS_NVIDIA:
            assert 'pytorch' in backend.name or 'cuda' in backend.name
        elif HAS_MPS:
            assert 'mps' in backend.name and 'ridge' in backend.name
        else:
            assert 'cpu' in backend.name
    
    def test_auto_backend_consistency(self):
        """Test auto backend produces reasonable results."""
        backend = get_backend('auto')
        
        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)
        y = np.random.randn(n)
        
        # May warn on MPS
        if HAS_MPS and not HAS_NVIDIA:
            with pytest.warns(UserWarning, match="ridge regression"):
                result = backend.fit_linear_model(X, y)
        else:
            result = backend.fit_linear_model(X, y)
        
        assert result.coef.shape == (p + 1,)
        assert result.rank > 0
    
    def test_gpu_backend_selection(self):
        """Test GPU backend selection logic."""
        if not HAS_ANY_GPU:
            pytest.skip("Requires GPU")
        
        backend = get_backend('gpu')
        
        if HAS_NVIDIA:
            assert 'pytorch' in backend.name or 'cuda' in backend.name
        elif HAS_MPS:
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
    
    @pytest.mark.skipif(HAS_MPS, reason="Test requires no Apple Silicon")
    def test_mps_without_apple(self):
        """Test error when requesting MPS without Apple Silicon."""
        with pytest.raises(RuntimeError):
            get_backend('mps')


class TestNumericalCorrectness:
    """Test numerical correctness across backends."""
    
    def test_perfect_fit(self):
        """Test all backends handle perfect fit correctly."""
        backends_to_test = ['cpu']
        if HAS_NVIDIA:
            backends_to_test.append('pytorch')
        if HAS_MPS:
            backends_to_test.append('mps')
        
        np.random.seed(42)
        n, p = 50, 3
        X = np.random.randn(n, p)
        beta_true = np.array([2.0, -1.0, 0.5])
        y = X @ beta_true  # Perfect fit, no noise
        
        for backend_name in backends_to_test:
            backend = get_backend(backend_name)
            
            # MPS will warn about ridge usage
            if backend_name == 'mps':
                with pytest.warns(UserWarning, match="ridge regression"):
                    result = backend.fit_linear_model(X, y)
                # Ridge penalty causes small deviation
                atol = 1e-2
            else:
                result = backend.fit_linear_model(X, y)
                atol = 1e-3
            
            # Should recover true coefficients
            assert np.allclose(result.coef[1:], beta_true, atol=atol)
            # Residuals should be near zero
            assert np.allclose(result.residuals, 0, atol=1e-3)
    
    def test_rank_deficient(self):
        """Test all backends handle rank deficiency.
        
        Note: MPS backend uses ridge regularization, so skip rank deficiency test.
        """
        backends_to_test = ['cpu']
        if HAS_NVIDIA:
            backends_to_test.append('pytorch')
        # MPS uses ridge regularization, so skip rank deficiency test
        
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