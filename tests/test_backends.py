"""
Test backend implementations.
"""

import pytest
import numpy as np
from pyregression._backends import get_backend, list_available_backends


def test_list_backends():
    """Test that we can list available backends."""
    backends = list_available_backends()
    assert 'cpu' in backends
    assert isinstance(backends, list)


def test_get_cpu_backend():
    """Test CPU backend initialization."""
    backend = get_backend('cpu')
    assert backend is not None
    
    info = backend.get_device_info()
    assert info['backend'] == 'cpu'
    assert info['precision'] == 'fp64'


def test_qr_simple():
    """Test QR decomposition on simple matrix."""
    backend = get_backend('cpu')
    
    # Simple 3x2 matrix
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    
    result = backend.qr_with_pivoting(X)
    
    assert result.R.shape == (3, 2)
    assert len(result.pivot) == 2
    assert result.rank <= 2
    assert result.rank > 0


def test_apply_qt():
    """Test applying Q' to vector."""
    backend = get_backend('cpu')
    
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0])
    
    qr_result = backend.qr_with_pivoting(X)
    qty = backend.apply_qt_to_vector(qr_result.Q_implicit, y)
    
    assert len(qty) == 3
    assert np.all(np.isfinite(qty))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
