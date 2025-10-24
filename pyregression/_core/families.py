"""
GLM family definitions.

Defines link functions, variance functions, etc.
"""

import numpy as np
from abc import ABC, abstractmethod


class Family(ABC):
    """Base class for GLM families."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Family name."""
        pass
    
    @abstractmethod
    def linkfun(self, mu: np.ndarray) -> np.ndarray:
        """Link function: η = g(μ)"""
        pass
    
    @abstractmethod
    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        """Inverse link: μ = g⁻¹(η)"""
        pass
    
    @abstractmethod
    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        """Derivative: dμ/dη"""
        pass
    
    @abstractmethod
    def variance(self, mu: np.ndarray) -> np.ndarray:
        """Variance function: V(μ)"""
        pass


class Gaussian(Family):
    """Gaussian family with identity link."""
    
    @property
    def name(self) -> str:
        return "gaussian"
    
    def linkfun(self, mu: np.ndarray) -> np.ndarray:
        return mu
    
    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        return eta
    
    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        return np.ones_like(eta)
    
    def variance(self, mu: np.ndarray) -> np.ndarray:
        return np.ones_like(mu)


class Binomial(Family):
    """Binomial family with logit link."""
    
    # Thresholds from R
    THRESH = 30.0
    MTHRESH = -30.0
    EPS = np.finfo(np.float64).eps
    
    @property
    def name(self) -> str:
        return "binomial"
    
    def linkfun(self, mu: np.ndarray) -> np.ndarray:
        return np.log(mu / (1 - mu))
    
    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        mu = np.empty_like(eta)
        mu[eta < self.MTHRESH] = self.EPS
        mu[eta > self.THRESH] = 1 - self.EPS
        mask = (eta >= self.MTHRESH) & (eta <= self.THRESH)
        mu[mask] = 1.0 / (1.0 + np.exp(-eta[mask]))
        return mu
    
    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        d = np.empty_like(eta)
        outside = (eta < self.MTHRESH) | (eta > self.THRESH)
        d[outside] = self.EPS
        inside = ~outside
        exp_eta = np.exp(eta[inside])
        d[inside] = exp_eta / (1.0 + exp_eta) ** 2
        return d
    
    def variance(self, mu: np.ndarray) -> np.ndarray:
        return mu * (1 - mu)


__all__ = ["Family", "Gaussian", "Binomial"]
