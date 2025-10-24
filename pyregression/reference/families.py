"""
GLM family objects.

Replicates R's family functions.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Callable


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
    
    @abstractmethod
    def dev_resids(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        wt: np.ndarray
    ) -> np.ndarray:
        """Deviance residuals."""
        pass
    
    def validmu(self, mu: np.ndarray) -> bool:
        """Check if μ values are valid."""
        return True
    
    def valideta(self, eta: np.ndarray) -> bool:
        """Check if η values are valid."""
        return True


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
    
    def dev_resids(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        wt: np.ndarray
    ) -> np.ndarray:
        return wt * (y - mu) ** 2


class Binomial(Family):
    """
    Binomial family with logit link.
    
    Replicates R's binomial() family exactly, including
    the thresholding at ±30 to prevent overflow.
    """
    
    # Thresholds from R's family.c
    THRESH = 30.0
    MTHRESH = -30.0
    EPS = np.finfo(np.float64).eps
    INVEPS = 1.0 / EPS
    
    @property
    def name(self) -> str:
        return "binomial"
    
    def linkfun(self, mu: np.ndarray) -> np.ndarray:
        """Logit link: η = log(μ/(1-μ))"""
        return np.log(mu / (1 - mu))
    
    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        """
        Inverse logit: μ = 1/(1 + exp(-η))
        
        With thresholding to prevent overflow (matches R exactly).
        """
        # Replicate R's family.c logic
        mu = np.empty_like(eta)
        
        # η < -30: return ε
        mu[eta < self.MTHRESH] = self.EPS
        
        # η > 30: return 1 - ε  
        mu[eta > self.THRESH] = 1 - self.EPS
        
        # -30 ≤ η ≤ 30: standard formula
        mask = (eta >= self.MTHRESH) & (eta <= self.THRESH)
        mu[mask] = 1.0 / (1.0 + np.exp(-eta[mask]))
        
        return mu
    
    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        """
        Derivative: dμ/dη = exp(η)/(1 + exp(η))²
        
        With thresholding (matches R exactly).
        """
        d = np.empty_like(eta)
        
        # Outside [-30, 30]: return ε
        outside = (eta < self.MTHRESH) | (eta > self.THRESH)
        d[outside] = self.EPS
        
        # Inside [-30, 30]: standard formula
        inside = ~outside
        exp_eta = np.exp(eta[inside])
        d[inside] = exp_eta / (1.0 + exp_eta) ** 2
        
        return d
    
    def variance(self, mu: np.ndarray) -> np.ndarray:
        """Variance: V(μ) = μ(1-μ)"""
        return mu * (1 - mu)
    
    def dev_resids(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        wt: np.ndarray
    ) -> np.ndarray:
        """Deviance residuals for binomial."""
        # TODO: Implement (matches R's binomial_dev_resids in family.c)
        raise NotImplementedError


class Poisson(Family):
    """Poisson family with log link."""
    
    @property
    def name(self) -> str:
        return "poisson"
    
    def linkfun(self, mu: np.ndarray) -> np.ndarray:
        return np.log(mu)
    
    def linkinv(self, eta: np.ndarray) -> np.ndarray:
        return np.exp(eta)
    
    def mu_eta(self, eta: np.ndarray) -> np.ndarray:
        return np.exp(eta)
    
    def variance(self, mu: np.ndarray) -> np.ndarray:
        return mu
    
    def dev_resids(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        wt: np.ndarray
    ) -> np.ndarray:
        # TODO: Implement
        raise NotImplementedError
