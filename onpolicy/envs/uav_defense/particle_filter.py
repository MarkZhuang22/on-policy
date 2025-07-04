# /home/fsc-jupiter/source/Mark/AER1810/uav-defense-mappo/onpolicy/envs/uav_defense/particle_filter.py
# Centralised Particle Filter (CPF) for bearing‑only tracking
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from numpy.random import Generator

from .kinematics import F_ca, Q_ca
from .aoa import bearing, sigma2_range


class ParticleFilter:
    """Centralised PF over ``N`` intruders (state dimension ``6N``)."""

    def __init__(
        self,
        n_intr: int,
        n_part: int,
        T: float,
        sigma_a: float,
        r0: float,
        sigma0: float,
        rng: Generator | None = None,
    ) -> None:
        self.n_intr = n_intr
        self.n_part = n_part
        self.T = T
        self.sigma_a = sigma_a
        self.r0 = r0
        self.sigma0 = sigma0
        self.rng = np.random.default_rng() if rng is None else rng

        self.state_dim = 6 * n_intr
        self.particles = np.zeros((n_part, self.state_dim), dtype=float)
        self.weights = np.full(n_part, 1.0 / n_part, dtype=float)

    def init_gaussian(self, mean: np.ndarray, cov: np.ndarray) -> None:
        """Initialise particles ~ N(mean, cov)."""
        self.particles = self.rng.multivariate_normal(mean, cov, size=self.n_part)
        self.weights.fill(1.0 / self.n_part)

    def predict(self) -> None:
        """
        Propagate each particle with CA model and process noise:
        x_{k+1} = F_ca @ x_k + noise, noise ~ N(0, Q_ca).
        """
        F = F_ca(self.T)
        Q = Q_ca(self.T, self.sigma_a)
        L = np.linalg.cholesky(Q)
        for idx in range(self.n_part):
            x = self.particles[idx]
            for i in range(self.n_intr):
                base = 6 * i
                xi = x[base:base + 6]
                xi = F @ xi + L @ self.rng.standard_normal(6)
                x[base:base + 6] = xi
            self.particles[idx] = x

    def update(
        self,
        bearings: List[Tuple[int, int, float]],
        p_defenders: np.ndarray,
    ) -> None:
        """Weight update using AoA measurements."""
        log_w = np.log(self.weights + 1e-300)
        for idx, part in enumerate(self.particles):
            lw = 0.0
            for d, i, theta_meas in bearings:
                pos_i = self._pos_of(part, i)
                r = np.linalg.norm(pos_i - p_defenders[d])
                var = sigma2_range(r, self.r0, self.sigma0)
                mu = bearing(pos_i, p_defenders[d])
                err = wrap_angle(theta_meas - mu)
                lw += -0.5 * (err ** 2) / var - 0.5 * math.log(2 * math.pi * var)
            log_w[idx] += lw

        log_w -= logsumexp(log_w)
        self.weights = np.exp(log_w)

        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < 0.5 * self.n_part:
            self._resample_stratified()


    def estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return posterior mean and covariance."""
        mu = np.average(self.particles, axis=0, weights=self.weights)
        Xc = self.particles - mu
        P = (self.weights[:, None] * Xc).T @ Xc
        return mu, P

    def _pos_of(self, part: np.ndarray, i: int) -> np.ndarray:
        """Extract ``[x, y]`` for intruder ``i`` from flat state."""
        base = 6 * i
        return part[[base, base + 3]]

    def _extract_position_mean(self, i: int) -> np.ndarray:
        """Weighted mean position of intruder ``i``."""
        cols = [6 * i, 6 * i + 3]
        return np.average(self.particles[:, cols], axis=0, weights=self.weights)

    def _resample_stratified(self) -> None:
        """Stratified resampling with uniform weights."""
        cum = np.cumsum(self.weights)
        u = (self.rng.random(self.n_part) + np.arange(self.n_part)) / self.n_part
        idx = np.searchsorted(cum, u)
        self.particles = self.particles[idx]
        self.weights.fill(1.0 / self.n_part)


# ----------------------------------------------------------------------
# Helper functions

def logsumexp(a: np.ndarray) -> float:
    """Stable log-sum-exp."""
    amax = np.max(a)
    return amax + math.log(np.sum(np.exp(a - amax)))


def wrap_angle(phi: float) -> float:
    """Wrap angle to the range (-pi, pi]."""
    return (phi + math.pi) % (2 * math.pi) - math.pi
