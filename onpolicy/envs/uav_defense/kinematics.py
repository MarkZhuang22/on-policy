# /home/fsc-jupiter/source/Mark/AER1810/uav-defense-mappo/onpolicy/envs/uav_defense/kinematics.py
# Constant‑Acceleration (CA / DWPA) kinematics
from __future__ import annotations

import numpy as np


def F_ca(T: float) -> np.ndarray:
    """State-transition matrix F_CA (6x6) for one time step T."""
    return np.array([
        [1, T, 0.5 * T**2, 0, 0, 0],
        [0, 1, T, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, T, 0.5 * T**2],
        [0, 0, 0, 0, 1, T],
        [0, 0, 0, 0, 0, 1],
    ], dtype=float)


def Q_ca(T: float, sigma_a: float) -> np.ndarray:
    """Process-noise covariance Q_CA for the CA model."""
    G = np.array([0.5 * T**2, T, 1.0], dtype=float)
    Q3 = sigma_a**2 * np.outer(G, G)
    Q = np.zeros((6, 6), dtype=float)
    Q[:3, :3] = Q3
    Q[3:, 3:] = Q3
    return Q


# Selector matrices to extract position/velocity/acceleration components
J_p = np.array([[1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0]], dtype=float)
J_v = np.array([[0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0]], dtype=float)
J_a = np.array([[0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1]], dtype=float)


def predict_state(x: np.ndarray, T: float) -> np.ndarray:
    """One-step mean propagation: x_{k+1} = F_ca(T) @ x_k."""
    return F_ca(T) @ x


def predict_cov(P: np.ndarray, T: float, sigma_a: float) -> np.ndarray:
    """One-step covariance propagation: P_{k+1} = F P F^T + Q_ca(T, sigma_a)."""
    F = F_ca(T)
    return F @ P @ F.T + Q_ca(T, sigma_a)


def mu_lookahead(p: np.ndarray, v: np.ndarray, a: np.ndarray, tau: float) -> np.ndarray:
    """
    Dead-reckoned position mean after look-ahead ``tau``:

        mu = p + v * tau + 0.5 * a * tau^2
    """
    return p + v * tau + 0.5 * a * tau**2


def cov_lookahead(Px: np.ndarray, tau: float, sigma_a: float) -> np.ndarray:
    """Dead-reckoned marginal position covariance after look-ahead ``tau``."""
    F_tau = F_ca(tau)
    Q_tau = Q_ca(tau, sigma_a)
    P_full = F_tau @ Px @ F_tau.T + Q_tau
    return J_p @ P_full @ J_p.T
