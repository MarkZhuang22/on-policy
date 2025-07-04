# /home/fsc-jupiter/source/Mark/AER1810/uav-defense-mappo/onpolicy/envs/uav_defense/aoa.py
# Angle-of-Arrival (AoA) utilities
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# small epsilon to avoid division by zero
_EPS = 1e-12


def bearing(p_intr: np.ndarray, p_def: np.ndarray) -> float:
    """
    Deterministic bearing:
        theta = atan2(y_i - y_d, x_i - x_d)  (Eq. 4.7)

    Parameters
    ----------
    p_intr : ndarray, shape (2,)
        Intruder position [x, y].
    p_def  : ndarray, shape (2,)
        Defender position [x, y].

    Returns
    -------
    theta : float
        Bearing in radians in (-pi, pi].
    """
    dx, dy = p_intr - p_def
    return math.atan2(dy, dx)


def sigma2_range(r: float, r0: float, sigma0: float) -> float:
    """
    Distance‑dependent AoA variance (Eq. 4.10):

      sigma^2(r) = sigma0^2                , r <= r0
                    sigma0^2 * (r / r0)^2  , r >  r0
    """
    if r <= r0:
        return sigma0 * sigma0
    return sigma0 * sigma0 * (r / r0) ** 2


def noisy_bearing(
    p_intr: np.ndarray,
    p_def: np.ndarray,
    r0: float,
    sigma0: float,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float]:
    """
    Generate one noisy AoA measurement:

        theta = bearing + n,  n ~ N(0, sigma^2(r))

    Returns measurement and the variance used.
    """
    if rng is None:
        rng = np.random.default_rng()
    r = np.linalg.norm(p_intr - p_def)
    var = sigma2_range(r, r0, sigma0)
    noise = rng.normal(0.0, math.sqrt(var))
    return bearing(p_intr, p_def) + noise, var


def jacobian(p_intr: np.ndarray, p_def: np.ndarray) -> np.ndarray:
    """
    Analytic Jacobian H = d(theta)/d(p) with respect to [x, y]
    (Eq. 4.11):

      H = 1 / r^2 * [ -delta_y , delta_x ]
    """
    dx, dy = p_intr - p_def
    r2 = dx * dx + dy * dy + _EPS
    return np.array([[-dy / r2, dx / r2]], dtype=float)


def fim(p_intr: np.ndarray, p_def: np.ndarray, var: float) -> np.ndarray:
    """
    2x2 Fisher information for one defender (Eq. 4.12):

      I = 1/(sigma^2 * r^4) [[dy^2, -dx*dy],
                             [-dx*dy, dx^2]]
    """
    dx, dy = p_intr - p_def
    r4 = (dx * dx + dy * dy + _EPS) ** 2
    c = 1.0 / (var * r4)
    return c * np.array([[dy * dy, -dx * dy],
                         [-dx * dy, dx * dx]], dtype=float)
