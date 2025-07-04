# /home/fsc-jupiter/source/Mark/AER1810/uav-defense-mappo/onpolicy/envs/uav_defense/threat_metrics.py
# Threat metric calculations
#  - 3σ confidence ellipse -> circumscribed circle radius rho(tau)
#  - signed distance d(tau) to protected zone
#  - earliest collision time tau_hit via bracket + Brent
#  - threat weight T = exp(-lambda * tau_hit)

from __future__ import annotations

import math

import numpy as np
from scipy import optimize

from .kinematics import mu_lookahead, cov_lookahead


def circumscribed_radius(Pp: np.ndarray) -> float:
    """
    3σ ellipse major semi-axis -> circumscribed circle radius ``rho``.
    Eq. (2.15): ``rho = 3 * sqrt(lambda_max(Pp))``.
    """
    lam_max = np.linalg.eigvalsh(Pp).max()
    return 3.0 * math.sqrt(lam_max)


def signed_distance(mu: np.ndarray, rho: float, p_a: np.ndarray, r_zone: float) -> float:
    """
    Signed distance ``d(tau) = ||mu - p_a|| - r_zone - rho``.
    ``d <= 0`` indicates the first intersection.
    """
    return np.linalg.norm(mu - p_a) - r_zone - rho


def earliest_collision_time(
    p0: np.ndarray,
    v0: np.ndarray,
    a0: np.ndarray,
    Px0: np.ndarray,
    p_a: np.ndarray,
    r_zone: float,
    tau_max: float,
    sigma_a: float,
    brent_tol: float = 1e-3,
    coarse_steps: int = 200,
) -> float:
    """
    Find ``tau_hit = min tau in (0, tau_max]`` such that ``d(tau) = 0``.
    If already intersecting at ``tau = 0`` return 0.
    """
    Pp0 = cov_lookahead(Px0, 0.0, sigma_a)
    rho0 = circumscribed_radius(Pp0)
    d0 = signed_distance(p0, rho0, p_a, r_zone)
    if d0 <= 0.0:
        return 0.0

    tau_prev = 0.0
    d_prev = d0
    for k in range(1, coarse_steps + 1):
        tau = (k / coarse_steps) * tau_max
        mu = mu_lookahead(p0, v0, a0, tau)
        Pp = cov_lookahead(Px0, tau, sigma_a)
        rho = circumscribed_radius(Pp)
        d = signed_distance(mu, rho, p_a, r_zone)
        if d <= 0.0 and d_prev > 0.0:
            def f(t: float) -> float:
                mu_t = mu_lookahead(p0, v0, a0, t)
                Pp_t = cov_lookahead(Px0, t, sigma_a)
                rho_t = circumscribed_radius(Pp_t)
                return signed_distance(mu_t, rho_t, p_a, r_zone)
            try:
                return optimize.brentq(f, tau_prev, tau, xtol=brent_tol)
            except ValueError:
                return tau
        tau_prev, d_prev = tau, d

    return tau_max


def threat_weight(tau_hit: float, lam: float = 0.05) -> float:
    """Exponential threat weight ``T = exp(-lambda * tau_hit)``."""
    return math.exp(-lam * tau_hit)
