from __future__ import annotations
import math
from typing import Iterable, List, Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse
from matplotlib.animation import FuncAnimation

# 从父包中导入
from ..aoa import bearing
from ..threat_metrics import circumscribed_radius


def draw_world(
    ax: plt.Axes,
    pos_def: np.ndarray,
    pos_intr: np.ndarray,
    Rz: float,
    title: str | None = None,
):
    """2D 顶视：保护圈、defenders（蓝）、intruders（橙×）"""
    ax.clear()
    ax.set_aspect("equal", adjustable="box")
    # 保护圈
    ax.add_patch(Circle((0, 0), Rz, fill=False, linestyle="--", color="red", lw=2))
    # defenders
    ax.scatter(pos_def[:, 0], pos_def[:, 1], c="blue", label="Defenders")
    # intruders
    ax.scatter(pos_intr[:, 0], pos_intr[:, 1], c="orange", marker="x", label="Intruders")
    # 视野范围
    lim = Rz * 5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    if title:
        ax.set_title(title)
    ax.legend(loc="upper right")


def draw_aoa_beams(
    ax: plt.Axes,
    pos_def: np.ndarray,
    measurements: Iterable[Tuple[int, float]],
    length: float = 100.0,
    **kwargs,
):
    """
    画 AoA 测向线。
    measurements: iterable of (defender_index d, bearing θ).
    length: 线段长度.
    kwargs: 透传给 ax.plot(), e.g. color, linestyle。
    """
    for d, theta in measurements:
        x0, y0 = pos_def[d]
        x1 = x0 + length * math.cos(theta)
        y1 = y0 + length * math.sin(theta)
        ax.plot([x0, x1], [y0, y1], **kwargs)


def draw_confidence(
    ax: plt.Axes,
    mean: np.ndarray,
    cov: np.ndarray,
    nsig: float = 3.0,
    draw_circle: bool = True,
    ellipse_kwargs: Dict[str, Any] = None,
    circle_kwargs: Dict[str, Any] = None,
):
    """
    在 ax 上画 3σ 椭圆及其外接圆。
    - mean: (2,) 均值
    - cov: (2,2) 协方差
    - nsig: 椭圆 sigma 数
    - draw_circle: 是否同时画 circumscribed circle
    - ellipse_kwargs / circle_kwargs: 额外样式
    """
    ellipse_kwargs = ellipse_kwargs or {"edgecolor": "green", "fc": "none", "ls": "--", "lw": 1.5}
    circle_kwargs = circle_kwargs or {"edgecolor": "magenta", "fc": "none", "ls": ":", "lw": 1.5}

    # 椭圆
    lam, vec = np.linalg.eigh(cov)
    order = lam.argsort()[::-1]
    lam = lam[order]
    vec = vec[:, order]
    width, height = 2 * nsig * np.sqrt(lam)
    angle = math.degrees(math.atan2(vec[1, 0], vec[0, 0]))
    e = Ellipse(xy=mean, width=width, height=height, angle=angle, **ellipse_kwargs)
    ax.add_patch(e)

    if draw_circle:
        rho = circumscribed_radius(cov)
        c = Circle(tuple(mean), rho, **circle_kwargs)
        ax.add_patch(c)


def animate_run(
    frames: List[Dict[str, Any]],
    Rz: float,
    interval: int = 150,
):
    """
    frames: 每帧一个 dict，至少包含：
        'pos_def': np.ndarray (M,2)
        'pos_intr': np.ndarray (N,2)
      可选字段：
        'measurements': List[(d:int, θ:float)]
        'means': List[np.ndarray (2,)]      # N intruder mean
        'covs':   List[np.ndarray (2,2)]    # N intruder cov
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    lim = Rz * 5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")

    def _update(idx):
        fr = frames[idx]
        draw_world(ax, fr["pos_def"], fr["pos_intr"], Rz, title=f"Step {idx}")
        # AoA beams
        if "measurements" in fr:
            draw_aoa_beams(ax, fr["pos_def"], fr["measurements"],
                           length=Rz * 2, color="gray", alpha=0.7)
        # 置信椭圆 + 外接圆
        if "means" in fr and "covs" in fr:
            for mu, cov in zip(fr["means"], fr["covs"]):
                draw_confidence(ax, mu, cov)
        return []

    ani = FuncAnimation(fig, _update, frames=len(frames), interval=interval, blit=True)
    return ani
