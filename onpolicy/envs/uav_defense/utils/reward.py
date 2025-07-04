from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np


class RewardConfig:
    # 拦截成功 +R_cap
    R_cap: float = 1.0
    # 任何 intruder 进入保护区 ⇒ −R_violate 给所有 defenders
    R_violate: float = -1.0
    # 同一 intruder 被 ≥2 个 defenders 测向且夹角 > thresh ⇒ 每个参与测量的 defender +R_bearing_pair
    R_bearing_pair: float = 0.1

    # 几何参数
    capture_radius: float = 2.0
    protected_radius: float = 50.0
    # 同一入侵者两条测向的最小夹角阈值（rad）
    bearing_pair_angle_thresh: float = 15.0 * math.pi / 180.0  


def _pairwise_angles(vecs: np.ndarray) -> np.ndarray:
    """
    计算 n 个向量两两夹角（只返回上三角部分），单位 rad。
    """
    n = len(vecs)
    if n < 2:
        return np.zeros(0, dtype=float)
    angs = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            v1 = vecs[i] / (np.linalg.norm(vecs[i]) + 1e-9)
            v2 = vecs[j] / (np.linalg.norm(vecs[j]) + 1e-9)
            cos_ = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angs.append(math.acos(cos_))
    return np.array(angs, dtype=float)


def compute_capture_reward(
    pos_def: np.ndarray,
    pos_intr: np.ndarray,
    cfg: RewardConfig,
) -> List[float]:
    """
    对每个 defender 计算捕获奖励：
      若 any intruder 与 defender 距离 ≤ capture_radius ⇒ +R_cap
    返回长度 = n_def 的列表。
    """
    n_def = pos_def.shape[0]
    rewards = [0.0] * n_def
    for d in range(n_def):
        for p_i in pos_intr:
            if np.linalg.norm(p_i - pos_def[d]) <= cfg.capture_radius:
                rewards[d] += cfg.R_cap
    return rewards


def compute_violate_penalty(
    pos_intr: np.ndarray,
    cfg: RewardConfig,
) -> float:
    """
    若 any intruder 进入保护区 ⇒ 对所有 defenders 扣 R_violate（一致值）
    """
    for p_i in pos_intr:
        if np.linalg.norm(p_i) <= cfg.protected_radius:
            return cfg.R_violate
    return 0.0


def compute_bearing_pair_bonus(
    bearings_by_def: List[List[Tuple[int, np.ndarray]]],
    cfg: RewardConfig,
) -> List[float]:
    """
    对每个 defender 计算“测向对”奖励：
    输入 bearings_by_def[d] = [(intr_id, vec), ...] 
      表示 defender d 本步对哪些 intruder 做了测向 (vec 为测向的 unit vector)
    若同一 intruder 被 ≥2 个 defenders 测向，且这些向量两两夹角 > threshold
      则给所有参与测向的 defenders 各 +R_bearing_pair
    返回长度 = n_def 的列表。
    """
    n_def = len(bearings_by_def)
    bonus = [0.0] * n_def

    # 按 intruder 聚合测向记录： intr_id -> list of (d, vec)
    by_intr: dict[int, List[Tuple[int, np.ndarray]]] = {}
    for d, recs in enumerate(bearings_by_def):
        for intr_id, vec in recs:
            by_intr.setdefault(intr_id, []).append((d, vec))

    # 对每个 intruder 的记录做角度判断
    for recs in by_intr.values():
        if len(recs) < 2:
            continue
        vecs = np.stack([v for _, v in recs], axis=0)
        if (_pairwise_angles(vecs) > cfg.bearing_pair_angle_thresh).any():
            for d, _ in recs:
                bonus[d] += cfg.R_bearing_pair

    return bonus


def total_reward(
    pos_def: np.ndarray,
    pos_intr: np.ndarray,
    bearings_by_def: List[List[Tuple[int, np.ndarray]]],
    cfg: RewardConfig = RewardConfig(),
) -> List[float]:
    """
    汇总三项奖励，返回长度 = n_def 的列表，与 env.step() 要求的 reward_list 对应。
    """
    # 捕获奖励
    r_cap = compute_capture_reward(pos_def, pos_intr, cfg)
    # 进入禁区惩罚
    r_zone = compute_violate_penalty(pos_intr, cfg)
    # 测向对奖励
    r_pair = compute_bearing_pair_bonus(bearings_by_def, cfg)

    # 合并，进入禁区惩罚作用于所有 defenders
    return [r_cap[i] + r_pair[i] + r_zone for i in range(len(r_cap))]
