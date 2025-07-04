# /home/fsc-jupiter/source/Mark/AER1810/uav-defense-mappo/onpolicy/envs/uav_defense/uav_defense_env.py
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .aoa import noisy_bearing


def random_outside_circle(rng: np.random.Generator,
                          r_min: float,
                          r_max: float) -> np.ndarray:
    """Return random point with radius in [r_min, r_max]."""
    r = rng.uniform(r_min, r_max)
    theta = rng.uniform(-math.pi, math.pi)
    return np.array([r * math.cos(theta), r * math.sin(theta)],
                    dtype=np.float32)


class UAVDefenseEnv(ParallelEnv):
    """
    Parallel PettingZoo environment for M defenders and N intruders.
    Each defender outputs a 2-D velocity command in [-v_def, +v_def]^2.
    Observations: [x, y, v_x, v_y, theta_0…theta_{N-1}].
    """

    metadata = {"name": "uav_defense_v0", "render_modes": ["human"]}

    def __init__(
        self,
        n_def: int = 5,
        n_intr: int = 2,
        protected_radius: float = 50.0,
        world_radius: float = 500.0,
        max_steps: int = 600,
        dt: float = 0.2,
        v_def: float = 15.0,
        v_intr: float = 30.0,
        r0: float = 150.0,
        sigma0: float = 0.02,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        # ---- parameters ----
        self.n_def = n_def
        self.n_intr = n_intr
        self.Rz = protected_radius
        self.Rw = world_radius
        self.max_steps = max_steps
        self.dt = dt
        self.v_def = v_def
        self.v_intr = v_intr
        self.r0 = r0
        self.sigma0 = sigma0

        self.rng = np.random.default_rng(seed)

        # agent names
        self.possible_agents = [f"D_{i}" for i in range(self.n_def)]
        self.agents: List[str] = []

        # positions, velocities and intruder headings
        self.pos_def = np.zeros((n_def, 2), dtype=np.float32)
        self.vel_def = np.zeros((n_def, 2), dtype=np.float32)
        self.pos_intr = np.zeros((n_intr, 2), dtype=np.float32)
        self.head_intr = np.zeros(n_intr, dtype=np.float32)
        self.intruder_active = np.ones(n_intr, dtype=bool)

        self._step_count = 0

        # ---- observation & action spaces ----
        low = np.array(
            [-np.inf, -np.inf, -self.v_def, -self.v_def]
            + [-math.pi] * self.n_intr,
            dtype=np.float32,
        )
        high = np.array(
            [+np.inf, +np.inf, +self.v_def, +self.v_def]
            + [math.pi] * self.n_intr,
            dtype=np.float32,
        )

        obs_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = [obs_space for _ in range(self.n_def)]
        self.share_observation_space = [obs_space for _ in range(self.n_def)]
        act_space = spaces.Box(
            low=np.array([-self.v_def, -self.v_def], dtype=np.float32),
            high=np.array([self.v_def, self.v_def], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = [act_space for _ in range(self.n_def)]

    # ------------------------------------------------------------------

    def reset(
        self, seed: Optional[int] = None, options=None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._step_count = 0
        self.agents = self.possible_agents[:]
        self.intruder_active[:] = True

        # defenders on the protected circle, zero velocity
        for i in range(self.n_def):
            theta = 2 * math.pi * i / self.n_def
            self.pos_def[i] = self.Rz * np.array(
                [math.cos(theta), math.sin(theta)], dtype=np.float32
            )
            self.vel_def[i].fill(0.0)

        # intruders randomly outside, heading toward zone centre
        for j in range(self.n_intr):
            self.pos_intr[j] = random_outside_circle(
                self.rng, self.Rz + 80, self.Rw
            )
            tgt = -self.pos_intr[j]
            self.head_intr[j] = (
                math.atan2(tgt[1], tgt[0]) + self.rng.normal(0, 0.3)
            )

        observations = self._get_obs_dict()
        infos = {a: {} for a in self.agents}
        return observations, infos

    # ------------------------------------------------------------------

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        self._step_count += 1

        # defenders move according to commands
        for i, a in enumerate(self.agents):
            v_cmd = actions.get(a, np.zeros(2, dtype=np.float32))
            v = np.clip(v_cmd, -self.v_def, self.v_def)
            self.vel_def[i] = v
            self.pos_def[i] += v * self.dt

        # intruders move if still active
        for j in range(self.n_intr):
            if not self.intruder_active[j]:
                continue
            move = (
                self.v_intr
                * self.dt
                * np.array(
                    [math.cos(self.head_intr[j]), math.sin(self.head_intr[j])],
                    dtype=np.float32,
                )
            )
            self.pos_intr[j] += move

        rewards = self._compute_simple_reward()

        observations = self._get_obs_dict()
        reward_dict = {a: rewards[i] for i, a in enumerate(self.agents)}
        terminations = {a: False for a in self.agents}
        truncations = {
            a: self._step_count >= self.max_steps for a in self.agents
        }
        infos = {a: {} for a in self.agents}

        if all(truncations.values()):
            self.agents = []

        return observations, reward_dict, terminations, truncations, infos

    # ------------------------------------------------------------------

    def _get_obs_dict(self) -> Dict[str, np.ndarray]:
        obs: Dict[str, np.ndarray] = {}
        for i, a in enumerate(self.agents):
            vec = np.empty(4 + self.n_intr, dtype=np.float32)
            vec[0:2] = self.pos_def[i]
            vec[2:4] = self.vel_def[i]
            for j in range(self.n_intr):
                if self.intruder_active[j]:
                    theta, _ = noisy_bearing(
                        self.pos_intr[j],
                        self.pos_def[i],
                        self.r0,
                        self.sigma0,
                        self.rng,
                    )
                else:
                    theta = 0.0
                vec[4 + j] = theta
            obs[a] = vec
        return obs

    # ------------------------------------------------------------------

    def _compute_simple_reward(self) -> List[float]:
        # Intruder inside zone => negative reward for all defenders.
        # Interception within 2 m => positive reward for interceptor.
        rewards = [0.0] * self.n_def
        for j in range(self.n_intr):
            if not self.intruder_active[j]:
                continue
            if np.linalg.norm(self.pos_intr[j]) <= self.Rz:
                for i in range(self.n_def):
                    rewards[i] -= 1.0
                self.intruder_active[j] = False
                continue
        for i in range(self.n_def):
            for j in range(self.n_intr):
                if (
                    self.intruder_active[j]
                    and np.linalg.norm(self.pos_intr[j] - self.pos_def[i])
                    <= 2.0
                ):
                    rewards[i] += 1.0
                    self.intruder_active[j] = False
        return rewards

    # ------------------------------------------------------------------

    def render(self, mode: str = "human") -> None:
        print(f"Step {self._step_count}")
        for i, a in enumerate(self.agents):
            print(f"{a}: pos={self.pos_def[i]}, vel={self.vel_def[i]}")

    def close(self) -> None:  # noqa: D401
        """Placeholder for API completeness."""
        pass
