"""Synthetic shuttle trajectories with known stroke frames.

Image-plane physics: gravity plus quadratic drag, so the shuttle leaves fast
and floats back slowly -- the property the off-screen rule relies on. Strokes
are instantaneous velocity changes. Tracker behaviour (jitter, dropouts,
frozen latches, frame exits) is layered on afterwards.
"""
from __future__ import annotations

import numpy as np

from hits.badminton_ai.types import ShuttlePosition


class Rally:
    def __init__(self, fps: float = 25, width: int = 1920, height: int = 1080,
                 g: float = 0.5, terminal: float = 12.0, seed: int = 0):
        self.fps, self.width, self.height = fps, width, height
        self.g = g
        self.k = g / terminal ** 2
        self.rng = np.random.default_rng(seed)
        self.P: list[np.ndarray] = []
        self.v = np.zeros(2)
        self.contacts: list[int] = []
        self.hidden: set[int] = set()
        self.frozen: dict[int, tuple[float, float]] = {}

    # --- building ---------------------------------------------------------
    def start(self, x: float, y: float) -> "Rally":
        self.P.append(np.array([x, y], float))
        return self

    def hold(self, n: int) -> "Rally":
        for _ in range(n):
            self.P.append(self.P[-1].copy())
        self.v = np.zeros(2)
        return self

    def set_velocity(self, vx: float, vy: float) -> "Rally":
        """Velocity change at the current (last) frame, not recorded as a stroke."""
        self.v = np.array([vx, vy], float)
        return self

    def hit(self, vx: float, vy: float) -> "Rally":
        self.contacts.append(len(self.P) - 1)
        return self.set_velocity(vx, vy)

    def fly(self, n: int) -> "Rally":
        for _ in range(n):
            p = self.P[-1] + self.v
            self.P.append(p)
            sp = np.linalg.norm(self.v)
            self.v = self.v + np.array([0.0, self.g]) - self.k * sp * self.v
        return self

    def skid(self, n: int, friction: float = 0.85) -> "Rally":
        """Landed: slides along the floor, never rises."""
        self.v = np.array([self.v[0] * 0.6, 0.0])
        for _ in range(n):
            self.P.append(self.P[-1] + self.v)
            self.v = self.v * friction
        return self

    @property
    def now(self) -> int:
        return len(self.P) - 1

    def hide(self, a: int, b: int) -> "Rally":
        self.hidden.update(range(a, b + 1))
        return self

    def freeze(self, a: int, b: int, x: float, y: float) -> "Rally":
        for f in range(a, b + 1):
            self.frozen[f] = (x, y)
        return self

    # --- output -------------------------------------------------------------
    def positions(self, jitter: float = 1.0) -> list[ShuttlePosition]:
        out = []
        for f, p in enumerate(self.P):
            if f in self.frozen:
                x, y = self.frozen[f]
                out.append(ShuttlePosition(f, True, x, y))
                continue
            x, y = p + self.rng.normal(0, jitter, 2)
            inside = 0 <= x < self.width and 0 <= y < self.height
            if f in self.hidden or not inside:
                out.append(ShuttlePosition(f, False))
            else:
                out.append(ShuttlePosition(f, True, float(x), float(y)))
        return out


def standard_rally(seed: int = 0) -> Rally:
    """Serve + six strokes between a near player (bottom) and far player (top),
    including one slow far-court block and a final floor landing."""
    r = Rally(seed=seed).start(900, 820)
    r.hold(10)
    r.set_velocity(1, -12).fly(8)            # serve release: not a stroke
    r.hit(4, -34).fly(22)                    # serve, near -> far
    r.hit(-6, 30).fly(18)                    # far player drive
    r.hit(8, -40).fly(24)                    # near player clear
    r.hit(-3, 3).fly(24)                     # far player block, slow
    r.hit(-5, -30).fly(20)                   # near lift
    r.hit(8, 50).fly(28)                     # far smash -> lands
    r.skid(12)
    return r
