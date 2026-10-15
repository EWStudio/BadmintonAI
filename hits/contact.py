from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .adapter import REGION_TO_PLAYER, GBRun
from .badminton_ai.types import KP_L_WRIST, KP_R_WRIST, RejectedCandidate, ShotEvent

REACH_M = 1.3
WINDOW = (-4, 2)
SAME_STRIKER_GAP_S = 0.5
COURT_M = ((0.0, 0.0), (6.1, 0.0), (6.1, 13.4), (0.0, 13.4))


@dataclass
class Reach:
    metres: dict[int, float]
    checked: bool

    @property
    def nearest(self) -> Optional[float]:
        return min(self.metres.values()) if self.metres else None

    def within(self, limit: float) -> list[int]:
        return [p for p, d in self.metres.items() if d <= limit]


@dataclass
class Confirmed:
    events: list[ShotEvent]
    scores: Optional[list[dict[int, float]]] = None
    rejected: list[RejectedCandidate] = field(default_factory=list)
    note: Optional[str] = None


class CourtScale:
    def __init__(self, corners):
        import cv2
        self.to_image = np.linalg.inv(cv2.getPerspectiveTransform(np.float32(corners), np.float32(COURT_M)))

    def _project(self, x: float, y: float) -> np.ndarray:
        p = self.to_image @ np.array([x, y, 1.0])
        return p[:2] / p[2]

    def px_per_m(self, court_xy) -> float:
        x, y = float(court_xy[0]), float(court_xy[1])
        return float(np.linalg.norm(self._project(x + 0.5, y) - self._project(x - 0.5, y)))


def court_scale(run: GBRun) -> Optional[CourtScale]:
    try:
        with open(os.path.join(run.run_dir, "metadata.json"), encoding="utf-8") as fh:
            corners = json.load(fh)["court"]["corners"]
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return CourtScale(corners) if len(corners) == 4 else None


def _wrists(run: GBRun, f: int, pid: int) -> list[tuple[float, float]]:
    k = run.poses.get(f, {}).get(pid)
    if k is None:
        return []
    return [(float(k[i, 0]), float(k[i, 1])) for i in (KP_L_WRIST, KP_R_WRIST) if k[i, 2] > 0]


def _px_per_m(run: GBRun, scale: CourtScale, f: int, pid: int) -> Optional[float]:
    region = next(r for r, p in REGION_TO_PLAYER.items() if p == pid)
    c = (((run.records.get(f) or {}).get("players") or {}).get(region) or {}).get("court")
    return scale.px_per_m(c) if c else None


def _next_sighting(run: GBRun, f: int) -> int:
    return next((g for g in range(f + 1, len(run.positions)) if run.positions[g].visible), f)


def reach(event: ShotEvent, run: GBRun, scale: CourtScale) -> Reach:
    f = event.frame_index
    last = _next_sighting(run, f) if event.inferred else f
    metres: dict[int, float] = {}
    checked = False
    for g in range(f + WINDOW[0], last + WINDOW[1] + 1):
        if not 0 <= g < len(run.positions):
            continue
        p = run.positions[g]
        if not event.inferred and not p.visible:
            continue
        for pid in REGION_TO_PLAYER.values():
            hands = _wrists(run, g, pid)
            ppm = _px_per_m(run, scale, g, pid)
            if not hands or not ppm:
                continue
            checked = True
            if event.inferred:
                d = min(y for _, y in hands)
            else:
                d = min(np.hypot(x - p.x, y - p.y) for x, y in hands)
            metres[pid] = min(metres.get(pid, np.inf), d / ppm)
    return Reach(metres, checked)


def confirm_contacts(events: list[ShotEvent], run: GBRun, reach_m: float = REACH_M) -> Confirmed:
    if reach_m <= 0:
        return Confirmed(events)
    scale = court_scale(run)
    if scale is None:
        return Confirmed(events, note="Reach check skipped: no court corners in metadata.json.")

    kept: list[tuple[ShotEvent, Reach]] = []
    rejected: list[RejectedCandidate] = []
    gap = SAME_STRIKER_GAP_S * run.fps
    for e in events:
        r = reach(e, run, scale)
        if r.checked and r.nearest is not None and r.nearest > reach_m:
            reason = ("off-screen stroke out of reach (nearest wrist to top edge, m)" if e.inferred
                      else "no racket in reach (nearest wrist, m)")
            rejected.append(RejectedCandidate(e.frame_index, e.detector, reason, round(r.nearest, 2)))
            continue
        if kept and not e.inferred and not kept[-1][0].inferred:
            prev, prev_r = kept[-1]
            striker, prev_striker = r.within(reach_m), prev_r.within(reach_m)
            if (e.frame_index - prev.frame_index <= gap and len(striker) == 1 and striker == prev_striker):
                rejected.append(RejectedCandidate(e.frame_index, e.detector,
                                                  "same player again, too soon (s after the previous hit)",
                                                  round((e.frame_index - prev.frame_index) / run.fps, 2)))
                continue
        kept.append((e, r))

    scores = [{p: max(0.0, 1.0 - d / reach_m) for p, d in r.metres.items()} for _, r in kept]
    return Confirmed([e for e, _ in kept], scores, rejected)
