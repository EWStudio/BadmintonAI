"""Credit each stroke to a player (singles).

Two rules from measured failures, then one rule of the sport:

* **Distance limit** (9% of the frame diagonal). Without it a stroke goes to
  whichever player has a pose -- including one most of a court away when the
  striker wasn't detected. Every far-court stroke in one broadcast clip was
  credited to the near player.

* **3-frame lookback.** Contact precedes the acceleration peak the detector
  fires on, so at the detected frame the shuttle has already left the racket.
  Measured on 14 verified strokes: lookback 0 -> 12 correct, 2 unattributed;
  3 -> 14 correct; 4 -> 1 wrong; a symmetric +-3 search -> 2 wrong, because
  searching forward finds the *receiver*.

* **Alternation.** Nearest-wrist got 2 of 5 wrong on one clip and no
  threshold could fix it: image distance conflates depth (both players at the
  net), and overlapping players come back from YOLO as one box. In singles,
  consecutive strokes are played by different players, so the
  highest-scoring *alternating* assignment wins. Strong evidence pins the
  chain, weak evidence yields, and an undetected striker is attributed by
  elimination. If the greedy answer already alternates it is already optimal
  and nothing changes.

  The cost: alternation is only as safe as recall. A missed stroke inverts
  every attribution after it until a strong anchor pulls the chain back.

Pixel distance, not ground-plane metres: shuttle and wrist are both ~2 m
above the floor at contact, so the court homography doesn't help ranking.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .types import KP_L_WRIST, KP_R_WRIST, ShotEvent, ShuttlePosition

ATTRIBUTION_FRACTION = 0.09            # of frame diagonal, observed strokes
OFFSCREEN_ATTRIBUTION_FRACTION = 0.20  # inferred strokes: existence already established;
                                       # wrist sat 238-298 px away through motion blur
LOOKBACK = 3
NEAR_TIE = 0.05                        # inferred strokes: closer than this -> leave unattributed
MAX_CHAIN_PAUSE_S = 3.0                # a new rally can start with either player
MIN_WRIST_CONF = 0.3

# frame -> {player_id: keypoints (17 x 3)}
PoseIndex = dict[int, dict[int, np.ndarray]]


@dataclass
class Attribution:
    player_id: Optional[int]
    scores: dict[int, float]
    method: str


def _wrist_distance(kps: np.ndarray, x: float, y: float) -> Optional[float]:
    best = None
    for k in (KP_L_WRIST, KP_R_WRIST):
        if kps[k, 2] >= MIN_WRIST_CONF:
            d = float(np.hypot(kps[k, 0] - x, kps[k, 1] - y))
            best = d if best is None else min(best, d)
    return best


def _shuttle_at(track: dict[int, tuple[float, float]], frame: int, back_to: int) -> Optional[tuple[float, float]]:
    """Shuttle position at ``frame``, or the nearest sighting between it and ``back_to``."""
    for f in range(frame, back_to + 1):
        if f in track:
            return track[f]
    return None


def score_stroke(event: ShotEvent, poses: PoseIndex, track: dict[int, tuple[float, float]],
                 diagonal: float, players: tuple[int, ...] = (0, 1)) -> dict[int, float]:
    """Per-player evidence in [0, 1]: 1 = wrist on the shuttle, 0 = beyond the limit."""
    if event.inferred:
        frame, limit = event.frame_index, OFFSCREEN_ATTRIBUTION_FRACTION * diagonal
    else:
        frame, limit = max(0, event.frame_index - LOOKBACK), ATTRIBUTION_FRACTION * diagonal
    pos = _shuttle_at(track, frame, event.frame_index)
    if pos is None and event.x is not None:
        pos = (event.x, event.y)
    scores = {p: 0.0 for p in players}
    if pos is None:
        return scores
    for p in players:
        for f in (frame, frame - 1, frame + 1):            # tolerate a single missing pose frame
            kps = poses.get(f, {}).get(p)
            if kps is None:
                continue
            d = _wrist_distance(kps, *pos)
            if d is not None:
                scores[p] = max(0.0, 1.0 - d / limit)
                break
    return scores


def _greedy(scores: dict[int, float], inferred: bool) -> Optional[int]:
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if ranked[0][1] <= 0:
        return None
    if inferred and len(ranked) > 1 and ranked[0][1] - ranked[1][1] < NEAR_TIE:
        return None
    return ranked[0][0]


def _chains(events: list[ShotEvent], cut_frames: set[int], fps: float) -> list[list[int]]:
    """Indices of events that belong to one rally: split at cuts and long pauses."""
    chains, cur = [], []
    for i, e in enumerate(events):
        if cur:
            prev = events[cur[-1]]
            cut = any(prev.frame_index < c <= e.frame_index for c in cut_frames)
            if cut or (e.frame_index - prev.frame_index) / fps > MAX_CHAIN_PAUSE_S:
                chains.append(cur)
                cur = []
        cur.append(i)
    if cur:
        chains.append(cur)
    return chains


def attribute_strokes(
    events: list[ShotEvent],
    poses: PoseIndex,
    shuttle: list[ShuttlePosition],
    diagonal: float,
    fps: float,
    cut_frames: Optional[list[int]] = None,
    alternation: bool = True,
    n_players: int = 2,
) -> list[ShotEvent]:
    """Return copies of ``events`` with ``player_id`` / ``attribution_*`` filled in."""
    track = {p.frame_index: (p.x, p.y) for p in shuttle if p.visible and p.x is not None}
    players = tuple(range(n_players))
    all_scores = [score_stroke(e, poses, track, diagonal, players) for e in events]
    result = [Attribution(_greedy(s, e.inferred), s, "distance") for e, s in zip(events, all_scores)]

    if alternation and n_players == 2:
        for chain in _chains(events, set(cut_frames or []), fps):
            if len(chain) < 2:
                continue
            totals = {}
            for first in (0, 1):
                totals[first] = sum(all_scores[i][(first + k) % 2] for k, i in enumerate(chain))
            if totals[0] == totals[1]:
                continue                       # no evidence either way: keep greedy
            first = max(totals, key=totals.get)
            for k, i in enumerate(chain):
                pid = (first + k) % 2
                if result[i].player_id == pid:
                    continue
                method = "elimination" if all_scores[i][pid] == 0 else "alternation"
                result[i] = Attribution(pid, all_scores[i], method)

    out = []
    for e, a in zip(events, result):
        out.append(ShotEvent(**{**e.to_dict(),
                                "player_id": a.player_id,
                                "attribution_score": round(a.scores.get(a.player_id, 0.0), 3)
                                if a.player_id is not None else None,
                                "attribution_method": a.method if a.player_id is not None else None}))
    return out
