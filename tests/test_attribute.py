import numpy as np

from hits.badminton_ai import attribute as A
from hits.badminton_ai.types import KP_R_WRIST, ShotEvent, ShuttlePosition

W, H, FPS = 1920, 1080, 25
DIAG = (W ** 2 + H ** 2) ** 0.5
LIMIT = A.ATTRIBUTION_FRACTION * DIAG


def _kps(wx, wy, conf=0.9):
    k = np.zeros((17, 3))
    k[:, 2] = conf
    k[:, 0], k[:, 1] = wx, wy
    k[KP_R_WRIST] = (wx, wy, conf)
    k[9, 2] = 0.0                       # left wrist unseen; right wrist does the work
    return k


def _scene(scores_per_event, frames, inferred=False):
    """Events whose per-player evidence equals the given scores exactly."""
    events, poses, shuttle = [], {}, []
    for f, scores in zip(frames, scores_per_event):
        look = f if inferred else f - A.LOOKBACK
        sx, sy = 1000.0, 500.0
        shuttle.append(ShuttlePosition(look, True, sx, sy))
        limit = (A.OFFSCREEN_ATTRIBUTION_FRACTION if inferred else A.ATTRIBUTION_FRACTION) * DIAG
        for pid, s in enumerate(scores):
            if s > 0:
                poses.setdefault(look, {})[pid] = _kps(sx + (1 - s) * limit, sy)
        events.append(ShotEvent(f, f / FPS, 1.0, "accel", sx, sy, inferred=inferred))
    return events, poses, shuttle


def test_alternation_reproduces_the_documented_case():
    """The 1_00_05 table: greedy gets 43 and 64 wrong; alternation fixes both."""
    table = [(0.80, 0.0), (0.54, 0.36), (0.0, 0.56), (0.19, 0.64), (0.63, 0.0)]
    frames = [25, 43, 64, 83, 111]
    events, poses, shuttle = _scene(table, frames)

    greedy = A.attribute_strokes(events, poses, shuttle, DIAG, FPS, alternation=False)
    assert [e.player_id for e in greedy] == [0, 0, 1, 1, 0]

    alt = A.attribute_strokes(events, poses, shuttle, DIAG, FPS)
    assert [e.player_id for e in alt] == [0, 1, 0, 1, 0]
    assert [e.attribution_method for e in alt] == ["distance", "alternation", "elimination",
                                                   "distance", "distance"]


def test_alternation_leaves_an_already_alternating_answer_alone():
    table = [(0.9, 0.1), (0.2, 0.7), (0.8, 0.0), (0.1, 0.6)]
    events, poses, shuttle = _scene(table, [10, 30, 50, 70])
    out = A.attribute_strokes(events, poses, shuttle, DIAG, FPS)
    assert [e.player_id for e in out] == [0, 1, 0, 1]
    assert all(e.attribution_method == "distance" for e in out)


def test_chain_is_cut_at_long_pauses():
    """A new rally can begin with either player."""
    table = [(0.9, 0.0), (0.0, 0.8), (0.9, 0.0), (0.9, 0.0), (0.0, 0.8)]
    frames = [10, 30, 50, 50 + int(4 * FPS), 70 + int(4 * FPS)]
    events, poses, shuttle = _scene(table, frames)
    out = A.attribute_strokes(events, poses, shuttle, DIAG, FPS)
    assert [e.player_id for e in out] == [0, 1, 0, 0, 1]


def test_distance_limit_leaves_far_striker_unattributed():
    ev = ShotEvent(40, 40 / FPS, 1.0, "accel", 1000, 500)
    shuttle = [ShuttlePosition(40 - A.LOOKBACK, True, 1000.0, 500.0)]
    poses = {40 - A.LOOKBACK: {0: _kps(1000 + 2 * LIMIT, 500)}}       # nearest wrist far away
    out = A.attribute_strokes([ev], poses, shuttle, DIAG, FPS)
    assert out[0].player_id is None


def test_lookback_uses_contact_not_detected_frame():
    """At the detected frame the shuttle is already nearer the receiver."""
    f = 60
    shuttle = [ShuttlePosition(f - A.LOOKBACK, True, 1000.0, 500.0),
               ShuttlePosition(f, True, 1300.0, 400.0)]
    poses = {f - A.LOOKBACK: {0: _kps(1010, 505), 1: _kps(1305, 405)},
             f: {0: _kps(1010, 505), 1: _kps(1305, 405)}}
    ev = ShotEvent(f, f / FPS, 1.0, "accel", 1300, 400)
    assert A.attribute_strokes([ev], poses, shuttle, DIAG, FPS)[0].player_id == 0


def test_inferred_stroke_uses_looser_limit_and_refuses_near_ties():
    loose = A.OFFSCREEN_ATTRIBUTION_FRACTION * DIAG
    ev = ShotEvent(80, 80 / FPS, 1.0, "offscreen", 1000, 200, inferred=True)
    shuttle = [ShuttlePosition(80, True, 1000.0, 200.0)]
    d = 0.15 * DIAG                                   # beyond 9%, inside 20%
    poses = {80: {1: _kps(1000 + d, 200)}}
    assert A.attribute_strokes([ev], poses, shuttle, DIAG, FPS)[0].player_id == 1
    tie = {80: {0: _kps(1000 + 0.5 * loose, 200), 1: _kps(1000 - 0.5 * loose - 1, 200)}}
    assert A.attribute_strokes([ev], tie, shuttle, DIAG, FPS)[0].player_id is None
