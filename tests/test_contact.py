import json

import numpy as np

from hits.adapter import FAR, NEAR, GBRun
from hits.badminton_ai.types import KP_R_WRIST, ShotEvent, ShuttlePosition
from hits.contact import confirm_contacts

FPS, W, H = 25, 1280, 720
CORNERS = [[439, 269], [837, 269], [981, 641], [312, 641]]
NEAR_COURT, FAR_COURT = [3.0, 10.5], [3.0, 3.0]


def _wrist(x, y):
    k = np.zeros((17, 3))
    k[KP_R_WRIST] = (x, y, 1.0)
    return k


def _run(tmp_path, shuttle, near_wrist, far_wrist, n=60, corners=True, poses=True):
    meta = {"video": {"name": "t", "fps": FPS, "total_frames": n, "width": W, "height": H}}
    if corners:
        meta["court"] = {"corners": CORNERS}
    (tmp_path / "metadata.json").write_text(json.dumps(meta))
    players = {"lower": {"court": NEAR_COURT}, "upper": {"court": FAR_COURT}}
    records = {f: {"players": players} for f in range(n)}
    positions = [ShuttlePosition(f, True, *shuttle(f)) if shuttle(f) else ShuttlePosition(f, False)
                 for f in range(n)]
    pose = {f: {NEAR: _wrist(*near_wrist(f)), FAR: _wrist(*far_wrist(f))} for f in range(n)} if poses else {}
    return GBRun(str(tmp_path), "t", FPS, W, H, n, records, positions, pose)


def _event(f, inferred=False):
    return ShotEvent(f, f / FPS, 1.0, "offscreen" if inferred else "accel", inferred=inferred)


def test_hit_with_no_wrist_in_reach_is_rejected(tmp_path):
    run = _run(tmp_path, lambda f: (640.0, 80.0), lambda f: (650.0, 520.0), lambda f: (640.0, 250.0))
    out = confirm_contacts([_event(30)], run)
    assert out.events == []
    assert "no racket in reach" in out.rejected[0].reason and out.rejected[0].value > 1.3


def test_hit_next_to_a_wrist_is_kept_and_scored_for_that_player(tmp_path):
    run = _run(tmp_path, lambda f: (640.0, 240.0), lambda f: (650.0, 520.0), lambda f: (650.0, 250.0))
    out = confirm_contacts([_event(30)], run)
    assert len(out.events) == 1
    assert out.scores[0][FAR] > 0.8 and out.scores[0].get(NEAR, 0.0) == 0.0


def test_offscreen_stroke_needs_a_wrist_near_the_top_edge(tmp_path):
    shuttle = lambda f: None if 30 < f < 45 else (600.0, 20.0)
    low = _run(tmp_path, shuttle, lambda f: (650.0, 520.0), lambda f: (640.0, 250.0))
    assert confirm_contacts([_event(30, inferred=True)], low).events == []
    high = _run(tmp_path, shuttle, lambda f: (650.0, 520.0), lambda f: (640.0, 30.0))
    assert len(confirm_contacts([_event(30, inferred=True)], high).events) == 1


def test_same_player_again_within_half_a_second_is_one_hit(tmp_path):
    run = _run(tmp_path, lambda f: (640.0, 240.0), lambda f: (650.0, 520.0), lambda f: (650.0, 250.0))
    out = confirm_contacts([_event(20), _event(28), _event(40)], run)
    assert [e.frame_index for e in out.events] == [20, 40]
    assert "same player again" in out.rejected[0].reason


def test_nothing_is_rejected_without_poses_or_court_corners(tmp_path):
    far_away = lambda f: (640.0, 80.0)
    no_pose = _run(tmp_path, far_away, lambda f: (0, 0), lambda f: (0, 0), poses=False)
    assert len(confirm_contacts([_event(30)], no_pose).events) == 1
    no_court = _run(tmp_path, far_away, lambda f: (650.0, 520.0), lambda f: (640.0, 250.0), corners=False)
    out = confirm_contacts([_event(30)], no_court)
    assert len(out.events) == 1 and "skipped" in out.note


def test_reach_zero_turns_the_check_off(tmp_path):
    run = _run(tmp_path, lambda f: (640.0, 80.0), lambda f: (650.0, 520.0), lambda f: (640.0, 250.0))
    out = confirm_contacts([_event(30)], run, reach_m=0)
    assert len(out.events) == 1 and out.scores is None
