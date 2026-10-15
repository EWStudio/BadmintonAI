"""Stage 5 on synthetic trajectories with known contact frames.

Each test isolates one rule from the write-up, so a regression names the rule
that broke. Synthetic physics is not broadcast footage: these prove the logic,
not the constants. The constants are validated only against hand-labelled clips.
"""
import csv

import numpy as np
import pytest

from hits.badminton_ai.detect_shots import DEFAULT, detect_shots, load_ball_csv, trajectory_stats
from hits.badminton_ai.types import ShuttlePosition
from tests.synth import Rally, standard_rally

FPS, W, H = 25, 1920, 1080


def _frames(res):
    return [e.frame_index for e in res.events]


def _match(truth, got, tol=1):
    hits = [c for c in truth if any(abs(c - g) <= tol for g in got)]
    fps = [g for g in got if not any(abs(c - g) <= tol for c in truth)]
    return hits, fps


# --- end-to-end on a full rally ---------------------------------------------

def test_rally_recall_and_precision_over_seeds():
    tp = fp = total = 0
    for seed in range(20):
        r = standard_rally(seed)
        res = detect_shots(r.positions(jitter=1.0), FPS, W, H)
        hits, false = _match(r.contacts, _frames(res))
        tp, fp, total = tp + len(hits), fp + len(false), total + len(r.contacts)
    assert tp / total >= 0.95, f"recall {tp}/{total}"
    assert tp / (tp + fp) >= 0.95, f"precision {tp}/{tp + fp}"


def test_clean_rally_is_exact():
    r = standard_rally(7)
    res = detect_shots(r.positions(jitter=0.5), FPS, W, H)
    hits, false = _match(r.contacts, _frames(res))
    assert hits == r.contacts and false == []


# --- detector behaviour ------------------------------------------------------

def test_apex_of_a_lob_is_not_a_stroke():
    r = Rally(seed=1).start(900, 900).set_velocity(0, 14).fly(12)
    r.hit(3, -38).fly(45)              # high lob: rises, turns over, falls
    r.hit(-2, -30).fly(15)
    res = detect_shots(r.positions(1.0), FPS, W, H)
    hits, false = _match(r.contacts, _frames(res))
    assert false == [], f"apex reported as stroke: {false}"
    assert hits == r.contacts


def test_ballistic_finds_slow_stroke_that_acceleration_misses():
    """Far-court touch: |dv| is under the acceleration threshold, but the
    trajectory departs from its arc over the following frames."""
    r = Rally(seed=3, g=0.15, terminal=20).start(900, 300).set_velocity(2, -4).fly(14)
    r.hit(-3, 3.5).fly(14)
    pos = r.positions(0.6)
    strict_accel = DEFAULT.__class__(ballistic_threshold=1e9)   # acceleration only
    acc_only = detect_shots(pos, FPS, W, H, config=strict_accel)
    both = detect_shots(pos, FPS, W, H)
    assert _match(r.contacts, _frames(acc_only))[0] == []
    hits, _ = _match(r.contacts, _frames(both))
    assert hits == r.contacts
    assert "ballistic" in both.events[0].detector


def test_sensitivity_below_one_finds_at_least_as_many():
    r = standard_rally(5)
    pos = r.positions(1.0)
    base = len(detect_shots(pos, FPS, W, H).events)
    loose = len(detect_shots(pos, FPS, W, H, sensitivity=0.7).events)
    assert loose >= base


# --- rejection rules -----------------------------------------------------------

def test_serve_release_is_rejected_and_real_contact_kept():
    r = standard_rally(0)                         # release at frame 10, serve at 18
    res = detect_shots(r.positions(1.0), FPS, W, H)
    assert 10 not in _frames(res)
    assert any(abs(f - 18) <= 1 for f in _frames(res))
    assert any(c.frame_index == 10 and c.reason.startswith("released") for c in res.rejected)


def test_floor_landing_with_skid_is_rejected():
    r = standard_rally(1)
    landing = r.contacts[-1] + 28
    res = detect_shots(r.positions(1.0), FPS, W, H)
    assert not any(abs(f - landing) <= 2 for f in _frames(res))
    assert any(abs(c.frame_index - landing) <= 2 and "floor" in c.reason for c in res.rejected)


def test_low_retrieval_that_rises_again_is_kept():
    r = Rally(seed=2).start(900, 700).set_velocity(1, 12).fly(16)   # dropping toward the floor
    y = r.P[-1][1]
    assert y / H > DEFAULT.floor_y_fraction
    r.hit(2, -32).fly(25)                                            # lifted back up
    res = detect_shots(r.positions(1.0), FPS, W, H)
    assert _match(r.contacts, _frames(res))[0] == r.contacts


def test_frozen_tracker_stretch_is_dropped_without_phantoms():
    r = standard_rally(2)
    r.freeze(64, 76, 1275.0, 465.0)              # tracker pinned mid-rally
    res = detect_shots(r.positions(1.0), FPS, W, H)
    assert res.stats["frozen_points_dropped"] >= 13
    # The jumps into and out of a latched stretch are huge fake accelerations.
    near_edges = [f for f in _frames(res) if min(abs(f - 64), abs(f - 76)) <= 2]
    assert near_edges == []


def test_camera_cut_is_not_a_stroke_when_cuts_are_known():
    r = Rally(seed=4).start(900, 820).hit(3, -30).fly(60)
    pos = r.positions(1.0)
    cut = 30
    for p in pos[cut:]:                            # new camera angle: everything jumps
        if p.visible:
            p.x, p.y = p.x - 500, p.y + 250
    without = detect_shots(pos, FPS, W, H)
    with_cuts = detect_shots(pos, FPS, W, H, cut_frames=[cut])
    assert any(abs(f - cut) <= 1 for f in _frames(without)), "the cut should look like a stroke"
    assert not any(abs(f - cut) <= 1 for f in _frames(with_cuts))


def test_run_where_shuttle_barely_moves_is_dropped():
    rng = np.random.default_rng(0)
    pos = [ShuttlePosition(f, True, 600 + rng.normal(0, 0.6), 400 + rng.normal(0, 0.6))
           for f in range(60)]
    res = detect_shots(pos, FPS, W, H)
    assert res.events == [] and res.stats["dropped_runs"] >= 1


# --- off-screen inference ------------------------------------------------------

def _struck_offscreen(seed=0):
    """Far player strikes as the tracker loses the shuttle; it leaves the top
    of the frame and returns at roughly the speed it left."""
    r = Rally(seed=seed).start(950, 430).set_velocity(1, -16).fly(16)
    c = r.now
    r.hit(-2, -45).hide(c + 1, c + 10).fly(75)
    return r


def _free_flight_offscreen(seed=0):
    """A visible stroke drives the shuttle out of the top; it floats back slowly."""
    r = Rally(seed=seed).start(950, 420).set_velocity(1, -16).fly(14)
    r.hit(1, -60).fly(80)
    return r


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_offscreen_stroke_is_inferred_at_last_sighting(seed):
    r = _struck_offscreen(seed)
    res = detect_shots(r.positions(1.0), FPS, W, H)
    inferred = [e for e in res.events if e.inferred]
    assert len(inferred) == 1
    assert abs(inferred[0].frame_index - r.contacts[0]) <= 2
    assert inferred[0].detector == "offscreen"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_free_flight_offscreen_is_not_a_stroke(seed):
    r = _free_flight_offscreen(seed)
    res = detect_shots(r.positions(1.0), FPS, W, H)
    assert not any(e.inferred for e in res.events)
    assert any(c.detector == "offscreen" and c.value >= DEFAULT.offscreen_speed_ratio for c in res.rejected)
    assert _match(r.contacts, _frames(res))[0] == r.contacts


def test_offscreen_never_bridges_a_camera_cut():
    r = _struck_offscreen()
    gap_mid = r.contacts[0] + 20
    res = detect_shots(r.positions(1.0), FPS, W, H, cut_frames=[gap_mid])
    assert not any(e.inferred for e in res.events)


# --- event features and ingest checks -------------------------------------------

def test_events_carry_trajectory_features_for_classification():
    r = standard_rally(0)
    ev = detect_shots(r.positions(0.5), FPS, W, H).events[1]
    assert ev.pre_velocity is not None and ev.post_velocity is not None
    assert ev.post_travel_px > 0
    assert ev.time_s == pytest.approx(ev.frame_index / FPS)


def test_padded_video_triggers_duplicate_warning():
    r = standard_rally(0)
    pos = r.positions(1.0)
    padded = []
    for p in pos:                                  # every frame doubled: 25 fps content in a 50 fps file
        padded.append(ShuttlePosition(2 * p.frame_index, p.visible, p.x, p.y))
        padded.append(ShuttlePosition(2 * p.frame_index + 1, p.visible, p.x, p.y))
    res = detect_shots(padded, 2 * FPS, W, H)
    assert res.stats["duplicate_fraction"] > 0.4
    assert "padded" in res.stats["warning"]


def test_trajectory_stats_reports_tracking_and_dropouts():
    pos = [ShuttlePosition(f, not (20 <= f < 40), 1.0 * f, 2.0 * f) for f in range(100)]
    st = trajectory_stats(pos)
    assert st["tracked_fraction"] == pytest.approx(0.8)
    assert st["dropouts"] == [(20, 39)]


def test_load_ball_csv(tmp_path):
    path = tmp_path / "clip_ball.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Frame", "Visibility", "X", "Y"])
        w.writerow([0, 1, 100, 200])
        w.writerow([1, 0, 0, 0])
        w.writerow([2, 1, 0, 0])           # visible at the origin = TrackNet's "nothing"
    pos = load_ball_csv(str(path))
    assert [p.visible for p in pos] == [True, False, False]
    assert (pos[0].x, pos[0].y) == (100.0, 200.0)
