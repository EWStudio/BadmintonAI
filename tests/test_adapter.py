import csv
import json

import cv2
import numpy as np
import pytest

from hits.adapter import (FAR, NEAR, decimate, find_duplicate_frames, load_run, padding_period,
                          with_tracknet)
from hits.badminton_ai.attribute import attribute_strokes
from hits.badminton_ai.detect_shots import detect_shots
from tests.synth import standard_rally

FPS, W, H = 25, 1920, 1080


def _write_run(tmp_path, rally, skip_frames=(), jitter=0.5, total=None):
    pos = rally.positions(jitter)
    n = total or len(pos)
    strikers = [NEAR if k % 2 == 0 else FAR for k in range(len(rally.contacts))]
    near_home, far_home = (900.0, 950.0), (1000.0, 300.0)
    lines = []
    for p in pos:
        f = p.frame_index
        if f in skip_frames:
            continue
        hands = {NEAR: near_home, FAR: far_home}
        for c, who in zip(rally.contacts, strikers):
            if c - 5 <= f <= c + 1 and p.visible:
                hands[who] = (p.x + 8, p.y + 6)
        rec = {"schema_version": "1.0", "frame": f + 1, "time_sec": (f + 1) / FPS, "detect_frame": f + 1,
               "players": {
                   "lower": {"image": [900, 1000], "court": [3, 10], "speed": 0.0,
                             "hands": {"left": [880, 960], "right": list(hands[NEAR])}},
                   "upper": {"image": [1000, 330], "court": [3, 3], "speed": 0.0,
                             "hands": {"left": [990, 290], "right": list(hands[FAR])}}},
               "shuttlecock": {"image": [p.x, p.y] if p.visible else None}}
        lines.append(json.dumps(rec))
    (tmp_path / "detections.jsonl").write_text("\n".join(lines) + "\n")
    (tmp_path / "metadata.json").write_text(json.dumps({"video": {
        "path": "missing.mp4", "name": "synth", "fps": FPS, "total_frames": n, "width": W, "height": H}}))
    return strikers


def _hits(run, **kw):
    res = detect_shots(run.positions, run.fps, run.width, run.height, cut_frames=run.cut_frames, **kw)
    return attribute_strokes(res.events, run.poses, run.positions, run.diagonal, run.fps,
                             cut_frames=run.cut_frames)


def _matched(events, contacts, strikers, tol=1):
    out = []
    for c, who in zip(contacts, strikers):
        e = next((e for e in events if abs(e.frame_index - c) <= tol), None)
        out.append((c, who, None if e is None else e.player_id))
    return out


def test_frames_are_zero_based_and_players_map_lower_near_upper_far(tmp_path):
    r = standard_rally(7)
    _write_run(tmp_path, r)
    run = load_run(str(tmp_path))
    assert min(run.records) == 0
    assert run.positions[r.contacts[0]].x == pytest.approx(r.P[r.contacts[0]][0], abs=3)
    f = r.contacts[0] - 2
    assert run.poses[f][NEAR][10, 0] == pytest.approx(run.positions[f].x + 8)
    assert run.poses[f][FAR][10, 1] == pytest.approx(300.0)


def test_end_to_end_finds_every_stroke_and_its_striker(tmp_path):
    r = standard_rally(7)
    strikers = _write_run(tmp_path, r)
    events = _hits(load_run(str(tmp_path)))
    got = _matched(events, r.contacts, strikers)
    assert all(pid is not None for _, _, pid in got), got
    assert all(pid == who for _, who, pid in got), got
    assert len(events) == len(r.contacts)


def test_reading_the_one_based_frames_as_zero_based_would_break_it(tmp_path):
    r = standard_rally(7)
    _write_run(tmp_path, r)
    run = load_run(str(tmp_path))
    shifted = {f + 1: p for f, p in run.poses.items()}
    res = detect_shots(run.positions, FPS, W, H)
    good = attribute_strokes(res.events, run.poses, run.positions, run.diagonal, FPS, alternation=False)
    bad = attribute_strokes(res.events, shifted, run.positions, run.diagonal, FPS, alternation=False)
    score = lambda evs: sum(e.attribution_score or 0 for e in evs)
    assert score(bad) < score(good)


def test_gaps_in_records_become_cuts(tmp_path):
    r = standard_rally(7)
    _write_run(tmp_path, r, skip_frames=set(range(60, 66)))
    run = load_run(str(tmp_path))
    assert run.cut_frames == [66]
    assert not run.positions[62].visible


def test_padding_period_needs_regular_spacing():
    assert padding_period(list(range(6, 120, 6))) == 6
    assert padding_period([5, 9, 30, 31, 70]) is None
    assert padding_period([6, 12]) is None


def test_find_duplicate_frames_on_a_padded_video(tmp_path):
    path = str(tmp_path / "padded.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
    yy, xx = np.mgrid[0:180, 0:320]
    base = np.dstack([60 + xx // 4, 120 + yy // 3, 70 + 0 * xx]).astype(np.uint8)
    img, t = None, 0
    for f in range(72):
        if f % 6 != 0 or img is None:
            t += 1
            img = base.copy()
            cv2.rectangle(img, (40 + 3 * t, 100), (70 + 3 * t, 170), (230, 230, 230), -1)
            cv2.rectangle(img, (250 - 2 * t, 20), (270 - 2 * t, 60), (60, 40, 120), -1)
            cv2.circle(img, (20 + 5 * t, 30 + (t * 7) % 120), 4, (255, 255, 255), -1)
        vw.write(img)
    vw.release()
    dups = find_duplicate_frames(path)
    assert dups == list(range(6, 72, 6))
    assert padding_period(dups) == 6


def test_decimation_maps_hits_back_to_original_frames(tmp_path):
    r = standard_rally(7)
    strikers = _write_run(tmp_path, r)
    clean = load_run(str(tmp_path))
    recs, orig_of, dups, f_out = [], {}, [], 0
    for f in range(clean.n_frames):
        rec = dict(clean.records[f]); rec["frame"] = f_out + 1
        recs.append(json.dumps(rec)); orig_of[f] = f_out; f_out += 1
        if f % 5 == 4:
            dup = dict(rec); dup["frame"] = f_out + 1
            recs.append(json.dumps(dup)); dups.append(f_out); f_out += 1
    (tmp_path / "detections.jsonl").write_text("\n".join(recs) + "\n")
    meta = json.loads((tmp_path / "metadata.json").read_text())
    meta["video"].update(total_frames=f_out, fps=30)
    (tmp_path / "metadata.json").write_text(json.dumps(meta))

    padded = load_run(str(tmp_path))
    assert padding_period(dups) == 6
    dec = decimate(padded, dups)
    assert dec.true_fps == pytest.approx(25, abs=0.2)
    events = _hits(dec.run)
    mapped = [dec.to_original(e.frame_index) for e in events]
    for c in r.contacts:
        assert any(abs(m - orig_of[c]) <= 1 for m in mapped), (c, orig_of[c], mapped)
    got = _matched(events, r.contacts, strikers)
    assert all(pid == who for _, who, pid in got), got


def test_tracknet_csv_replaces_the_shuttle_and_keeps_the_wrists(tmp_path):
    r = standard_rally(7)
    strikers = _write_run(tmp_path, r)
    run = load_run(str(tmp_path))
    for f in range(r.contacts[2] - 10, r.contacts[2] + 10):
        run.positions[f].visible, run.positions[f].x, run.positions[f].y = True, 1000.0, 330.0
    tn = tmp_path / "synth_ball.csv"
    with open(tn, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Frame", "Visibility", "X", "Y"])
        for p in r.positions(0.5):
            w.writerow([p.frame_index, int(p.visible), round(p.x or 0), round(p.y or 0)])
    fixed, warn = with_tracknet(run, str(tn))
    assert warn is None
    assert fixed.poses is run.poses
    got = _matched(_hits(fixed), r.contacts, strikers)
    assert all(pid == who for _, who, pid in got), got
