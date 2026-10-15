"""Regression on the two Good-Badminton sample clips.

tests/data/<clip>/detections.jsonl is Good-Badminton's own tracking output
(shuttle, wrists, court position only); answer_key.json is hand-labelled.
"""
import json
import os

import pytest

from hits.adapter import PLAYER_NAMES, load_run
from hits.pipeline import find_hits, padded
from hits.score import load_answer_key, score

DATA = os.path.join(os.path.dirname(__file__), "data")


def _score(clip, **kw):
    d = os.path.join(DATA, clip)
    run = load_run(d)
    with open(os.path.join(d, "metadata.json"), encoding="utf-8") as fh:
        dups = json.load(fh)["duplicate_frames"]
    found = find_hits(run, padded(run, dups), **kw)
    rows = [(e.frame_index, c, PLAYER_NAMES[e.player_id]) for e, c in zip(found.events, found.contacts)]
    return score(rows, load_answer_key(os.path.join(d, "answer_key.json")), run.fps)


@pytest.mark.parametrize("clip", ["demo", "demo1"])
def test_no_false_hits_and_only_the_serve_missed(clip):
    s = _score(clip)
    key = load_answer_key(os.path.join(DATA, clip, "answer_key.json"))
    assert s.false == [], s.summary()
    assert s.missed == [key[0][0]], s.summary()


@pytest.mark.parametrize("clip", ["demo", "demo1"])
def test_every_hit_is_credited_to_the_right_player(clip):
    s = _score(clip)
    assert s.right_player == s.found, s.summary()


def test_reach_check_removes_false_hits_on_its_own():
    assert len(_score("demo", reach_m=0).false) > len(_score("demo").false)
