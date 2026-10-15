#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hits.score import TOLERANCE_S, load_answer_key, score  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Score hits.csv against a hand-labelled answer key.")
    ap.add_argument("run", help="run directory containing hits.csv and metadata.json")
    ap.add_argument("answer_key", help='JSON file: {"hits": [[frame, "near"|"far"], ...]}, 0-based frames')
    ap.add_argument("--tolerance", type=float, default=TOLERANCE_S, help=f"seconds (default {TOLERANCE_S})")
    a = ap.parse_args()

    with open(os.path.join(a.run, "metadata.json"), encoding="utf-8") as fh:
        fps = float(json.load(fh)["video"]["fps"])
    with open(os.path.join(a.run, "hits.csv"), newline="", encoding="utf-8") as fh:
        rows = [(int(r["frame"]), int(r["contact_frame"]), r["player"]) for r in csv.DictReader(fh)]
    s = score(rows, load_answer_key(a.answer_key), fps, a.tolerance)
    print(s.summary())
    for label, frames in (("False hits", s.false), ("Missed", s.missed), ("Wrong player", s.wrong_player)):
        if frames:
            print(f"{label}: frames {', '.join(map(str, frames))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
