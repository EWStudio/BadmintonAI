#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hits.adapter import (PLAYER_NAMES, PLAYER_REGIONS, decimate, find_duplicate_frames,  # noqa: E402
                          load_run, padding_period, with_tracknet)
from hits.badminton_ai.attribute import LOOKBACK, attribute_strokes  # noqa: E402
from hits.badminton_ai.detect_shots import detect_shots  # noqa: E402


def _ingest(run, source: str | None, mode: str):
    if mode == "off":
        return None
    src = source or run.source_video
    if not src or not os.path.exists(src):
        print(f"Ingest check skipped: source video not found ({src}). Pass --source to enable it.")
        return None
    dups = find_duplicate_frames(src)
    period = padding_period(dups)
    share = len(dups) / max(1, run.n_frames)
    if period and share >= 0.05:
        dec = decimate(run, dups)
        print(f"Ingest: {len(dups)} of {run.n_frames} frames are copies of the previous frame "
              f"(every {period}th) -> {dec.true_fps:.2f} fps content in a {run.fps:g} fps file. "
              f"Detecting on the {len(dec.kept)} real frames.")
        return dec
    if dups:
        print(f"Ingest: {len(dups)} near-identical frames, not periodic -- left in place "
              f"(freeze frames or very slow play, not padding).")
    else:
        print("Ingest: no duplicated frames.")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Detect hits and attribute them to players from a Good-Badminton run.")
    ap.add_argument("run", help="Good-Badminton output directory (or its detections.jsonl)")
    ap.add_argument("--tracknet", metavar="BALL_CSV",
                    help="use this TrackNetV3 <clip>_ball.csv for the shuttle instead of Good-Badminton's")
    ap.add_argument("--sensitivity", type=float, default=1.0,
                    help="< 1 finds more hits and more false positives (default 1.0)")
    ap.add_argument("--no-alternation", action="store_true",
                    help="attribute by wrist distance only (no singles alternation rule)")
    ap.add_argument("--source", help="source video for the ingest check (default: path in metadata.json)")
    ap.add_argument("--decimate", choices=["auto", "off"], default="auto",
                    help="drop padded duplicate frames before detection (default auto)")
    ap.add_argument("--video", help="video to draw on (default: the run's detect_<name>.mp4)")
    ap.add_argument("--no-video", action="store_true", help="skip the annotated video")
    ap.add_argument("--no-montage", action="store_true", help="skip hits_verify.jpg")
    a = ap.parse_args()

    run = load_run(a.run)
    if a.tracknet:
        run, warn = with_tracknet(run, a.tracknet)
        print(f"Shuttle: TrackNetV3 ({a.tracknet}); wrists: Good-Badminton")
        if warn:
            print("WARNING:", warn)
    else:
        print("Shuttle and wrists: Good-Badminton (pass --tracknet <clip>_ball.csv if you have it)")
    dec = _ingest(run, a.source, a.decimate)
    work = dec.run if dec else run
    to_orig = dec.to_original if dec else (lambda f: f)

    res = detect_shots(work.positions, work.fps, work.width, work.height,
                       cut_frames=work.cut_frames, sensitivity=a.sensitivity)
    attributed = attribute_strokes(res.events, work.poses, work.positions, work.diagonal, work.fps,
                                   cut_frames=work.cut_frames, alternation=not a.no_alternation)

    events, contacts = [], []
    for e in attributed:
        contact = e.frame_index if e.inferred else max(0, e.frame_index - LOOKBACK)
        f = to_orig(e.frame_index)
        events.append(replace(e, frame_index=f, time_s=f / run.fps))
        contacts.append(to_orig(contact))
    rejected = [replace(c, frame_index=to_orig(c.frame_index)) for c in res.rejected]
    s = dict(res.stats)
    s["dropouts"] = [(to_orig(x), to_orig(y)) for x, y in s.get("dropouts", [])]
    if dec:
        s["decimated"] = {"dropped_frames": dec.dropped, "true_fps": dec.true_fps}

    print(f"\nShuttle tracked in {s['tracked_fraction']:.0%} of {s['frames']} frames | "
          f"{s.get('runs', 0)} usable runs ({s.get('dropped_runs', 0)} dropped) | "
          f"{len(run.cut_frames)} cuts | {run.width}x{run.height} @ {work.fps:.4g} fps")
    if s["dropouts"]:
        print("Dropouts >= 5 frames:", ", ".join(f"{x}-{y}" for x, y in s["dropouts"]))
    if "warning" in s:
        print("WARNING:", s["warning"])

    print(f"\n{len(events)} hits  (0-based frames of the original video)")
    if events:
        print(f"{'#':>3} {'frame':>6} {'contact':>7} {'time':>7}  {'player':<12} {'how':<12} {'score':>5}  "
              f"{'detector':<20} {'conf':>4}")
        for n, (e, c) in enumerate(zip(events, contacts), 1):
            score = f"{e.attribution_score:.2f}" if e.attribution_score is not None else "-"
            print(f"{n:>3} {e.frame_index:>6} {c:>7} {e.time_s:>6.2f}s  {PLAYER_NAMES[e.player_id]:<12} "
                  f"{(e.attribution_method or '-'):<12} {score:>5}  "
                  f"{e.detector + (' *' if e.inferred else ''):<20} {e.confidence:>4.2f}")
        counts = {k: sum(1 for e in events if e.player_id == k) for k in (0, 1, None)}
        print(f"\nnear {counts[0]}, far {counts[1]}" + (f", unattributed {counts[None]}" if counts[None] else ""))
        if any(e.inferred for e in events):
            print("* inferred: the stroke happened while the shuttle was off-screen")
        weak = sum(1 for e in events if e.attribution_method == "elimination")
        if weak:
            print(f"{weak} attributed by elimination only (no wrist near the shuttle) -- weakest evidence.")
    if rejected:
        print(f"\nRejected candidates ({len(rejected)}):")
        for c in rejected:
            print(f"  f{c.frame_index:<5} {c.detector:<10} {c.reason} = {c.value:.2f}")

    out_json = os.path.join(run.run_dir, "hits.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump({"frame_base": 0,
                   "players": {"0": "near (Good-Badminton 'lower')", "1": "far (Good-Badminton 'upper')"},
                   "events": [{**e.to_dict(), "contact_frame": c, "player": PLAYER_NAMES[e.player_id],
                               "gb_region": PLAYER_REGIONS[e.player_id]} for e, c in zip(events, contacts)],
                   "rejected": [c.__dict__ for c in rejected],
                   "stats": s, "cut_frames": run.cut_frames,
                   "settings": {"sensitivity": a.sensitivity, "alternation": not a.no_alternation,
                                "decimate": a.decimate,
                                "shuttle_source": a.tracknet or "good-badminton"}},
                  fh, indent=2)
    out_csv = os.path.join(run.run_dir, "hits.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["hit", "frame", "contact_frame", "time_s", "player", "gb_region", "attribution_method",
                    "attribution_score", "detector", "confidence", "inferred", "x", "y"])
        for n, (e, c) in enumerate(zip(events, contacts), 1):
            w.writerow([n, e.frame_index, c, f"{e.time_s:.3f}", PLAYER_NAMES[e.player_id],
                        PLAYER_REGIONS[e.player_id] or "", e.attribution_method or "",
                        e.attribution_score if e.attribution_score is not None else "",
                        e.detector, e.confidence, e.inferred,
                        f"{e.x:.1f}" if e.x is not None else "", f"{e.y:.1f}" if e.y is not None else ""])
    written = [out_json, out_csv]

    video = a.video or os.path.join(run.run_dir, f"detect_{run.name}.mp4")
    if events and not (a.no_video and a.no_montage):
        if not os.path.exists(video):
            print(f"\nNo video at {video}; pass --video to draw on one.")
        else:
            import cv2
            from hits.render import write_montage, write_video
            cap = cv2.VideoCapture(video)
            n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            order = sorted(run.records)
            if a.video and abs(n_video - run.n_frames) <= 2 and abs(n_video - len(order)) > 2:
                order = list(range(run.n_frames))
            elif abs(n_video - len(order)) > 2:
                print(f"\nWARNING: {video} has {n_video} frames but the run has {len(order)} "
                      "court-view records; markers may be misplaced.")
            if not a.no_montage:
                m = os.path.join(run.run_dir, "hits_verify.jpg")
                write_montage(video, order, events, contacts, run.positions, m)
                written.append(m)
            if not a.no_video:
                v = os.path.join(run.run_dir, f"detect_{run.name}_hits.mp4")
                print("\nWriting annotated video...")
                write_video(video, order, events, contacts, run.positions, run.fps, v)
                written.append(v)

    print("\nWrote:\n  " + "\n  ".join(os.path.relpath(p) for p in written))
    if events and not a.no_montage:
        print("Check every tile in hits_verify.jpg before trusting the numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
