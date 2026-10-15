#!/usr/bin/env python3
import argparse
import csv
import json
import os
import statistics
import sys


def load_gb(path):
    records = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                records[int(rec["frame"]) - 1] = rec
    return records


def total_frames(jsonl_path, records):
    meta = os.path.join(os.path.dirname(jsonl_path), "metadata.json")
    try:
        with open(meta, encoding="utf-8") as fh:
            return int(json.load(fh)["video"]["total_frames"])
    except (OSError, KeyError, ValueError):
        return max(records) + 1 if records else 0


def gb_shuttle(rec):
    if rec is None:
        return None
    img = (rec.get("shuttlecock") or {}).get("image")
    if not img or (float(img[0]) == 0 and float(img[1]) == 0):
        return None
    return float(img[0]), float(img[1])


def player_at(rec, half):
    p = (rec or {}).get("players", {}).get(half) or {}
    return p if p.get("image") is not None else None


def load_tracknet(path):
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            frame = int(float(row["Frame"]))
            vis = float(row.get("Visibility", 0) or 0) >= 0.5
            x, y = float(row.get("X", 0) or 0), float(row.get("Y", 0) or 0)
            out[frame] = (x, y) if vis and not (x == 0 and y == 0) else None
    return out


def gaps(present, n, min_gap):
    runs, start = [], None
    for f in range(n):
        if not present(f):
            if start is None:
                start = f
        elif start is not None:
            if f - start >= min_gap:
                runs.append((start, f - 1))
            start = None
    if start is not None and n - start >= min_gap:
        runs.append((start, n - 1))
    return runs


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "n/a"


def fmt_runs(runs):
    return ", ".join(f"{a}-{b}" for a, b in runs) if runs else "none"


def main():
    ap = argparse.ArgumentParser(description="Summarise a Good-Badminton run, optionally against a "
                                             "TrackNetV3 trajectory. Frames are 0-based.")
    ap.add_argument("detections", help="Good-Badminton detections.jsonl")
    ap.add_argument("--tracknet", help="TrackNetV3 <stem>_ball.csv to compare against")
    ap.add_argument("--min-gap", type=int, default=5, help="shortest gap to report (default 5)")
    ap.add_argument("--frames", help="print per-frame detail for a 0-based range, e.g. 60-66")
    args = ap.parse_args()

    gb = load_gb(args.detections)
    n = total_frames(args.detections, gb)
    analysed = len(gb)
    both = sum(1 for r in gb.values() if player_at(r, "upper") and player_at(r, "lower"))
    ball = {f for f, r in gb.items() if gb_shuttle(r)}
    unanalysed = gaps(lambda f: f in gb, n, 1)

    print(f"Good-Badminton  {args.detections}")
    print(f"  video frames          {n}")
    print(f"  analysed (court view) {analysed}  ({pct(analysed, n)})"
          + (f"   skipped: {fmt_runs(unanalysed)}" if analysed < n else ""))
    print(f"  both players found    {both}  ({pct(both, analysed)} of analysed)")
    print(f"  shuttle present       {len(ball)}  ({pct(len(ball), n)} of all frames)")
    print(f"  gaps >= {args.min_gap}           {fmt_runs(gaps(lambda f: f in ball, n, args.min_gap))}")

    if args.tracknet:
        tn = load_tracknet(args.tracknet)
        tn_ball = {f for f, p in tn.items() if p}
        n_cmp = max(n, max(tn) + 1 if tn else 0)
        both_see = ball & tn_ball
        only_gb = ball - tn_ball
        only_tn = tn_ball - ball
        union = ball | tn_ball
        tn_missing = set(range(n_cmp)) - tn_ball

        print(f"\nTrackNetV3      {args.tracknet}")
        print(f"  shuttle present       {len(tn_ball)}  ({pct(len(tn_ball), n_cmp)})")
        print(f"  gaps >= {args.min_gap}           {fmt_runs(gaps(lambda f: f in tn_ball, n_cmp, args.min_gap))}")

        print("\nComparison (0-based frames)")
        print(f"  both see it           {len(both_see)}")
        print(f"  only TrackNetV3       {len(only_tn)}")
        print(f"  only Good-Badminton   {len(only_gb)}")
        print(f"  union                 {len(union)}  ({pct(len(union), n_cmp)})"
              f"   vs TrackNetV3 alone {pct(len(tn_ball), n_cmp)}")
        print(f"  TrackNet misses that GB fills: {len(tn_missing & ball)} of {len(tn_missing)}")
        print(f"  gaps left in the union: {fmt_runs(gaps(lambda f: f in union, n_cmp, args.min_gap))}")

        if both_see:
            d = sorted(((gb_shuttle(gb[f])[0] - tn[f][0]) ** 2 +
                        (gb_shuttle(gb[f])[1] - tn[f][1]) ** 2) ** 0.5 for f in both_see)
            p90 = d[int(0.9 * (len(d) - 1))]
            far = sum(1 for x in d if x > 30)
            print(f"\n  position agreement where both see it:"
                  f" median {statistics.median(d):.1f} px, p90 {p90:.1f} px,"
                  f" {far} frames > 30 px apart")
            print("  -> If the median is more than a few px, the two are often tracking")
            print("     different things; don't trust 'GB fills' until that's explained.")

    if args.frames:
        a, b = (int(x) for x in args.frames.split("-"))
        print(f"\nPer-frame detail {a}-{b} (0-based; Good-Badminton frame = +1)")
        print("  frame  upper(px)          lower(px)          shuttle(px)")
        for f in range(a, b + 1):
            r = gb.get(f)
            if r is None:
                print(f"  {f:5d}  -- not analysed (non-court frame) --")
                continue
            def p(pl):
                return f"({pl['image'][0]:.0f},{pl['image'][1]:.0f})" if pl else "MISSING"
            s = gb_shuttle(r)
            print(f"  {f:5d}  {p(player_at(r, 'upper')):18s} {p(player_at(r, 'lower')):18s} "
                  f"{f'({s[0]:.0f},{s[1]:.0f})' if s else '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
